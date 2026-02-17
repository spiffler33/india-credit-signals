# WHY THIS: Async LLM labeling engine shared by all three phases (calibration,
# bulk, audit). Uses asyncio + semaphore for controlled concurrency — 10
# parallel API calls gives ~10x throughput vs sequential, while staying under
# Anthropic rate limits. Progressive JSONL writes mean a crash at article 8,000
# loses zero work (resume picks up from where it stopped).

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from loguru import logger

# 🎓 load_dotenv() reads .env file and sets OS environment variables.
# The anthropic SDK then picks up ANTHROPIC_API_KEY automatically.
# Without this, Python ignores .env files entirely — they're a convention,
# not a language feature.
load_dotenv()

from src.data.label_models import (
    ArticleLabel,
    get_completed_urls,
    parse_llm_response,
    write_label_jsonl,
)

# --- Constants ---
CONFIG_PATH = Path("configs/labeling_config.yaml")
LABELING_CSV = Path("data/processed/gdelt_for_labeling.csv")


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_articles_for_phase(
    phase: str, config: dict
) -> tuple[list[dict[str, str]], Path]:
    """Load the right CSV and output path for the given phase.

    🎓 WHY separate input sources: Calibration reads the 300-article sample,
    bulk reads all 17K. Same labeling engine, different input/output.
    """
    if phase == "calibration":
        input_path = Path(config["paths"]["calibration_sample"])
        output_path = Path(config["paths"]["calibration_labels"])
    elif phase == "bulk":
        input_path = LABELING_CSV
        output_path = Path(config["paths"]["bulk_labels"])
    elif phase == "audit":
        # Audit uses its own candidate list — handled by label_audit.py
        input_path = Path(config["paths"]["audit_candidates"])
        output_path = Path(config["paths"]["audit_labels"])
    else:
        raise ValueError(f"Unknown phase: {phase}. Use calibration, bulk, or audit.")

    with open(input_path, "r", encoding="utf-8") as f:
        articles = list(csv.DictReader(f))
    return articles, output_path


def build_prompt(
    article: dict[str, str], config: dict
) -> tuple[str, str]:
    """Build (system_message, user_message) for one article.

    🎓 WHY separate system/user: Anthropic's API treats system messages as
    persistent context (the analyst persona) and user messages as the specific
    task. This separation improves instruction following.

    CRITICAL: We never include rating_windows in the prompt — that's held-back
    ground truth for evaluation. Including it would be data leakage.
    """
    prompt_cfg = config["prompt"]
    max_chars = config["text_truncation"]["max_chars"]

    # System message with optional few-shot examples
    system = prompt_cfg["system"].strip()
    few_shot = prompt_cfg.get("few_shot_examples", [])
    if few_shot:
        system += "\n\nHere are examples of correct labeling:\n"
        for i, ex in enumerate(few_shot, 1):
            system += f"\nExample {i}:\nInput: {ex['input']}\nOutput: {json.dumps(ex['output'])}\n"

    # User message with article content
    text = article.get("article_text", "")
    if len(text) > max_chars:
        text = text[:max_chars] + "... [truncated]"

    # Get first entity from comma-separated list
    entities = article.get("entities", "Unknown")

    user = prompt_cfg["user_template"].format(
        entity=entities,
        title=article.get("article_title", "No title"),
        date=article.get("article_date", "Unknown"),
        source=article.get("source_domain", "Unknown"),
        text=text if text else "No article text available.",
    )

    return system, user


async def label_one_article(
    article: dict[str, str],
    config: dict,
    model: str,
    phase: str,
    semaphore: asyncio.Semaphore,
    client: "anthropic.AsyncAnthropic",  # type: ignore[name-defined]
) -> ArticleLabel:
    """Label a single article via the Anthropic API.

    🎓 WHY semaphore: Without it, asyncio would fire ALL 17K requests at once
    and get rate-limited hard. The semaphore acts as a bouncer — only 10
    requests in-flight at a time. When one finishes, the next one enters.
    """
    system_msg, user_msg = build_prompt(article, config)
    url = article.get("article_url", "")
    api_cfg = config["api"]

    async with semaphore:
        for attempt in range(api_cfg["max_retries"]):
            try:
                response = await client.messages.create(
                    model=model,
                    max_tokens=300,
                    system=system_msg,
                    messages=[{"role": "user", "content": user_msg}],
                )
                raw = response.content[0].text
                label = parse_llm_response(raw, url=url, model=model, phase=phase)
                return label

            except Exception as e:
                error_name = type(e).__name__
                # Retry on rate limit (429) and server errors (500+)
                if "rate" in error_name.lower() or "overloaded" in str(e).lower() or attempt < api_cfg["max_retries"] - 1:
                    delay = api_cfg["retry_base_delay"] * (2 ** attempt)
                    logger.warning(
                        f"Retry {attempt + 1}/{api_cfg['max_retries']} for {url[:60]}... "
                        f"({error_name}: {str(e)[:80]}). Waiting {delay:.0f}s."
                    )
                    await asyncio.sleep(delay)
                else:
                    # Final attempt failed — return label with error
                    label = ArticleLabel(
                        url=url, model=model, phase=phase,
                        parse_error=f"API error after {api_cfg['max_retries']} attempts: {error_name}: {str(e)[:200]}",
                    )
                    return label

    # Should not reach here, but safety net
    return ArticleLabel(url=url, model=model, phase=phase, parse_error="Max retries exhausted")


async def label_batch(
    articles: list[dict[str, str]],
    config: dict,
    model: str,
    phase: str,
    output_path: Path,
    dry_run: bool = False,
) -> list[ArticleLabel]:
    """Label all articles with controlled concurrency. Resumes from existing JSONL.

    🎓 WHY progressive writes: Each label is appended to JSONL immediately after
    the API returns. If the process crashes at article 8,000, the first 7,999
    labels are safe on disk. On restart, get_completed_urls() reads the JSONL
    and skips already-labeled articles. This is standard for long-running
    data pipelines — never buffer everything in memory.
    """
    if dry_run:
        # Print first 3 prompts and exit
        for article in articles[:3]:
            system_msg, user_msg = build_prompt(article, config)
            print(f"\n{'='*60}")
            print(f"URL: {article.get('article_url', '')[:80]}")
            print(f"{'='*60}")
            print(f"SYSTEM ({len(system_msg)} chars):")
            print(system_msg[:500] + "..." if len(system_msg) > 500 else system_msg)
            print(f"\nUSER ({len(user_msg)} chars):")
            print(user_msg[:500] + "..." if len(user_msg) > 500 else user_msg)
        print(f"\n🔍 Dry run complete. Would label {len(articles)} articles.")
        return []

    # Import here so --dry-run works without ANTHROPIC_API_KEY
    import anthropic

    # Resume support: skip already-labeled articles
    completed = get_completed_urls(output_path)
    remaining = [a for a in articles if a.get("article_url", "") not in completed]

    if completed:
        logger.info(f"Resuming: {len(completed)} already done, {len(remaining)} remaining")

    if not remaining:
        logger.info("All articles already labeled — nothing to do")
        return []

    api_cfg = config["api"]
    semaphore = asyncio.Semaphore(api_cfg["max_concurrent"])
    client = anthropic.AsyncAnthropic()  # Reads ANTHROPIC_API_KEY from env

    labels: list[ArticleLabel] = []
    errors = 0
    start_time = time.monotonic()

    # 🎓 asyncio.as_completed gives results as they finish (not in order).
    # This means the progress counter updates as fast as possible, and we
    # write to JSONL in completion order (which is fine for JSONL).
    tasks = {
        asyncio.ensure_future(
            label_one_article(article, config, model, phase, semaphore, client)
        ): article
        for article in remaining
    }

    done_count = 0
    total = len(remaining)

    for coro in asyncio.as_completed(tasks):
        label = await coro
        labels.append(label)
        write_label_jsonl(label, output_path)

        if label.parse_error:
            errors += 1

        done_count += 1
        elapsed = time.monotonic() - start_time
        rate = done_count / elapsed if elapsed > 0 else 0
        eta_sec = (total - done_count) / rate if rate > 0 else 0
        eta_min = eta_sec / 60

        # Progress line every 50 articles (or every article for small batches)
        if done_count % max(1, min(50, total // 20)) == 0 or done_count == total:
            pct = done_count / total * 100
            logger.info(
                f"[{phase.capitalize()}] {done_count}/{total} ({pct:.1f}%) | "
                f"{errors} errors | {rate:.1f}/sec | ETA: {eta_min:.0f} min"
            )

    await client.close()

    elapsed_total = time.monotonic() - start_time
    logger.info(
        f"Done: {len(labels)} labels in {elapsed_total:.0f}s | "
        f"{errors} parse errors ({errors/len(labels)*100:.1f}%)"
    )
    return labels


def print_summary(labels: list[ArticleLabel], phase: str) -> None:
    """Print labeling results per OUTPUT_STYLE.md."""
    total = len(labels)
    if total == 0:
        print("No labels to summarize.")
        return

    credit = sum(1 for l in labels if l.credit_relevant == 1)
    neg = sum(1 for l in labels if l.signal_direction == -1)
    pos = sum(1 for l in labels if l.signal_direction == 1)
    neutral = sum(1 for l in labels if l.signal_direction == 0)
    errors = sum(1 for l in labels if l.parse_error)

    from collections import Counter
    type_counts = Counter(l.signal_type for l in labels if l.credit_relevant == 1)
    conf_counts = Counter(l.confidence for l in labels)

    print(f"\n┌──────────────────────────────────────────────────────────┐")
    print(f"│ Labeling Results: Phase {phase.capitalize():<33}│")
    print(f"├──────────────────────────────────────────────────────────┤")
    print(f"│ Total labeled:           {total:>7}                        │")
    print(f"│ Credit-relevant:         {credit:>7}  ({credit/total*100:>5.1f}%)              │")
    print(f"│ Not credit-relevant:     {total-credit:>7}  ({(total-credit)/total*100:>5.1f}%)              │")
    print(f"│ Parse errors:            {errors:>7}  ({errors/total*100:>5.1f}%)              │")
    print(f"├──────────────────────────────────────────────────────────┤")
    print(f"│ Signal direction (credit-relevant only):                 │")
    print(f"│   Deterioration (-1):    {neg:>7}                        │")
    print(f"│   Neutral (0):           {neutral:>7}                        │")
    print(f"│   Improvement (+1):      {pos:>7}                        │")
    print(f"├──────────────────────────────────────────────────────────┤")
    print(f"│ Signal types:                                            │")
    for stype, count in type_counts.most_common():
        print(f"│   {stype:<24} {count:>5}                          │")
    print(f"├──────────────────────────────────────────────────────────┤")
    print(f"│ Confidence:                                              │")
    for conf in ["high", "medium", "low"]:
        c = conf_counts.get(conf, 0)
        print(f"│   {conf:<24} {c:>5}                          │")
    print(f"└──────────────────────────────────────────────────────────┘")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM-based article labeling for credit signal pipeline"
    )
    parser.add_argument(
        "--phase", required=True, choices=["calibration", "bulk", "audit"],
        help="Labeling phase: calibration (300, Sonnet), bulk (17K, Haiku), audit (5K, Sonnet)",
    )
    parser.add_argument(
        "--config", type=Path, default=CONFIG_PATH,
        help=f"Path to labeling config (default: {CONFIG_PATH})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print first 3 prompts without making API calls",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model = config["models"][args.phase]
    articles, output_path = load_articles_for_phase(args.phase, config)

    logger.info(
        f"Phase 1.3 Step 4: LLM labeling ({args.phase}) | "
        f"{len(articles)} articles | model={model}"
    )

    labels = asyncio.run(
        label_batch(articles, config, model, args.phase, output_path, dry_run=args.dry_run)
    )

    if labels:
        print_summary(labels, args.phase)
        print(f"\n✅ DONE: Labeled {len(labels)} articles ({args.phase} phase)")
        print(f"📁 Output: {output_path}")

        if args.phase == "calibration":
            print(f"\n⏭️  NEXT: Review labels in {output_path}")
            print(f"   Pick 8-10 good few-shot examples → add to configs/labeling_config.yaml")
            print(f"   Then run: python -m src.data.label_articles --phase bulk")
        elif args.phase == "bulk":
            print(f"\n⏭️  NEXT: Run audit selection + labeling:")
            print(f"   python -m src.data.label_audit select")
            print(f"   python -m src.data.label_audit run")
            print(f"   python -m src.data.label_audit merge")
