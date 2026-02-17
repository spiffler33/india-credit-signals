# WHY THIS: Haiku is cheap but makes mistakes on borderline cases. This audit
# re-labels a targeted subset with Sonnet, then merges. Two modes:
#   --targeted (default): Only low-confidence + parse errors + 300 stratified
#     sample. The 300 sample measures Haiku↔Sonnet agreement. If >90%, Haiku
#     is trustworthy. Cost: ~$2 vs ~$58 for full audit.
#   --full: Re-label ALL credit-relevant articles (~9K). Thorough but expensive.
# This Calibrate→Bulk→Audit pattern is standard in ML data pipelines.

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from loguru import logger

from src.data.label_models import ArticleLabel, get_completed_urls, read_labels_jsonl


# --- Constants ---
CONFIG_PATH = Path("configs/labeling_config.yaml")
LABELING_CSV = Path("data/processed/gdelt_for_labeling.csv")
TARGETED_SAMPLE_SIZE = 300
TARGETED_SEED = 42  # 🎓 Fixed seed = reproducible random sampling


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --- SELECT: Identify articles needing Sonnet re-check ---

def select_audit_candidates(config: dict) -> list[str]:
    """Read bulk labels, return ALL "interesting" URLs for full audit.

    🎓 WHY these criteria: We audit any article that is "interesting" —
    credit-relevant, has a directional signal, low confidence, or had a
    parse error. Articles Haiku marked as not-credit-relevant with high
    confidence are probably correct and don't need a second opinion.
    """
    bulk_path = Path(config["paths"]["bulk_labels"])
    labels = read_labels_jsonl(bulk_path)

    if not labels:
        logger.error(f"No bulk labels found at {bulk_path}. Run bulk labeling first.")
        return []

    candidates: list[str] = []
    for label in labels:
        needs_audit = (
            label.credit_relevant == 1
            or label.signal_direction != 0
            or label.confidence == "low"
            or label.parse_error is not None
        )
        if needs_audit:
            candidates.append(label.url)

    logger.info(f"Full audit candidates: {len(candidates)} / {len(labels)} total labels")
    return candidates


def select_targeted_candidates(config: dict) -> tuple[list[str], dict[str, str]]:
    """Select a small, targeted audit subset. Returns (urls, url→reason_map).

    🎓 WHY targeted: Full audit re-labels ~9K articles ($58). Targeted picks
    only ~313: all low-confidence, all parse errors, plus a stratified 300
    sample from credit-relevant articles. The 300 sample is a STATISTICAL
    VALIDATION — it measures Haiku↔Sonnet agreement rate. If >90%, we trust
    Haiku on the remaining 9K. If <90%, we expand the audit.

    The stratified sampling ensures we test every signal_type, not just the
    dominant asset_quality bucket.
    """
    bulk_path = Path(config["paths"]["bulk_labels"])
    labels = read_labels_jsonl(bulk_path)

    if not labels:
        logger.error(f"No bulk labels found at {bulk_path}. Run bulk labeling first.")
        return [], {}

    # Categorize labels
    low_conf: list[ArticleLabel] = []
    parse_errors: list[ArticleLabel] = []
    credit_pool: list[ArticleLabel] = []  # credit_relevant=1 AND confidence != low

    for label in labels:
        if label.parse_error is not None:
            parse_errors.append(label)
        elif label.confidence == "low":
            low_conf.append(label)
        elif label.credit_relevant == 1:
            credit_pool.append(label)

    # Stratified sample from credit pool by signal_type
    # 🎓 Stratified sampling: instead of random 300 from 9K (which would over-
    # represent asset_quality at 30%), we sample proportionally from each
    # signal_type. This way we validate Haiku's accuracy across ALL categories.
    rng = random.Random(TARGETED_SEED)
    by_type: dict[str, list[ArticleLabel]] = defaultdict(list)
    for label in credit_pool:
        by_type[label.signal_type].append(label)

    sample_size = min(TARGETED_SAMPLE_SIZE, len(credit_pool))
    sampled: list[ArticleLabel] = []

    if credit_pool:
        # Proportional allocation per signal_type
        type_counts = {t: len(lst) for t, lst in by_type.items()}
        total_pool = sum(type_counts.values())
        allocation: dict[str, int] = {}
        remainder_pool: list[tuple[str, float]] = []

        for stype, count in type_counts.items():
            exact = sample_size * count / total_pool
            allocation[stype] = int(exact)
            remainder_pool.append((stype, exact - int(exact)))

        # Distribute remaining slots by largest fractional remainder
        remaining = sample_size - sum(allocation.values())
        remainder_pool.sort(key=lambda x: x[1], reverse=True)
        for i in range(remaining):
            allocation[remainder_pool[i][0]] += 1

        for stype, n in allocation.items():
            pool = by_type[stype]
            rng.shuffle(pool)
            sampled.extend(pool[:n])

        logger.info(
            f"Stratified sample allocation: "
            + ", ".join(f"{t}={n}" for t, n in sorted(allocation.items()))
        )

    # Build URL → reason map for merge step
    url_reason: dict[str, str] = {}
    for label in low_conf:
        url_reason[label.url] = "low_conf"
    for label in parse_errors:
        url_reason[label.url] = "parse_error"
    for label in sampled:
        if label.url not in url_reason:  # Don't overwrite low_conf/parse_error
            url_reason[label.url] = "sample"

    urls = list(url_reason.keys())
    logger.info(
        f"Targeted audit: {len(urls)} total | "
        f"{len(low_conf)} low-conf | {len(parse_errors)} parse-errors | "
        f"{len(sampled)} stratified sample"
    )
    return urls, url_reason


def write_audit_candidates_csv(
    candidate_urls: list[str], config: dict,
    url_reasons: dict[str, str] | None = None,
) -> Path:
    """Write audit candidate articles to CSV (for label_articles.py --phase audit).

    Reads the full article data from the labeling CSV, filters to candidates.
    Optionally includes audit_reason column for targeted audit tracking.
    """
    output_path = Path(config["paths"]["audit_candidates"])
    candidate_set = set(candidate_urls)

    # Read original articles to get full text
    articles: list[dict[str, str]] = []
    with open(LABELING_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["article_url"] in candidate_set:
                if url_reasons:
                    row["audit_reason"] = url_reasons.get(row["article_url"], "unknown")
                articles.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "article_url", "article_title", "article_date", "source_domain",
        "gdelt_tone", "entities", "rating_windows", "article_text", "source_bucket",
    ]
    if url_reasons:
        fieldnames.append("audit_reason")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for a in articles:
            writer.writerow(a)

    logger.info(f"Wrote {len(articles)} audit candidates to {output_path}")
    return output_path


# --- RUN: handled by label_articles.py --phase audit ---


# --- MERGE: Combine Haiku bulk + Sonnet audit into final labels ---

AGREEMENT_THRESHOLD = 0.90  # 🎓 If Haiku↔Sonnet agree >90% on sample, Haiku is trustworthy


def merge_labels(config: dict, targeted: bool = False) -> tuple[list[ArticleLabel], dict]:
    """Merge bulk (Haiku) and audit (Sonnet) labels.

    🎓 WHY two merge modes:
    - Full audit: Sonnet overrides Haiku on ALL audited articles.
    - Targeted audit: Sonnet overrides only low-conf/parse-error articles.
      The 300 "sample" articles are used to MEASURE agreement — if >90%,
      we keep Haiku's labels for all non-audited articles. The sample labels
      are still used (Sonnet's version) since we already paid for them.

    Returns (final_labels, stats_dict) for reporting.
    """
    bulk_path = Path(config["paths"]["bulk_labels"])
    audit_path = Path(config["paths"]["audit_labels"])
    candidates_path = Path(config["paths"]["audit_candidates"])

    bulk_labels = read_labels_jsonl(bulk_path)
    audit_labels = read_labels_jsonl(audit_path)

    # Build URL → label maps
    bulk_map: dict[str, ArticleLabel] = {l.url: l for l in bulk_labels}
    audit_map: dict[str, ArticleLabel] = {l.url: l for l in audit_labels}

    # Load audit reasons if targeted mode (from candidates CSV)
    url_reasons: dict[str, str] = {}
    if targeted and candidates_path.exists():
        with open(candidates_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "audit_reason" in (reader.fieldnames or []):
                for row in reader:
                    url_reasons[row["article_url"]] = row.get("audit_reason", "unknown")

    # Agreement tracking — split by reason category
    stats: dict[str, int] = {
        "total_audited": 0,
        "sample_agree": 0, "sample_disagree": 0,
        "low_conf_agree": 0, "low_conf_disagree": 0,
        "parse_error_count": 0,
        "sonnet_overrides": 0,
    }
    # Track detailed disagreements for inspection
    disagreement_examples: list[dict] = []

    final: list[ArticleLabel] = []

    for url, haiku_label in bulk_map.items():
        if url in audit_map:
            sonnet_label = audit_map[url]
            stats["total_audited"] += 1
            reason = url_reasons.get(url, "full")
            agrees = haiku_label.credit_relevant == sonnet_label.credit_relevant

            if reason == "sample":
                if agrees:
                    stats["sample_agree"] += 1
                else:
                    stats["sample_disagree"] += 1
                    disagreement_examples.append({
                        "url": url,
                        "reason": reason,
                        "haiku_cr": haiku_label.credit_relevant,
                        "sonnet_cr": sonnet_label.credit_relevant,
                        "haiku_dir": haiku_label.signal_direction,
                        "sonnet_dir": sonnet_label.signal_direction,
                        "haiku_type": haiku_label.signal_type,
                        "sonnet_type": sonnet_label.signal_type,
                        "haiku_reasoning": haiku_label.reasoning[:100],
                        "sonnet_reasoning": sonnet_label.reasoning[:100],
                    })
            elif reason == "low_conf":
                if agrees:
                    stats["low_conf_agree"] += 1
                else:
                    stats["low_conf_disagree"] += 1
            elif reason == "parse_error":
                stats["parse_error_count"] += 1

            # Override logic:
            # - low_conf and parse_error: ALWAYS use Sonnet
            # - sample: use Sonnet (we already paid for it)
            # - full mode: Sonnet always wins
            stats["sonnet_overrides"] += 1
            final.append(sonnet_label)
        else:
            # Not audited — Haiku label stands
            final.append(haiku_label)

    # Compute agreement rate on the 300 sample
    sample_total = stats["sample_agree"] + stats["sample_disagree"]
    if sample_total > 0:
        agree_pct = stats["sample_agree"] / sample_total * 100
        stats["sample_agreement_pct"] = agree_pct
        logger.info(
            f"Sample agreement (credit_relevant): {stats['sample_agree']}/{sample_total} "
            f"({agree_pct:.1f}%) — threshold: {AGREEMENT_THRESHOLD*100:.0f}%"
        )
        if agree_pct >= AGREEMENT_THRESHOLD * 100:
            logger.info("✅ Agreement above threshold — Haiku labels are trustworthy")
        else:
            logger.warning(
                f"⚠️ Agreement below {AGREEMENT_THRESHOLD*100:.0f}% — "
                f"consider expanding audit to full set"
            )

    stats["disagreement_examples"] = disagreement_examples  # type: ignore[assignment]

    total_audited = stats["total_audited"]
    if total_audited > 0:
        logger.info(
            f"Merge: {total_audited} audited | "
            f"{stats['sonnet_overrides']} Sonnet overrides"
        )

    return final, stats


def write_final_labels(labels: list[ArticleLabel], config: dict) -> None:
    """Write merged labels to both JSONL (for training) and CSV (for review)."""
    jsonl_path = Path(config["paths"]["final_labels_jsonl"])
    csv_path = Path(config["paths"]["final_labels_csv"])

    # JSONL
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for label in labels:
            f.write(json.dumps(label.to_dict(), ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(labels)} labels to {jsonl_path}")

    # CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "url", "credit_relevant", "signal_direction", "signal_type",
        "sector_wide", "confidence", "reasoning", "model", "phase", "parse_error",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for label in labels:
            d = label.to_dict()
            # Don't write raw_response to CSV (too large)
            writer.writerow({k: d.get(k, "") for k in fieldnames})
    logger.info(f"Wrote {len(labels)} labels to {csv_path}")


def print_summary(
    labels: list[ArticleLabel], bulk_count: int, audit_count: int,
    stats: dict | None = None,
) -> None:
    """Print final merge summary per OUTPUT_STYLE.md."""
    total = len(labels)
    credit = sum(1 for l in labels if l.credit_relevant == 1)
    neg = sum(1 for l in labels if l.signal_direction == -1)
    pos = sum(1 for l in labels if l.signal_direction == 1)
    errors = sum(1 for l in labels if l.parse_error)

    type_counts = Counter(l.signal_type for l in labels if l.credit_relevant == 1)
    model_counts = Counter(l.model for l in labels)

    print(f"\n┌──────────────────────────────────────────────────────────┐")
    print(f"│ Final Merged Labels                                      │")
    print(f"├──────────────────────────────────────────────────────────┤")
    print(f"│ Total labels:            {total:>7}                        │")
    print(f"│ From Haiku (bulk):       {bulk_count:>7}                        │")
    print(f"│ Overridden by Sonnet:    {audit_count:>7}                        │")
    print(f"│ Credit-relevant:         {credit:>7}  ({credit/total*100:>5.1f}%)              │")
    print(f"│ Parse errors remaining:  {errors:>7}                        │")
    print(f"├──────────────────────────────────────────────────────────┤")
    print(f"│ Signal direction (credit-relevant only):                 │")
    print(f"│   Deterioration (-1):    {neg:>7}                        │")
    print(f"│   Improvement (+1):      {pos:>7}                        │")
    print(f"├──────────────────────────────────────────────────────────┤")
    print(f"│ Signal types:                                            │")
    for stype, count in type_counts.most_common():
        print(f"│   {stype:<24} {count:>5}                          │")
    print(f"├──────────────────────────────────────────────────────────┤")
    print(f"│ Model provenance:                                        │")
    for model, count in model_counts.most_common():
        name = model.split("-")[1] if "-" in model else model
        print(f"│   {name:<24} {count:>5}                          │")
    print(f"└──────────────────────────────────────────────────────────┘")

    # Print targeted audit agreement stats if available
    if stats and "sample_agreement_pct" in stats:
        sample_total = stats["sample_agree"] + stats["sample_disagree"]
        agree_pct = stats["sample_agreement_pct"]
        threshold = AGREEMENT_THRESHOLD * 100

        print(f"\n┌──────────────────────────────────────────────────────────┐")
        print(f"│ Haiku ↔ Sonnet Agreement (300 Stratified Sample)         │")
        print(f"├──────────────────────────────────────────────────────────┤")
        print(f"│ Sample size:             {sample_total:>7}                        │")
        print(f"│ Agree (credit_relevant): {stats['sample_agree']:>7}  ({agree_pct:>5.1f}%)              │")
        print(f"│ Disagree:                {stats['sample_disagree']:>7}  ({100-agree_pct:>5.1f}%)              │")
        print(f"│ Threshold:               {threshold:>5.0f}%                         │")
        verdict = "PASS — Haiku labels trustworthy" if agree_pct >= threshold else "FAIL — consider full audit"
        print(f"│ Verdict:  {verdict:<46}│")
        print(f"├──────────────────────────────────────────────────────────┤")
        print(f"│ Low-confidence overrides: {stats.get('low_conf_agree', 0) + stats.get('low_conf_disagree', 0):>5}                       │")
        print(f"│ Parse error overrides:    {stats.get('parse_error_count', 0):>5}                       │")
        print(f"└──────────────────────────────────────────────────────────┘")

        # Show disagreement examples
        examples = stats.get("disagreement_examples", [])
        if examples:
            print(f"\n🔍 Disagreement Examples (Haiku vs Sonnet):")
            for i, ex in enumerate(examples[:10], 1):
                print(f"\n  [{i}] {ex['url'][:70]}...")
                print(f"      Haiku: cr={ex['haiku_cr']} dir={ex['haiku_dir']} type={ex['haiku_type']}")
                print(f"             {ex['haiku_reasoning']}")
                print(f"      Sonnet: cr={ex['sonnet_cr']} dir={ex['sonnet_dir']} type={ex['sonnet_type']}")
                print(f"              {ex['sonnet_reasoning']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 3 audit: select candidates, run Sonnet re-labeling, merge"
    )
    parser.add_argument(
        "command", choices=["select", "run", "merge"],
        help="select: identify audit candidates | run: label with Sonnet | merge: combine labels",
    )
    parser.add_argument(
        "--config", type=Path, default=CONFIG_PATH,
        help=f"Path to labeling config (default: {CONFIG_PATH})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print first 3 prompts without making API calls (for 'run' command)",
    )
    parser.add_argument(
        "--targeted", action="store_true", default=True,
        help="Targeted audit: low-conf + parse-errors + 300 stratified sample (default)",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Full audit: re-label ALL credit-relevant articles (~9K, ~$58)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    targeted = not args.full  # Default to targeted unless --full is passed

    if args.command == "select":
        if targeted:
            logger.info("Phase 1.3 Step 4c: Targeted audit candidate selection")
            urls, url_reasons = select_targeted_candidates(config)
            if urls:
                write_audit_candidates_csv(urls, config, url_reasons=url_reasons)
                reasons = Counter(url_reasons.values())
                print(f"\n✅ DONE: Selected {len(urls)} targeted audit candidates")
                for reason, count in reasons.most_common():
                    print(f"   {reason}: {count}")
                print(f"⏭️  NEXT: python -m src.data.label_audit run")
            else:
                print("❌ No candidates found. Run bulk labeling first.")
        else:
            logger.info("Phase 1.3 Step 4c: Full audit candidate selection")
            urls = select_audit_candidates(config)
            if urls:
                write_audit_candidates_csv(urls, config)
                print(f"\n✅ DONE: Selected {len(urls)} audit candidates")
                print(f"⏭️  NEXT: python -m src.data.label_audit run")
            else:
                print("❌ No candidates found. Run bulk labeling first.")

    elif args.command == "run":
        logger.info("Phase 1.3 Step 4c: Audit labeling with Sonnet")
        # Reuse label_articles.py engine
        from src.data.label_articles import label_batch, load_articles_for_phase, print_summary as lab_summary

        model = config["models"]["audit"]
        articles, output_path = load_articles_for_phase("audit", config)
        logger.info(f"Audit: {len(articles)} candidates | model={model}")

        labels = asyncio.run(
            label_batch(articles, config, model, "audit", output_path, dry_run=args.dry_run)
        )
        if labels:
            lab_summary(labels, "audit")
            print(f"\n✅ DONE: Audit-labeled {len(labels)} articles")
            print(f"⏭️  NEXT: python -m src.data.label_audit merge")

    elif args.command == "merge":
        logger.info("Phase 1.3 Step 4c: Merging Haiku + Sonnet labels")
        final, stats = merge_labels(config, targeted=targeted)
        write_final_labels(final, config)

        # Count how many are from each source
        audit_labels = read_labels_jsonl(Path(config["paths"]["audit_labels"]))
        audit_urls = {l.url for l in audit_labels}
        from_haiku = sum(1 for l in final if l.url not in audit_urls)
        from_sonnet = sum(1 for l in final if l.url in audit_urls)

        print_summary(final, from_haiku, from_sonnet, stats=stats)
        print(f"\n✅ DONE: Merged {len(final)} final labels")
        print(f"📁 JSONL: {config['paths']['final_labels_jsonl']}")
        print(f"📁 CSV:   {config['paths']['final_labels_csv']}")
        print(f"\n🎓 KEY CONCEPT: The CSV is for your manual review in Excel/Sheets.")
        print(f"   The JSONL is what the training pipeline reads in Phase 2.")
        print(f"\n⏭️  NEXT: Spot-check 50 labels against rating_windows ground truth,")
        print(f"   then move to Phase 2 (training data formatting).")
