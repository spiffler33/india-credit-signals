# WHY THIS: Spot-check LLM labels against rating_windows ground truth.
# The model never saw rating_windows during labeling (would be data leakage).
# Now we check: does signal_direction align with actual rating outcomes?
# This is NOT a precision/recall eval (that's Phase 2.4) — it's a sanity check
# on label quality before we build training data.

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from src.data.label_models import ArticleLabel, read_labels_jsonl


# --- Data Structures ---

@dataclass
class RatingWindow:
    """One rating action near an article."""
    entity: str
    date: str
    action_type: str  # initial, upgrade, downgrade, default, watchlist, outlook
    outcome: str      # positive, neutral, negative
    days_before: int   # article published this many days BEFORE the rating action


@dataclass
class SpotCheckRow:
    """Joined label + article + rating window for review."""
    url: str
    title: str
    article_date: str
    entity: str
    label_direction: int        # model's signal_direction (-1, 0, +1)
    label_credit_relevant: int  # model's credit_relevant (0, 1)
    label_signal_type: str
    label_confidence: str
    label_reasoning: str
    ground_truth_outcome: str   # worst rating outcome (negative > neutral > positive)
    ground_truth_action: str    # action_type for worst outcome
    ground_truth_days: int      # days_before for worst outcome
    alignment: str              # correct, missed, false_alarm, irrelevant_correct


# --- Loading ---

def load_articles(path: Path) -> dict[str, dict]:
    """Load gdelt_for_labeling.csv into a URL-keyed dict."""
    articles: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row.get("article_url", "").strip()
            if url:
                articles[url] = row
    logger.info(f"Loaded {len(articles)} articles from {path.name}")
    return articles


def parse_rating_windows(raw: str) -> list[RatingWindow]:
    """Parse the rating_windows JSON string from CSV."""
    if not raw or raw.strip() in ("", "[]"):
        return []
    try:
        windows = json.loads(raw)
        return [
            RatingWindow(
                entity=w.get("entity", ""),
                date=w.get("date", ""),
                action_type=w.get("action_type", ""),
                outcome=w.get("outcome", "neutral"),
                days_before=int(w.get("days_before", 0)),
            )
            for w in windows
        ]
    except (json.JSONDecodeError, TypeError):
        return []


def worst_outcome(windows: list[RatingWindow], max_days: int = 90) -> RatingWindow | None:
    """Find the most negative rating window within max_days.

    🎓 WHY "worst": For credit early-warning, we care most about whether
    the model detected deterioration before a downgrade/default. If an entity
    had both an upgrade and a default within 6 months, the default is what matters.

    Priority: negative > neutral > positive. Within same outcome, prefer closest date.
    """
    # Filter to articles published BEFORE rating action (predictive window)
    predictive = [w for w in windows if 0 < w.days_before <= max_days]
    if not predictive:
        return None

    outcome_rank = {"negative": 0, "neutral": 1, "positive": 2}
    predictive.sort(key=lambda w: (outcome_rank.get(w.outcome, 1), w.days_before))
    return predictive[0]


# --- Analysis ---

def outcome_to_direction(outcome: str) -> int:
    """Map rating outcome to expected signal direction."""
    return {"negative": -1, "neutral": 0, "positive": 1}.get(outcome, 0)


def classify_alignment(label: ArticleLabel, window: RatingWindow) -> str:
    """Classify how well the label aligns with ground truth.

    🎓 This is NOT binary right/wrong. The categories:
    - correct: model direction matches rating outcome direction
    - missed: rating says negative, model said neutral/positive (missed a real signal)
    - false_alarm: model said negative, rating says neutral/positive (noise)
    - irrelevant_correct: both say neutral/no signal (boring but correct)
    """
    expected = outcome_to_direction(window.outcome)
    predicted = label.signal_direction

    if expected == -1 and predicted == -1:
        return "correct"          # Detected deterioration before downgrade
    elif expected == 1 and predicted == 1:
        return "correct"          # Detected improvement before upgrade
    elif expected == -1 and predicted >= 0:
        return "missed"           # Downgrade coming but model didn't flag it
    elif expected >= 0 and predicted == -1:
        return "false_alarm"      # Model flagged deterioration, but no downgrade
    elif expected == 0 and predicted == 0:
        return "irrelevant_correct"
    elif expected == 1 and predicted <= 0:
        return "missed_positive"  # Upgrade coming but model didn't flag improvement
    else:
        return "other"


def build_spotcheck(
    labels: list[ArticleLabel],
    articles: dict[str, dict],
    max_days: int = 90,
) -> list[SpotCheckRow]:
    """Join labels with articles, filter to those with rating windows in range."""
    rows: list[SpotCheckRow] = []

    for label in labels:
        article = articles.get(label.url)
        if not article:
            continue

        windows = parse_rating_windows(article.get("rating_windows", ""))
        window = worst_outcome(windows, max_days=max_days)
        if window is None:
            continue

        alignment = classify_alignment(label, window)

        rows.append(SpotCheckRow(
            url=label.url,
            title=article.get("article_title", "")[:100],
            article_date=article.get("article_date", ""),
            entity=window.entity,
            label_direction=label.signal_direction,
            label_credit_relevant=label.credit_relevant,
            label_signal_type=label.signal_type,
            label_confidence=label.confidence,
            label_reasoning=label.reasoning[:120],
            ground_truth_outcome=window.outcome,
            ground_truth_action=window.action_type,
            ground_truth_days=window.days_before,
            alignment=alignment,
        ))

    return rows


def sample_50(rows: list[SpotCheckRow], seed: int = 42) -> list[SpotCheckRow]:
    """Stratified sample of 50 rows for human review.

    🎓 WHY stratified: Random sampling would give mostly "irrelevant_correct" (boring).
    We want a balanced view — equal parts correct detections, misses, and false alarms.
    """
    by_alignment: dict[str, list[SpotCheckRow]] = defaultdict(list)
    for r in rows:
        by_alignment[r.alignment].append(r)

    rng = random.Random(seed)
    sample: list[SpotCheckRow] = []

    # Priority order: missed signals are most important to review
    targets = [
        ("missed", 15),           # Downgrades the model missed
        ("correct", 15),          # Downgrades the model caught
        ("false_alarm", 10),      # Model cried wolf
        ("missed_positive", 5),   # Upgrades the model missed
        ("irrelevant_correct", 5),  # Filler
    ]

    for alignment, n in targets:
        pool = by_alignment.get(alignment, [])
        take = min(n, len(pool))
        sample.extend(rng.sample(pool, take))

    # Fill remaining slots from any category
    remaining = 50 - len(sample)
    if remaining > 0:
        used_urls = {r.url for r in sample}
        leftovers = [r for r in rows if r.url not in used_urls]
        sample.extend(rng.sample(leftovers, min(remaining, len(leftovers))))

    return sample


# --- Reporting ---

def print_confusion_matrix(rows: list[SpotCheckRow]) -> None:
    """Print a confusion matrix: predicted direction vs ground truth outcome."""
    # 🎓 Confusion matrix rows = predicted, columns = actual
    # Standard ML convention: rows are what the model said, columns are reality
    matrix: dict[int, Counter] = {-1: Counter(), 0: Counter(), 1: Counter()}
    for r in rows:
        gt = outcome_to_direction(r.ground_truth_outcome)
        matrix[r.label_direction][gt] += 1

    gt_labels = {"negative": -1, "neutral": 0, "positive": 1}
    total = len(rows)

    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│         CONFUSION MATRIX: Model vs Rating Outcome          │")
    print("│         (articles with rating action within 90 days)        │")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│                    Ground Truth (Rating Action)             │")
    print("│  Predicted    │  Negative  │  Neutral   │  Positive  │ Tot │")
    print("├───────────────┼────────────┼────────────┼────────────┼─────┤")

    for pred in [-1, 0, 1]:
        pred_label = {-1: "Deterioration", 0: "Neutral     ", 1: "Improvement "}[pred]
        neg = matrix[pred][-1]
        neu = matrix[pred][0]
        pos = matrix[pred][1]
        row_total = neg + neu + pos
        print(f"│  {pred_label} │  {neg:>6}    │  {neu:>6}    │  {pos:>6}    │{row_total:>4} │")

    print("├───────────────┼────────────┼────────────┼────────────┼─────┤")
    neg_tot = sum(matrix[p][-1] for p in [-1, 0, 1])
    neu_tot = sum(matrix[p][0] for p in [-1, 0, 1])
    pos_tot = sum(matrix[p][1] for p in [-1, 0, 1])
    print(f"│  Total        │  {neg_tot:>6}    │  {neu_tot:>6}    │  {pos_tot:>6}    │{total:>4} │")
    print("└───────────────┴────────────┴────────────┴────────────┴─────┘")

    # Key metrics
    print()
    if neg_tot > 0:
        true_neg_detect = matrix[-1][-1]
        recall = true_neg_detect / neg_tot * 100
        print(f"  Deterioration recall:  {true_neg_detect}/{neg_tot} = {recall:.1f}%")
        print(f"    (Of articles near downgrades, how many did the model flag?)")

    pred_neg_total = matrix[-1][-1] + matrix[-1][0] + matrix[-1][1]
    if pred_neg_total > 0:
        precision = matrix[-1][-1] / pred_neg_total * 100
        print(f"  Deterioration precision: {matrix[-1][-1]}/{pred_neg_total} = {precision:.1f}%")
        print(f"    (Of articles model flagged as deterioration, how many had downgrades?)")


def print_alignment_summary(rows: list[SpotCheckRow]) -> None:
    """Print alignment category breakdown."""
    counts = Counter(r.alignment for r in rows)
    total = len(rows)

    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│           ALIGNMENT SUMMARY (within 90-day window)      │")
    print("├─────────────────────────────────────────────────────────┤")
    for cat, desc in [
        ("correct", "Model direction matches rating outcome"),
        ("missed", "Downgrade coming, model didn't flag"),
        ("false_alarm", "Model flagged deterioration, no downgrade"),
        ("missed_positive", "Upgrade coming, model didn't flag"),
        ("irrelevant_correct", "Both say neutral/no signal"),
        ("other", "Other combinations"),
    ]:
        n = counts.get(cat, 0)
        pct = n / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"│  {desc[:45]:<45} {n:>5} ({pct:>5.1f}%) {bar}")
    print(f"├─────────────────────────────────────────────────────────┤")
    print(f"│  Total articles with rating action ≤90 days:   {total:>7} │")
    print(f"└─────────────────────────────────────────────────────────┘")


def print_examples(sample: list[SpotCheckRow], n: int = 10) -> None:
    """Print interesting examples for human review."""
    # Show the most interesting ones first: misses, then correct catches
    priority = {"missed": 0, "correct": 1, "false_alarm": 2, "missed_positive": 3}
    sample_sorted = sorted(sample, key=lambda r: (priority.get(r.alignment, 9), r.ground_truth_days))

    print(f"\n{'='*80}")
    print(f"  DETAILED EXAMPLES ({n} most interesting)")
    print(f"{'='*80}")

    for i, r in enumerate(sample_sorted[:n], 1):
        dir_emoji = {-1: "🔴", 0: "⚪", 1: "🟢"}.get(r.label_direction, "⚪")
        gt_emoji = {"negative": "🔴", "neutral": "⚪", "positive": "🟢"}.get(r.ground_truth_outcome, "⚪")
        align_emoji = {
            "correct": "✅", "missed": "❌", "false_alarm": "⚠️",
            "missed_positive": "❌", "irrelevant_correct": "✅",
        }.get(r.alignment, "❓")

        print(f"\n  [{i}] {align_emoji} {r.alignment.upper()}")
        print(f"  Entity:     {r.entity}")
        print(f"  Title:      {r.title}")
        print(f"  Date:       {r.article_date} → rating action {r.ground_truth_days} days later")
        print(f"  Model:      {dir_emoji} direction={r.label_direction:+d} | type={r.label_signal_type} | conf={r.label_confidence}")
        print(f"  Rating:     {gt_emoji} {r.ground_truth_action} → {r.ground_truth_outcome}")
        print(f"  Reasoning:  {r.label_reasoning}")
        print(f"  URL:        {r.url[:80]}...")


def save_sample_csv(sample: list[SpotCheckRow], path: Path) -> None:
    """Save the 50-row sample as CSV for Excel review."""
    fieldnames = [
        "alignment", "entity", "title", "article_date",
        "label_direction", "label_credit_relevant", "label_signal_type",
        "label_confidence", "label_reasoning",
        "ground_truth_outcome", "ground_truth_action", "ground_truth_days",
        "url",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sample:
            writer.writerow({
                "alignment": r.alignment,
                "entity": r.entity,
                "title": r.title,
                "article_date": r.article_date,
                "label_direction": r.label_direction,
                "label_credit_relevant": r.label_credit_relevant,
                "label_signal_type": r.label_signal_type,
                "label_confidence": r.label_confidence,
                "label_reasoning": r.label_reasoning,
                "ground_truth_outcome": r.ground_truth_outcome,
                "ground_truth_action": r.ground_truth_action,
                "ground_truth_days": r.ground_truth_days,
                "url": r.url,
            })
    logger.info(f"Saved {len(sample)} sample rows to {path.name}")


# --- Main ---

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spot-check LLM labels against rating_windows ground truth"
    )
    parser.add_argument(
        "--max-days", type=int, default=90,
        help="Max days_before for rating window (default: 90)"
    )
    parser.add_argument(
        "--sample-size", type=int, default=50,
        help="Number of examples to sample for review (default: 50)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling"
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent.parent
    labels_path = base / "data" / "processed" / "labels_final.jsonl"
    articles_path = base / "data" / "processed" / "gdelt_for_labeling.csv"
    sample_csv_path = base / "data" / "processed" / "spotcheck_sample.csv"

    # Load data
    logger.info("Loading labels and articles...")
    labels = read_labels_jsonl(labels_path)
    articles = load_articles(articles_path)
    logger.info(f"Labels: {len(labels)} | Articles: {len(articles)}")

    # Join and filter
    logger.info(f"Building spot-check rows (max {args.max_days} days before rating action)...")
    rows = build_spotcheck(labels, articles, max_days=args.max_days)
    logger.info(f"Articles with rating action within {args.max_days} days: {len(rows)}")

    if not rows:
        logger.error("No articles matched! Check max_days or data files.")
        sys.exit(1)

    # Full-dataset analysis
    print_confusion_matrix(rows)
    print_alignment_summary(rows)

    # Signal type breakdown for correctly detected deterioration
    correct_types = Counter(
        r.label_signal_type for r in rows
        if r.alignment == "correct" and r.label_direction == -1
    )
    if correct_types:
        print("\n  Signal types in correctly detected deterioration:")
        for stype, count in correct_types.most_common():
            print(f"    {stype:<20} {count:>4}")

    # Sample and display
    sample = sample_50(rows, seed=args.seed)
    print_examples(sample, n=10)

    # Save CSV
    save_sample_csv(sample, sample_csv_path)

    # Summary
    print(f"\n{'='*80}")
    print(f"  ✅ SPOT-CHECK COMPLETE")
    print(f"  📊 {len(rows)} articles with rating actions within {args.max_days} days")
    print(f"  📝 {len(sample)} sample rows saved to {sample_csv_path.name}")
    print(f"  ⏭️  NEXT: Review spotcheck_sample.csv, then move to Phase 1.4")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
