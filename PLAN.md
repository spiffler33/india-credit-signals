# India NBFC Credit Signal Engine — Master Plan

## What We're Building
A fine-tuned LLM that extracts credit deterioration signals from English-language news/filings for Indian NBFCs, with regulator-to-sector contagion logic. Backtested against actual rating actions.

**End goal:** FinRL/FinAI 2026 contest entry + workplace tool + learning vehicle for ML fine-tuning.

## Learning Philosophy
**Learn by building existing things first, then adapt.** We clone FinRLlama, run it, understand it, THEN swap in our credit risk data. Every Claude Code output teaches you what it did and why.

---

## Phase 0: Foundation (Days 1-2)
**Goal:** Run someone else's fine-tuning pipeline end-to-end. Understand every step.

### 0.1 Clone & Run FinRLlama
```
Repo: https://github.com/Arnav-Gr0ver/ICAIF_FinRL-2024
Model: https://huggingface.co/SesameStreet/FinRLlama-3.2-3B-Instruct
```
- Clone repo, read every file, understand the RLMF training loop
- Run inference on the pre-trained model with sample financial news
- **Learning checkpoint:** You should be able to explain LoRA rank, learning rate schedules, and what RLMF does differently from standard SFT

### 0.2 Clone & Read FinGPT
```
Repo: https://github.com/AI4Finance-Foundation/FinGPT
Focus: FinGPT v3 series (LoRA fine-tuning on sentiment)
```
- Don't run the full training — just read the data pipeline and training scripts
- Understand how they structure training data (instruction format, labels)
- **Learning checkpoint:** You should understand the difference between SFT, RLHF, and RLMF

### 0.3 Set Up Development Environment
- Python 3.11+, PyTorch, transformers, peft (for LoRA), datasets
- GPU access: Colab Pro ($12/mo for A100) or Lambda Labs ($1.10/hr for A10G)
- **IMPORTANT:** Don't buy expensive GPU time yet. Phase 0-2 run on CPU or free tier.

**Discuss in Claude Chat (not Code):**
- Review your understanding of LoRA, RLMF, SFT after reading the repos
- Clarify anything confusing about the training pipeline architecture

---

## Phase 1: Data Collection (Days 3-8)
**Goal:** Build the labeled training dataset. This is 60% of the project.

### 1.1 Credit Event Timeline
**Source:** CRISIL, ICRA, CARE, India Ratings — rating action histories

| Entity | Rating Agency | Date | Action | From | To |
|--------|--------------|------|--------|------|------|
| IL&FS | ICRA | Sep 2018 | Downgrade | AAA | D |
| DHFL | CRISIL | Jun 2019 | Downgrade | AA | D |
| ... | ... | ... | ... | ... | ... |

**Target:** 100+ rating actions across 50+ NBFCs from 2016-2024.

How to get this data:
- CRISIL: https://www.crisil.com/en/home/our-analysis/ratings/rating-list.html
- ICRA: https://www.icra.in/Rationale/Search
- CARE: https://www.careedge.in/ratings/rating-rationale
- India Ratings: https://www.indiaratings.co.in/PressRelease

Scrape rating action pages. Each has entity name, date, old rating, new rating, rationale PDF.

### 1.2 News Data Collection
**Primary source:** GDELT (free, covers Indian English press well)
```
GDELT API: https://api.gdeltproject.org/api/v2/doc/doc
Query pattern: "NBFC" OR "shadow bank" OR entity_name
Filter: sourcelang:english, sourcecountry:IN
Date range: 6 months before each rating action
```

**Secondary source:** BSE/NSE Corporate Filings
```
BSE API: https://api.bseindia.com/
NSE: https://www.nseindia.com/companies-listing/corporate-filings-announcements
```
Material event disclosures, profit warnings, board meeting outcomes.

**Tertiary source:** RBI Circulars & Press Releases
```
RBI: https://www.rbi.org.in/Scripts/NotificationUser.aspx
Focus: NBFC regulatory changes, risk weight modifications, PCA framework
```

### 1.3 Label Construction
For each article/filing, label with:
- `credit_relevant`: 0 or 1 (is this about credit quality, not general business?)
- `signal_direction`: -1 (deterioration), 0 (neutral), +1 (improvement)
- `signal_type`: one of [liquidity, asset_quality, regulatory, contagion, governance, funding]
- `entity`: which NBFC(s) does this affect?
- `sector_wide`: 0 or 1 (does this affect all NBFCs or just this one?)

**Ground truth:** Did the entity get downgraded within 6 months? Binary label.

**CRITICAL:** Use Claude (prompted, not fine-tuned) to do initial labeling of articles. Then manually verify 200+ labels. This is your quality gate.

### 1.4 Training Data Format
Follow FinRLlama instruction format:
```json
{
  "instruction": "Assess the credit risk signal in the following news article for Indian NBFCs.",
  "input": "[article text]",
  "output": "CREDIT_DETERIORATION | Reason: [specific credit-relevant factor]. Signal type: [type]. Affected entity: [name]. Sector-wide: [yes/no]."
}
```
**Target:** 5,000+ labeled examples. 70/15/15 train/val/test split.

---

## Phase 2: Model Training (Days 9-14)
**Goal:** Fine-tune a model that distinguishes credit signals from generic sentiment.

### 2.1 Base Model Selection
**Primary:** Qwen 2.5-7B-Instruct (strong on financial text, multilingual for future Hindi extension)
**Fallback:** LLaMA 3.1-8B-Instruct (more community support, FinRLlama was built on LLaMA)

### 2.2 Training Configuration
Adapt FinRLlama's training script:
- LoRA: rank=16, alpha=32, target_modules=["q_proj", "v_proj"]
- Learning rate: 2e-4 with cosine scheduler
- Batch size: 4 (gradient accumulation 8)
- Epochs: 3 (with early stopping on val loss)
- Estimated cost: ~$50-100 on Colab Pro or Lambda Labs

### 2.3 RLMF Adaptation (Advanced — Week 3)
Instead of market feedback (price movements), use **rating feedback**:
- Reward = model's signal prediction vs actual rating action 6 months later
- Positive reward: model flagged deterioration AND entity was downgraded
- Negative reward: model flagged deterioration BUT entity maintained/upgraded
- This is the novel contribution — RLMF with credit outcomes instead of trading returns

### 2.4 Evaluation Metrics
- **Precision/Recall on credit events:** Can it correctly flag entities that will be downgraded?
- **Lead time:** How many days/weeks before the rating action did the model flag it?
- **False positive rate:** How many entities flagged but NOT downgraded?
- **Baseline comparison:** Prompted Opus on same test set (is fine-tuning actually better?)

---

## Phase 3: Contagion Layer (Days 15-20)
**Goal:** When RBI issues a regulatory change, propagate signals across all exposed NBFCs.

### 3.1 Entity Graph
Build a simple graph of Indian NBFCs:
- Nodes: Each NBFC (name, type, size, primary asset class)
- Edges: Shared characteristics (funding profile, asset class exposure, geography)
- Weight: Similarity score (two housing finance NBFCs are more connected than a housing NBFC and a gold loan NBFC)

### 3.2 Contagion Rules
```python
# When entity X gets a credit signal:
if signal.sector_wide:
    for entity in same_subsector(X):
        entity.score += signal.strength * edge_weight(X, entity)

# When RBI issues regulatory change:
for entity in affected_entities(regulation):
    entity.score += regulation.impact * entity.exposure
```

### 3.3 Rolling Scores
- Per-entity: 7-day, 30-day, 90-day rolling average of signal scores
- Per-subsector: Housing finance, infrastructure, microfinance, vehicle finance
- Threshold alerts: When rolling score crosses -0.5 (warning) or -0.8 (critical)

---

## Phase 4: Dashboard & Interface (Days 15-20, parallel with Phase 3)
**Goal:** Visual interface to see signals, entities, scores in real-time.

### 4.1 Tech Stack
- **Frontend:** React + Tailwind + Recharts (or Plotly)
- **Backend:** FastAPI (you already know this)
- **Database:** Supabase (you already know this) or SQLite for simplicity
- **Deployment:** Vercel (frontend) + Railway/Fly.io (backend)

### 4.2 Key Views
1. **Entity Dashboard:** List of NBFCs with current rolling credit scores, sparklines, last 5 signals
2. **Signal Feed:** Real-time feed of articles with credit relevance scores and signal types
3. **Contagion Map:** Network graph showing how signals propagate (use D3 or vis.js)
4. **Backtest View:** Timeline showing model predictions vs actual rating actions
5. **Model Comparison:** Side-by-side: fine-tuned model vs prompted Opus vs generic sentiment

### 4.3 Backtest Visualization
This is the money shot for any demo/hackathon:
- X-axis: time
- Y-axis: model's rolling credit score for entity
- Vertical lines: actual rating actions
- Show that the score drops BEFORE the vertical line = early warning works

---

## Phase 5: Contest Preparation (Days 21-30)
**Goal:** Package for FinRL/FinAI 2026 or SecureFinAI 2026 submission.

### 5.1 Watch These Repos
```
https://github.com/Open-Finance-Lab/FinRL_Contest_2025
https://github.com/Open-Finance-Lab/FinAI_Contest_2025
https://github.com/Open-Finance-Lab/SecureFinAI_Contest_2025
```
SecureFinAI 2026 repo exists but dates not announced. Check weekly.

### 5.2 Submission Requirements (based on past contests)
- Open-source code on GitHub
- Trained model weights on HuggingFace
- Technical report (4-6 pages, IEEE format)
- Reproducible results with clear instructions

### 5.3 What Makes This Win
Past winners won because of novel data pipelines, not model architecture. Your edge:
- **Credit-specific signals** (not generic sentiment) — nobody else is doing this
- **Regulator-to-sector contagion** — novel contribution to the field
- **EM focus with English-language data** — accessible and reproducible
- **Backtested against real rating actions** — concrete, verifiable results

---

## What to Discuss in Claude Chat vs Build in Claude Code

| Topic | Where |
|-------|-------|
| Architecture decisions, tradeoffs | Claude Chat (this project) |
| Understanding LoRA, RLMF, training concepts | Claude Chat |
| "Why did this fail?" debugging strategy | Claude Chat |
| Writing actual code, scripts, pipelines | Claude Code |
| Data scraping, cleaning, labeling | Claude Code |
| Model training, evaluation scripts | Claude Code |
| Frontend/dashboard building | Claude Code |
| Git workflow, deployment | Claude Code |

---

## Key Reference Links
| Resource | URL |
|----------|-----|
| FinRLlama Code | https://github.com/Arnav-Gr0ver/ICAIF_FinRL-2024 |
| FinRLlama Model | https://huggingface.co/SesameStreet/FinRLlama-3.2-3B-Instruct |
| FinGPT Framework | https://github.com/AI4Finance-Foundation/FinGPT |
| FinRL Contest 2025 | https://open-finance-lab.github.io/FinRL_Contest_2025/ |
| FinAI Contest 2025 | https://open-finance-lab.github.io/FinAI_Contest_2025/ |
| Benchmarking Paper | https://arxiv.org/abs/2504.02281 |
| LLM Credit Risk Survey (60 papers) | https://arxiv.org/abs/2506.04290 |
| Graph ML for Credit | https://pubsonline.informs.org/doi/10.1287/ijds.2022.00018 |
| GDELT API | https://api.gdeltproject.org/api/v2/doc/doc |
| CRISIL Ratings | https://www.crisil.com/en/home/our-analysis/ratings/rating-list.html |
| ICRA Rationale Search | https://www.icra.in/Rationale/Search |
| RBI NBFC Regulations | https://www.rbi.org.in/Scripts/NotificationUser.aspx |
| NSE Corporate Filings | https://www.nseindia.com/companies-listing/corporate-filings-announcements |
