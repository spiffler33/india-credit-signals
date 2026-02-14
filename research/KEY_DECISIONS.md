# Key Decisions Log — From Research Phase

## Decision 1: India NBFCs over Chinese Property Developers
**Why:** English-language data availability. India's financial press, BSE/NSE filings, RBI circulars, and rating agency reports (CRISIL, ICRA, CARE) are all in English. Chinese property required accepting gaps from Chinese intranet sources.

**Bonus:** Richer contagion map. IL&FS → DHFL → Altico → Reliance Capital → SREI shows contagion across subsectors (infrastructure → housing → real estate → diversified), unlike Chinese property which was one sector imploding.

## Decision 2: Fine-Tune + Traditional ML Hybrid (Not Pure LLM)
**Why:** The 2025 systematic review of 60 papers on LLM credit risk (Golec et al.) found hybrid approaches (LLM extracts features → XGBoost/CatBoost classifies) outperform pure LLM approaches. Fine-tuned model extracts credit-specific signals; aggregation/prediction layer uses proven ML.

**Contested:** Some evidence that prompted frontier models (Opus) nearly match fine-tuned small models. We build both as comparison — this is itself a valuable finding.

## Decision 3: RLMF with Rating Feedback (Novel Contribution)
**Why:** FinRLlama used Reinforcement Learning from Market Feedback (price movements as reward). We adapt this to credit: reward = alignment between model's signal and subsequent rating action. This is novel — nobody has published RLMF with credit outcomes.

## Decision 4: Learn by Building Existing Pipelines First
**Why:** Clone FinRLlama → run it → understand it → adapt for credit risk. Not "study theory then build from scratch." The codebase IS the textbook.

## Decision 5: Contest Target = FinRL/FinAI/SecureFinAI 2026
**Why:** Open-Finance-Lab runs all three contest series. Past winners submitted open-source code + HuggingFace model + IEEE-format paper. The credit risk signal approach maps directly to their "FinGPT Agents" and "Digital Regulatory Reporting" tracks.

## Decision 6: Adani Group as Phase 2 Extension
**Why:** Tests conglomerate-level contagion (within a group's entities) vs sector-level contagion (across NBFCs). Hindenburg report → DoJ indictment → TotalEnergies pullout is a rich news-driven credit event cascade with abundant English-language data.

## Decision 7: Colab Pro over Local Mac for GPU Work
**Why:** 8GB M1 Mac can't fit 3B+ models in memory for inference (unified memory shared with OS, MPS has per-tensor 4GB limit). Colab Pro ($12/mo) gives A100 with 40GB VRAM. Local Mac is kept for code editing, data scraping, and dashboard dev — no GPU work.

**Tried and failed locally:** fp16 on MPS (warmup allocation rejected), fp32 on CPU (model memory-mapped from SSD, unusably slow). The right tool for the right job.

## Key People to Follow
- **Xiao-Yang Liu** (Columbia/SecureFinAI Lab) — FinRL/FinGPT creator, runs all contests
- **Muhammed Golec** — Systematic review of 60 LLM credit risk papers, identified gaps
- **Sanjiv Das** (Santa Clara/INFORMS) — Graph ML for credit, contagion modeling pioneer
