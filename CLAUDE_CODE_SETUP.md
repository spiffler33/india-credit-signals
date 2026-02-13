# Claude Code Setup & Best Practices for This Project

## Initial Setup

### 1. Create the project folder
```bash
mkdir india-credit-signals
cd india-credit-signals
git init
```

### 2. Copy all files from this delivery into the folder
```
india-credit-signals/
├── CLAUDE.md              ← Claude reads this every session
├── PLAN.md                ← Master execution plan
├── .claude/
│   └── OUTPUT_STYLE.md    ← Teaching-mode output rules
├── research/
│   ├── EM_Credit_Risk_LLM_Briefing.docx   ← Your research briefing
│   └── conversation_transcript.md          ← This chat's key decisions
└── .gitignore
```

### 3. First Claude Code session
```bash
cd india-credit-signals
claude
```
Then type: `Read CLAUDE.md and PLAN.md. We're starting Phase 0. Set up the project structure and dev environment.`

Claude Code will scaffold the full directory structure, create pyproject.toml, install dependencies.

---

## Features You Should Know About

### Plan Mode (Shift+Tab twice)
**USE THIS FOR EVERY MAJOR TASK.** Plan Mode makes Claude think before coding. It reads files, analyzes, proposes a plan — but can't write anything. You review the plan, then approve.

When to use: Starting any new Phase. Building a new module. Debugging a complex issue.

### /compact
Your context window fills up during long sessions. When Claude starts forgetting earlier context, type `/compact`. It summarizes the conversation and frees up space. Use it proactively every 30-40 messages.

### /clear
Resets conversation but keeps CLAUDE.md loaded. Use when switching between very different tasks (e.g., from data scraping to frontend work).

### /cost
Shows how much the current session has cost. Track this — ML projects can burn tokens fast. Budget: aim for <$5/session.

### --continue and --resume
```bash
claude --continue    # Resume last session
claude --resume      # Pick a specific past session to resume
```
Great for multi-day work. Resume yesterday's session to keep context.

### Subagents (Claude spawns workers)
Claude Code can spin up sub-tasks that run in parallel. For this project, useful when:
- Scraping multiple data sources simultaneously
- Running tests while building new features
- Searching codebase while writing code

You don't need to configure this — Claude does it automatically. But know it happens so you're not confused by parallel activity.

### Hooks (Advanced — set up after Phase 1)
Hooks run scripts automatically at trigger points. Useful hooks for this project:

**After every file edit — run type checking:**
```json
// .claude/settings.json
{
  "hooks": {
    "postToolUse": [{
      "tool": "edit",
      "command": "pyright --project . 2>&1 | tail -5"
    }]
  }
}
```

**Before git commit — run tests:**
```json
{
  "hooks": {
    "preToolUse": [{
      "tool": "bash",
      "matcher": "git commit",
      "command": "pytest tests/ -x -q"
    }]
  }
}
```

Set these up by typing `/hooks` in Claude Code, or ask Claude to write them for you.

---

## Workflow for Each Phase

### Starting a Phase
1. Open Claude Code: `cd india-credit-signals && claude`
2. Say: `Read PLAN.md Phase {N}. Enter Plan Mode and propose your approach.`
3. Review the plan. Push back if needed.
4. Say: `Execute. Start with step {N}.1.`

### During a Phase
- Commit frequently: `Commit what we have so far.`
- When confused: `Explain what you just did. I don't understand [concept].`
- When stuck: `Stop coding. Let's discuss the approach in plain English.`
- When output is too terse: `Follow OUTPUT_STYLE.md — show me the concept card for [term].`

### Ending a Phase
1. `Summarize what we built in Phase {N}. List what works and what's pending.`
2. `Update CLAUDE.md to reflect we're now in Phase {N+1}.`
3. `Commit everything with message "[Phase N] Complete: {summary}".`

---

## Project-Specific Tips

### For Data Scraping (Phase 1)
- Always save raw responses to `data/raw/` before processing
- Use `tenacity` for retry logic on HTTP requests
- Rate-limit scrapers (1 req/sec for GDELT, respectful on Indian sites)
- Claude Code is great at writing scrapers. Say: `Write a scraper for CRISIL rating actions. Follow PLAN.md section 1.1. Handle pagination and rate limiting.`

### For Model Training (Phase 2)
- Start with a TINY dataset (100 examples) to verify the pipeline works
- Then scale to full dataset
- Save checkpoints every epoch
- Log everything with Weights & Biases (free tier) or just CSV
- Claude Code can write training scripts but CAN'T run GPU training (no GPU in terminal). It writes the script, you run it on Colab/Lambda.

### For the Dashboard (Phase 4)
- Use Vite + React + Tailwind (Claude Code is very good at this)
- Start with static/mock data, wire up real data later
- Say: `Build the entity dashboard view from PLAN.md section 4.2. Use mock data. I want to see it working in the browser.`

### For Learning
- When Claude writes something you don't understand, say: `Pause. Explain [this specific thing] like I'm a finance person who's never trained a model.`
- Ask: `What would happen if we changed [parameter] from X to Y? Why did you choose X?`
- After each Phase: `Give me a quiz. 5 questions on what I should now understand from Phase {N}.`

---

## Things Claude Code Can Do That You Might Not Know

1. **Read and analyze existing codebases.** Say: `Read the FinRLlama repo at data/repos/finrllama/. Explain the training pipeline architecture.`

2. **Write tests first.** Say: `Write tests for the GDELT scraper first, then implement the scraper to pass them.`

3. **Create diagrams.** Say: `Create a Mermaid diagram of the data pipeline from PLAN.md.`

4. **Search the web.** Claude Code can't search, but you're in Claude Chat right now which CAN. Use Chat for research, Code for building.

5. **Generate Jupyter notebooks.** For exploration: `Create a notebook that loads our training data and shows 10 examples of each label type with visualizations.`

6. **Write documentation.** Say: `Write a README.md that explains how to set up and run this project from scratch.`

7. **Profile and optimize.** Say: `This script takes 45 minutes. Profile it and find the bottleneck.`

---

## Recommended .gitignore
```
data/raw/
data/models/
*.pyc
__pycache__/
.env
node_modules/
dist/
.venv/
wandb/
*.ckpt
*.safetensors
```

## Recommended pyproject.toml starter
Ask Claude Code to generate this on first session. Key dependencies:
- `transformers`, `peft`, `datasets`, `accelerate` — HuggingFace ecosystem
- `bitsandbytes` — quantization for QLoRA
- `fastapi`, `uvicorn` — API
- `httpx`, `tenacity` — scraping with retries
- `loguru` — logging
- `pandas`, `polars` — data manipulation
- `pyyaml` — config
- `pytest` — testing
