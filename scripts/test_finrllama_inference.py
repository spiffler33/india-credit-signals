# WHY THIS: Phase 0.1 learning exercise — load an LLM and run inference on
# sample financial news using the FinRLlama prompt template. This verifies our
# environment works and establishes a BASELINE before we fine-tune our own model.
# We test both the FinRLlama prompt (stock sentiment) and credit-risk headlines.

"""
Test LLM inference on Apple M1 MPS using the FinRLlama prompt template.

Supports multiple models:
  --model finrllama   → FinRLlama-3.2-3B (requires LLaMA license)
  --model qwen        → Qwen 2.5-3B-Instruct (no auth needed, our planned base)
"""

import argparse
import re
import time
from pathlib import Path

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

# 🎓 Model registry: maps friendly names to HuggingFace model IDs
MODELS: dict[str, str] = {
    "finrllama": "Arnav-Gr0ver/FinRLlama-3.2-3B-Instruct",
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen": "Qwen/Qwen2.5-1.5B-Instruct",  # 🎓 Default: 1.5B fits in 8GB Mac RAM
}


# 🎓 This is the same prompt template from FinRLlama's task2_signal.py
# The model was fine-tuned to respond to this exact format.
SIGNAL_PROMPT = """
Task: Analyze the following news headline about a stock and provide a sentiment score between -{signal_strength} and {signal_strength}, where:
-{signal_strength} represents a highly negative sentiment, likely indicating a substantial decline in stock performance.
-{threshold} represents a moderate negative sentiment, suggesting a slight potential decline in stock performance.
0 represents neutral sentiment, indicating no significant impact on stock performance.
{threshold} represents a moderate positive sentiment, indicating potential for slight stock growth.
{signal_strength} represents a highly positive sentiment, indicating significant potential for stock appreciation.

Consider the likely influence of market feedback from previous price movements and sentiment trends:
How has the stock's price responded to similar news in the past?
Does the headline align with prevailing market sentiment, or does it contradict current trends?
How might this sentiment lead to a change in the stock's behavior, considering both historical price patterns and market expectations?

Examples of sentiment scoring:
"Company X announces layoffs amidst economic downturn." Score: -8
"Company Y reports record revenue growth in Q1." Score: 7
"Market sees strong response to Company Z's new product release." Score: 5

Do not provide any explanations or reasoning. Output only a single integer in the range of -{signal_strength} to {signal_strength} based on the sentiment of the news and its potential impact on stock performance.

News headline: "{news}"

Price Data: "{prices}"

SENTIMENT SCORE:
"""

# 🎓 Sample test cases: mix of clearly positive, negative, and ambiguous news
# We include both US equity headlines (what the model was trained on) and
# credit-risk-style headlines (what we want to ADAPT it for) to see the difference.
TEST_CASES: list[dict[str, str]] = [
    {
        "name": "Strong positive — earnings beat",
        "news": "Apple reports Q4 revenue of $94.9 billion, beating analyst estimates by 5%. "
                "iPhone sales surge 12% year-over-year driven by strong demand in emerging markets.",
        "prices": "AAPL: Open=175.20, High=178.50, Low=174.80, Close=177.90, Volume=82M",
    },
    {
        "name": "Strong negative — fraud/governance",
        "news": "SEC charges Wirecard executives with massive accounting fraud. "
                "Company files for insolvency after revealing 1.9 billion euros missing from accounts.",
        "prices": "WDI: Open=104.50, High=104.50, Low=1.28, Close=1.28, Volume=350M",
    },
    {
        "name": "Ambiguous — mixed signals",
        "news": "Tesla announces 10% workforce reduction while simultaneously revealing "
                "record vehicle deliveries of 1.8 million units in 2023.",
        "prices": "TSLA: Open=248.50, High=252.30, Low=245.10, Close=246.80, Volume=115M",
    },
    {
        "name": "NBFC credit test — IL&FS style crisis (out of domain)",
        "news": "IL&FS defaults on Rs 1,000 crore commercial paper. RBI expresses concern "
                "about liquidity in NBFC sector. DHFL share price crashes 60% on contagion fears.",
        "prices": "ILFS: Open=25.50, High=25.50, Low=12.20, Close=12.80, Volume=45M",
    },
    {
        "name": "Regulatory — RBI action (out of domain for stock model)",
        "news": "RBI increases risk weights on NBFC lending by 25 basis points, citing "
                "rapid credit growth concerns. Banking stocks fall 2-3% across the board.",
        "prices": "NIFTYBANK: Open=44250, High=44300, Low=43100, Close=43200, Volume=200M",
    },
]


def run_inference(model_key: str = "qwen") -> None:
    """Load an LLM and generate sentiment scores for test cases."""
    model_name = MODELS[model_key]
    signal_strength = 10
    threshold = signal_strength // 3  # 3 — same as FinRLlama's default

    # 🎓 Device selection: CUDA for cloud GPU, MPS for larger M1 Macs, CPU as fallback.
    # On 8GB M1, MPS hits a per-tensor 4GB limit during attention on long prompts.
    # CPU is slower (~15-30s/query) but has no allocation limits. Fine for testing.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float16
        logger.info("Using CUDA GPU")
    elif torch.backends.mps.is_available() and torch.mps.driver_allocated_memory() == 0:
        # 🎓 MPS works on 16GB+ Macs. On 8GB, attention tensors exceed the 4GB
        # per-tensor limit for long prompts. We force CPU on 8GB machines.
        import psutil
        total_gb = psutil.virtual_memory().total / (1024**3)
        if total_gb > 12:
            device = torch.device("mps")
            dtype = torch.float16
            logger.info(f"Using MPS (Apple Metal GPU) — {total_gb:.0f}GB RAM")
        else:
            device = torch.device("cpu")
            dtype = torch.float32
            logger.info(f"Using CPU — 8GB Mac, MPS per-tensor limit too tight for long prompts")
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        logger.info("Using CPU")

    # Load tokenizer
    logger.info(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 🎓 Loading a 3B model: ~5.75GB in fp16.
    # On 8GB M1, we load to CPU first then move to MPS — avoids the
    # caching_allocator_warmup crash (MPS rejects one big allocation).
    # low_cpu_mem_usage=True loads weights layer-by-layer instead of all at once.
    logger.info(f"Loading model (this will download ~6GB on first run)...")
    t0 = time.time()
    if device.type == "cpu":
        # 🎓 On CPU: load in fp32 directly, no device_map tricks needed
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
    else:
        # 🎓 On GPU (MPS/CUDA): load to CPU first, then move to GPU
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model = model.to(device)
        logger.info(f"Model moved to {device}")
    load_time = time.time() - t0
    logger.info(f"Model loaded in {load_time:.1f}s")

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {param_count / 1e9:.2f}B")

    # Run inference on each test case
    print("\n" + "=" * 70)
    print(f"Inference Results — {model_key} ({model_name})")
    print("=" * 70)

    for i, case in enumerate(TEST_CASES, 1):
        prompt = SIGNAL_PROMPT.format(
            signal_strength=signal_strength,
            threshold=threshold,
            news=case["news"],
            prices=case["prices"],
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,  # 🎓 Greedy decoding for deterministic eval
            )
        gen_time = time.time() - t0

        # Decode only the new tokens (not the prompt)
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        print(f"\n[{i}/5] {case['name']}")
        print(f"  Raw output: {repr(response)}")
        print(f"  Inference time: {gen_time:.2f}s")

        # Try to extract numeric score
        match = re.search(r"-?\d+", response)
        if match:
            score = int(match.group())
            direction = "BULLISH" if score >= threshold else "BEARISH" if score <= -threshold else "NEUTRAL"
            print(f"  Score: {score} → {direction}")
        else:
            print(f"  Score: PARSE FAILED (model didn't output a number)")

    print("\n" + "=" * 70)
    print("✅ DONE: FinRLlama inference test complete")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LLM inference with FinRLlama prompt")
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default="qwen",
        help="Which model to test (default: qwen)",
    )
    args = parser.parse_args()
    run_inference(model_key=args.model)
