import { useState } from "react";

const C = {
  bg: "#fafafa",
  white: "#ffffff",
  text: "#1a1a2e",
  textSec: "#5a5a72",
  textDim: "#8e8ea0",
  border: "#e8e8ee",
  accent: "#2563eb",
  accentLight: "#eff4ff",
  green: "#059669",
  greenBg: "#ecfdf5",
  greenBorder: "#a7f3d0",
  amber: "#d97706",
  amberBg: "#fffbeb",
  amberBorder: "#fde68a",
  red: "#dc2626",
  redBg: "#fef2f2",
  redBorder: "#fecaca",
};

const STAGES = [
  {
    num: "01",
    title: "Collect Known Outcomes",
    what: "Gather historical credit rating actions (upgrades, downgrades, defaults) from regulators as our benchmark.",
    why: "You can't measure prediction accuracy without knowing what actually happened. This is our answer key.",
    input: "Regulatory filings",
    output: "1,654 rated events across 39 companies",
    port: "Swap regulator source per market",
    tag: "data",
  },
  {
    num: "02",
    title: "Gather & Filter News",
    what: "Pull news articles from a global index, then filter in three passes: remove junk sources → keep financial articles → match to our companies.",
    why: "Start broad (74K articles), filter aggressively (→17K usable). Filters are tuned to never miss a negative signal — false alarms are cheap, missed warnings are not.",
    input: "74,028 raw articles",
    output: "17,299 relevant articles with full text",
    port: "Swap keyword lists & company names",
    tag: "data",
  },
  {
    num: "03",
    title: "Label Articles with AI",
    what: "Use AI to read each article and tag it: is this credit-relevant? Positive or negative? What type of risk signal?",
    why: "Three-phase approach keeps costs at ~$100 instead of scaling linearly. Phase 1: test on 300 articles with premium AI to calibrate quality. Phase 2: process all 17K with fast AI. Phase 3: spot-check disagreements with premium AI.",
    input: "17,299 articles + labeling criteria",
    output: "17,274 labeled articles (direction, risk type, confidence)",
    port: "Rewrite labeling criteria for new sector",
    tag: "config",
  },
  {
    num: "04",
    title: "Structure for Training",
    what: "Format labeled articles into question-answer pairs a model can learn from. Split by time period so the model is always tested on future data it hasn't seen.",
    why: "Time-based splitting prevents the model from 'peeking' at future events — the same discipline as out-of-sample testing in any forecasting exercise.",
    input: "17,274 labeled articles",
    output: "9,591 training + 2,247 validation + 2,133 test examples",
    port: "Reuse as-is",
    tag: "reuse",
  },
  {
    num: "05",
    title: "Train a Custom Model",
    what: "Take an existing open-source language model and teach it our specific task — producing structured credit risk assessments from news text.",
    why: "The base model writes essays when asked for a risk assessment (0% usable output). After 12–15 hours of GPU training, it produces structured, parseable ratings. Once trained, the model runs locally — no per-article API fees, no vendor dependency, no data leaving your environment.",
    input: "Training examples + base model",
    output: "Fine-tuned model (~50MB of custom weights)",
    port: "Reuse as-is (retrain with new data)",
    tag: "reuse",
  },
  {
    num: "06",
    title: "Validate & Benchmark",
    what: "Test the model against known outcomes and compare to three baselines: the untrained model, a finance-tuned competitor, and a premium commercial AI.",
    why: "Multiple benchmarks prevent false confidence. Also test on 3 companies the model never saw during training — proving it learned patterns, not just company names.",
    input: "Test data + known rating outcomes",
    output: "75.5% detection rate for credit deterioration",
    port: "Reuse as-is",
    tag: "reuse",
  },
];

const tagStyle = {
  data: { color: C.red, bg: C.redBg, border: C.redBorder, label: "Sector-specific data" },
  config: { color: C.amber, bg: C.amberBg, border: C.amberBorder, label: "Reconfigure per sector" },
  reuse: { color: C.green, bg: C.greenBg, border: C.greenBorder, label: "Reuse across sectors" },
};

function Stage({ stage, isLast }) {
  const t = tagStyle[stage.tag];
  return (
    <div style={{ display: "flex", gap: 20, position: "relative" }}>
      {/* Timeline */}
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", width: 40, flexShrink: 0 }}>
        <div style={{
          width: 36, height: 36, borderRadius: "50%",
          background: C.accent, color: "#fff",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 13, fontWeight: 700, fontFamily: "'DM Mono', monospace",
          flexShrink: 0,
        }}>{stage.num}</div>
        {!isLast && <div style={{ width: 2, flex: 1, background: C.border, marginTop: 4 }} />}
      </div>

      {/* Card */}
      <div style={{
        flex: 1, marginBottom: isLast ? 0 : 12,
        background: C.white,
        border: `1px solid ${C.border}`,
        borderRadius: 12,
        padding: "20px 24px",
      }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 8 }}>
          <h3 style={{ margin: 0, fontSize: 17, fontWeight: 600, color: C.text }}>{stage.title}</h3>
          <span style={{
            fontSize: 11, fontWeight: 600, padding: "3px 10px", borderRadius: 20,
            color: t.color, background: t.bg, border: `1px solid ${t.border}`,
            whiteSpace: "nowrap",
          }}>{t.label}</span>
        </div>

        <p style={{ margin: "10px 0 0", fontSize: 14, lineHeight: 1.65, color: C.textSec }}>{stage.what}</p>

        <div style={{
          margin: "14px 0 0", padding: "10px 14px", borderRadius: 8,
          background: C.accentLight, borderLeft: `3px solid ${C.accent}`,
        }}>
          <span style={{ fontSize: 12, fontWeight: 600, color: C.accent }}>Rationale: </span>
          <span style={{ fontSize: 13, color: C.text, lineHeight: 1.55 }}>{stage.why}</span>
        </div>

        <div style={{
          display: "grid", gridTemplateColumns: "1fr 1fr",
          gap: 10, marginTop: 14,
        }}>
          <div style={{ padding: "8px 12px", borderRadius: 8, background: C.bg }}>
            <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.06em", color: C.textDim, textTransform: "uppercase", marginBottom: 3 }}>In</div>
            <div style={{ fontSize: 13, color: C.text }}>{stage.input}</div>
          </div>
          <div style={{ padding: "8px 12px", borderRadius: 8, background: C.bg }}>
            <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.06em", color: C.textDim, textTransform: "uppercase", marginBottom: 3 }}>Out</div>
            <div style={{ fontSize: 13, color: C.text }}>{stage.output}</div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function PipelineExecutive() {
  const [tab, setTab] = useState("process");

  return (
    <div style={{
      minHeight: "100vh", background: C.bg,
      fontFamily: "'DM Sans', -apple-system, sans-serif",
      color: C.text,
    }}>
      <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet" />

      <div style={{ maxWidth: 740, margin: "0 auto", padding: "40px 24px 64px" }}>

        {/* Header */}
        <div style={{ marginBottom: 32 }}>
          <p style={{ fontSize: 12, fontWeight: 600, letterSpacing: "0.1em", color: C.accent, textTransform: "uppercase", margin: "0 0 6px", fontFamily: "'DM Mono', monospace" }}>
            Project Overview
          </p>
          <h1 style={{ fontSize: 28, fontWeight: 700, margin: "0 0 12px", color: C.text, lineHeight: 1.2 }}>
            AI-Driven Credit Early Warning System
          </h1>
          <p style={{ fontSize: 15, color: C.textSec, lineHeight: 1.6, margin: 0 }}>
            An automated system that reads financial news and identifies credit risk signals before rating agencies act — detecting 75.5% of deterioration events at a build cost of ~$100.
          </p>
        </div>

        {/* Summary strip */}
        <div style={{
          display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginBottom: 32,
        }}>
          {[
            { label: "Build Cost", value: "~$100", sub: "AI labeling + compute" },
            { label: "Detection Rate", value: "75.5%", sub: "credit deterioration" },
            { label: "Training Time", value: "12–15 hrs", sub: "GPU fine-tuning" },
            { label: "Articles", value: "17.3K", sub: "processed & labeled" },
          ].map(m => (
            <div key={m.label} style={{
              background: C.white, border: `1px solid ${C.border}`,
              borderRadius: 10, padding: "16px 14px", textAlign: "center",
            }}>
              <div style={{ fontSize: 22, fontWeight: 700, color: C.accent, fontFamily: "'DM Mono', monospace" }}>{m.value}</div>
              <div style={{ fontSize: 12, fontWeight: 600, color: C.text, marginTop: 2 }}>{m.label}</div>
              <div style={{ fontSize: 11, color: C.textDim }}>{m.sub}</div>
            </div>
          ))}
        </div>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 0, marginBottom: 28, borderBottom: `1px solid ${C.border}` }}>
          {[
            { id: "process", label: "How It Works" },
            { id: "whynot", label: "Why Not Real-Time AI?" },
            { id: "reuse", label: "Cross-Sector Portability" },
          ].map(t => (
            <button key={t.id} onClick={() => setTab(t.id)} style={{
              padding: "12px 20px", fontSize: 14, fontWeight: 600, cursor: "pointer",
              color: tab === t.id ? C.accent : C.textDim,
              background: "transparent", border: "none",
              borderBottom: tab === t.id ? `2px solid ${C.accent}` : "2px solid transparent",
              marginBottom: -1, transition: "all 0.15s",
            }}>{t.label}</button>
          ))}
        </div>

        {/* Process tab */}
        {tab === "process" && (
          <div>
            {/* Color legend */}
            <div style={{
              display: "flex", gap: 20, marginBottom: 24, flexWrap: "wrap",
            }}>
              {Object.values(tagStyle).map(t => (
                <div key={t.label} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <span style={{ width: 10, height: 10, borderRadius: "50%", background: t.color }} />
                  <span style={{ fontSize: 12, color: C.textSec }}>{t.label}</span>
                </div>
              ))}
            </div>

            {STAGES.map((s, i) => (
              <Stage key={s.num} stage={s} isLast={i === STAGES.length - 1} />
            ))}
          </div>
        )}

        {/* Why Not Real-Time AI tab */}
        {tab === "whynot" && (
          <div>
            <p style={{ fontSize: 14, color: C.textSec, lineHeight: 1.6, marginTop: 0, marginBottom: 24 }}>
              The obvious question: why not just send every article to a commercial AI (Gemini, GPT-4, Claude) in real time and ask it to score credit risk? It works — but it doesn't scale, and it creates dependencies you don't want in a risk system.
            </p>

            {/* Comparison cards */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 24 }}>
              <div style={{
                background: C.redBg, border: `1px solid ${C.redBorder}`,
                borderRadius: 12, padding: "20px 22px",
              }}>
                <div style={{ fontSize: 14, fontWeight: 700, color: C.red, marginBottom: 14 }}>Real-Time API Calls</div>
                {[
                  { label: "Cost per article", value: "$0.01–0.05" },
                  { label: "17K articles (one-off)", value: "$170–850" },
                  { label: "Running daily (est. 50/day)", value: "$150–900/yr" },
                  { label: "Latency", value: "2–10 sec each" },
                  { label: "Data leaves your environment", value: "Yes" },
                  { label: "Vendor dependency", value: "Complete" },
                  { label: "Consistency", value: "Model updates can change output format without warning" },
                ].map((r, i) => (
                  <div key={i} style={{ display: "flex", justifyContent: "space-between", padding: "7px 0", borderBottom: i < 6 ? `1px solid ${C.redBorder}` : "none", gap: 12 }}>
                    <span style={{ fontSize: 12, color: C.textSec }}>{r.label}</span>
                    <span style={{ fontSize: 12, fontWeight: 600, color: C.red, textAlign: "right" }}>{r.value}</span>
                  </div>
                ))}
              </div>

              <div style={{
                background: C.greenBg, border: `1px solid ${C.greenBorder}`,
                borderRadius: 12, padding: "20px 22px",
              }}>
                <div style={{ fontSize: 14, fontWeight: 700, color: C.green, marginBottom: 14 }}>Fine-Tuned Model (This Pipeline)</div>
                {[
                  { label: "Cost per article", value: "~$0 (runs locally)" },
                  { label: "17K articles (one-off)", value: "~$100 (build cost)" },
                  { label: "Running daily (est. 50/day)", value: "~$0/yr" },
                  { label: "Latency", value: "<1 sec each" },
                  { label: "Data leaves your environment", value: "No" },
                  { label: "Vendor dependency", value: "None" },
                  { label: "Consistency", value: "Frozen — output format is guaranteed" },
                ].map((r, i) => (
                  <div key={i} style={{ display: "flex", justifyContent: "space-between", padding: "7px 0", borderBottom: i < 6 ? `1px solid ${C.greenBorder}` : "none", gap: 12 }}>
                    <span style={{ fontSize: 12, color: C.textSec }}>{r.label}</span>
                    <span style={{ fontSize: 12, fontWeight: 600, color: C.green, textAlign: "right" }}>{r.value}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Key arguments */}
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              {[
                {
                  icon: "💰",
                  title: "Economics flip at scale",
                  body: "API calls are cheap per article but scale linearly — every new article, every new sector, every day costs the same. A fine-tuned model has a one-time build cost, then runs for free. Across multiple sectors and years, the difference is orders of magnitude.",
                },
                {
                  icon: "🔒",
                  title: "Data stays in-house",
                  body: "Sending potentially market-moving news articles to a third-party API creates information leakage risk. With a local model, no article text ever leaves your environment. For a bank running this on client portfolios, this isn't optional — it's a compliance requirement.",
                },
                {
                  icon: "⚡",
                  title: "Speed and reliability",
                  body: "API calls are subject to rate limits, outages, and latency spikes. A local model processes articles in under a second with no network dependency. When you're scanning breaking news for credit signals, waiting in a queue defeats the purpose.",
                },
                {
                  icon: "🎯",
                  title: "Output consistency",
                  body: "Commercial AI providers update their models without notice. An output format that worked yesterday might break tomorrow. A fine-tuned model is frozen — it produces the exact same structured output format every time, which means your downstream systems never break.",
                },
                {
                  icon: "🏗️",
                  title: "The real comparison",
                  body: "Real-time AI is like hiring an expensive consultant for every single decision. Fine-tuning is like training a junior analyst who then works for free. The upfront investment in training pays for itself almost immediately, and keeps paying.",
                },
              ].map((item, i) => (
                <div key={i} style={{
                  background: C.white, border: `1px solid ${C.border}`,
                  borderRadius: 10, padding: "16px 20px",
                  display: "flex", gap: 14, alignItems: "flex-start",
                }}>
                  <span style={{ fontSize: 20, flexShrink: 0, lineHeight: 1.4 }}>{item.icon}</span>
                  <div>
                    <div style={{ fontSize: 14, fontWeight: 600, color: C.text, marginBottom: 4 }}>{item.title}</div>
                    <div style={{ fontSize: 13, color: C.textSec, lineHeight: 1.6 }}>{item.body}</div>
                  </div>
                </div>
              ))}
            </div>

            {/* Bottom line */}
            <div style={{
              marginTop: 24, padding: "16px 20px", borderRadius: 12,
              background: C.accentLight, border: `1px solid #bfdbfe`,
            }}>
              <div style={{ fontSize: 14, fontWeight: 600, color: C.text, marginBottom: 4 }}>Bottom line</div>
              <div style={{ fontSize: 13, color: C.textSec, lineHeight: 1.6 }}>
                Real-time AI works for prototyping and one-off analysis. For a production credit surveillance system running daily across multiple sectors, a fine-tuned model is cheaper, faster, more private, and more reliable. The ~$100 build cost replaces thousands in ongoing API spend.
              </div>
            </div>
          </div>
        )}

        {/* Portability tab */}
        {tab === "reuse" && (
          <div>
            <p style={{ fontSize: 14, color: C.textSec, lineHeight: 1.6, marginTop: 0, marginBottom: 24 }}>
              ~70% of the system is reusable across sectors. To apply this to a new market, you replace the data sources, update the labeling criteria, and retrain. The core infrastructure — collection, filtering, labeling workflow, training, and evaluation — stays the same.
            </p>

            {/* Visual: what changes */}
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              {[
                {
                  label: "Replace data sources",
                  tag: "data",
                  examples: [
                    ["Indian NBFCs", "SEBI filings, Indian rating agencies, 39 NBFC watchlist"],
                    ["US Regional Banks", "FDIC call reports, Moody's/S&P, ~200 bank watchlist"],
                    ["Chinese Property", "CSRC filings, local agencies, ~50 developer watchlist"],
                  ],
                },
                {
                  label: "Update labeling criteria",
                  tag: "config",
                  examples: [
                    ["Indian NBFCs", "NPAs, RBI directives, provisioning, liquidity"],
                    ["US Regional Banks", "CRE exposure, deposit flight, NIM compression"],
                    ["Chinese Property", "Presales, LGFV exposure, trust product defaults"],
                  ],
                },
                {
                  label: "Reuse everything else",
                  tag: "reuse",
                  items: [
                    "News collection & three-pass filtering",
                    "Calibrate → Bulk → Audit labeling workflow",
                    "Time-based data splitting",
                    "Model training pipeline",
                    "Evaluation & benchmarking framework",
                  ],
                },
              ].map((section, si) => {
                const t = tagStyle[section.tag];
                return (
                  <div key={si} style={{
                    background: C.white, border: `1px solid ${C.border}`,
                    borderRadius: 12, overflow: "hidden",
                  }}>
                    <div style={{
                      padding: "14px 20px",
                      borderBottom: `1px solid ${C.border}`,
                      display: "flex", alignItems: "center", gap: 10,
                    }}>
                      <span style={{ width: 10, height: 10, borderRadius: "50%", background: t.color }} />
                      <span style={{ fontSize: 14, fontWeight: 600, color: C.text }}>{section.label}</span>
                    </div>
                    <div style={{ padding: "14px 20px" }}>
                      {section.examples ? (
                        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                          {section.examples.map(([market, detail], i) => (
                            <div key={i} style={{ display: "flex", gap: 12, alignItems: "baseline" }}>
                              <span style={{
                                fontSize: 12, fontWeight: 600, color: C.accent,
                                minWidth: 130, flexShrink: 0,
                              }}>{market}</span>
                              <span style={{ fontSize: 13, color: C.textSec }}>{detail}</span>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                          {section.items.map((item, i) => (
                            <span key={i} style={{
                              fontSize: 13, padding: "6px 14px", borderRadius: 20,
                              background: t.bg, color: t.color, border: `1px solid ${t.border}`,
                              fontWeight: 500,
                            }}>{item}</span>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Bottom line */}
            <div style={{
              marginTop: 24, padding: "16px 20px", borderRadius: 12,
              background: C.accentLight, border: `1px solid #bfdbfe`,
              display: "flex", alignItems: "center", gap: 12,
            }}>
              <span style={{ fontSize: 22 }}>⏱</span>
              <div>
                <div style={{ fontSize: 14, fontWeight: 600, color: C.text }}>Estimated time to port to a new sector</div>
                <div style={{ fontSize: 13, color: C.textSec }}>~1–2 weeks for data sourcing & prompt calibration. Infrastructure reuse is immediate.</div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
