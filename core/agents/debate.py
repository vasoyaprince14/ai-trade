"""
Bull vs Bear Debate Engine
===========================
Inspired by TradingAgents (TauricResearch/TradingAgents).
Two Ollama agents argue the case for and against a trade.
A third judge agent produces the final verdict with confidence.

Architecture:
  Bull Agent  →  bullish case from market data
  Bear Agent  →  bearish case + rebuttals
  2-round debate
  Evidence Agent → checks which claims have data support
  Judge Agent    → final verdict: direction + confidence + invalidation level

Usage:
    from core.agents.debate import run_debate
    result = run_debate(market_context, symbol="NIFTY")
"""

import json
import time
from datetime import datetime
from typing import Optional
from loguru import logger

try:
    import ollama as _ollama
    _OLLAMA_OK = True
except ImportError:
    _OLLAMA_OK = False

OLLAMA_MODEL = "llama3.2:3b"


def _call_llm(prompt: str, role_label: str) -> str:
    """Call Ollama with a prompt. Returns response text."""
    if not _OLLAMA_OK:
        return f"[{role_label}] Ollama not available"
    try:
        resp = _ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.6, "num_predict": 300},
        )
        return resp["message"]["content"].strip()
    except Exception as e:
        logger.warning(f"Ollama {role_label} failed: {e}")
        return f"[{role_label} unavailable: {e}]"


def _build_context(ctx: dict) -> str:
    """Format market context dict into readable string for LLM."""
    lines = []
    for k, v in ctx.items():
        if v is not None and v != "" and v != {}:
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def run_debate(market_context: dict, symbol: str = "NIFTY", rounds: int = 2) -> dict:
    """
    Run a Bull vs Bear debate and return:
    {
      bull_case:    str,
      bear_case:    str,
      bull_rebuttal: str,
      bear_rebuttal: str,
      evidence:     str,
      verdict:      {direction, confidence, reason, invalidated_if},
      timestamp:    str,
    }
    """
    ctx_str = _build_context(market_context)
    ts = datetime.now().strftime("%d %b %Y %H:%M")

    # ── Bull Agent ────────────────────────────────────────────────────────────
    bull_prompt = f"""You are a BULL analyst for {symbol}. Your job is to argue the bullish case.
Be specific and cite the data below. Keep response under 150 words.

Market Data:
{ctx_str}

Make the strongest possible bullish argument. Focus on: trend, momentum, support levels, flow signals.
End with: "Bull target: [price/level]" and "Bull confidence: [%]"
"""
    bull_case = _call_llm(bull_prompt, "Bull")
    logger.info(f"[Debate] Bull: {bull_case[:100]}...")

    # ── Bear Agent ────────────────────────────────────────────────────────────
    bear_prompt = f"""You are a BEAR analyst for {symbol}. Your job is to argue the bearish case.
Be specific and cite the data below. Keep response under 150 words.

Market Data:
{ctx_str}

Bull's argument:
{bull_case}

Counter the bull's points and make the strongest possible bearish argument.
Focus on: resistance, overbought signals, risk factors, headwinds.
End with: "Bear target: [price/level]" and "Bear confidence: [%]"
"""
    bear_case = _call_llm(bear_prompt, "Bear")
    logger.info(f"[Debate] Bear: {bear_case[:100]}...")

    bull_rebuttal = bear_rebuttal = ""
    if rounds >= 2:
        # ── Bull Rebuttal ─────────────────────────────────────────────────────
        bull_rebuttal = _call_llm(f"""You are the BULL analyst. Rebut the bear's argument in 80 words max.

Bear said: {bear_case}

Your rebuttal (be specific, use data):""", "BullRebuttal")

        # ── Bear Rebuttal ─────────────────────────────────────────────────────
        bear_rebuttal = _call_llm(f"""You are the BEAR analyst. Final rebuttal in 80 words max.

Bull rebuttal: {bull_rebuttal}

Your counter (be specific):""", "BearRebuttal")

    # ── Evidence Check ────────────────────────────────────────────────────────
    evidence_prompt = f"""You are the EVIDENCE ANALYST for {symbol}. Review both sides and check which claims are data-supported.

Market Data (ground truth):
{ctx_str}

Bull case: {bull_case}
Bear case: {bear_case}

List in 80 words: which claims are TRUE (supported by data), which are WEAK (not in data), and which side has stronger data support.
"""
    evidence = _call_llm(evidence_prompt, "Evidence")

    # ── Judge / Decision Agent ────────────────────────────────────────────────
    judge_prompt = f"""You are the HEAD TRADER making the final {symbol} decision.

You've heard:
BULL: {bull_case}
BEAR: {bear_case}
EVIDENCE: {evidence}

Market Data:
{ctx_str}

Output EXACTLY this JSON (no other text):
{{
  "direction": "BULLISH" or "BEARISH" or "NEUTRAL",
  "confidence": 0-100,
  "reason": "2-sentence summary",
  "invalidated_if": "specific price level or event that would flip this view",
  "agreed_with": "BULL" or "BEAR" or "SPLIT"
}}
"""
    judge_raw = _call_llm(judge_prompt, "Judge")

    # Parse verdict JSON
    verdict = {
        "direction": "NEUTRAL",
        "confidence": 50,
        "reason": judge_raw[:200] if judge_raw else "Debate inconclusive",
        "invalidated_if": "N/A",
        "agreed_with": "SPLIT",
    }
    try:
        start = judge_raw.find("{")
        end   = judge_raw.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(judge_raw[start:end])
            verdict.update(parsed)
    except Exception:
        pass

    result = {
        "symbol":        symbol,
        "bull_case":     bull_case,
        "bear_case":     bear_case,
        "bull_rebuttal": bull_rebuttal,
        "bear_rebuttal": bear_rebuttal,
        "evidence":      evidence,
        "verdict":       verdict,
        "timestamp":     ts,
    }

    logger.info(
        f"[Debate] Verdict: {verdict['direction']} {verdict['confidence']}% | {verdict['agreed_with']}"
    )
    return result
