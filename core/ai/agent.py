"""
AI Trading Copilot Agent
=========================
Uses Claude claude-sonnet-4-6 (or GPT-4 as fallback) as an AI agent
that can reason about trade signals using:

  - ML model predictions (binary, 5-class, regression)
  - Order flow data (OI, PCR, Max Pain, IV skew)
  - News sentiment (VADER + FinBERT if available)
  - Technical analysis summary
  - Current positions and P&L

The agent can:
  - Explain why a signal was generated
  - Suggest trade parameters (entry/target/SL)
  - Warn about risk factors
  - Answer natural language queries about market state
  - Summarize daily performance

Inspired by FinGPT's financial reasoning + NSE-MCP server pattern.
"""
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any

from loguru import logger

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))


class TradingAgent:
    """
    AI copilot that reasons about trade signals and market state.
    Uses Anthropic Claude via the Anthropic SDK.
    Falls back to rule-based reasoning if API key not available.
    """

    SYSTEM_PROMPT = """You are an expert Indian stock market analyst and algo trader specializing in Nifty F&O.

You have access to:
- Real-time NSE option chain data (OI, PCR, Max Pain, IV Skew)
- ML model predictions (XGBoost + LightGBM ensemble)
- News sentiment from ET, Moneycontrol, Mint
- Technical indicators (EMA, RSI, MACD, Bollinger Bands, VWAP)
- FII/DII institutional positioning (participant-wise futures, calls, puts bias)
- Strike-level institutional zone analysis (call walls, put walls, gamma pin)
- Current positions and P&L

Your role:
1. Analyze trade signals and provide clear reasoning
2. Suggest precise entry, target, and stop-loss levels
3. Highlight key risks and market conditions
4. Summarize market state in plain language
5. Interpret FII/DII positioning honestly — treat it as one factor, not a direct copy signal

Rules:
- Always mention VIX level and its impact
- Always check PCR for market direction confirmation
- Always mention FII regime if institutional data is present
- Treat FII call-selling as a bearish signal only when confirmed by futures/cash
- Never claim to know exactly which FII sold a specific strike — this is inference, not fact
- Consider time-of-day (avoid F&O trades in last 30 min)
- Be concise — traders need quick, actionable insights
- Format numbers clearly: ₹24,500, PCR=1.23, VIX=14.5, FII=BEARISH

Respond in 3-5 sentences unless asked for detailed analysis."""

    def __init__(self):
        self._client = None
        self._model  = "claude-sonnet-4-6"
        self._history: List[Dict] = []
        self._init_client()

    def _init_client(self):
        """Initialize Anthropic client."""
        try:
            import anthropic
            import os
            api_key = os.environ.get("ANTHROPIC_API_KEY") or self._load_from_env()
            if api_key:
                self._client = anthropic.Anthropic(api_key=api_key)
                logger.info("AI Agent initialized with Claude claude-sonnet-4-6")
            else:
                logger.warning("ANTHROPIC_API_KEY not set — AI agent will use rule-based fallback")
        except ImportError:
            logger.warning("anthropic package not installed — run: pip install anthropic")

    def _load_from_env(self) -> Optional[str]:
        try:
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env")
            import os
            return os.environ.get("ANTHROPIC_API_KEY")
        except Exception:
            return None

    # ── Core query ────────────────────────────────────────────────────────────

    def query(self, prompt: str, context: Optional[Dict] = None) -> str:
        """
        Ask the AI agent a question with optional market context.

        Args:
            prompt  : Natural language question
            context : Dict with market data (prediction, oi, news, positions)

        Returns:
            AI response as string
        """
        full_prompt = self._build_prompt(prompt, context)

        if self._client:
            return self._query_claude(full_prompt)
        else:
            return self._rule_based_response(prompt, context)

    def _build_prompt(self, prompt: str, context: Optional[Dict]) -> str:
        if not context:
            return prompt

        ctx_lines = [f"=== Market Context ({datetime.now().strftime('%H:%M IST')}) ==="]

        if "prediction" in context:
            p = context["prediction"]
            ctx_lines.append(
                f"ML Signal: {p.get('direction','?')} | "
                f"Confidence: {p.get('confidence',0):.0%} | "
                f"Predicted Return: {p.get('predicted_return',0):+.3f}% | "
                f"5-Class: {p.get('label_5class','?')}"
            )

        if "market" in context:
            m = context["market"]
            ctx_lines.append(
                f"Market: LTP={m.get('ltp',0):,.0f} | VIX={m.get('vix',0):.1f} | "
                f"PCR={m.get('pcr',0):.2f} | MaxPain={m.get('max_pain',0):,.0f}"
            )

        if "technicals" in context:
            t = context["technicals"]
            ctx_lines.append(
                f"Technicals: EMA21={t.get('ema21',0):.0f} EMA50={t.get('ema50',0):.0f} | "
                f"RSI={t.get('rsi',0):.1f} | MACD={t.get('macd',0):+.1f}"
            )

        if "sentiment" in context:
            s = context["sentiment"]
            ctx_lines.append(
                f"News: {s.get('sentiment','?')} (score={s.get('score',0):.3f}, "
                f"articles={s.get('count',0)})"
            )

        if "positions" in context:
            pos = context["positions"]
            if pos:
                ctx_lines.append(f"Open Positions: {len(pos)} | " +
                    " | ".join(f"{p.get('symbol','?')} {p.get('pnl',0):+.0f}" for p in pos[:3]))

        if "order_flow" in context:
            of = context["order_flow"]
            ctx_lines.append(
                f"Order Flow: {of.get('signal','?')} | "
                f"OI Change: CE={of.get('ce_oi_chg',0):+,} PE={of.get('pe_oi_chg',0):+,}"
            )

        if "institutional" in context:
            inst = context["institutional"]
            regime    = inst.get("regime", "?")
            composite = inst.get("composite", 0)
            fut_bias  = inst.get("futures_bias", 0)
            call_bias = inst.get("call_bias", 0)
            put_bias  = inst.get("put_bias", 0)
            fii_cash  = inst.get("fii_cash_net", 0)
            dii_cash  = inst.get("dii_cash_net", 0)
            divs      = ", ".join(inst.get("divergences", [])) or "none"
            ctx_lines.append(
                f"Institutional: FII={regime} (composite={composite:+.3f}) | "
                f"Futures={fut_bias:+.3f} Calls={call_bias:+.3f} Puts={put_bias:+.3f} | "
                f"Cash: FII={fii_cash:+,}Cr DII={dii_cash:+,}Cr | "
                f"Divergences: {divs}"
            )

        if "strike_analysis" in context:
            sa = context["strike_analysis"]
            call_walls = [str(w["strike"]) for w in sa.get("call_walls", [])[:3]]
            put_walls  = [str(w["strike"]) for w in sa.get("put_walls", [])[:3]]
            gamma_pin  = sa.get("gamma_pin_strike", "?")
            direction  = sa.get("inferred_direction", "?")
            conf       = sa.get("direction_confidence", 0)
            ctx_lines.append(
                f"Strike Analysis: Inferred={direction} (conf={conf:.0%}) | "
                f"Call Walls={','.join(call_walls) or '?'} | "
                f"Put Walls={','.join(put_walls) or '?'} | "
                f"GammaPin={gamma_pin}"
            )

        ctx_block = "\n".join(ctx_lines)
        return f"{ctx_block}\n\n{prompt}"

    def _query_claude(self, prompt: str) -> str:
        """Query Claude via Anthropic API."""
        try:
            self._history.append({"role": "user", "content": prompt})
            # Keep last 10 turns to manage context
            messages = self._history[-10:]

            response = self._client.messages.create(
                model=self._model,
                max_tokens=512,
                system=self.SYSTEM_PROMPT,
                messages=messages,
            )
            reply = response.content[0].text
            self._history.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return f"[AI unavailable: {e}]"

    def _rule_based_response(self, prompt: str, context: Optional[Dict]) -> str:
        """Fallback rule-based analysis when API key not available."""
        if not context:
            return "No market context provided. Load market data first."

        p   = context.get("prediction", {})
        m   = context.get("market", {})
        direction   = p.get("direction", "NEUTRAL")
        confidence  = p.get("confidence", 0.5)
        vix         = m.get("vix", 0)
        pcr         = m.get("pcr", 1.0)
        ltp         = m.get("ltp", 0)
        max_pain    = m.get("max_pain", 0)

        lines = []

        if direction == "BUY" and confidence > 0.6:
            lines.append(f"BULLISH signal with {confidence:.0%} confidence. Consider buying ATM Call.")
            target = ltp * 1.005
            sl     = ltp * 0.997
            lines.append(f"Entry near ₹{ltp:,.0f} | Target ₹{target:,.0f} | SL ₹{sl:,.0f}.")
        elif direction == "SELL" and confidence > 0.6:
            lines.append(f"BEARISH signal with {confidence:.0%} confidence. Consider buying ATM Put.")
            target = ltp * 0.995
            sl     = ltp * 1.003
            lines.append(f"Entry near ₹{ltp:,.0f} | Target ₹{target:,.0f} | SL ₹{sl:,.0f}.")
        else:
            lines.append("No clear directional signal. Market appears range-bound.")

        if vix > 20:
            lines.append(f"VIX at {vix:.1f} is elevated — option premiums are expensive, consider spreads.")
        elif vix < 12:
            lines.append(f"VIX at {vix:.1f} is low — option buying is cheap.")

        if pcr > 1.3:
            lines.append(f"PCR={pcr:.2f} is bullish (heavy put writing by institutions).")
        elif pcr < 0.8:
            lines.append(f"PCR={pcr:.2f} is bearish (heavy call writing).")

        if max_pain and ltp:
            dist = ((ltp - max_pain) / max_pain) * 100
            lines.append(f"Price is {dist:+.1f}% from Max Pain ₹{max_pain:,.0f} (gravitational pull expected).")

        return " ".join(lines)

    # ── Convenience methods ───────────────────────────────────────────────────

    def explain_signal(self, context: Dict) -> str:
        return self.query("Explain this trading signal and should I take this trade?", context)

    def daily_summary(self, context: Dict) -> str:
        return self.query(
            "Summarize today's market action, key signals, and tomorrow's outlook in 5 bullet points.",
            context
        )

    def risk_check(self, context: Dict, trade_details: Dict) -> str:
        prompt = (
            f"I'm planning to trade: {json.dumps(trade_details)}. "
            "What are the key risks and should I reduce position size given current market conditions?"
        )
        return self.query(prompt, context)

    def clear_history(self):
        self._history.clear()


# ── Singleton ─────────────────────────────────────────────────────────────────

_agent: Optional[TradingAgent] = None


def get_agent() -> TradingAgent:
    global _agent
    if _agent is None:
        _agent = TradingAgent()
    return _agent
