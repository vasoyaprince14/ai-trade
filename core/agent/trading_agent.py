"""
AI Trading Agent — Powered by Claude claude-sonnet-4-6
======================================
Uses Anthropic's Claude with tool_use to:
  1. Read live tape (institutional order flow)
  2. Check FII/DII positioning
  3. Query vector memory (similar historical states)
  4. Analyze option chain (OI, PCR, IV, max pain)
  5. Check risk limits
  6. Make a structured trade decision

The agent reasons step by step, like an experienced F&O trader watching the tape.

Decision output (TradeDecision):
  action    : BUY_CE | BUY_PE | SELL_STRADDLE | SELL_CE | SELL_PE | WAIT | EXIT
  strike    : int (option strike)
  expiry    : str
  qty_lots  : int
  entry_price_range : (low, high)
  stop_loss : float (option premium level)
  target    : float (option premium level)
  sl_spot   : float (spot level for SL)
  target_spot: float (spot level for target)
  confidence: float 0-1
  reasoning : str (Claude's full reasoning)
  risk_reward: float

Usage:
  agent = TradingAgent()
  decision = agent.decide(oi_tracker, risk_manager)
  if decision.action != "WAIT":
      broker.place(decision)
"""

import os
import json
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple

from loguru import logger

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("anthropic not installed. Run: pip install anthropic")

from core.memory.vector_store import get_vector_store


# ── Trade Decision ─────────────────────────────────────────────────────────────

@dataclass
class TradeDecision:
    action:       str      # BUY_CE | BUY_PE | SELL_STRADDLE | SELL_CE | SELL_PE | WAIT | EXIT
    symbol:       str      = "NIFTY"
    strike:       int      = 0
    expiry:       str      = ""
    qty_lots:     int      = 1
    entry_low:    float    = 0.0    # lower bound of entry price range
    entry_high:   float    = 0.0    # upper bound
    stop_loss:    float    = 0.0    # in option premium terms
    target:       float    = 0.0    # in option premium terms
    sl_spot:      float    = 0.0    # stop loss in spot (Nifty level)
    target_spot:  float    = 0.0    # target in spot
    risk_reward:  float    = 0.0
    confidence:   float    = 0.0    # 0.0 - 1.0
    reasoning:    str      = ""
    timestamp:    str      = field(default_factory=lambda: datetime.now().isoformat())

    def is_trade(self) -> bool:
        return self.action not in ("WAIT", "EXIT", "")

    def to_dict(self) -> Dict:
        return asdict(self)

    def summary(self) -> str:
        if not self.is_trade():
            return f"[AGENT] {self.action} — {self.reasoning[:100]}"
        return (
            f"[AGENT] {self.action} {self.symbol} {self.strike} {self.expiry} | "
            f"Entry: {self.entry_low:.0f}-{self.entry_high:.0f} | "
            f"SL: {self.stop_loss:.0f} (spot:{self.sl_spot:.0f}) | "
            f"Tgt: {self.target:.0f} (spot:{self.target_spot:.0f}) | "
            f"RR: {self.risk_reward:.1f} | Conf: {self.confidence:.0%}"
        )

    def signal_message(self) -> str:
        """Clean WhatsApp/Telegram-style signal card."""
        ts = datetime.now().strftime("%d %b %Y  %H:%M")
        if not self.is_trade():
            return (
                f"⏳ WAIT — {self.symbol}\n"
                f"Time   : {ts}\n"
                f"Conf   : {self.confidence:.0%}\n"
                f"Reason : {self.reasoning[:200]}"
            )
        emoji = {"BUY_CE": "🟢", "BUY_PE": "🔴", "SELL_STRADDLE": "🟡",
                 "SELL_CE": "🟠", "SELL_PE": "🟠"}.get(self.action, "⚪")
        opt_type = "CE" if "CE" in self.action else "PE" if "PE" in self.action else "STRADDLE"
        lines = [
            f"{emoji} {self.action} | {self.symbol} {self.strike} {opt_type}",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"Entry  : {self.entry_low:.0f} – {self.entry_high:.0f}",
            f"SL     : {self.stop_loss:.0f}  (Nifty below {self.sl_spot:.0f})" if "BUY" in self.action
                else f"SL     : {self.stop_loss:.0f}  (Nifty above {self.sl_spot:.0f})",
            f"Target : {self.target:.0f}  (Nifty at {self.target_spot:.0f})",
            f"R:R    : 1:{self.risk_reward:.1f}",
            f"Conf   : {self.confidence:.0%}",
            f"Time   : {ts}",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"Reason : {self.reasoning[:300]}",
        ]
        return "\n".join(lines)


# ── Tool Definitions (Claude tool_use) ────────────────────────────────────────

TOOLS = [
    {
        "name": "get_tape_and_flow",
        "description": (
            "Get the live institutional tape reading: what big players are doing, "
            "which strikes they're active at, their entry prices, inferred SL and targets, "
            "FII/DII positioning (long/short in futures and options), PCR, OI build-up."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Index symbol: NIFTY or BANKNIFTY",
                    "enum": ["NIFTY", "BANKNIFTY"],
                }
            },
            "required": [],
        },
    },
    {
        "name": "query_market_memory",
        "description": (
            "Query the vector database for historical market states similar to now. "
            "Returns up to 10 most similar past situations and what happened next "
            "(market direction, P&L of trades). Use this to validate current signal "
            "with historical precedent."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "top_k": {
                    "type": "integer",
                    "description": "Number of similar states to retrieve (5-15)",
                    "default": 10,
                }
            },
            "required": [],
        },
    },
    {
        "name": "get_option_chain",
        "description": (
            "Get current option chain data: all strikes with OI, OI change, volume, "
            "LTP, IV, delta. Useful to pick the right strike for the trade."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol":  {"type": "string", "default": "NIFTY"},
                "expiry":  {"type": "string", "description": "Expiry date (optional, defaults to nearest)"},
                "strikes": {"type": "integer", "description": "Number of strikes each side of ATM (5-15)", "default": 8},
            },
            "required": [],
        },
    },
    {
        "name": "check_risk",
        "description": (
            "Check current risk status: daily P&L, open positions, drawdown, "
            "whether we're allowed to take a new trade, available capital."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "make_decision",
        "description": (
            "Submit the final trade decision. Call this LAST after analyzing "
            "all data. Provide complete trade details."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["BUY_CE", "BUY_PE", "SELL_STRADDLE", "SELL_CE", "SELL_PE", "WAIT", "EXIT"],
                    "description": "Trade action to take",
                },
                "strike": {
                    "type": "integer",
                    "description": "Option strike price (0 for WAIT)",
                },
                "expiry": {"type": "string", "description": "Expiry date string"},
                "qty_lots": {"type": "integer", "description": "Number of lots", "default": 1},
                "entry_low": {"type": "number", "description": "Lower bound of acceptable entry price"},
                "entry_high": {"type": "number", "description": "Upper bound of acceptable entry price"},
                "stop_loss": {"type": "number", "description": "SL in option premium (e.g. 50 means exit if option hits ₹50)"},
                "target": {"type": "number", "description": "Target in option premium"},
                "sl_spot": {"type": "number", "description": "Underlying spot level at which SL triggers"},
                "target_spot": {"type": "number", "description": "Underlying spot level at which to take profit"},
                "confidence": {"type": "number", "description": "Confidence 0.0-1.0"},
                "reasoning": {"type": "string", "description": "Full reasoning for this decision"},
            },
            "required": ["action", "reasoning", "confidence"],
        },
    },
]


# ── Trading Agent ─────────────────────────────────────────────────────────────

class TradingAgent:
    """
    Claude-powered F&O trading agent with market memory.

    The agent is given a system prompt describing the trading context,
    then uses tools to gather data and make a structured decision.
    """

    SYSTEM_PROMPT = """You are an expert Nifty F&O trader and order flow analyst.
You trade index options (Nifty, BankNifty) using institutional order flow, tape reading,
and FII/DII positioning as your primary signals.

Your decision process:
1. Read the tape — what are big players (FII/Pro) doing? Which strikes? Are they buying or selling?
2. Check FII/DII F&O positioning from NSE participant data — are foreigners net long or short?
3. Query market memory — in similar past situations, what happened?
4. Check the option chain — pick the best strike (ATM or slightly OTM based on signal)
5. Check risk — can we trade? What's the daily P&L?
6. Make a decision

Key rules:
- Only trade when you have CONFLUENCE: tape + FII/DII + historical memory all agree
- Prefer selling premium (straddle/strangle) when IV is high and market is range-bound
- Prefer buying options when there's a clear directional move with big OI buildup
- WAIT is a valid and often the best decision — no trade is better than a bad trade
- Always set SL at 50% of premium for bought options, or 100% for sold options
- Target minimum 1:1.5 risk-reward for bought options

Risk rules:
- Maximum 5% of capital per trade
- Stop all trading if daily loss exceeds 2%
- Maximum 5 open positions

When you call make_decision with action=WAIT, explain why (e.g. "conflicting signals",
"FII neutral", "no clear tape", "near max pain - wait for breakout").
"""

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self._client = None
        self._vector_store = get_vector_store()
        self._last_decision: Optional[TradeDecision] = None

        # Context injected before each decision run
        self._current_oi_tracker = None
        self._current_risk_manager = None
        self._current_features: Dict = {}
        self._current_tape_summary: Dict = {}
        self._current_participant: Dict = {}

        if not ANTHROPIC_AVAILABLE:
            logger.warning("anthropic SDK not available. Agent will use fallback mode.")
            return

        if not self._api_key:
            logger.warning("ANTHROPIC_API_KEY not set. Agent will use fallback mode.")
            return

        try:
            self._client = anthropic.Anthropic(api_key=self._api_key)
            logger.info("TradingAgent initialized with Claude claude-sonnet-4-6")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")

    # ── Main Entry Point ───────────────────────────────────────────────────────

    def decide(
        self,
        oi_tracker,
        risk_manager=None,
        symbol: str = "NIFTY",
    ) -> TradeDecision:
        """
        Run the agent to make a trade decision.

        Args:
          oi_tracker   : OITracker instance (has tape_reader, scraper, etc.)
          risk_manager : RiskManager instance (optional)
          symbol       : NIFTY or BANKNIFTY

        Returns TradeDecision.
        """
        self._current_oi_tracker  = oi_tracker
        self._current_risk_manager = risk_manager

        # Pre-fetch data for tool handlers
        try:
            self._current_features    = oi_tracker.get_model_features()
            self._current_tape_summary = oi_tracker.tape_reader.get_flow_summary()
            self._current_participant  = oi_tracker._participant.get_full_picture()
        except Exception as e:
            logger.error(f"Pre-fetch failed: {e}")
            self._current_features     = {}
            self._current_tape_summary = {}
            self._current_participant  = {}

        # Store current state in vector DB
        tape_bias = self._current_tape_summary.get("bias", "NEUTRAL")
        fii_bias  = self._current_participant.get("fno", {}).get("fii_bias", "NEUTRAL")
        smart     = self._current_participant.get("smart_money_bias", "NEUTRAL")

        state_id = self._vector_store.store_state(
            features=self._current_features,
            symbol=symbol,
            tape_bias=tape_bias,
            fii_bias=fii_bias,
            smart_money=smart,
        )

        # Run agent
        if self._client:
            decision = self._run_agent(symbol)
        else:
            decision = self._fallback_decision(symbol)

        # Tag decision with signal in vector store
        if state_id and decision.action != "WAIT":
            self._vector_store.record_outcome(
                state_id,
                direction="UP" if "CE" in decision.action else "DOWN",
                move_pct=0.0,   # filled later when we know the result
                pnl=0.0,
            )

        self._last_decision = decision
        logger.info(decision.summary())
        return decision

    # ── Agent Loop ─────────────────────────────────────────────────────────────

    def _run_agent(self, symbol: str) -> TradeDecision:
        """Run Claude with tool_use loop until make_decision is called."""
        messages = [
            {
                "role": "user",
                "content": (
                    f"Analyze the current {symbol} F&O market and make a trade decision. "
                    f"Current time: {datetime.now().strftime('%H:%M IST, %A %d %b %Y')}. "
                    f"Use your tools in order: tape → memory → option chain → risk → decide."
                ),
            }
        ]

        decision = None
        max_iterations = 8

        for iteration in range(max_iterations):
            try:
                response = self._client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=4096,
                    system=self.SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=messages,
                )
            except Exception as e:
                logger.error(f"Claude API error: {e}")
                return self._fallback_decision(symbol)

            # Add assistant response to messages
            messages.append({"role": "assistant", "content": response.content})

            # Check stop reason
            if response.stop_reason == "end_turn":
                break

            if response.stop_reason != "tool_use":
                break

            # Process tool calls
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool_name  = block.name
                tool_input = block.input
                tool_id    = block.id

                result, decision = self._handle_tool(
                    tool_name, tool_input, decision, symbol
                )

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": json.dumps(result, default=str),
                })

            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            # Stop if decision was made
            if decision is not None:
                break

        return decision or TradeDecision(
            action="WAIT",
            symbol=symbol,
            reasoning="Agent did not reach a decision",
            confidence=0.0,
        )

    def _handle_tool(
        self,
        tool_name: str,
        tool_input: Dict,
        decision: Optional[TradeDecision],
        symbol: str,
    ) -> Tuple[Any, Optional[TradeDecision]]:
        """Execute a tool call and return (result_dict, decision_or_None)."""

        if tool_name == "get_tape_and_flow":
            return self._tool_tape(), None

        elif tool_name == "query_market_memory":
            top_k = tool_input.get("top_k", 10)
            return self._tool_memory(top_k), None

        elif tool_name == "get_option_chain":
            return self._tool_option_chain(
                symbol=tool_input.get("symbol", symbol),
                expiry=tool_input.get("expiry"),
                strikes=tool_input.get("strikes", 8),
            ), None

        elif tool_name == "check_risk":
            return self._tool_risk(), None

        elif tool_name == "make_decision":
            d = self._tool_make_decision(tool_input, symbol)
            return {"status": "decision_recorded", "action": d.action}, d

        else:
            return {"error": f"Unknown tool: {tool_name}"}, None

    # ── Tool Implementations ───────────────────────────────────────────────────

    def _tool_tape(self) -> Dict:
        """Return tape + FII/DII data for the agent."""
        tape = self._current_tape_summary
        participant = self._current_participant

        # Format big player positions readably
        big_players = []
        for bp in tape.get("big_players", [])[:5]:
            big_players.append({
                "strike":         bp["strike"],
                "type":           bp["option_type"],
                "side":           bp.get("side", "?"),
                "market_impact":  bp.get("market_impact", "?"),
                "oi_contracts":   bp["net_oi"],
                "avg_fill_price": round(bp.get("avg_fill_price", 0), 2),
                "inferred_sl":    bp.get("inferred_sl", 0),
                "sl_spot":        bp.get("sl_spot", 0),
                "inferred_target": bp.get("inferred_target", 0),
                "target_spot":    bp.get("target_spot", 0),
            })

        fno = participant.get("fno", {})
        fii = fno.get("participants", {}).get("FII", {})
        dii = fno.get("participants", {}).get("DII", {})

        return {
            "tape_bias":         tape.get("bias", "NEUTRAL"),
            "bullish_oi":        tape.get("bullish_oi", 0),
            "bearish_oi":        tape.get("bearish_oi", 0),
            "total_tape_events": tape.get("total_events", 0),
            "big_player_positions": big_players,
            "hot_strikes":       tape.get("hot_strikes", [])[:5],
            "recent_events":     tape.get("recent_tape", [])[:5],
            "fii_positioning": {
                "net_futures": fii.get("net_futures", 0),
                "net_calls":   fii.get("net_calls",   0),
                "net_puts":    fii.get("net_puts",    0),
                "bias":        fii.get("bias",        "NEUTRAL"),
            },
            "dii_positioning": {
                "net_futures": dii.get("net_futures", 0),
                "net_calls":   dii.get("net_calls",   0),
                "net_puts":    dii.get("net_puts",    0),
                "bias":        dii.get("bias",        "NEUTRAL"),
            },
            "smart_money_bias": participant.get("smart_money_bias", "NEUTRAL"),
            "cash_flows": participant.get("cash", {}),
        }

    def _tool_memory(self, top_k: int = 10) -> Dict:
        """Query vector DB for similar historical states."""
        if not self._vector_store.is_available:
            return {
                "available": False,
                "message": "Vector store not connected. Start Qdrant with: docker-compose up qdrant",
            }

        similar = self._vector_store.find_similar(
            self._current_features,
            top_k=top_k,
        )
        stats = self._vector_store.get_outcome_stats(similar)

        # Format for the agent
        formatted = []
        for s in similar[:8]:
            formatted.append({
                "similarity":  s["score"],
                "when":        s["timestamp"][:16],
                "tape_bias":   s["tape_bias"],
                "fii_bias":    s["fii_bias"],
                "pcr_oi":      s["pcr_oi"],
                "atm_iv":      s["atm_iv"],
                "vix":         s["vix"],
                "signal":      s["signal"],
                "outcome":     s["outcome"],
            })

        return {
            "similar_states": formatted,
            "outcome_stats":  stats,
            "total_stored":   self._vector_store.get_stats().get("total_states", 0),
        }

    def _tool_option_chain(
        self,
        symbol: str = "NIFTY",
        expiry: Optional[str] = None,
        strikes: int = 8,
    ) -> Dict:
        """Return formatted option chain for the agent."""
        try:
            scraper = self._current_oi_tracker.scraper
            records = scraper.parse_option_chain(symbol, expiry, strikes_range=strikes)
            if not records:
                return {"error": "No option chain data available"}

            spot = records[0]["underlying"]
            atm  = round(spot / 50) * 50
            exp  = records[0]["expiry"]

            # Build strike table
            strikes_data = {}
            for r in records:
                s = r["strike"]
                if s not in strikes_data:
                    strikes_data[s] = {"strike": s, "CE": {}, "PE": {}}
                strikes_data[s][r["option_type"]] = {
                    "oi":      r["oi"],
                    "oi_chg":  r["oi_change"],
                    "vol":     r["volume"],
                    "ltp":     r["ltp"],
                    "iv":      r["iv"],
                }

            # Format as list sorted by strike
            chain = sorted(strikes_data.values(), key=lambda x: x["strike"])

            # Compute PCR and max pain hint
            ce_oi = sum(r["oi"] for r in records if r["option_type"] == "CE")
            pe_oi = sum(r["oi"] for r in records if r["option_type"] == "PE")
            pcr   = round(pe_oi / ce_oi, 3) if ce_oi else 0

            # Max OI strikes
            ce_by_oi = sorted(
                [r for r in records if r["option_type"] == "CE"],
                key=lambda x: x["oi"], reverse=True
            )[:3]
            pe_by_oi = sorted(
                [r for r in records if r["option_type"] == "PE"],
                key=lambda x: x["oi"], reverse=True
            )[:3]

            return {
                "spot":   spot,
                "atm":    atm,
                "expiry": exp,
                "pcr_oi": pcr,
                "max_call_oi_strikes": [r["strike"] for r in ce_by_oi],
                "max_put_oi_strikes":  [r["strike"] for r in pe_by_oi],
                "chain":  chain,
            }
        except Exception as e:
            return {"error": str(e)}

    def _tool_risk(self) -> Dict:
        """Return current risk status."""
        base = {
            "can_trade":         True,
            "reason":            "OK",
            "daily_pnl":         0,
            "daily_loss_limit":  -10000,
            "open_positions":    0,
            "max_positions":     5,
            "available_capital": 500000,
        }

        if not self._current_risk_manager:
            return base

        try:
            rm = self._current_risk_manager
            can_trade = rm.can_trade() if hasattr(rm, "can_trade") else True
            pnl = rm.daily_pnl if hasattr(rm, "daily_pnl") else 0
            positions = len(rm.open_positions) if hasattr(rm, "open_positions") else 0

            return {
                "can_trade":         can_trade,
                "daily_pnl":         pnl,
                "open_positions":    positions,
                "max_positions":     5,
                "available_capital": getattr(rm, "available_capital", 500000),
            }
        except Exception as e:
            base["error"] = str(e)
            return base

    def _tool_make_decision(self, inp: Dict, symbol: str) -> TradeDecision:
        """Parse the agent's final decision."""
        action = inp.get("action", "WAIT")
        return TradeDecision(
            action=action,
            symbol=symbol,
            strike=int(inp.get("strike", 0)),
            expiry=str(inp.get("expiry", "")),
            qty_lots=int(inp.get("qty_lots", 1)),
            entry_low=float(inp.get("entry_low", 0)),
            entry_high=float(inp.get("entry_high", 0)),
            stop_loss=float(inp.get("stop_loss", 0)),
            target=float(inp.get("target", 0)),
            sl_spot=float(inp.get("sl_spot", 0)),
            target_spot=float(inp.get("target_spot", 0)),
            confidence=float(inp.get("confidence", 0)),
            reasoning=str(inp.get("reasoning", "")),
            risk_reward=(
                (float(inp.get("target", 0)) - float(inp.get("entry_high", 0)))
                / max(float(inp.get("entry_high", 0)) - float(inp.get("stop_loss", 0.01)), 0.01)
            ) if inp.get("target") and inp.get("entry_high") and inp.get("stop_loss") else 0.0,
        )

    # ── Fallback (no API key) ──────────────────────────────────────────────────

    def _fallback_decision(self, symbol: str) -> TradeDecision:
        """
        Rule-based fallback when Claude API is not available.
        Uses the pre-fetched features directly.
        """
        features = self._current_features
        tape     = self._current_tape_summary
        smart    = self._current_participant.get("smart_money_bias", "NEUTRAL")

        bias_num   = features.get("f_tape_bias_numeric", 0)
        smart_num  = {"STRONGLY_BULLISH": 2, "BULLISH": 1, "NEUTRAL": 0,
                      "BEARISH": -1, "STRONGLY_BEARISH": -2}.get(smart, 0)
        pcr        = features.get("f_pcr_oi", 1.0)
        atm_iv     = features.get("f_atm_iv", 15)
        spot       = features.get("f_spot", 0)
        atm        = features.get("f_atm_strike", round(spot / 50) * 50 if spot else 0)

        combined = bias_num + smart_num

        # Check risk
        can_trade = True
        if self._current_risk_manager:
            try:
                can_trade = self._current_risk_manager.can_trade()
            except Exception:
                pass

        if not can_trade:
            return TradeDecision(
                action="WAIT", symbol=symbol,
                reasoning="Risk limit hit — trading halted",
                confidence=1.0,
            )

        # High IV + neutral bias → sell straddle
        if atm_iv > 20 and abs(combined) <= 1 and 0.7 <= pcr <= 1.3:
            return TradeDecision(
                action="SELL_STRADDLE",
                symbol=symbol,
                strike=int(atm),
                confidence=0.6,
                reasoning=f"High IV ({atm_iv:.1f}%) + neutral bias + PCR={pcr:.2f} → sell premium",
            )

        # Strong bullish → buy CE
        if combined >= 3:
            strike = int(atm)
            return TradeDecision(
                action="BUY_CE",
                symbol=symbol,
                strike=strike,
                confidence=min(combined / 4.0, 0.9),
                reasoning=(
                    f"Strong bullish confluence: tape={tape.get('bias')} "
                    f"smart_money={smart} PCR={pcr:.2f}"
                ),
            )

        # Strong bearish → buy PE
        if combined <= -3:
            strike = int(atm)
            return TradeDecision(
                action="BUY_PE",
                symbol=symbol,
                strike=strike,
                confidence=min(abs(combined) / 4.0, 0.9),
                reasoning=(
                    f"Strong bearish confluence: tape={tape.get('bias')} "
                    f"smart_money={smart} PCR={pcr:.2f}"
                ),
            )

        return TradeDecision(
            action="WAIT",
            symbol=symbol,
            reasoning=f"No clear confluence. Tape={tape.get('bias')}, Smart={smart}, PCR={pcr:.2f}",
            confidence=0.5,
        )

    @property
    def last_decision(self) -> Optional[TradeDecision]:
        return self._last_decision


# Singleton
_agent: Optional[TradingAgent] = None


def get_agent(api_key: Optional[str] = None) -> TradingAgent:
    global _agent
    if _agent is None:
        _agent = TradingAgent(api_key)
    return _agent
