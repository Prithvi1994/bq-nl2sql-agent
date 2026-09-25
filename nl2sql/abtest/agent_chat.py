"""Agentic chat over the experiment results.

The model runs a tool-calling loop: it reads the question and the conversation, decides
which of the eight tools to call, gets numbers back, and then answers. It never writes
SQL and never computes a statistic.

Two things make this trustworthy rather than merely impressive:

1. **Numbers are checked against the tool output.** After the model writes its answer,
   every numeric token in it is matched back to a value that a tool actually returned.
   An invented number fails the answer and the model is asked to rewrite it. This is
   the difference between a readout and a plausible paragraph.

2. **The ship decision is not the model's.** `variant_lift` returns a verdict computed
   from the pre-registered rule, and the model is instructed to report that verdict
   rather than form its own.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .ask import ask, load_experiments
from .chat_llm import LLMError, create_llm
from .tools import ToolBox, TOOL_BY_NAME, TOOL_SPECS, catalog

SYSTEM = """\
You are an experiment analyst. You answer questions about A/B test results by calling
tools. You do not have a database connection, you cannot write SQL, and you cannot do
arithmetic: every number you report must come from a tool result.

## Tools
{catalog}

## How to work
- Start with `list_experiments` if you do not know which experiment the user means or
  what its variants are. Use `metric_definitions` if you are unsure which metric matches
  their wording.
- For any "which variant won", "how much lift", "did it beat control" question, call
  `variant_lift`. It returns the lift, confidence interval, p-value, SRM result and a
  ship/no-ship verdict.
- Call `check_srm` before trusting an effect. If SRM is detected, the arms are not
  comparable and no variant can be called a winner, no matter how large the lift is.
- Use `variant_trend` when the user asks about trend, stability, or whether an effect
  held over time. Use `segment_lift` for "where", "which country", "by device".
- `required_sample_size` answers "how long do we need to run". `data_quality` is worth
  calling before you interpret a number on an experiment with low activity.

## Rules you must follow
- Report the `verdict.decision` from `variant_lift` verbatim. Do not form your own.
- Never name a variant a winner if `verdict.shipping_variant` is null, even when the
  point estimate looks large. Say what the tool says.
- Report confidence intervals. A lift without its CI is not a result.
- If a tool returns an error, say what is missing and what the user should try. Do not
  guess at a number.
- Never invent a number. If it is not in a tool result, do not say it.
- Be concise. A short answer with a CI beats a long one without.

## Output format
Every reply that needs data must be a single JSON tool call on its own:
`{{"tool": "variant_lift", "arguments": {{"experiment_id": "checkout_flow_v2", "metric": "revenue_per_user"}}}}`
Do not prefix it with an explanation. Do not write prose about what you are about to do.
Only after a tool result comes back do you write the final answer in markdown, at most
~200 words unless asked for detail. Lead with the decision. Use a small table for
per-variant numbers. Do not describe your tool calls unless the user asks.

If a question needs no data (e.g. "what metrics do you have?"), call
`metric_definitions` or `list_experiments` anyway rather than answering from memory.
"""


@dataclass
class Turn:
    user: str
    answer: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    numbers_verified: bool = True
    rejected_numbers: List[str] = field(default_factory=list)
    retries: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user": self.user,
            "answer": self.answer,
            "tool_calls": self.tool_calls,
            "numbers_verified": self.numbers_verified,
            "rejected_numbers": self.rejected_numbers,
            "retries": self.retries,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Number verification
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"(?<![\w.])[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])[-+]?\d+\.\d+|(?<![\w.])[-+]?\d+")


def extract_numbers(text: str) -> List[str]:
    """Numeric tokens from the answer, ignoring years and small ordinals."""
    out: List[str] = []
    for m in _NUM_RE.finditer(text or ""):
        tok = m.group(0)
        try:
            val = float(tok.replace(",", ""))
        except ValueError:
            continue
        # A bare year in a date range is not a claim about a result.
        if 1990 <= val <= 2100 and val.is_integer() and "." not in tok:
            continue
        out.append(tok)
    return out


def _tolerances(value: float) -> List[float]:
    """Plausible renderings of one measured value."""
    out = {
        value,
        round(value, 1), round(value, 2), round(value, 3), round(value, 4),
        round(value * 100, 1), round(value * 100, 2), round(value * 100, 3),
        round(value * 1000, 1), round(value * 1000, 2),
        round(value * 10000, 1), round(value * 10000, 2),
        round(value / 1000, 2), round(value / 1000, 3),
    }
    return [abs(x) for x in out if x is not None]


def collect_grounded_values(payload: Any, acc: Optional[List[float]] = None) -> List[float]:
    """Every numeric value anywhere in a tool result, including nested dicts."""
    if acc is None:
        acc = []
    if isinstance(payload, bool):
        return acc
    if isinstance(payload, (int, float)):
        if payload == payload and abs(payload) != float("inf"):
            acc.append(float(payload))
    elif isinstance(payload, dict):
        for v in payload.values():
            collect_grounded_values(v, acc)
    elif isinstance(payload, (list, tuple)):
        for v in payload:
            collect_grounded_values(v, acc)
    return acc


def verify_numbers(answer: str, grounded: Sequence[float]) -> Tuple[bool, List[str]]:
    """Every number in the answer must be traceable to a tool result.

    Percentages, basis points and thousands renderings of the same value are all
    accepted, because `0.2984`, `29.84` and `2984 bps` are the same measurement.
    """
    claimed = extract_numbers(answer)
    if not claimed:
        return True, []
    if not grounded:
        return False, claimed
    flat: List[float] = []
    for g in grounded:
        flat.extend(_tolerances(g))
    unverified: List[str] = []
    for tok in claimed:
        val = float(tok.replace(",", ""))
        hit = False
        for cand in (val, val * 100, val / 100, val * 1000, val / 1000, val * 10000):
            if any(abs(cand - g) <= max(abs(g) * 1e-3, 5e-4) for g in flat):
                hit = True
                break
        if not hit:
            unverified.append(tok)
    return not unverified, unverified


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ExperimentAgent:
    def __init__(
        self,
        db_path: str,
        llm: Any,
        *,
        max_tool_rounds: int = 8,
        max_retries: int = 2,
        history_limit: int = 12,
    ) -> None:
        self.db_path = db_path
        self.llm = llm
        self.max_tool_rounds = max_tool_rounds
        self.max_retries = max_retries
        self.history_limit = history_limit
        self.experiments = load_experiments(db_path)
        self.history: List[Dict[str, str]] = []
        self.last_grounded: List[float] = []
        self.last_turn: Optional[Turn] = None

    def _messages(self, user: str) -> List[Dict[str, str]]:
        msgs = [{"role": "system", "content": SYSTEM.format(catalog=catalog())}]
        msgs.extend(self.history[-self.history_limit:])
        msgs.append({"role": "user", "content": user})
        return msgs

    def _parse_calls(self, text: str) -> Tuple[Optional[Dict[str, Any]], str]:
        """Pull a tool call out of the model output.

        Accepts a bare JSON object, a fenced json block, or a call embedded in prose.
        Returns (call, cleaned_text). A call with an unknown tool or bad arguments is
        returned anyway so the error goes back to the model and it can correct itself.
        """
        if not text:
            return None, ""
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
        candidates: List[str] = []
        if fenced:
            candidates.append(fenced.group(1))
        stripped = text.strip()
        if stripped.startswith("{"):
            candidates.append(stripped)
        obj = re.search(r"(\{[^{}]*\"tool\"\s*:\s*\"[a-z_]+\"[^{}]*\})", text, re.S)
        if obj:
            candidates.append(obj.group(1))
        for raw in candidates:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and ("tool" in data or "name" in data):
                name = data.get("tool") or data.get("name")
                args = data.get("arguments") or data.get("args") or data.get("parameters") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                return {"tool": str(name), "arguments": args if isinstance(args, dict) else {}}, text
        return None, text

    def _run_tool(self, box: ToolBox, call: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
        name = call.get("tool", "")
        args = call.get("arguments") or {}
        try:
            result = box.call(name, args)
            return {"tool": name, "arguments": args, "ok": True, "result": result}, None
        except Exception as exc:  # noqa: BLE001 - surfaced to the model so it can recover
            msg = f"{type(exc).__name__}: {exc}"
            return {"tool": name, "arguments": args, "ok": False, "error": msg}, msg

    def ask(self, user: str) -> Turn:
        turn = Turn(user=user)
        box = ToolBox(self.db_path, self.experiments)
        self.last_grounded = []
        messages = self._messages(user)
        nudged = False

        for _ in range(self.max_tool_rounds):
            try:
                text = self.llm.complete(messages)
            except LLMError as exc:
                turn.error = f"model error: {exc}"
                return turn
            except Exception as exc:  # noqa: BLE001
                turn.error = f"{type(exc).__name__}: {exc}"
                return turn

            call, cleaned = self._parse_calls(text)
            if call is None and not turn.tool_calls:
                # The model described what it was going to do instead of doing it.
                # Announcing intent is not a tool call, so push it back once.
                if not nudged:
                    nudged = True
                    messages.append({"role": "assistant", "content": text.strip()})
                    messages.append(
                        {"role": "user",
                         "content": (
                             "Do not describe what you will do. Emit the tool call now, as a "
                             "single JSON object: "
                             '{"tool": "<tool name>", "arguments": {...}}'
                         )}
                    )
                    continue
            if call is None:
                turn.answer = cleaned.strip()
                break

            record, err = self._run_tool(box, call)
            turn.tool_calls.append(record)
            if err is None:
                self.last_grounded.extend(collect_grounded_values(record["result"]))
            rendered = json.dumps(record.get("result", record.get("error")), default=str)
            if len(rendered) > 6000:
                rendered = rendered[:6000] + "\n... (truncated)"
            messages.append({"role": "assistant", "content": cleaned.strip() or json.dumps(call)})
            messages.append(
                {"role": "user",
                 "content": f"Tool result:\n```json\n{rendered}\n```\n"
                            "Continue. If you have enough, write the final answer in markdown."}
            )
        else:
            turn.answer = "I could not finish this in the tool budget. Try asking one thing at a time."

        # Grounding check on the prose the model produced.
        if turn.answer:
            for attempt in range(self.max_retries + 1):
                ok, bad = verify_numbers(turn.answer, self.last_grounded)
                if ok:
                    turn.numbers_verified = True
                    break
                turn.numbers_verified = False
                turn.rejected_numbers = bad
                turn.retries = attempt
                if attempt >= self.max_retries:
                    break
                messages = list(messages)
                messages.append({"role": "assistant", "content": turn.answer})
                messages.append(
                    {"role": "user",
                     "content": (
                         f"Those numbers are not in any tool result: {bad}. "
                         "Rewrite the answer using only numbers that appeared in the tool "
                         "results above. If you cannot support a claim, drop it."
                     )}
                )
                try:
                    turn.answer = (self.llm.complete(messages) or "").strip()
                except LLMError as exc:
                    turn.error = f"model error on rewrite: {exc}"
                    break

        self.history.append({"role": "user", "content": user})
        if turn.answer:
            self.history.append({"role": "assistant", "content": turn.answer})
        self.last_turn = turn
        self.last_box = box
        return turn


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Agentic chat over A/B experiment results.")
    ap.add_argument("question", nargs="*", help="Question; omit to enter an interactive session")
    ap.add_argument("--db", default="./abtest_data/abtest.duckdb")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--chat", action="store_true", help="interactive REPL")
    args = ap.parse_args()

    try:
        llm = create_llm(args.model, base_url=args.base_url)
    except LLMError as exc:
        print(f"Cannot start: {exc}")
        raise SystemExit(2)

    agent = ExperimentAgent(args.db, llm)

    if args.chat or not args.question:
        print(f"Experiment agent — model {llm.name}. Ctrl-D to exit.\n")
        while True:
            try:
                q = input("you> ").strip()
            except EOFError:
                print()
                return
            if not q or q in ("exit", "quit"):
                return
            turn = agent.ask(q)
            _print_turn(turn, verbose=not args.json)
        return

    turn = agent.ask(" ".join(args.question))
    _print_turn(turn, verbose=args.json)


def _print_turn(turn: Turn, verbose: bool = False) -> None:
    if turn.error and not turn.answer:
        print(f"error: {turn.error}")
        return
    if verbose:
        print(json.dumps(turn.to_dict(), indent=2, default=str))
        return
    print(turn.answer)
    for call in turn.tool_calls:
        status = "ok" if call.get("ok") else f"error: {call.get('error')}"
        print(f"  · {call['tool']}({json.dumps(call.get('arguments', {}), default=str)}) -> {status}")
    if not turn.numbers_verified:
        print(f"  ! unverified numbers: {turn.rejected_numbers} "
              f"(after {turn.retries} rewrite attempts)")


if __name__ == "__main__":
    main()
