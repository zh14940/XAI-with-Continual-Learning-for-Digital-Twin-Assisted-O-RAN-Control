"""
llm_summariser.py  --  D2: LLM that can ONLY summarise verified tool outputs
----------------------------------------------------------------------------
Uses Anthropic's tool-use API. The model has no access to the events file
itself. Every fact in its answer must come from a tool call we executed
locally with query_tools.py.

The full trace of (tool_name, args, result) is returned so the UI can show
exactly what the LLM saw -- making the answer auditable.

Env:
  ANTHROPIC_API_KEY  required
  ANTHROPIC_MODEL    optional, default "claude-opus-4-7"
"""

import os
import json
from typing import Tuple, List, Dict, Any

import query_tools as qt


# -------- Anthropic client (lazy import so the module loads without the SDK) --------

def _client():
    try:
        from anthropic import Anthropic
    except ImportError as e:
        raise RuntimeError("anthropic SDK not installed. pip install anthropic") from e
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set.")
    return Anthropic(api_key=api_key)


MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-7")


SYSTEM_PROMPT = """You are an analysis assistant for a long-running O-RAN MARL + LMUT experiment.

You DO NOT have direct access to logs or metrics. The ONLY way to obtain facts
is to call the provided tools. Every numeric claim in your final answer MUST
come from a value that appeared in a tool result during THIS conversation.

Rules:
1. Plan: think about which tool answers the question, then call it.
2. Multiple calls are fine. Stop once you have enough to answer.
3. Do NOT invent numbers, timestamps, step counts, rule strings, or modes.
4. If the tools return nothing useful, say so. Do not guess.
5. Keep the final answer concise (3-6 sentences). Reference the tools you used.
6. When the user gives a relative time like 'around step 12000' or 'the last
   minute', first call run_summary to learn the time range, then convert.
"""


def _dispatch(store: qt.EventStore, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if name not in qt.TOOLS:
        return {"ok": False, "error": f"unknown tool: {name}"}
    fn = qt.TOOLS[name]
    try:
        return fn(store, **args)
    except TypeError as e:
        return {"ok": False, "error": f"bad args for {name}: {e}"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def answer_question(store: qt.EventStore, question: str,
                    max_steps: int = 6) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Returns (final_answer_text, tool_call_trace).
    tool_call_trace = [{"name": ..., "args": {...}, "result": {...}}, ...]
    """
    client = _client()

    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": question},
    ]
    trace: List[Dict[str, Any]] = []

    for _step in range(max_steps):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=qt.TOOL_SCHEMAS,
            messages=messages,
        )

        # Collect tool_use blocks; record assistant message verbatim.
        assistant_blocks = resp.content  # list of blocks
        messages.append({"role": "assistant", "content": assistant_blocks})

        tool_uses = [b for b in assistant_blocks if getattr(b, "type", None) == "tool_use"]
        if not tool_uses:
            # Terminal: gather text.
            text_parts = [b.text for b in assistant_blocks if getattr(b, "type", None) == "text"]
            return ("\n".join(text_parts).strip(), trace)

        # Execute every tool_use the model requested.
        tool_results_block: List[Dict[str, Any]] = []
        for tu in tool_uses:
            name = tu.name
            args = tu.input or {}
            result = _dispatch(store, name, args)
            trace.append({"name": name, "args": args, "result": result})
            tool_results_block.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": json.dumps(result, default=str),
            })

        messages.append({"role": "user", "content": tool_results_block})

    return ("(stopped: hit max_steps without a final answer)", trace)
