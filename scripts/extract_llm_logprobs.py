#!/usr/bin/env python3
"""Extract per-token logprobs from OpenHands logs (llm_completions or output.jsonl).

OpenHands can record logprobs in two places:

1) One JSON file per LLM call under:
     <eval_output_dir>/.../llm_completions/<instance_id>/*.json

2) Embedded inside the eval output.jsonl (EvalOutput), under:
     history[*].tool_call_metadata.model_response.choices[0].logprobs.content

For GRPO-style training you often want a single concatenated stream of:
  - token_id
  - logprob (prefer sampling_logprob if present)
  - (optional) token
plus a mask (default: all ones).

This script supports both sources:
  - --llm-completions-dir / --eval-run-dir  (per-call JSONs)
  - --output-jsonl                          (embedded logprobs in history)

Usage examples:
  # Point directly at an instance directory
  python scripts/extract_llm_logprobs.py \
    --llm-completions-dir /path/to/llm_completions/django__django-13230 \
    --out rollout.json

  # Or point at an eval run dir and instance id
  python scripts/extract_llm_logprobs.py \
    --eval-run-dir /path/to/.../qwen*_maxiter_40 \
    --instance-id django__django-13230 \
    --out rollout.json

Output format (JSON):
  {
    "instance_id": "...",            # if provided
    "num_calls": 12,
    "calls": [{"timestamp": ..., "file": "...", "n_tokens": ...}, ...],
    "token_ids": [...],
    "logprobs": [...],
    "tokens": [...],                # only if --include-tokens
    "mask": [...]                   # 1s, same length as token_ids
  }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from datetime import datetime


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_from_call(d: dict[str, Any]) -> list[dict[str, Any]]:
    """Return logprobs.content list for a single OpenHands completion log JSON."""
    resp = d.get("response") or {}
    choices = resp.get("choices") or []
    if not choices:
        return []

    c0 = choices[0]
    lp = c0.get("logprobs")
    if not isinstance(lp, dict):
        return []

    content = lp.get("content")
    return content if isinstance(content, list) else []


def _parse_iso_ts(ts: Any) -> float:
    """Parse ISO timestamp string to epoch seconds; fall back to 0.0."""
    if isinstance(ts, (int, float)):
        return float(ts)
    if not isinstance(ts, str):
        return 0.0
    try:
        # Handle trailing Z if present
        s = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return 0.0


def extract_instance_rollout_from_output_jsonl(
    output_jsonl: Path,
    instance_id: str | None = None,
    include_tokens: bool = False,
) -> dict[str, Any]:
    """Extract concatenated token_ids/logprobs from EvalOutput output.jsonl."""
    if not output_jsonl.exists():
        raise FileNotFoundError(f"output_jsonl not found: {output_jsonl}")

    records: list[dict[str, Any]] = []
    with output_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if instance_id is not None:
        records = [r for r in records if r.get("instance_id") == instance_id]
        if not records:
            raise ValueError(f"instance_id {instance_id} not found in {output_jsonl}")
    else:
        if len(records) != 1:
            raise ValueError(
                f"{output_jsonl} contains {len(records)} records; pass --instance-id to select one"
            )

    rec = records[0]
    inferred_instance_id = rec.get("instance_id")

    history = rec.get("history") or []
    if not isinstance(history, list):
        raise ValueError("record.history is not a list")

    calls: list[tuple[float, int, dict[str, Any]]] = []
    # Keep original order index as tie-breaker if timestamps collide
    for idx, ev in enumerate(history):
        if not isinstance(ev, dict):
            continue
        tcm = ev.get("tool_call_metadata")
        if not isinstance(tcm, dict):
            continue
        mr = tcm.get("model_response")
        if not isinstance(mr, dict):
            continue
        choices = mr.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            continue
        lp = choices[0].get("logprobs")
        if not isinstance(lp, dict):
            continue
        content = lp.get("content")
        if not isinstance(content, list) or not content:
            continue

        ts = _parse_iso_ts(ev.get("timestamp"))
        calls.append((ts, idx, {"event": ev, "content": content}))

    # Sort by parsed timestamp then history index
    calls.sort(key=lambda x: (x[0], x[1]))

    token_ids: list[int] = []
    logprobs: list[float] = []
    tokens: list[str] | None = [] if include_tokens else None
    call_summaries: list[dict[str, Any]] = []

    for ts, idx, payload in calls:
        content = payload["content"]
        n_before = len(token_ids)
        for t in content:
            if not isinstance(t, dict):
                continue
            tid = t.get("token_id")
            lpv = t.get("sampling_logprob", t.get("logprob"))
            if tid is None or lpv is None:
                continue
            token_ids.append(int(tid))
            logprobs.append(float(lpv))
            if tokens is not None:
                tok = t.get("token")
                tokens.append(tok if isinstance(tok, str) else "")

        call_summaries.append(
            {
                "timestamp": ts,
                "history_index": idx,
                "event_id": payload["event"].get("id"),
                "function_name": (payload["event"].get("tool_call_metadata") or {}).get("function_name"),
                "n_tokens": len(token_ids) - n_before,
            }
        )

    mask = [1] * len(token_ids)
    out: dict[str, Any] = {
        "instance_id": inferred_instance_id,
        "output_jsonl": str(output_jsonl),
        "num_calls": len(calls),
        "calls": call_summaries,
        "token_ids": token_ids,
        "logprobs": logprobs,
        "mask": mask,
    }
    if tokens is not None:
        out["tokens"] = tokens
    return out


def extract_instance_rollout(
    llm_completions_dir: Path,
    instance_id: str | None = None,
    include_tokens: bool = False,
) -> dict[str, Any]:
    if not llm_completions_dir.exists():
        raise FileNotFoundError(f"llm_completions_dir not found: {llm_completions_dir}")
    if not llm_completions_dir.is_dir():
        raise ValueError(f"llm_completions_dir is not a directory: {llm_completions_dir}")

    files = sorted(p for p in llm_completions_dir.iterdir() if p.suffix == ".json")
    calls: list[tuple[float, str, Path, dict[str, Any]]] = []
    for p in files:
        d = _load_json(p)
        ts = d.get("timestamp")
        ts_f = float(ts) if isinstance(ts, (int, float)) else 0.0
        calls.append((ts_f, p.name, p, d))

    # Sort strictly by timestamp then filename (deterministic)
    calls.sort(key=lambda x: (x[0], x[1]))

    token_ids: list[int] = []
    logprobs: list[float] = []
    tokens: list[str] | None = [] if include_tokens else None

    call_summaries: list[dict[str, Any]] = []

    for ts, name, _path, d in calls:
        content = _extract_from_call(d)
        n_before = len(token_ids)

        for t in content:
            if not isinstance(t, dict):
                continue
            tid = t.get("token_id")
            lpv = t.get("sampling_logprob", t.get("logprob"))
            if tid is None or lpv is None:
                continue
            token_ids.append(int(tid))
            logprobs.append(float(lpv))
            if tokens is not None:
                tok = t.get("token")
                tokens.append(tok if isinstance(tok, str) else "")

        call_summaries.append(
            {
                "timestamp": ts,
                "file": name,
                "n_tokens": len(token_ids) - n_before,
            }
        )

    mask = [1] * len(token_ids)

    out: dict[str, Any] = {
        "instance_id": instance_id,
        "llm_completions_dir": str(llm_completions_dir),
        "num_calls": len(calls),
        "calls": call_summaries,
        "token_ids": token_ids,
        "logprobs": logprobs,
        "mask": mask,
    }
    if tokens is not None:
        out["tokens"] = tokens

    return out


def main() -> None:
    ap = argparse.ArgumentParser()

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--llm-completions-dir",
        type=str,
        help="Path to a single instance llm_completions directory (contains *.json)",
    )
    g.add_argument(
        "--eval-run-dir",
        type=str,
        help="Path to an eval run directory that contains llm_completions/<instance_id>/",
    )
    g.add_argument(
        "--output-jsonl",
        type=str,
        help="Path to OpenHands eval output.jsonl (extracts embedded logprobs from history)",
    )

    ap.add_argument(
        "--instance-id",
        type=str,
        default=None,
        help="Instance id (required with --eval-run-dir). With --output-jsonl, selects one record if multiple.",
    )
    ap.add_argument("--include-tokens", action="store_true", help="Include token strings in output")
    ap.add_argument("--out", type=str, default=None, help="Output JSON path (default: stdout)")
    ap.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON (indent=2). Default: enabled when writing --out, disabled on stdout.",
    )

    args = ap.parse_args()

    if args.output_jsonl:
        result = extract_instance_rollout_from_output_jsonl(
            Path(args.output_jsonl),
            instance_id=args.instance_id,
            include_tokens=args.include_tokens,
        )
    elif args.eval_run_dir:
        if not args.instance_id:
            raise SystemExit("--instance-id is required when using --eval-run-dir")
        llm_dir = Path(args.eval_run_dir) / "llm_completions" / args.instance_id
        instance_id = args.instance_id
        result = extract_instance_rollout(llm_dir, instance_id=instance_id, include_tokens=args.include_tokens)
    else:
        llm_dir = Path(args.llm_completions_dir)
        # If user doesn't pass instance-id, infer it from the directory name.
        instance_id = args.instance_id or llm_dir.name
        result = extract_instance_rollout(llm_dir, instance_id=instance_id, include_tokens=args.include_tokens)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # When writing to a file, pretty-print by default.
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    else:
        print(json.dumps(result, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
