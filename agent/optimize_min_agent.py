from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import dspy

from min_agent import DeepMiniAgent
from metrics import compute_metrics
from deep_agent import configure_lms  # reuse provider-specific LM setup


def _load_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _make_examples(rows: List[Dict[str, Any]]) -> List[dspy.Example]:
    examples: List[dspy.Example] = []
    for r in rows:
        # Use native Python types to match signatures (history: list, session: dict)
        sess_raw = r.get("session", {})
        hist_raw = r.get("history", [])
        sess = json.loads(sess_raw) if isinstance(sess_raw, str) else (sess_raw or {})
        hist = json.loads(hist_raw) if isinstance(hist_raw, str) else (hist_raw or [])
        ex = dspy.Example(
            user_input=r.get("user_input", ""),
            session=sess,
            history=hist,
        ).with_inputs("user_input", "session", "history")
        examples.append(ex)
    return examples


def _metric_wrapper(agent: DeepMiniAgent):
    def metric_fn(example: dspy.Example, pred: Any, trace: Any = None, **_) -> float:
        # Here example.session is dict and example.history is list to match signatures
        sess = example.session if isinstance(example.session, dict) else {}
        hist = example.history if isinstance(example.history, list) else []
        out = agent(user_input=example.user_input, session=sess, history=hist)
        m = compute_metrics(out["session"], out["plan"], out["agent_response"])
        return float(m.get("composite", 0.0))

    return metric_fn


def evaluate(agent: DeepMiniAgent, rows: List[Dict[str, Any]]) -> Dict[str, float]:
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for r in rows:
        sess_raw = r.get("session", {})
        hist_raw = r.get("history", [])
        sess = json.loads(sess_raw) if isinstance(sess_raw, str) else (sess_raw or {})
        hist = json.loads(hist_raw) if isinstance(hist_raw, str) else (hist_raw or [])
        out = agent(user_input=r.get("user_input", ""), session=sess, history=hist)
        m = compute_metrics(out["session"], out["plan"], out["agent_response"])
        for k, v in m.items():
            sums[k] = sums.get(k, 0.0) + float(v)
            counts[k] = counts.get(k, 0) + 1
    avg = {k: (sums[k] / max(1, counts.get(k, 1))) for k in sums}
    print(json.dumps(avg, ensure_ascii=False, indent=2))
    return avg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize/evaluate DeepMiniAgent with DSPy."
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["nebius", "azure"],
        required=True,
        help="LLM provider to use for DSPy.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=os.environ.get("TRAINING_DATA", "training_data.json"),
        help="Path to training data (JSON list).",
    )
    parser.add_argument(
        "--val",
        type=str,
        default=os.environ.get("VAL_DATA", ""),
        help="Optional validation data path (JSON list).",
    )
    parser.add_argument(
        "--tuned",
        type=str,
        default="tuned_min_agent.json",
        help="Path to save/load tuned agent.",
    )
    parser.add_argument(
        "--train", action="store_true", help="Run MIPROv2 teleprompting on --data."
    )
    parser.add_argument(
        "--eval", action="store_true", help="Evaluate the (tuned) agent on --data."
    )
    args = parser.parse_args()

    # Configure LMs globally (you can still pass per-agent LMs to DeepMiniAgent if desired)
    configure_lms(args.provider)

    agent: dspy.Module = DeepMiniAgent()

    if args.train and os.path.exists(args.data):
        train_rows = _load_rows(args.data)
        val_rows = (
            _load_rows(args.val) if args.val and os.path.exists(args.val) else None
        )
        trainset = _make_examples(train_rows)
        valset = _make_examples(val_rows) if val_rows else None

        tele = dspy.teleprompt.MIPROv2(
            metric=_metric_wrapper(agent), max_bootstrapped_demos=6, max_labeled_demos=6
        )
        tuned = tele.compile(agent, trainset=trainset, valset=valset)

        try:
            if hasattr(tuned, "save"):
                tuned.save(args.tuned)  # type: ignore[attr-defined]
            else:
                dspy.save(tuned, args.tuned)
            print(f"Saved tuned agent to {args.tuned}")
            agent = tuned
        except Exception as e:
            print(f"Warning: could not save tuned agent: {e}")

    # Try loading tuned agent if present and not just trained
    if not args.train and os.path.exists(args.tuned):
        try:
            if hasattr(DeepMiniAgent, "load"):
                agent = DeepMiniAgent.load(args.tuned)  # type: ignore[attr-defined]
            else:
                agent = dspy.load(args.tuned)
            print(f"Loaded tuned agent from {args.tuned}")
        except Exception as e:
            print(f"Failed to load tuned agent: {e}")

    if args.eval and os.path.exists(args.data):
        rows = _load_rows(args.data)
        evaluate(agent, rows)


if __name__ == "__main__":
    main()
