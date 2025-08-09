from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import dspy

from helpers import (
    read_json,
    GLOBAL_MIN_REQUIRED,
    ensure_field_meta_from_schema,
    flat_required_from_schema,
    find_form_schema,
    provider_from_catalog,
    group_of,
    pick_next_group,
)
from adapters import register_adapter, get_adapter, ElementAdapter, HelvetiaAdapter


def load_prompts(path: str = "prompts.json") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


PROMPTS = load_prompts()


def _ctx_for(lm: Optional[dspy.LM]):
    # Best-effort DSPy context manager for per-agent LMs
    try:
        context = getattr(dspy, "context", None) or getattr(dspy.settings, "context", None)  # type: ignore[attr-defined]
        return context(lm=lm) if (context and lm is not None) else None
    except Exception:
        return None


"""
Lean agent design: three sub-agents (Planner, Catalog, Order) with unified LLM calls.
Catalog surfaces overview/list/details via one prompt. Order handles flow + extraction via one prompt.
"""


class PlannerAgent(dspy.Module):
    def __init__(self, lm: Optional[dspy.LM] = None) -> None:
        super().__init__()
        self.lm = lm
        self.predict = dspy.Predict(
            signature=dspy.Signature(
                PROMPTS["planner"]["signature"],
                PROMPTS["planner"]["instructions"],
            )
        )

    def forward(
        self, history: List[Dict[str, Any]], user_input: str, session: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        ctx = _ctx_for(self.lm)
        if ctx:
            with ctx:
                out = self.predict(
                    history=history,
                    user_input=user_input,
                    session=json.dumps(session, ensure_ascii=False),
                )
        else:
            out = self.predict(
                history=history,
                user_input=user_input,
                session=json.dumps(session, ensure_ascii=False),
            )
        fn = getattr(out, "function", "schutzgarant_catalog") or "schutzgarant_catalog"
        args_json = getattr(out, "arguments", "") or "{}"
        try:
            args = json.loads(args_json)
        except Exception:
            args = {}
        return fn, args


class CatalogAgent(dspy.Module):
    def __init__(
        self, lm_call: Optional[dspy.LM] = None, lm_respond: Optional[dspy.LM] = None
    ) -> None:
        super().__init__()
        self.lm_call = lm_call
        self.lm_respond = lm_respond
        # Single unified catalog predictor to reduce branching and hardcoded logic
        self.unified = dspy.Predict(
            signature=dspy.Signature(
                PROMPTS["catalog_unified"]["signature"],
                PROMPTS["catalog_unified"]["instructions"],
            )
        )

    def overview(self) -> str:
        payload = read_json("catalog.json")
        catalog_json = json.dumps(payload, ensure_ascii=False)
        ctx = _ctx_for(self.lm_respond)
        if ctx:
            with ctx:
                out = self.unified(
                    catalog=catalog_json,
                    query="",
                    session=json.dumps({}, ensure_ascii=False),
                    previous_items="[]",
                )
        else:
            out = self.unified(
                catalog=catalog_json,
                query="",
                session=json.dumps({}, ensure_ascii=False),
                previous_items="[]",
            )
        return (
            getattr(out, "response", None)
            or "Hier ist eine kurze Übersicht unseres Portfolios."
        )

    # No other helpers; unified prompt handles categorization vs details

    def forward(
        self, query: str, session: Dict[str, Any]
    ) -> Tuple[str, Optional[str], Optional[str], List[Dict[str, Any]]]:
        payload = read_json("catalog.json")
        catalog_json = json.dumps(payload, ensure_ascii=False)
        sjson = json.dumps(session or {}, ensure_ascii=False)
        prev_items = json.dumps(
            session.get("_last_shown_items", []), ensure_ascii=False
        )
        ctx = _ctx_for(self.lm_respond)
        if ctx:
            with ctx:
                out = self.unified(
                    catalog=catalog_json,
                    query=query or "",
                    session=sjson,
                    previous_items=prev_items,
                )
        else:
            out = self.unified(
                catalog=catalog_json,
                query=query or "",
                session=sjson,
                previous_items=prev_items,
            )
        text = getattr(out, "response", None) or ""
        pid = getattr(out, "selected_product_id", None) or None
        cat = getattr(out, "category_key", None) or None
        displayed = []
        try:
            displayed = json.loads(getattr(out, "displayed_items", "[]") or "[]")
        except Exception:
            displayed = []
        return (
            text,
            (str(pid).strip() if pid else None),
            (str(cat).strip() if cat else None),
            displayed if isinstance(displayed, list) else [],
        )


class OrderAgent(dspy.Module):
    def __init__(self, lm_unified: Optional[dspy.LM] = None) -> None:
        super().__init__()
        self.lm_unified = lm_unified
        self.unified = dspy.Predict(
            signature=dspy.Signature(
                PROMPTS["order_unified"]["signature"],
                PROMPTS["order_unified"]["instructions"],
            )
        )

    def forward(
        self, history: List[Dict[str, Any]], user_input: str, session: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        s = dict(session or {})

        # Load/derive schema meta once (labels, required)
        schema = (
            find_form_schema(product=s.get("product"), form_id=s.get("product_id"))
            or {}
        )
        if schema:
            s["labels"] = ensure_field_meta_from_schema(s.get("labels") or {}, schema)
            s["required_now"] = list(sorted(flat_required_from_schema(schema)))
        else:
            s.setdefault("labels", {})
            s.setdefault("required_now", list(sorted(GLOBAL_MIN_REQUIRED)))
        # Provide friendly options for certain fields (e.g., payment)
        opts = s.get("field_options") or {}
        opts.setdefault("payment_method", ["SEPA/IBAN", "Kreditkarte", "PayPal"])
        s["field_options"] = opts

        # Delegate flow + extraction + response to a single LLM call
        ctx = _ctx_for(self.lm_unified)
        sjson = json.dumps(s, ensure_ascii=False)
        schema_json = json.dumps(schema, ensure_ascii=False)
        if ctx:
            with ctx:
                out = self.unified(
                    history=history,
                    user_input=user_input,
                    session=sjson,
                    schema=schema_json,
                )
        else:
            out = self.unified(
                history=history,
                user_input=user_input,
                session=sjson,
                schema=schema_json,
            )

        # Merge updated form and stage if present
        try:
            updated_form_raw = getattr(out, "updated_form", "{}") or "{}"
            updated_form = (
                json.loads(updated_form_raw)
                if isinstance(updated_form_raw, str)
                else updated_form_raw
            )
            if isinstance(updated_form, dict) and updated_form:
                base = s.get("form_data") or {}
                base.update(updated_form)
                s["form_data"] = base
        except Exception:
            # Ignore malformed updated_form from the model
            pass
        stage = getattr(out, "stage", None)
        if stage:
            s["stage"] = stage

        # Maintain robust tracking for provided/missing and grouping hints
        form = s.get("form_data") or {}
        filled_keys = [k for k, v in form.items() if v]
        req_now = s.get("required_now") or []
        missing_keys = [k for k in req_now if k not in form or not form.get(k)]
        s["provided"] = sorted(filled_keys)
        s["missing"] = sorted(missing_keys)
        s["groups"] = {k: group_of(k) for k in set(filled_keys) | set(missing_keys)}
        s.setdefault(
            "group_order", ["person", "kontakt", "geraet", "zahlung", "vertrag"]
        )
        s["next_group"] = pick_next_group(missing_keys) if missing_keys else None

        # Surface last inner decision for metric alignment
        inner_fn = getattr(out, "inner_function", None)
        if isinstance(inner_fn, str) and inner_fn.strip():
            s["last_inner_function"] = inner_fn.strip()

        surface = getattr(out, "response", None) or ""
        # Optional provider adapter step on submit/finalize
        inner_fn = getattr(out, "inner_function", None)
        if s.get("stage") == "finalized" or (
            isinstance(inner_fn, str) and inner_fn.strip() == "submit_form"
        ):
            form = s.get("form_data") or {}
            prov_key = (
                s.get("provider")
                or provider_from_catalog(s.get("product"), s.get("product_id"))
                or "ELEMENT"
            )
            adapter = get_adapter(prov_key)
            if adapter:
                extra_missing = adapter.validate(form)
                if extra_missing:
                    # Return to form stage and ensure these are included in required_now
                    s["stage"] = "form"
                    req = set(s.get("required_now") or []) | set(extra_missing)
                    s["required_now"] = list(sorted(req))
                else:
                    try:
                        payload = adapter.to_payload(form)
                        # Optionally attach for downstream submission
                        s["_provider_payload"] = payload
                    except Exception:
                        s["stage"] = "form"
            else:
                # No adapter available → stay in form to avoid false finalize
                s["stage"] = "form"
        s.setdefault("stage", "form")
        return surface, s


class DeepMiniAgent(dspy.Module):
    """A compact, hierarchical deep-agent with sub-agents and per-agent prompts/LLMs."""

    def __init__(
        self,
        planner_lm: Optional[dspy.LM] = None,
        catalog_call_lm: Optional[dspy.LM] = None,
        catalog_respond_lm: Optional[dspy.LM] = None,
        order_call_lm: Optional[dspy.LM] = None,
        order_respond_lm: Optional[dspy.LM] = None,
    ) -> None:
        super().__init__()
        self.planner = PlannerAgent(lm=planner_lm)
        self.catalog = CatalogAgent(
            lm_call=catalog_call_lm, lm_respond=catalog_respond_lm
        )
        self.order = OrderAgent(lm_unified=order_respond_lm)

    def forward(
        self, user_input: str, session: Dict[str, Any], history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        s = dict(session or {})
        fn, args = self.planner(history=history, user_input=user_input, session=s)

        plan: Dict[str, Any] = {"function": fn, "arguments": args}
        # Heuristic override to avoid getting stuck in endless overview
        if fn == "provide_category_overview" and (
            s.get("stage") == "portfolio" and (user_input or "").strip()
        ):
            fn = "schutzgarant_catalog"
            args = {"query": user_input}
            plan = {"function": fn, "arguments": args}

        if fn == "provide_category_overview":
            sub = self.catalog.overview()
            s.setdefault("stage", "portfolio")
        elif fn == "schutzgarant_catalog":
            query = args.get("query") or user_input
            sub, pid, cat, items = self.catalog(query=query, session=s)
            # Optionally keep light state updates based on LLM hints
            if pid and not s.get("product_id"):
                s["product_id"] = int(pid) if pid.isdigit() else pid
            if cat and not s.get("product"):
                s["product"] = cat
            if items:
                s["_last_shown_items"] = items
            s.setdefault("stage", "recommend")
        elif fn == "schutzgarant_order":
            sub, s = self.order(history=history, user_input=user_input, session=s)
            # Backfill plan inner function and arguments for metrics
            inner_guess = s.get("last_inner_function") or (
                "submit_form"
                if s.get("stage") == "finalized"
                else ("_mantix" if (user_input or "").strip() else "get_form")
            )
            plan.setdefault("arguments", {})["function"] = inner_guess
            plan["arguments"]["arguments"] = {
                "provided": s.get("provided", []),
                "missing": s.get("missing", []),
            }
        else:
            # Default to catalog if unsure
            sub = self.catalog(query=user_input)
            s.setdefault("stage", "recommend")

        return {"plan": plan, "agent_response": sub, "session": s}


# Register default adapters for provider payloads
register_adapter("ELEMENT", ElementAdapter())
register_adapter("HELVETIA", HelvetiaAdapter())


if __name__ == "__main__":
    # Minimal interactive REPL for DeepMiniAgent
    from deep_agent import configure_lms

    parser = argparse.ArgumentParser(description="DeepMiniAgent REPL")
    parser.add_argument(
        "--provider",
        type=str,
        choices=["nebius", "azure"],
        required=True,
        help="LLM provider to use.",
    )
    parser.add_argument(
        "--tuned",
        type=str,
        default=os.environ.get("TUNED_MIN_AGENT", ""),
        help="Optional path to a tuned agent (dspy.load).",
    )
    args = parser.parse_args()

    configure_lms(args.provider)
    agent: dspy.Module = DeepMiniAgent()
    if args.tuned and os.path.exists(args.tuned):
        try:
            agent.load(args.tuned)
            print(f"Loaded tuned agent from {args.tuned}")
        except Exception as e:
            print(f"Failed to load tuned agent: {e}")

    session: Dict[str, Any] = {}
    history: List[Dict[str, Any]] = []
    print("\nAgent: Hello! Type 'exit' to quit.")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in {"exit", "quit"}:
            break
        history.append({"role": "user", "content": user_input})
        out = agent(user_input=user_input, session=session, history=history)
        reply = out["agent_response"]
        print("Agent:", reply)
        session = out["session"]
        history.append({"role": "agent", "content": reply})
