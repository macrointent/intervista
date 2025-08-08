# metrics.py
"""
Evaluation metrics and shared policy constants for the DeepAgent system.

This module centralizes:
- Policy constants used across planning/slot-filling.
- Per-turn metrics (each in [0,1] unless stated otherwise).
- A weighted composite metric for CI/dashboards.

Metric set:
1) state_machine_valid
2) inner_tool_valid
3) arg_completeness
4) provided_missing_consistent
5) slot_coverage_ratio
6) response_stage_alignment

Composite = weighted average with WEIGHTS.
"""

from typing import Dict, Any, Set

# ---------------------------------------------
# Policy constants (single source of truth)
# ---------------------------------------------

# Which outer tools are legal by stage.
ALLOWED_BY_STAGE: Dict[str, Set[str]] = {
    "portfolio": {"provide_category_overview", "schutzgarant_catalog"},
    "recommend": {"schutzgarant_order", "schutzgarant_catalog"},
    "form": {"schutzgarant_order"},
    "finalized": {"_mantix"},
}

# Allowed inner tools for the ORDER agent.
ORDER_INNER_TOOLS: Set[str] = {"get_form", "_mantix", "submit_form", "form_submit"}

# Minimal fields required to submit the order form.
MIN_REQUIRED: Set[str] = {
    "product_name",
    "first_name",
    "last_name",
    "email",
    "device_model",
    "purchase_date",
    "payment_method",
}

# Weights for the composite score.
WEIGHTS = {
    "state_machine_valid": 0.28,
    "inner_tool_valid": 0.17,
    "arg_completeness": 0.25,
    "provided_missing_consistent": 0.15,
    "slot_coverage_ratio": 0.10,
    "response_stage_alignment": 0.05,
}

# ---------------------------------------------
# Helpers
# ---------------------------------------------


def _filled_slots(session: Dict[str, Any]) -> Set[str]:
    """Keys in session.form_data that have truthy values."""
    form = session.get("form_data") or {}
    return {k for k, v in form.items() if v}


# ---------------------------------------------
# Individual metrics
# ---------------------------------------------


def state_machine_valid(session: Dict[str, Any], plan: Dict[str, Any]) -> float:
    """1.0 iff plan.function is allowed for session.stage; else 0.0."""
    stage = session.get("stage")
    fn = plan.get("function")
    return float(fn in ALLOWED_BY_STAGE.get(stage, set()))


def inner_tool_valid(plan: Dict[str, Any]) -> float:
    """If top-level is order agent, inner tool must be allowed; otherwise 1.0."""
    if plan.get("function") != "schutzgarant_order":
        return 1.0
    inner = (plan.get("arguments") or {}).get("function")
    return float(inner in ORDER_INNER_TOOLS)


def arg_completeness(session: Dict[str, Any], plan: Dict[str, Any]) -> float:
    """For submit_form: fraction of required fields already present; else 1.0."""
    if plan.get("function") != "schutzgarant_order":
        return 1.0
    args = plan.get("arguments") or {}
    inner_fn = args.get("function")
    if inner_fn not in {"submit_form", "form_submit"}:
        return 1.0
    inner_args = args.get("arguments") or {}
    provided = set(inner_args.get("provided", []) or [])
    have = provided | _filled_slots(session)
    need = MIN_REQUIRED
    return len(need & have) / max(1, len(need))


def provided_missing_consistent(session: Dict[str, Any], plan: Dict[str, Any]) -> float:
    """provided ∩ missing == ∅ and filled ⊆ (provided ∪ missing)."""
    if plan.get("function") != "schutzgarant_order":
        return 1.0
    inner_args = (plan.get("arguments") or {}).get("arguments") or {}
    prov = set(inner_args.get("provided", []) or [])
    miss = set(inner_args.get("missing", []) or [])
    if prov & miss:
        return 0.0
    filled = _filled_slots(session)
    return float(filled <= (prov | miss))


def slot_coverage_ratio(session: Dict[str, Any], plan: Dict[str, Any]) -> float:
    """Progress toward minimal submittable form."""
    inner_args = {}
    if plan.get("function") == "schutzgarant_order":
        inner_args = (plan.get("arguments") or {}).get("arguments") or {}
    provided = set(inner_args.get("provided", []) or [])
    have = provided | _filled_slots(session)
    return len(MIN_REQUIRED & have) / max(1, len(MIN_REQUIRED))


def response_stage_alignment(
    session: Dict[str, Any], agent_response: str, plan: Dict[str, Any]
) -> float:
    """In form stage with missing fields, does reply ask a question or mention a missing field?"""
    if session.get("stage") != "form":
        return 1.0
    inner = (
        (plan.get("arguments") or {}).get("arguments")
        if plan.get("function") == "schutzgarant_order"
        else {}
    )
    missing = set((inner or {}).get("missing", []) or [])
    if not missing:
        return 1.0
    resp = (agent_response or "").lower()
    mentions = any(m.lower() in resp for m in missing if isinstance(m, str))
    return 1.0 if ("?" in resp or mentions) else 0.0


# ---------------------------------------------
# Composite
# ---------------------------------------------


def compute_metrics(
    session: Dict[str, Any], plan: Dict[str, Any], agent_response: str
) -> Dict[str, float]:
    """Compute all per-turn metrics and the weighted composite."""
    m = {
        "state_machine_valid": state_machine_valid(session, plan),
        "inner_tool_valid": inner_tool_valid(plan),
        "arg_completeness": arg_completeness(session, plan),
        "provided_missing_consistent": provided_missing_consistent(session, plan),
        "slot_coverage_ratio": slot_coverage_ratio(session, plan),
        "response_stage_alignment": response_stage_alignment(
            session, agent_response, plan
        ),
    }
    m["composite"] = sum(WEIGHTS[k] * m[k] for k in WEIGHTS)
    return m
