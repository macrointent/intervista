# deep_agent.py
"""
DeepAgent (simplified):

- Sub-agents: Catalog, Order, ProductExplainer; Planner; FinalComposer
- No prompt override function, no pagination
- Contract details are shown when a product is selected (or inferred)
- LLM-only slot extraction guided by expected fields (no regex)
- Training with DSPy MIPROv2 and per-example metric breakdown
- Provider-specific LM setup for Nebius / Azure

Dependencies:
- helpers.py
- adapters.py
- metrics.py
- prompts.json
"""

from __future__ import annotations

import argparse
import json
import os
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

import dspy
from dspy import Module
from dspy.teleprompt import MIPROv2
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

# External modules
from metrics import MIN_REQUIRED, ORDER_INNER_TOOLS, compute_metrics
from adapters import register_adapter, get_adapter, ElementAdapter, HelvetiaAdapter
from helpers import (
    # catalog helpers
    read_json,
    warm_catalog,
    get_catalog_indices,
    provider_from_catalog,
    compact_catalog,
    top_k,
    find_catalog_entry_by_query,
    stringify_product_details,
    # category helpers
    normalize_category,
    filter_by_category,
    # forms + labels helpers
    FieldMeta,
    FIELD_META,
    GLOBAL_MIN_REQUIRED,
    SCHEMA_TO_FLAT,
    FALLBACK_META,
    load_forms,
    find_form_schema,
    label_for,
    group_of,
    pick_next_group,
    format_group_ask,
    derive_required,
    ensure_field_meta_from_schema,
    flat_required_from_schema,
    user_wants_to_skip,
)

load_dotenv()

CATALOG_FILE = os.getenv("CATALOG_FILE", "catalog.json")
PROMPTS_FILE = os.getenv("PROMPTS_FILE", "prompts.json")


# =========================
# Provider-specific LM init
# =========================
def configure_lms(provider: str) -> None:
    """
    Initialize DSPy with a main LM and an evaluator LM for the selected provider.

    Providers:
      - "nebius": OpenAI-compatible endpoint (NEBIUS_BASE / NEBIUS_KEY)
      - "azure":  Azure OpenAI (AZURE_* envs). Use deployment names, not raw model ids.

    Nebius env:
      NEBIUS_BASE, NEBIUS_KEY
      NEBIUS_MAIN_MODEL (default: "nebius/Qwen/Qwen3-235B-A22B-Instruct-2507")
      NEBIUS_EVAL_MODEL (default: "nebius/Qwen/Qwen3-4B-fast")

    Azure env:
      AZURE_OPENAI_API_BASE, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION
      AZURE_OPENAI_DEPLOYMENT_NAME (default: "gpt-4.1")
      AZURE_OPENAI_EVALUATOR_DEPLOYMENT_NAME (default: "gpt-4.1-mini")
    """
    provider = (provider or "").strip().lower()

    if provider == "nebius":
        base = os.getenv("NEBIUS_BASE")
        key = os.getenv("NEBIUS_KEY")
        main_model = os.getenv(
            "NEBIUS_MAIN_MODEL", "nebius/Qwen/Qwen3-235B-A22B-Instruct-2507"
        )
        eval_model = os.getenv("NEBIUS_EVAL_MODEL", "nebius/Qwen/Qwen3-4B-fast")
        if not base or not key:
            raise RuntimeError("Nebius config missing: set NEBIUS_BASE and NEBIUS_KEY")

        print(f"Using Nebius provider.\n - main: {main_model}\n - eval: {eval_model}")
        main_llm = dspy.LM(
            model=main_model, api_base=base, api_key=key, model_provider="openai"
        )
        evaluator_llm = dspy.LM(
            model=eval_model, api_base=base, api_key=key, model_provider="openai"
        )

    elif provider == "azure":
        base = os.getenv("AZURE_OPENAI_API_BASE")
        key = os.getenv("AZURE_OPENAI_API_KEY")
        ver = os.getenv("AZURE_OPENAI_API_VERSION")
        main_depl = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1")
        eval_depl = os.getenv("AZURE_OPENAI_EVALUATOR_DEPLOYMENT_NAME", "gpt-4.1-mini")
        if not base or not key or not ver:
            raise RuntimeError(
                "Azure config missing: set AZURE_OPENAI_API_BASE, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION"
            )

        main_model = f"azure/{main_depl}"
        eval_model = f"azure/{eval_depl}"
        print(
            f"Using Azure OpenAI provider.\n - main: {main_model}\n - eval: {eval_model}"
        )
        main_llm = dspy.LM(
            model=main_model, api_base=base, api_key=key, api_version=ver
        )
        evaluator_llm = dspy.LM(
            model=eval_model, api_base=base, api_key=key, api_version=ver
        )

    else:
        raise ValueError("Unsupported --provider. Use 'nebius' or 'azure'.")

    dspy.configure(lm=main_llm, llm=evaluator_llm)


# ===============
# Prompt loading
# ===============
class PromptPack(BaseModel):
    """Typed container for a single prompt pack (signature + instructions)."""

    signature: str
    instructions: str


class Prompts(BaseModel):
    """Typed container for all prompts used by the agent."""

    category_overview: PromptPack
    product_explainer: PromptPack
    catalog_responder: PromptPack
    form_updater: PromptPack
    order_responder: PromptPack
    planner: PromptPack
    final_composer: PromptPack


def load_prompts(path: str = PROMPTS_FILE) -> Prompts:
    """Load and validate prompts.json into a typed Prompts object."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Prompts.model_validate(data)


PROMPTS: Prompts = load_prompts()


# ======================
# Core runtime classes
# ======================
class Stage(str, Enum):
    """Conversation stages used by the state machine."""

    portfolio = "portfolio"
    recommend = "recommend"
    form = "form"
    finalized = "finalized"


class Session(BaseModel):
    """
    Conversational session object exchanged between turns.
    """

    stage: Stage = Stage.portfolio
    product: str = ""
    form_data: Dict[str, Any] = Field(default_factory=dict)
    form_schema_present: bool = False
    labels: Dict[str, str] = Field(default_factory=dict)
    required_now: Set[str] = Field(default_factory=set)
    provider: Optional[str] = None
    product_id: Optional[int] = None
    # Remember last selected product id to show details when user says "ich will den tarif"
    last_selected_product_id: Optional[int] = None


class ToolCall(BaseModel):
    """Represents a single tool call in the agent's nested trace."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    parent_id: Optional[str] = None


class ToolResult(BaseModel):
    """Represents a tool result in the nested trace."""

    tool_call_id: str
    output: Any


class Plan(BaseModel):
    """Planner output holding the chosen function, arguments, and trace."""

    function: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    tool_trace: List[ToolCall] = Field(default_factory=list)
    tool_results: List[ToolResult] = Field(default_factory=list)


class AgentReturn(BaseModel):
    """Unified return payload for a single turn."""

    plan: Plan
    agent_response: str
    session: Session


class ProductExplainer(Module):
    """Turns a single product entry into a concise, friendly details block."""

    def __init__(self) -> None:
        super().__init__()
        self.responder = dspy.Predict(
            signature=dspy.Signature(
                PROMPTS.product_explainer.signature,
                PROMPTS.product_explainer.instructions,
            )
        )

    def forward(self, product_entry: Dict[str, Any]) -> str:
        """Generate a short details text for a product entry."""
        try:
            pj = json.dumps(product_entry, ensure_ascii=False)
        except Exception:
            pj = "{}"
        out = self.responder(product_json=pj)
        if hasattr(out, "details") and out.details:
            return out.details
        return stringify_product_details(product_entry)


class SchutzgarantCatalog(Module):
    """
    Provides a compact recommendation list (up to 3) and appends a detail block
    when the query resolves to a single product.
    """

    def __init__(self) -> None:
        super().__init__()
        self.responder = dspy.Predict(
            signature=dspy.Signature(
                PROMPTS.catalog_responder.signature,
                PROMPTS.catalog_responder.instructions,
            )
        )
        self.explainer = ProductExplainer()

    def get_catalog_json(self) -> str:
        """Return the raw catalog JSON as a string."""
        payload = read_json(CATALOG_FILE)
        return json.dumps(payload, ensure_ascii=False)

    def forward(
        self,
        history: List[Dict[str, Any]],
        query: str,
        session: Session,
        inner_function: str = "get_catalog",
    ) -> Tuple[str, Session]:
        """
        Produce a compact recommendation list and, if applicable, a detail block.
        Also supports a basic category overview for recognized categories.
        """
        raw = read_json(CATALOG_FILE)
        q = (query or "").strip().lower()

        s = session.model_copy(deep=True)
        s.stage = Stage.recommend

        # Category query → show a simple portfolio list (no pagination)
        category = normalize_category(q)
        if category:
            items = filter_by_category(raw, category)
            if not items:
                return ("Ich habe keine passenden Produkte gefunden.", s)
            # Simple list (max 8 to avoid spam)
            items = items[:8]

            def line(e: Dict[str, Any]) -> str:
                bits = []
                if e.get("theft") is not None:
                    bits.append(f"Diebstahl: {e['theft']}")
                if e.get("deductible"):
                    bits.append(f"SB: {e['deductible']}")
                if e.get("price"):
                    bits.append(f"Beitrag: {e['price']}")
                tail = (", ".join(bits[:2])) if bits else "Details auf Anfrage"
                return f"• **{e['name']}** — {tail}"

            listing = "\n".join(line(e) for e in items)
            msg = f"Hier ist unser Portfolio **{category}**:\n\n{listing}\n\nMöchtest du einen Tarif genauer sehen?"
            return msg, s

        # Otherwise: relevance-ranked top 3
        picks = top_k(compact_catalog(raw), query, k=3)

        def line(e: Dict[str, Any]) -> str:
            bits = []
            if e.get("theft") is not None:
                bits.append(f"Diebstahl: {e['theft']}")
            if e.get("deductible"):
                bits.append(f"SB: {e['deductible']}")
            if e.get("price"):
                bits.append(f"Beitrag: {e['price']}")
            tail = (", ".join(bits[:2])) if bits else "Details auf Anfrage"
            return f"• **{e['name']}** — {tail}"

        fallback_list = (
            "\n".join(line(e) for e in picks)
            if picks
            else "Leider keine passenden Produkte gefunden."
        )

        try:
            items_json = json.dumps(picks, ensure_ascii=False)
            polished = self.responder(items=items_json, query=query).response
        except Exception:
            polished = ""

        summary = (polished or "").strip() or fallback_list

        # If query refers to a specific product, add details and remember its id
        details_block = ""
        entry = find_catalog_entry_by_query(query)
        if entry:
            s.last_selected_product_id = (
                int(entry.get("id")) if entry.get("id") is not None else None
            )
            details_block = "\n\n" + self.explainer(entry)

        msg = (
            summary.rstrip()
            + details_block
            + "\n\nSoll ich mit einem dieser Tarife starten?"
        )
        return msg, s


class FormUpdater(Module):
    """
    Parses the user's free-text reply to update form fields using the LLM only.
    We add hints (expected fields, labels) inside 'current_form' to steer extraction.
    """

    def __init__(self, minimal_required: Set[str] = None) -> None:
        super().__init__()
        self.minimal_required = minimal_required or MIN_REQUIRED
        self.responder = dspy.Predict(
            signature=dspy.Signature(
                PROMPTS.form_updater.signature, PROMPTS.form_updater.instructions
            )
        )

    def forward(
        self, current_form: Dict[str, Any], user_input: str
    ) -> Tuple[Dict[str, Any], List[str], List[str]]:
        """Update form data and compute provided/missing keys via LLM."""
        try:
            current_json = json.dumps(current_form, ensure_ascii=False)
            out = self.responder(current_form=current_json, user_input=user_input)
            upd = json.loads(getattr(out, "updated_form", current_json) or current_json)
            provided = json.loads(getattr(out, "provided", "[]") or "[]")
            missing = json.loads(getattr(out, "missing", "[]") or "[]")
        except Exception:
            upd, provided, missing = current_form, [], []

        provided = [p for p in provided if isinstance(p, str)]
        missing = [m for m in missing if isinstance(m, str)]

        filled = {k for k, v in (upd or {}).items() if v}
        for req in self.minimal_required:
            if req not in filled and req not in missing and req not in provided:
                missing.append(req)

        missing = sorted(list(dict.fromkeys(missing)))
        provided = sorted(list(dict.fromkeys(provided)))
        return upd, provided, missing


class SchutzgarantOrder(Module):
    """
    Order flow:
      - get_form: load schema or derive requireds; ask clustered questions; prepend product details
      - _mantix: slot-filling turns with LLM-only extraction guided by expected fields
      - submit_form: validate, build provider payload, finalize
    """

    def __init__(self, form_updater: Optional[FormUpdater] = None) -> None:
        super().__init__()
        self.form_updater = form_updater or FormUpdater()
        self.responder = dspy.Predict(
            signature=dspy.Signature(
                PROMPTS.order_responder.signature, PROMPTS.order_responder.instructions
            )
        )
        self.explainer = ProductExplainer()

    def _handle_get_form(self, session: Session) -> Tuple[str, Session, Dict[str, Any]]:
        """Bootstrap the form: schema, requireds, first cluster; prepend product details if known."""
        s = session.model_copy(deep=True)
        s.form_schema_present = True
        s.stage = Stage.form

        # Prefer last selected product id if product not set explicitly
        if not s.product_id and s.last_selected_product_id:
            s.product_id = s.last_selected_product_id

        schema = find_form_schema(product=s.product or None, form_id=s.product_id)
        if schema:
            s.labels = ensure_field_meta_from_schema(s.labels or {}, schema)
            s.required_now = flat_required_from_schema(schema)
        else:
            if not s.labels:
                s.labels = {k: v.label_de for k, v in FIELD_META.items()}
            s.required_now = derive_required(
                (s.product or "").lower(), schema_required=None
            )

        provided, missing = [], sorted(list(s.required_now))
        ask_text = (
            format_group_ask(pick_next_group(missing), missing, s.labels, max_items=3)
            if missing
            else "Super, ich habe schon alles. Soll ich den Antrag absenden?"
        )

        details_block = ""
        if s.product or s.product_id is not None:
            warm_catalog()
            by_id, by_name = get_catalog_indices()
            entry = None
            if s.product_id is not None:
                entry = by_id.get(str(s.product_id))
            if not entry and s.product:
                entry = by_name.get(s.product.strip().lower())
            if entry:
                details_block = self.explainer(entry) + "\n\n"

        surface = (
            self.responder(
                history=[],
                user_input="",
                session=s.model_dump_json(),
                inner_function="get_form",
                provided=json.dumps(provided, ensure_ascii=False),
                missing=json.dumps(missing, ensure_ascii=False),
            ).response
            or ask_text
        )
        text = (details_block + surface).strip()
        return text, s, {"provided": provided, "missing": missing}

    def _handle_mantix(
        self, session: Session, user_input: str
    ) -> Tuple[str, Session, Dict[str, Any]]:
        """Slot-filling turn: LLM-only extraction guided by expected fields; ask next cluster or propose submit."""
        # Align updater with current required set
        self.form_updater.minimal_required = set(
            session.required_now or GLOBAL_MIN_REQUIRED
        )

        current_form = session.form_data or {}
        filled = {k for k, v in current_form.items() if v}
        missing = sorted(list(self.form_updater.minimal_required - filled))

        # Build augmented form with hints for the LLM
        augmented_form = dict(current_form)
        augmented_form["__expected_fields"] = missing
        augmented_form["__all_required_min"] = list(self.form_updater.minimal_required)
        if session.labels:
            augmented_form["__labels"] = session.labels

        if user_wants_to_skip(user_input):
            updated = current_form
            provided = []
            # keep 'missing' as-is
        else:
            updated, provided, _ = self.form_updater(
                current_form=augmented_form, user_input=user_input
            )
            # Recompute missing after extraction
            filled = {k for k, v in (updated or {}).items() if v}
            missing = sorted(list(self.form_updater.minimal_required - filled))

        s = session.model_copy(deep=True)
        if not s.labels:
            s.labels = {k: v.label_de for k, v in FIELD_META.items()}
        s.form_data = updated

        draft = (
            format_group_ask(pick_next_group(missing), missing, s.labels, max_items=3)
            if missing
            else "Danke! Ich habe alle nötigen Angaben. Soll ich den Antrag jetzt absenden?"
        )

        surface = (
            self.responder(
                history=[],
                user_input=user_input,
                session=s.model_dump_json(),
                inner_function="_mantix",
                provided=json.dumps(provided, ensure_ascii=False),
                missing=json.dumps(missing, ensure_ascii=False),
            ).response
            or draft
        )

        return (surface or draft), s, {"provided": provided, "missing": missing}

    def _handle_submit(self, session: Session) -> Tuple[str, Session, Dict[str, Any]]:
        """Validate requireds, build provider payload, or ask for missing fields; finalize on success."""
        req = set(session.required_now or GLOBAL_MIN_REQUIRED)
        form = session.form_data or {}
        filled = {k for k, v in form.items() if v}
        missing = [f for f in req if f not in filled]

        s = session.model_copy(deep=True)
        if not s.labels:
            s.labels = {k: v.label_de for k, v in FIELD_META.items()}

        inner_args = {"provided": sorted(list(filled)), "missing": missing}
        if missing:
            nice = ", ".join(label_for(k, s.labels) for k in missing)
            base = f"Ich kann noch nicht absenden. Es fehlen: {nice}. Bitte ergänze diese Angaben."
            surface = (
                self.responder(
                    history=[],
                    user_input="",
                    session=s.model_dump_json(),
                    inner_function="submit_form",
                    provided=json.dumps(list(filled), ensure_ascii=False),
                    missing=json.dumps(missing, ensure_ascii=False),
                ).response
                or base
            )
            s.stage = Stage.form
            return (surface or base), s, inner_args

        provider_key = (
            s.provider or provider_from_catalog(s.product, s.product_id) or "ELEMENT"
        )
        adapter = get_adapter(provider_key)
        if not adapter:
            s.stage = Stage.form
            return ("Interner Fehler: kein passender Adapter gefunden.", s, inner_args)

        extra_missing = adapter.validate(form)
        if extra_missing:
            nice = ", ".join(label_for(k, s.labels) for k in extra_missing)
            s.stage = Stage.form
            msg = f"Es fehlt noch etwas für die Einreichung: {nice}."
            surface = (
                self.responder(
                    history=[],
                    user_input="",
                    session=s.model_dump_json(),
                    inner_function="submit_form",
                    provided=json.dumps(list(filled), ensure_ascii=False),
                    missing=json.dumps(extra_missing, ensure_ascii=False),
                ).response
                or msg
            )
            return (surface or msg), s, inner_args

        payload = adapter.to_payload(form)
        # TODO: submit payload to backend

        s.stage = Stage.finalized
        final_msg = "Top! Dein Antrag ist eingereicht. Du bekommst gleich eine Bestätigung per E-Mail."
        surface = (
            self.responder(
                history=[],
                user_input="",
                session=s.model_dump_json(),
                inner_function="submit_form",
                provided=json.dumps(list(filled), ensure_ascii=False),
                missing="[]",
            ).response
            or final_msg
        )
        return (surface or final_msg), s, inner_args

    def forward(
        self,
        history: List[Dict[str, Any]],
        user_input: str,
        session: Session,
        inner_function: str,
    ) -> Tuple[str, Session, Dict[str, Any]]:
        """Route to the appropriate inner function in the order flow."""
        inner_function = (inner_function or "").strip()
        if inner_function == "get_form":
            return self._handle_get_form(session)
        elif inner_function == "_mantix":
            return self._handle_mantix(session, user_input)
        elif inner_function in {"submit_form", "form_submit"}:
            return self._handle_submit(session)
        else:
            return self._handle_get_form(session)


class Planner(Module):
    """Decides which top-level function to call next and with what arguments."""

    def __init__(self) -> None:
        super().__init__()
        self.responder = dspy.Predict(
            signature=dspy.Signature(
                PROMPTS.planner.signature, PROMPTS.planner.instructions
            )
        )

    def forward(
        self, history: List[Dict[str, Any]], user_input: str, session: Session
    ) -> Plan:
        """Return a Plan with function, arguments, and first tool trace entry."""
        raw = self.responder(
            history=history, user_input=user_input, session=session.model_dump_json()
        )
        function = getattr(raw, "function", None) or "_mantix"
        arguments_str = getattr(raw, "arguments", "") or ""
        try:
            arguments = json.loads(arguments_str) if arguments_str else {}
        except Exception:
            arguments = {}

        if function == "schutzgarant_order":
            inner = arguments.get("function")
            if inner not in ORDER_INNER_TOOLS:
                arguments["function"] = "get_form"
        elif function == "schutzgarant_catalog":
            arguments["function"] = "get_catalog"

        tool_call = ToolCall(
            name="planner.route",
            arguments={
                "user_input": user_input,
                "session": session.model_dump(mode="json"),
            },
        )
        return Plan(function=function, arguments=arguments, tool_trace=[tool_call])


class FinalComposer(Module):
    """Lightly polishes the sub-response into the final user-facing message."""

    def __init__(self) -> None:
        super().__init__()
        self.responder = dspy.Predict(
            signature=dspy.Signature(
                PROMPTS.final_composer.signature, PROMPTS.final_composer.instructions
            )
        )

    def forward(
        self,
        sub_response: str,
        plan: Plan,
        session: Session,
        history: List[Dict[str, Any]],
    ) -> str:
        """Compose the final message from the sub-response."""
        out = self.responder(
            sub_response=sub_response,
            plan=json.dumps(plan.model_dump(mode="json"), ensure_ascii=False),
            session=session.model_dump_json(),
            history=history,
        )
        return (
            out.final_response
            if hasattr(out, "final_response")
            else (sub_response or "")
        )


class DeepAgent(Module):
    """Top-level orchestrator: planner → sub-agent → composer, with strict traces."""

    def __init__(self) -> None:
        super().__init__()
        self.planner = Planner()
        self.catalog = SchutzgarantCatalog()
        self.order = SchutzgarantOrder()
        self.composer = FinalComposer()
        self.explainer = ProductExplainer()

        # Collapsed small predictors
        self._category_overview_predictor = dspy.Predict(
            signature=dspy.Signature(
                PROMPTS.category_overview.signature,
                PROMPTS.category_overview.instructions,
            )
        )
        self._generic_predictor = dspy.Predict(
            signature=dspy.Signature(
                PROMPTS.catalog_responder.signature.replace(
                    "items: str, query: str -> response: str",
                    "history: list, user_input: str -> response: str",
                ),
                "Antworte hilfreich und kurz basierend auf dem bisherigen Verlauf.",
            )
        )

    def _maybe_append_product_details(
        self, text: str, session: Session, user_input: str
    ) -> str:
        """If a product is implied, prepend its details block."""
        warm_catalog()
        by_id, by_name = get_catalog_indices()
        entry = None
        if session.product_id is not None:
            entry = by_id.get(str(session.product_id))
        if not entry and session.product:
            entry = by_name.get(session.product.strip().lower())
        if not entry and user_input:
            entry = find_catalog_entry_by_query(user_input)
        if entry:
            details = self.explainer(entry)
            if details and details not in text:
                return f"{details}\n\n{text}"
        return text

    def forward(
        self, user_input: str, session: Dict[str, Any], history: List[Dict[str, Any]]
    ) -> AgentReturn:
        """Run a single turn: plan → sub-agent → compose; return updated session + strict trace."""
        try:
            s = Session.model_validate(session or {})
        except ValidationError:
            s = Session()

        plan = self.planner(history=history, user_input=user_input, session=s)
        tool_trace = list(plan.tool_trace)
        tool_results: List[ToolResult] = []

        intermediate_reply = ""
        new_session = s.model_copy(deep=True)
        plan_dict = plan.model_dump()

        # If entering order, use planner hints or last selected id
        if plan.function == "schutzgarant_order":
            maybe_product = plan.arguments.get("product")
            maybe_product_id = plan.arguments.get("product_id")
            if maybe_product:
                new_session.product = maybe_product.strip()
            if maybe_product_id is not None:
                try:
                    new_session.product_id = int(maybe_product_id)
                except Exception:
                    pass
            if not new_session.product_id and new_session.last_selected_product_id:
                new_session.product_id = new_session.last_selected_product_id
            prov = provider_from_catalog(
                new_session.product or None, new_session.product_id
            )
            if prov:
                new_session.provider = prov

        if plan.function == "provide_category_overview":
            catalog_json = self.catalog.get_catalog_json()
            call = ToolCall(
                name="provide_category_overview",
                arguments={"catalog_json_len": len(catalog_json)},
                parent_id=tool_trace[-1].id if tool_trace else None,
            )
            out = self._category_overview_predictor(catalog=catalog_json)
            sub = out.category_summary if hasattr(out, "category_summary") else str(out)
            tool_trace.append(call)
            tool_results.append(ToolResult(tool_call_id=call.id, output=sub))
            sub = self._maybe_append_product_details(sub, new_session, user_input)
            intermediate_reply = sub
            new_session.stage = Stage.portfolio

        elif plan.function == "schutzgarant_catalog":
            inner = plan.arguments.get("function", "get_catalog")
            query = plan.arguments.get("query", user_input)
            call = ToolCall(
                name="schutzgarant_catalog.get_catalog",
                arguments={"query": query},
                parent_id=tool_trace[-1].id if tool_trace else None,
            )
            sub, new_session = self.catalog(
                history=history, query=query, session=new_session, inner_function=inner
            )
            tool_trace.append(call)
            tool_results.append(ToolResult(tool_call_id=call.id, output=sub))
            intermediate_reply = sub

        elif plan.function == "schutzgarant_order":
            inner = plan.arguments.get("function") or "get_form"
            call = ToolCall(
                name=f"schutzgarant_order.{inner}",
                arguments={"inner_function": inner},
                parent_id=tool_trace[-1].id if tool_trace else None,
            )
            sub, new_session, inner_args = self.order(
                history=history,
                user_input=user_input,
                session=new_session,
                inner_function=inner,
            )
            plan_dict.setdefault("arguments", {})["arguments"] = inner_args
            plan = Plan.model_validate(plan_dict)
            tool_trace.append(call)
            tool_results.append(
                ToolResult(tool_call_id=call.id, output={"response": sub, **inner_args})
            )
            intermediate_reply = sub

        else:
            call = ToolCall(
                name="generic_responder",
                arguments={"user_input": user_input},
                parent_id=tool_trace[-1].id if tool_trace else None,
            )
            out = self._generic_predictor(history=history, user_input=user_input)
            sub = out.response if hasattr(out, "response") else str(out)
            sub = self._maybe_append_product_details(sub, new_session, user_input)
            tool_trace.append(call)
            tool_results.append(ToolResult(tool_call_id=call.id, output=sub))
            intermediate_reply = sub

        composer_call = ToolCall(
            name="final_composer.compose",
            arguments={
                "sub_response_len": len(intermediate_reply),
                "stage": new_session.stage.value,
            },
            parent_id=tool_trace[-1].id if tool_trace else None,
        )
        final_reply = self.composer(
            sub_response=intermediate_reply,
            plan=plan,
            session=new_session,
            history=history,
        )
        tool_trace.append(composer_call)
        tool_results.append(
            ToolResult(tool_call_id=composer_call.id, output=final_reply)
        )

        plan.tool_trace = tool_trace
        plan.tool_results = tool_results

        return AgentReturn(plan=plan, agent_response=final_reply, session=new_session)


# Register default adapters
register_adapter("ELEMENT", ElementAdapter())
register_adapter("HELVETIA", HelvetiaAdapter())


# ===========================
# Training / evaluation utils
# ===========================
def load_training_data(path: str) -> List[Dict[str, Any]]:
    """Load a JSON list of training/eval rows."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _make_examples(rows: List[Dict[str, Any]]) -> List[dspy.Example]:
    """Convert raw rows into DSPy Examples for teleprompting."""
    examples: List[dspy.Example] = []
    for r in rows:
        ex = dspy.Example(
            user_input=r.get("user_input", ""),
            session=(
                json.dumps(r.get("session", {}), ensure_ascii=False)
                if not isinstance(r.get("session", {}), str)
                else r.get("session", "{}")
            ),
            history=(
                json.dumps(r.get("history", []), ensure_ascii=False)
                if not isinstance(r.get("history", []), str)
                else r.get("history", "[]")
            ),
            gold_plan=json.dumps(r.get("plan", {}), ensure_ascii=False),
            gold_response=r.get("agent_response", ""),
        )
        ex = ex.with_inputs("user_input", "session", "history")
        examples.append(ex)
    return examples


def _metric_wrapper(agent: Module):
    """Build a metric function for MIPROv2 that returns the composite score."""

    def metric_fn(example: dspy.Example, pred: Any, trace: Any = None, **_) -> float:
        try:
            sess = (
                json.loads(example.session)
                if isinstance(example.session, str)
                else (example.session or {})
            )
        except Exception:
            sess = {}
        try:
            hist = (
                json.loads(example.history)
                if isinstance(example.history, str)
                else (example.history or [])
            )
        except Exception:
            hist = []

        out = agent(user_input=example.user_input, session=sess, history=hist)
        m = compute_metrics(
            out.session.model_dump(), out.plan.model_dump(), out.agent_response
        )
        return float(m.get("composite", 0.0))

    return metric_fn


def train_with_mipro(
    agent: Module,
    train_rows: List[Dict[str, Any]],
    val_rows: Optional[List[Dict[str, Any]]] = None,
) -> Module:
    """
    Run DSPy MIPROv2 teleprompting and print per-example metric breakdown for the training set.
    """
    trainset = _make_examples(train_rows)
    valset = _make_examples(val_rows) if val_rows else None

    tele = MIPROv2(
        metric=_metric_wrapper(agent), max_bootstrapped_demos=6, max_labeled_demos=6
    )
    tuned = tele.compile(agent, trainset=trainset, valset=valset)

    # Per-example metrics (all components)
    print("\n=== Per-example metrics on training set ===")
    for i, row in enumerate(train_rows):
        sess_raw = row.get("session", {})
        hist_raw = row.get("history", [])
        if isinstance(sess_raw, str):
            try:
                sess = json.loads(sess_raw)
            except Exception:
                sess = {}
        else:
            sess = sess_raw or {}
        if isinstance(hist_raw, str):
            try:
                hist = json.loads(hist_raw)
            except Exception:
                hist = []
        else:
            hist = hist_raw or []

        out = tuned(user_input=row.get("user_input", ""), session=sess, history=hist)
        m = compute_metrics(
            out.session.model_dump(), out.plan.model_dump(), out.agent_response
        )
        print(f"- Example {i+1}:")
        for k, v in m.items():
            print(f"  {k:>28}: {v:.3f}")
    print("=== End per-example metrics ===\n")

    return tuned


def evaluate_on_dataset(agent: Module, rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """Run the agent over a dataset and print averaged metrics."""
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    for r in rows:
        sess_raw = r.get("session", {})
        hist_raw = r.get("history", [])
        if isinstance(sess_raw, str):
            try:
                sess = json.loads(sess_raw)
            except Exception:
                sess = {}
        else:
            sess = sess_raw or {}
        if isinstance(hist_raw, str):
            try:
                hist = json.loads(hist_raw)
            except Exception:
                hist = []
        else:
            hist = hist_raw or []

        out = agent(user_input=r.get("user_input", ""), session=sess, history=hist)
        m = compute_metrics(
            out.session.model_dump(), out.plan.model_dump(), out.agent_response
        )
        for k, v in m.items():
            sums[k] = sums.get(k, 0.0) + float(v)
            counts[k] = counts.get(k, 0) + 1

    avg = {k: (sums[k] / max(1, counts.get(k, 1))) for k in sums}
    print("=== Offline metrics (agent on dataset) ===")
    print(json.dumps(avg, ensure_ascii=False, indent=2))
    return avg


# =============
# CLI entrypoint
# =============
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepAgent (simplified).")
    parser.add_argument(
        "--provider",
        type=str,
        choices=["nebius", "azure"],
        required=True,
        help="LLM provider to use.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run MIPROv2 teleprompting on --data (and --val if provided).",
    )
    parser.add_argument(
        "--eval", action="store_true", help="Eval on --data (and --val if provided)."
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
        default="tuned_agent.json",
        help="Path to save/load tuned agent.",
    )
    parser.add_argument(
        "--repl",
        action="store_true",
        help="Start interactive REPL after other actions.",
    )
    args = parser.parse_args()

    # Initialize LMs per provider
    configure_lms(args.provider)

    # Base agent
    agent: Module = DeepAgent()

    # Train
    if args.train and os.path.exists(args.data):
        try:
            train_rows = load_training_data(args.data)
            val_rows = (
                load_training_data(args.val)
                if args.val and os.path.exists(args.val)
                else None
            )
            agent = train_with_mipro(agent, train_rows, val_rows)
            try:
                if hasattr(agent, "save"):
                    agent.save(args.tuned)  # type: ignore[attr-defined]
                else:
                    dspy.save(agent, args.tuned)
                print(f"=== Training completed and saved to {args.tuned} ===")
            except Exception as e:
                print(f"Warning: could not save tuned agent: {e}")
        except Exception as e:
            print(f"Training failed: {e}")

    # Load tuned agent if exists (and not just trained)
    if not args.train and os.path.exists(args.tuned):
        try:
            if hasattr(DeepAgent, "load"):
                agent = DeepAgent.load(args.tuned)  # type: ignore[attr-defined]
            else:
                agent = dspy.load(args.tuned)
            print(f"Loaded tuned agent from {args.tuned}")
        except Exception as e:
            print(f"Failed to load tuned agent: {e}")

    # Eval
    if args.eval and os.path.exists(args.data):
        try:
            rows = load_training_data(args.data)
            evaluate_on_dataset(agent, rows)
        except Exception as e:
            print(f"Dataset eval skipped due to error: {e}")

    # REPL
    if args.repl or (not args.train and not args.eval):
        session = Session()
        history: List[Dict[str, Any]] = []
        print("\nAgent: Hallo! Wie kann ich Ihnen helfen? (tippe 'exit' zum Beenden)")
        while True:
            try:
                user_input = input("Sie: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if user_input.lower() in {"exit", "quit"}:
                break
            history.append({"role": "user", "content": user_input})

            out = agent(
                user_input=user_input, session=session.model_dump(), history=history
            )
            reply = out.agent_response
            plan_dict = out.plan.model_dump()

            try:
                session = Session.model_validate(out.session)
            except ValidationError:
                session = Session()

            try:
                m = compute_metrics(session.model_dump(), plan_dict, reply)
                print("METRICS:", json.dumps(m, ensure_ascii=False))
            except Exception as e:
                print(f"Metric error: {e}")

            print("Agent:", reply)
            history.append({"role": "agent", "content": reply, "tool_call": plan_dict})
            if session.stage == Stage.finalized:
                print("Agent: Gibt es noch etwas, das ich für Sie tun kann?")
                session = Session()
                history = []
