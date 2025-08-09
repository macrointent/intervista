# helpers.py
"""
Shared helpers for DeepAgent:
- Catalog I/O and indexing
- Category utilities
- Forms / labels / schema helpers
- Lightweight relevance and rendering utilities
- Conversation UX helpers

These helpers are deterministic and side-effect free, except for small in-memory caches.
Parsing of user-provided personal data is intentionally NOT implemented here; that is handled
by the LLM-based FormUpdater in deep_agent.py.

Environment variables (optional):
- CATALOG_FILE: path to catalog JSON (default: "catalog.json")
- FORMS_FILE:   path to forms JSON (default: "forms.json")
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

# -----------------------------------------------------------
# Files / paths
# -----------------------------------------------------------

CATALOG_FILE = os.getenv("CATALOG_FILE", "catalog.json")
FORMS_FILE = os.getenv("FORMS_FILE", "forms.json")

# -----------------------------------------------------------
# In-memory caches for catalog and forms
# -----------------------------------------------------------

_CATALOG_RAW: Optional[Any] = None
_CATALOG_INDEX_BY_ID: Dict[str, Dict[str, Any]] = {}
_CATALOG_INDEX_BY_NAME: Dict[str, Dict[str, Any]] = {}

_FORMS_CACHE: Optional[List[Dict[str, Any]]] = None


# -----------------------------------------------------------
# Basic file I/O
# -----------------------------------------------------------


def read_json(path: str) -> Any:
    """
    Read a JSON file from disk.

    Args:
        path: Filesystem path to a JSON file.

    Returns:
        Parsed JSON (dict/list/primitive) or {} on failure.

    Notes:
        - Returns {} if file is not found or parsing fails.
        - This function is intentionally forgiving to avoid hard crashes at runtime.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


# -----------------------------------------------------------
# Catalog: indexing and provider resolution
# -----------------------------------------------------------


def warm_catalog() -> None:
    """
    Load and index the catalog into memory if not already cached.

    Side effects:
        - Populates _CATALOG_RAW, _CATALOG_INDEX_BY_ID, and _CATALOG_INDEX_BY_NAME.
    """
    global _CATALOG_RAW, _CATALOG_INDEX_BY_ID, _CATALOG_INDEX_BY_NAME
    if _CATALOG_RAW is not None and _CATALOG_INDEX_BY_ID and _CATALOG_INDEX_BY_NAME:
        return

    raw = read_json(CATALOG_FILE) or {}
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        # Support dict-shaped catalogs keyed by id
        items = []
        for k, v in raw.items():
            if isinstance(v, dict) and "id" not in v:
                try:
                    v["id"] = int(k)
                except Exception:
                    v["id"] = k
            items.append(v)
    else:
        items = []

    _CATALOG_INDEX_BY_ID = {}
    _CATALOG_INDEX_BY_NAME = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        pid = it.get("id") or it.get("productId") or it.get("sku")
        name = it.get("displayName") or it.get("name") or it.get("title") or ""
        provider = (
            it.get("provider") or it.get("insurer") or it.get("riskCarrier") or ""
        )
        entry = {"id": pid, "name": name, "provider": provider, **it}
        if pid is not None:
            _CATALOG_INDEX_BY_ID[str(pid)] = entry
        if name:
            _CATALOG_INDEX_BY_NAME[name.strip().lower()] = entry

    _CATALOG_RAW = items


def get_catalog_indices() -> (
    Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]
):
    """
    Return catalog indices by id and by normalized name.

    Returns:
        Tuple (index_by_id, index_by_name) where:
        - index_by_id maps str(id) -> entry dict
        - index_by_name maps lowercased name -> entry dict
    """
    warm_catalog()
    return _CATALOG_INDEX_BY_ID, _CATALOG_INDEX_BY_NAME


_PROVIDER_NORMALIZE = {
    "element": "ELEMENT",
    "helvetia": "HELVETIA",
}


def _normalize_provider(text: str) -> Optional[str]:
    """
    Normalize a provider string into a canonical form.

    Args:
        text: Raw provider label from catalog.

    Returns:
        Canonical upper-case name (e.g., 'ELEMENT', 'HELVETIA') when possible,
        otherwise an upper-cased best-effort value.

    Notes:
        - This function is only used to standardize catalog-derived provider names.
        - It does not parse user-provided personal data.
    """
    if not text:
        return None
    s = text.strip().lower()
    if s in _PROVIDER_NORMALIZE:
        return _PROVIDER_NORMALIZE[s]
    m = re.search(
        r"(risikoträger|risk\s*carrier)\s*[:\-]?\s*([a-zäöüß]+)", s, flags=re.I
    )
    if m:
        cand = m.group(2).lower()
        return _PROVIDER_NORMALIZE.get(cand, cand.upper())
    for key in _PROVIDER_NORMALIZE:
        if key in s:
            return _PROVIDER_NORMALIZE[key]
    return s.upper()


def provider_from_catalog(
    product_name: Optional[str] = None, product_id: Optional[int | str] = None
) -> Optional[str]:
    """
    Resolve and normalize the provider for a product using catalog indices.

    Args:
        product_name: Display name of the product (optional).
        product_id:   Numeric or string id of the product (optional).

    Returns:
        Canonical provider name (e.g., 'ELEMENT') or None if not found.
    """
    by_id, by_name = get_catalog_indices()

    if product_id is not None:
        ent = by_id.get(str(product_id))
        if ent:
            raw_text = (
                ent.get("provider")
                or ent.get("insurer")
                or ent.get("riskCarrier")
                or ent.get("displayName")
                or ent.get("name")
                or ent.get("title")
                or ""
            )
            return _normalize_provider(str(raw_text)) or None

    if product_name:
        ent = by_name.get(product_name.strip().lower())
        if ent:
            raw_text = (
                ent.get("provider")
                or ent.get("insurer")
                or ent.get("riskCarrier")
                or ent.get("displayName")
                or ent.get("name")
                or ent.get("title")
                or ""
            )
            return _normalize_provider(str(raw_text)) or None

        # substring fallback
        pn = product_name.strip().lower()
        for nm, ent in by_name.items():
            if pn in nm:
                raw_text = (
                    ent.get("provider")
                    or ent.get("insurer")
                    or ent.get("riskCarrier")
                    or ent.get("displayName")
                    or ent.get("name")
                    or ent.get("title")
                    or ""
                )
                return _normalize_provider(str(raw_text)) or None

    return None


def find_catalog_entry_by_query(query: str) -> Optional[Dict[str, Any]]:
    """
    Attempt a simple lookup in the catalog by product name (exact or substring) or by id.

    Args:
        query: User-provided search string.

    Returns:
        Matching catalog entry dict or None if ambiguous/not found.

    Strategy:
        - Exact case-insensitive match on display name.
        - Single substring hit (query in name) → return it; multiple hits → None (ambiguous).
        - Single containment hit (name in query) → return it; multiple hits → None (ambiguous).
        - Numeric string treated as id lookup.
    """
    if not query:
        return None
    by_id, by_name = get_catalog_indices()
    q = query.strip().lower()

    ent = by_name.get(q)
    if ent:
        return ent

    # query contained in product name
    candidates = [entry for nm, entry in by_name.items() if q in nm]
    if len(candidates) == 1:
        return candidates[0]

    # product name contained in query (e.g., "<name> klingt interessant")
    candidates2 = [entry for nm, entry in by_name.items() if nm in q]
    if len(candidates2) == 1:
        return candidates2[0]

    if q.isdigit():
        ent2 = by_id.get(q)
        if ent2:
            return ent2

    return None


def compact_catalog(raw: Any) -> List[Dict[str, Any]]:
    """
    Convert raw catalog payload to a consistent, compact list of entries.

    Args:
        raw: Raw catalog payload (list or dict).

    Returns:
        List of normalized entries with at least keys: id, name, provider (optional), and
        common product attributes (if present).
    """
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        items = []
        for k, v in raw.items():
            if isinstance(v, dict) and "id" not in v:
                try:
                    v["id"] = int(k)
                except Exception:
                    v["id"] = k
            items.append(v)
    else:
        items = []

    result: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        pid = it.get("id") or it.get("productId") or it.get("sku")
        name = (
            it.get("displayName")
            or it.get("name")
            or it.get("title")
            or f"Produkt #{pid}"
        )
        entry = {
            "id": pid,
            "name": name,
            "provider": it.get("provider")
            or it.get("insurer")
            or it.get("riskCarrier"),
            # common attributes (best-effort)
            "theft": it.get("theft") or it.get("diebstahl") or it.get("coverageTheft"),
            "screen": it.get("screen") or it.get("display") or it.get("coverageScreen"),
            "accident": it.get("accident")
            or it.get("unfall")
            or it.get("coverageAccident"),
            "deductible": it.get("deductible") or it.get("selbstbeteiligung"),
            "billing": it.get("billing") or it.get("zahlweise"),
            "waiting": it.get("waitingPeriod") or it.get("wartezeit"),
            "cancel": it.get("cancellation") or it.get("kuendigungsfrist"),
            "term": it.get("term") or it.get("laufzeit"),
            "price": it.get("price") or it.get("premium"),
            # keep original for downstream
            **it,
        }
        result.append(entry)
    return result


def _score(entry: Dict[str, Any], query: str) -> int:
    """
    Compute a very lightweight relevance score for an entry w.r.t. a query.

    Args:
        entry: Normalized catalog entry.
        query: Search string.

    Returns:
        Integer score (higher is better). Deterministic and simple to keep UX snappy.
    """
    if not query:
        return 0
    q = query.lower()
    score = 0
    name = str(entry.get("name", "")).lower()
    prov = str(entry.get("provider", "")).lower()
    attrs = " ".join(
        str(entry.get(k, "")) for k in ("theft", "billing", "price", "term")
    ).lower()

    if q in name:
        score += 10
    if q in prov:
        score += 3
    if q in attrs:
        score += 2
    return score


def top_k(
    entries: List[Dict[str, Any]], query: str, k: int = 3
) -> List[Dict[str, Any]]:
    """
    Return top-k entries by a simple textual relevance score.

    Args:
        entries: List of compact catalog entries.
        query:   Search string.
        k:       Number of items to return (default: 3).

    Returns:
        Top-k list (ties resolved by original order).
    """
    if not entries:
        return []
    scored = [(e, _score(e, query)) for e in entries]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [e for e, _ in scored[: max(1, k)]]


def stringify_product_details(entry: Dict[str, Any]) -> str:
    """
    Render a compact, user-friendly detail block for a single product entry.

    Args:
        entry: Catalog entry dict (normalized or raw).

    Returns:
        Markdown-formatted detail block with bullets and a closing question.

    Notes:
        - This function is deterministic; for more polished text the ProductExplainer
          sub-agent can rewrite it using an LLM.
    """
    if not entry:
        return "Zu diesem Produkt liegen mir gerade keine Details vor."

    name = (
        entry.get("displayName")
        or entry.get("name")
        or entry.get("title")
        or f"Produkt #{entry.get('id')}"
    )
    provider = entry.get("provider") or entry.get("insurer") or entry.get("riskCarrier")
    pid = entry.get("id")

    theft = entry.get("theft") or entry.get("diebstahl") or entry.get("coverageTheft")
    accident = (
        entry.get("accident") or entry.get("unfall") or entry.get("coverageAccident")
    )
    screen = entry.get("screen") or entry.get("display") or entry.get("coverageScreen")
    deductible = entry.get("deductible") or entry.get("selbstbeteiligung")
    billing = entry.get("billing") or entry.get("zahlweise")
    waiting = entry.get("waitingPeriod") or entry.get("wartezeit")
    cancel = entry.get("cancellation") or entry.get("kuendigungsfrist")
    term = entry.get("term") or entry.get("laufzeit")
    price = entry.get("price") or entry.get("premium")

    bullets: List[str] = []
    if theft is not None:
        yn = str(theft).lower()
        bullets.append(
            f"Diebstahl: {'ja' if yn in {'1','true','ja','yes'} else ('nein' if yn in {'0','false','nein','no'} else str(theft))}"
        )
    if screen is not None:
        bullets.append(f"Displayschäden: {screen}")
    if accident is not None:
        bullets.append(f"Unfallschäden: {accident}")
    if deductible is not None:
        bullets.append(f"Selbstbeteiligung: {deductible}")
    if billing is not None:
        bullets.append(f"Zahlungsweise: {billing}")
    if term is not None:
        bullets.append(f"Laufzeit: {term}")
    if waiting is not None:
        bullets.append(f"Wartezeit: {waiting}")
    if cancel is not None:
        bullets.append(f"Kündigung: {cancel}")
    if price is not None:
        bullets.append(f"Beitrag: {price}")
    if provider:
        bullets.append(f"Risikoträger: {provider}")
    if pid is not None:
        bullets.append(f"Produkt-ID: {pid}")

    if not bullets:
        return f"**{name}** – weitere Details folgen auf Wunsch."

    lines = [f"**{name}** – kurze Details:"]
    lines += [f"• {b}" for b in bullets]
    lines.append("Sollen wir damit starten?")
    return "\n".join(lines)


# -----------------------------------------------------------
# Category helpers
# -----------------------------------------------------------

_CANON_CATEGORIES = {
    "handyversicherung": {"handy", "smartphone", "phone", "handyversicherung"},
    "laptop": {"laptop", "notebook", "macbook"},
    "kamera": {"kamera", "camera"},
    "e-bike": {"e-bike", "ebike", "pedelec"},
}


def normalize_category(query: str) -> Optional[str]:
    """
    Map a free-text query to a canonical category key, if recognizable.

    Args:
        query: Raw user query.

    Returns:
        Canonical category (e.g., 'handyversicherung') or None if not recognized.
    """
    if not query:
        return None
    q = query.strip().lower()
    for canon, synonyms in _CANON_CATEGORIES.items():
        if q in synonyms:
            return canon
    # substring heuristic
    for canon, synonyms in _CANON_CATEGORIES.items():
        if any(s in q for s in synonyms):
            return canon
    return None


def filter_by_category(raw_catalog: Any, category: str) -> List[Dict[str, Any]]:
    """
    Filter the catalog for entries belonging to a canonical category.

    Args:
        raw_catalog: Raw catalog payload (list/dict).
        category:    Canonical category key (e.g., 'handyversicherung').

    Returns:
        List of compact entries for this category.

    Notes:
        - This uses best-effort heuristics (by name/title) to infer category membership.
    """
    entries = compact_catalog(raw_catalog)
    if category == "handyversicherung":

        def is_match(e: Dict[str, Any]) -> bool:
            nm = str(e.get("name", "")).lower()
            return any(
                tok in nm for tok in ["handy", "smartphone", "iphone", "android"]
            )

        return [e for e in entries if is_match(e)]

    if category == "laptop":

        def is_match(e: Dict[str, Any]) -> bool:
            nm = str(e.get("name", "")).lower()
            return any(tok in nm for tok in ["laptop", "notebook", "macbook"])

        return [e for e in entries if is_match(e)]

    if category == "kamera":

        def is_match(e: Dict[str, Any]) -> bool:
            nm = str(e.get("name", "")).lower()
            return any(tok in nm for tok in ["kamera", "camera"])

        return [e for e in entries if is_match(e)]

    if category == "e-bike":

        def is_match(e: Dict[str, Any]) -> bool:
            nm = str(e.get("name", "")).lower()
            return any(tok in nm for tok in ["e-bike", "ebike", "pedelec"])

        return [e for e in entries if is_match(e)]

    # Fallback: unknown category → empty list
    return []


# -----------------------------------------------------------
# Forms / labels / schema helpers
# -----------------------------------------------------------


@dataclass
class FieldMeta:
    """
    UI metadata for a single flat conversational field.

    Attributes:
        label_de: Human label to display (German).
        group:    Group key for clustered questions ('person', 'kontakt', 'geraet', 'zahlung', 'vertrag').
        hint_de:  Optional inline hint (e.g., date format).
        options:  Optional list of allowed options (from schema enums).
    """

    label_de: str
    group: str
    hint_de: str = ""
    options: List[str] = None  # type: ignore[assignment]


# Human microcopy defaults
FIELD_META: Dict[str, FieldMeta] = {
    "product_name": FieldMeta(
        label_de="Produkt", group="vertrag", hint_de="gewählter Tarifname"
    ),
    "first_name": FieldMeta(label_de="Vorname", group="person", hint_de="z. B. Anna"),
    "last_name": FieldMeta(label_de="Nachname", group="person", hint_de="z. B. Müller"),
    "email": FieldMeta(
        label_de="E-Mail", group="kontakt", hint_de="z. B. anna.mueller@example.com"
    ),
    "device_brand": FieldMeta(label_de="Hersteller", group="geraet"),
    "device_model": FieldMeta(label_de="Modell", group="geraet"),
    "device_price": FieldMeta(label_de="Gerätepreis", group="geraet"),
    "purchase_date": FieldMeta(
        label_de="Kauf-/Lieferdatum", group="geraet", hint_de="YYYY-MM-DD"
    ),
    "device_type": FieldMeta(label_de="Gerätetyp", group="geraet"),
    "device_category": FieldMeta(label_de="Gerätekategorie", group="geraet"),
    "payment_method": FieldMeta(
        label_de="Zahlungsart",
        group="zahlung",
        hint_de="z. B. IBAN / Kreditkarte / PayPal",
        options=["SEPA/IBAN", "Kreditkarte", "PayPal"],
    ),
    "payment_account": FieldMeta(
        label_de="Zahlungskennung",
        group="zahlung",
        hint_de="IBAN / Kartennummer / PayPal-ID",
    ),
    "payment_owner": FieldMeta(label_de="Kontoinhaber", group="zahlung"),
    "payment_provider_name": FieldMeta(label_de="Kartenanbieter/Bank", group="zahlung"),
    "payment_bic": FieldMeta(label_de="BIC", group="zahlung"),
    "payment_expiry": FieldMeta(label_de="Ablaufdatum (Karte)", group="zahlung"),
    "street": FieldMeta(label_de="Straße", group="person"),
    "street_number": FieldMeta(label_de="Hausnummer", group="person"),
    "zipcode": FieldMeta(label_de="PLZ", group="person"),
    "city": FieldMeta(label_de="Ort", group="person"),
    "country_code": FieldMeta(
        label_de="Ländercode", group="person", hint_de="z. B. DE"
    ),
}

# Global minimal set for UX; product/schema adds more
GLOBAL_MIN_REQUIRED: Set[str] = {"first_name", "last_name", "email", "payment_method"}

# Product-type knobs if you have category-level extras (optional)
PRODUCT_REQUIRED: Dict[str, Set[str]] = {
    # For device-centric products collect key device fields upfront
    "handy": {"device_brand", "device_model", "purchase_date", "device_type"},
    "laptop": {"device_brand", "device_model", "purchase_date", "device_type"},
    "kamera": {"device_brand", "device_model", "purchase_date", "device_type"},
    "e-bike": set(),
}

# Ask groups in this order
GROUP_ORDER = ["person", "kontakt", "geraet", "zahlung", "vertrag"]

# Map schema field paths → flat conversational keys
SCHEMA_TO_FLAT = {
    "Person.firstName": ("first_name", "person"),
    "Person.lastName": ("last_name", "person"),
    "Modules.attribute_1": ("purchase_date", "geraet"),
    "Modules.attribute_70": ("device_brand", "geraet"),
    "Modules.attribute_130": ("device_model", "geraet"),
    "Modules.attribute_166": ("device_price", "geraet"),
    "Attributes.attribute_331": ("device_type", "geraet"),
    "Attributes.attribute_7": ("device_category", "geraet"),
    "Payment.providerType": ("payment_method", "zahlung"),
    "Payment.accountNumber": ("payment_account", "zahlung"),
    "Payment.providerName": ("payment_provider_name", "zahlung"),
    "Payment.providerNumber": ("payment_bic", "zahlung"),
    "Payment.accountOwner": ("payment_owner", "zahlung"),
    "Payment.expirationDate": ("payment_expiry", "zahlung"),
}

# Fallback labels if schema lacks titles
FALLBACK_META = {
    "email": ("E-Mail", "kontakt", "z. B. anna.mueller@example.com"),
    "payment_method": ("Zahlungsart", "zahlung", "z. B. IBAN / Kreditkarte / PayPal"),
}


def load_forms() -> List[Dict[str, Any]]:
    """
    Load and cache forms.json (list of form descriptors with JSON Schemas).

    Returns:
        List of form descriptors. Empty list on failure.
    """
    global _FORMS_CACHE
    if _FORMS_CACHE is not None:
        return _FORMS_CACHE
    try:
        with open(FORMS_FILE, "r", encoding="utf-8") as f:
            _FORMS_CACHE = json.load(f)
    except Exception:
        _FORMS_CACHE = []
    return _FORMS_CACHE


def find_form_schema(
    product: str | None = None, form_id: int | None = None
) -> Optional[Dict[str, Any]]:
    """
    Find a JSON Schema for a product by id or name substring.

    Args:
        product: Display name of the product (optional).
        form_id: Numeric schema id (optional).

    Returns:
        JSON Schema dict or None if not found.
    """
    forms = load_forms()
    if form_id is not None:
        for item in forms:
            try:
                if int(item.get("id", -1)) == int(form_id):
                    return item.get("schema")
            except Exception:
                continue
    p = (product or "").strip().lower()
    if not p:
        return None
    for item in forms:
        name = (item.get("name") or "").lower()
        if p in name:
            return item.get("schema")
    return None


def label_for(key: str, labels: Dict[str, str]) -> str:
    """
    Resolve a user-facing label for a flat key, using session labels → FIELD_META → key name.

    Args:
        key:    Flat field key.
        labels: Session labels map.

    Returns:
        Human label (German).
    """
    if key in labels:
        return labels[key]
    if key in FIELD_META:
        return FIELD_META[key].label_de
    return key.replace("_", " ")


def group_of(key: str) -> str:
    """
    Resolve group name for a flat key.

    Args:
        key: Flat field key.

    Returns:
        Group string (e.g., 'person', 'kontakt', 'geraet', 'zahlung', 'vertrag').
    """
    meta = FIELD_META.get(key)
    return meta.group if meta else "vertrag"


def pick_next_group(missing_keys: List[str]) -> Optional[str]:
    """
    Choose the next group to ask from the current set of missing fields.

    Args:
        missing_keys: Flat keys still missing.

    Returns:
        Group name to focus on next or None if no missing keys.
    """
    present = {group_of(k) for k in missing_keys}
    for g in GROUP_ORDER:
        if g in present:
            return g
    return None


def format_group_ask(
    group: str, missing_keys: List[str], labels: Dict[str, str], max_items: int = 3
) -> str:
    """
    Render a compact question cluster for up to `max_items` fields within the same group.

    Args:
        group:        Group key to ask about (e.g., 'person').
        missing_keys: Flat keys still missing.
        labels:       Label mapping for human-readable field names.
        max_items:    Maximum number of fields to ask in this turn.

    Returns:
        Markdown-formatted prompt text.
    """
    keys = [k for k in missing_keys if group_of(k) == group][:max_items]
    title = {
        "person": "Deine Personendaten",
        "kontakt": "Deine Kontaktdaten",
        "geraet": "Angaben zum Gerät",
        "zahlung": "Zahlungsweise",
        "vertrag": "Vertragsdetails",
    }.get(group, group.capitalize())

    bullets: List[str] = []
    for k in keys:
        m = FIELD_META.get(k)
        hint = f" ({m.hint_de})" if m and m.hint_de else ""
        opts = ""
        if m and m.options:
            opts = " – Optionen: " + " / ".join(m.options)
        bullets.append(f"- {label_for(k, labels)}{hint}{opts}")

    intro = (
        "Alles klar – ich sammle kurz die wichtigsten Angaben, dann geht’s weiter.\n\n"
    )
    ask = (
        f"{title}:\n"
        + "\n".join(bullets)
        + '\n\nAntworte einfach direkt oder schreib "überspringen", wenn du es später nachreichen möchtest.'
    )
    return intro + ask


def derive_required(
    product_type: str, schema_required: Optional[List[str]]
) -> Set[str]:
    """
    Compute a fallback required-set if the JSON schema is missing.

    Args:
        product_type: Lower-cased product type/category (best-effort).
        schema_required: A required list from schema (if available).

    Returns:
        Set of required flat keys to guide slot-filling.
    """
    if schema_required:
        return set(schema_required) | GLOBAL_MIN_REQUIRED
    return GLOBAL_MIN_REQUIRED | PRODUCT_REQUIRED.get(
        (product_type or "").lower(), set()
    )


def ensure_field_meta_from_schema(
    labels: Dict[str, str], schema: Dict[str, Any]
) -> Dict[str, str]:
    """
    Harvest titles/enums from JSON Schema into session labels and FIELD_META.

    Args:
        labels: Existing session labels (will be updated/returned).
        schema: JSON Schema dict with $defs and properties blocks.

    Returns:
        Updated labels dict.

    Notes:
        - Updates the global FIELD_META to enrich hints and enum options where available.
    """
    if labels is None:
        labels = {}
    defs = schema.get("$defs", {})
    props = schema.get("properties", {})

    def ref_props(ref: str) -> Dict[str, Any]:
        if not ref.startswith("#/$defs/"):
            return {}
        name = ref.split("/")[-1]
        return defs.get(name, {})

    def set_label(flat_key: str, title: str, group: str, hint: str = ""):
        labels[flat_key] = title
        if flat_key not in FIELD_META:
            FIELD_META[flat_key] = FieldMeta(
                label_de=title, group=group, hint_de=hint, options=None
            )
        else:
            fm = FIELD_META[flat_key]
            if not fm.label_de:
                fm.label_de = title
            if not fm.group:
                fm.group = group
            if hint and not fm.hint_de:
                fm.hint_de = hint

    for top_name, top_spec in props.items():
        block = ref_props(top_spec["$ref"]) if "$ref" in top_spec else top_spec
        block_props = block.get("properties", {})
        for sub_name, sub_spec in block_props.items():
            path = f"{top_name}.{sub_name}"
            if path in SCHEMA_TO_FLAT:
                flat_key, group = SCHEMA_TO_FLAT[path]
                title = sub_spec.get("title") or sub_spec.get("description") or flat_key
                hint = ""
                if sub_spec.get("pattern") == r"^\d{4}-\d{2}-\d{2}$":
                    hint = "YYYY-MM-DD"
                set_label(flat_key, title, group, hint)
                enum_vals = sub_spec.get("enum")
                if enum_vals:
                    if flat_key not in FIELD_META:
                        FIELD_META[flat_key] = FieldMeta(
                            label_de=title,
                            group=group,
                            hint_de=hint,
                            options=list(enum_vals),
                        )
                    else:
                        FIELD_META[flat_key].options = list(enum_vals)

    # stable fallbacks
    for k, (lbl, grp, hint) in FALLBACK_META.items():
        if k not in labels:
            if k not in FIELD_META:
                FIELD_META[k] = FieldMeta(
                    label_de=lbl, group=grp, hint_de=hint, options=None
                )
            labels[k] = lbl

    return labels


def flat_required_from_schema(schema: Dict[str, Any]) -> Set[str]:
    """
    Convert nested JSON Schema required paths into our flat conversational field set.

    Args:
        schema: JSON Schema dict.

    Returns:
        Set of required flat keys (merged with GLOBAL_MIN_REQUIRED).
    """
    required_flat: Set[str] = set()
    top_required = set(schema.get("required", []))
    defs = schema.get("$defs", {})

    def reqs_for_block(block_name: str) -> Set[str]:
        blk = defs.get(block_name, {})
        return set(blk.get("required", []))

    if "Person" in top_required:
        person_reqs = reqs_for_block("Person")
        if "lastName" in person_reqs:
            required_flat.add("last_name")
        # We add first_name for better UX even if schema doesn't require it
        required_flat.add("first_name")
    if "Payment" in top_required:
        pay_reqs = reqs_for_block("PaymentProvider")
        if "providerType" in pay_reqs:
            required_flat.add("payment_method")
        if "accountNumber" in pay_reqs:
            required_flat.add("payment_account")
    if "Modules" in top_required:
        mod_reqs = reqs_for_block("Modules")
        if "attribute_1" in mod_reqs:
            required_flat.add("purchase_date")
        if "attribute_70" in mod_reqs:
            required_flat.add("device_brand")
    if "Attributes" in top_required:
        attr_reqs = reqs_for_block("Attributes")
        if "attribute_331" in attr_reqs:
            required_flat.add("device_type")
        if "attribute_7" in attr_reqs:
            required_flat.add("device_category")

    required_flat |= GLOBAL_MIN_REQUIRED
    return required_flat


# -----------------------------------------------------------
# Conversation UX helpers
# -----------------------------------------------------------

_SKIP_TOKENS = {
    "überspringen",
    "ueberspringen",
    "später",
    "spaeter",
    "weiss nicht",
    "weiß nicht",
    "keine ahnung",
    "später nachreichen",
    "spater nachreichen",
}


def user_wants_to_skip(text: str) -> bool:
    """
    Detect whether a user intends to skip the current question cluster.

    Args:
        text: Raw user reply.

    Returns:
        True if a skip token is present, False otherwise.

    Notes:
        - This is the only intent heuristic here; personal data extraction is handled by the LLM.
    """
    t = (text or "").strip().lower()
    return any(tok in t for tok in _SKIP_TOKENS)
