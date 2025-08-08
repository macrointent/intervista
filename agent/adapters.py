# adapters.py
"""
Provider adapters for building and validating payloads for external insurers.

This module defines:
- A base FormAdapter interface with default validation behavior.
- Concrete adapters: ElementAdapter, HelvetiaAdapter.
- A small registry (register_adapter / get_adapter) to look up adapters by key.

Design goals:
- Keep mapping/validation logic decoupled from orchestration code.
- Maintain predictable, testable transformations from flat form data
  (what the conversational layer collects) to provider payloads.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple


# Used by providers that expect contact info typed; -23 commonly denotes "email".
EMAIL_TYPEID = -23


class FormAdapter:
    """
    Abstract adapter for a provider, mapping flat conversational fields
    -> provider-specific API payloads.

    Subclasses are expected to:
      - implement provider_name()
      - implement required_fields()
      - implement to_payload(flat)
    They may optionally override validate(flat) for custom checks.
    """

    def provider_name(self) -> str:
        """
        Return the canonical provider name this adapter supports.

        Examples:
            "ELEMENT", "HELVETIA"
        """
        raise NotImplementedError

    def required_fields(self) -> Set[str]:
        """
        Return the minimal set of flat fields that must be present
        in order to build a valid submission payload for this provider.

        Returns:
            A set of flat field keys (e.g., {"first_name","email","payment_method"}).
        """
        raise NotImplementedError

    def to_payload(self, flat: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map the flat conversational form data to the provider's JSON schema.

        Args:
            flat: Flat form dictionary where keys are conversational field names.

        Returns:
            Provider-specific payload dictionary ready for submission.

        Raises:
            KeyError/ValueError if essential fields are missing or malformed.
        """
        raise NotImplementedError

    def validate(self, flat: Dict[str, Any]) -> List[str]:
        """
        Validate the flat data for provider-specific requirements.

        Default behavior:
          - Ensure that all `required_fields()` have truthy values in `flat`.

        Args:
            flat: Flat form dictionary.

        Returns:
            A list of missing field keys (empty list if valid).
        """
        return [k for k in self.required_fields() if not flat.get(k)]


# Internal adapter registry
_ADAPTERS: Dict[str, FormAdapter] = {}


def register_adapter(key: str, adapter: FormAdapter) -> None:
    """
    Register an adapter under a canonical provider key.

    Args:
        key: Registry key, e.g., "ELEMENT".
        adapter: Concrete FormAdapter instance.

    Returns:
        None
    """
    _ADAPTERS[key] = adapter


def get_adapter(key: str) -> Optional[FormAdapter]:
    """
    Retrieve a registered adapter by key.

    Args:
        key: Registry key as used in register_adapter().

    Returns:
        The FormAdapter instance if found; otherwise None.
    """
    return _ADAPTERS.get(key)


# -----------------------------
# ELEMENT adapter implementation
# -----------------------------


class ElementAdapter(FormAdapter):
    """
    Adapter for ELEMENT payload mapping.

    This adapter translates flat conversational fields into the nested structure
    expected by ELEMENT's API (Person, Payment, Attributes, Modules).
    """

    # Mapping of payment method tokens -> ELEMENT providerType codes
    PAYMENT_CODES = {
        "bank": -161,
        "iban": -161,
        "sepa": -161,
        "kreditkarte": -160,
        "credit": -160,
        "credit card": -160,
        "visa": -160,
        "mastercard": -160,
        "amex": -160,
        "paypal": -163,
        "amazon pay": -162,
        "google pay": -165,
    }

    def provider_name(self) -> str:
        """Return the provider name supported by this adapter."""
        return "ELEMENT"

    def required_fields(self) -> Set[str]:
        """
        Minimal flat fields typically required by ELEMENT for successful binding.

        Returns:
            A set of required flat keys.
        """
        return {
            "first_name",
            "last_name",
            "email",
            "payment_method",
            "payment_account",
            "device_brand",
        }

    def _normalize_brand(self, brand: str) -> Tuple[str, Optional[str]]:
        """
        Normalize brand to ELEMENT's enum where possible; otherwise return ('Sonstige', <free-text>).

        Args:
            brand: Raw brand string.

        Returns:
            Tuple of (enum_brand, other_text_or_None).
        """
        if not brand:
            return "", None
        b = brand.strip()
        allowed = {
            "Samsung",
            "Apple",
            "Huawei",
            "Xiaomi",
            "Acer",
            "AEG",
            "Alcatel",
            "Allview",
            "Amica",
            "Asus",
            "ATAG",
            "Bauknecht",
            "BEKO",
            "Blackberry",
            "Blomberg",
            "Bomann",
            "Bosch (BSH)",
            "Candy",
            "Canon",
            "CAT",
            "Caterpillar",
            "Clatronic",
            "Constructa",
            "Cubot",
            "DELL",
            "Doro",
            "Electrolux",
            "Elektra Bregenz",
            "Epson",
            "Exquisit",
            "Fagor",
            "Fairphone",
            "Fujifilm",
            "Fujitsu",
            "Gaggenau",
            "Gigaset",
            "Google",
            "GoPro",
            "Gorenje",
            "Grundig",
            "Haier",
            "Hewlett Packard/HP",
            "Honor",
            "Hoover",
            "HTC",
            "Ignis",
            "Ikea",
            "Indesit",
            "ISY",
            "Juno",
            "Koenic",
            "Körting",
            "Küppersbusch",
            "Leica",
            "Lenovo",
            "LG",
            "Liebherr",
            "Microsoft",
            "Miele",
            "Motorola",
            "Neff",
            "Nikon",
            "Nokia",
            "ok",
            "Olympus",
            "OnePlus",
            "OPPO",
            "Oranier",
            "Panasonic",
            "PEAQ",
            "Privileg",
            "Progress",
            "Schaub Lorenz",
            "Sharp",
            "Sibir",
            "Siemens",
            "Smeg",
            "Sony",
            "Teka",
            "Termikel",
            "UleFone",
            "Vivo",
            "Whirlpool",
            "Wiko",
            "Zanker",
            "Zanussi",
            "ZTE",
            "Sonstige",
        }
        if b in allowed:
            return b, None
        bt = b.title()
        if bt in allowed:
            return bt, None
        if b.lower() in {"hp", "hewlett packard", "hewlett packard/hp"}:
            return "Hewlett Packard/HP", None
        return "Sonstige", b

    def _provider_type(self, raw: Any) -> Optional[int]:
        """
        Map a free-text payment method to an ELEMENT providerType code.

        Args:
            raw: Raw input (e.g., 'iban', 'paypal', 'visa').

        Returns:
            Integer code or None if not recognized.
        """
        if raw is None:
            return None
        key = str(raw).strip().lower()
        return self.PAYMENT_CODES.get(key)

    def _iso_date(self, s: str) -> Optional[str]:
        """
        Convert DD.MM.YYYY or YYYY-MM-DD to ISO date (YYYY-MM-DD). Otherwise return None.

        Args:
            s: Raw date string.

        Returns:
            Normalized ISO date string or None.
        """
        if not s:
            return None
        s = s.strip()
        try:
            if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
                datetime.strptime(s, "%Y-%m-%d")
                return s
            if re.match(r"^\d{2}\.\d{2}\.\d{4}$", s):
                dt = datetime.strptime(s, "%d.%m.%Y")
                return dt.strftime("%Y-%m-%d")
        except Exception:
            return None
        return None

    def to_payload(self, flat: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build the ELEMENT API payload from flat conversational values.

        Structure:
            {
              "Person": {...},
              "Payment": {...},
              "Attributes": {...},
              "Modules": {...}
            }

        Args:
            flat: Flat field dict (e.g., 'first_name', 'payment_method', 'device_brand', ...).

        Returns:
            ELEMENT-specific payload dictionary.
        """
        # Person block
        person = {
            "typeId": -2101,
            "salutationId": None,
            "title": None,
            "firstName": flat.get("first_name"),
            "lastName": flat.get("last_name") or "",
            "dateOfBirth": flat.get("date_of_birth") or None,
            "addresses": [],
            "contactInformations": [],
        }
        if flat.get("email"):
            person["contactInformations"].append(
                {"typeId": EMAIL_TYPEID, "value": flat["email"]}
            )
        if flat.get("street") or flat.get("zipcode") or flat.get("city"):
            addr = {
                "typeId": -10,
                "street": flat.get("street", ""),
                "streetNumber": flat.get("street_number", ""),
                "addition": flat.get("address_addition"),
                "zipcode": flat.get("zipcode", ""),
                "city": flat.get("city", ""),
                "countryCode": flat.get("country_code", "DE"),
            }
            person["addresses"].append(addr)

        # Payment block
        ptype = self._provider_type(flat.get("payment_method"))
        payment = {
            "providerType": ptype,
            "providerNumber": flat.get("payment_bic"),
            "providerName": flat.get("payment_provider_name"),
            "accountNumber": flat.get("payment_account"),
            "accountOwner": flat.get("payment_owner"),
            "expirationDate": flat.get("payment_expiry"),
        }

        # Attributes
        attributes: Dict[str, Any] = {}
        if flat.get("device_type"):
            attributes["attribute_331"] = flat["device_type"]
        if flat.get("device_category"):
            attributes["attribute_7"] = flat["device_category"]

        # Modules
        brand_raw = flat.get("device_brand")
        brand_enum, brand_other = self._normalize_brand(brand_raw or "")
        purchase = self._iso_date(
            flat.get("purchase_date", "") or flat.get("lieferdatum", "")
        )

        modules = {
            "attribute_604": bool(flat.get("imei_unknown", False)),
            "attribute_67": flat.get("imei") or flat.get("serial_number") or "",
            "attribute_70": brand_enum or "",
            "attribute_217": brand_other or "",
            "attribute_130": flat.get("device_model") or "",
            "attribute_166": flat.get("device_price") or "",
            "attribute_1": purchase or None,
            "attribute_16": flat.get("manufacturer_warranty_months", 24),
        }

        return {
            "Person": person,
            "Payment": payment,
            "Attributes": attributes,
            "Modules": modules,
        }


# -------------------------------
# HELVETIA adapter implementation
# -------------------------------


class HelvetiaAdapter(FormAdapter):
    """
    Adapter for HELVETIA payload mapping.

    Transforms flat conversational fields into a simplified HELVETIA structure:
      {
        "customer": {...},
        "product": {...},
        "payment": {...}
      }
    """

    PAYMENT_CODES = {
        "sepa": "SEPA",
        "bank": "SEPA",
        "iban": "SEPA",
        "credit": "CREDIT_CARD",
        "credit card": "CREDIT_CARD",
        "kreditkarte": "CREDIT_CARD",
        "visa": "CREDIT_CARD",
        "mastercard": "CREDIT_CARD",
        "paypal": "PAYPAL",
    }

    def provider_name(self) -> str:
        """Return the provider name supported by this adapter."""
        return "HELVETIA"

    def required_fields(self) -> Set[str]:
        """
        Minimal flat fields for HELVETIA (example baseline).

        Returns:
            A set of required flat keys.
        """
        return {
            "first_name",
            "last_name",
            "email",
            "street",
            "zipcode",
            "city",
            "country_code",
            "payment_method",
            "payment_account",
            "product_name",
        }

    def _provider_type(self, raw: Any) -> Optional[str]:
        """
        Map a free-text payment method to a HELVETIA payment code.

        Args:
            raw: Raw input (e.g., 'sepa', 'paypal', 'visa').

        Returns:
            String code (e.g., 'SEPA', 'PAYPAL') or None if not recognized.
        """
        if not raw:
            return None
        return self.PAYMENT_CODES.get(str(raw).strip().lower())

    def to_payload(self, flat: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build the HELVETIA payload from flat conversational values.

        Args:
            flat: Flat field dict (e.g., 'first_name', 'product_name', 'payment_method').

        Returns:
            HELVETIA-specific payload dictionary.
        """
        customer = {
            "firstName": flat.get("first_name"),
            "lastName": flat.get("last_name"),
            "email": flat.get("email"),
            "address": {
                "street": flat.get("street"),
                "streetNumber": flat.get("street_number"),
                "zipcode": flat.get("zipcode"),
                "city": flat.get("city"),
                "country": flat.get("country_code", "DE"),
            },
        }
        product = {
            "name": flat.get("product_name") or flat.get("product") or "",
            "model": flat.get("device_model"),
            "purchaseDate": flat.get("purchase_date"),
            "price": flat.get("device_price"),
        }
        payment = {
            "method": self._provider_type(flat.get("payment_method")),
            "account": flat.get("payment_account"),
            "owner": flat.get("payment_owner"),
        }
        return {"customer": customer, "product": product, "payment": payment}
