import logging
import json
from fastapi import FastAPI, Request
from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_request
from starlette.requests import Request
from httpx import AsyncClient
from typing import Optional, Literal, List
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    create_model,
    constr,
    model_validator,
)

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("INTERVISTA_URL")
path = os.getenv("INTERVISTA_PATH")
user = os.getenv("INTERVISTA_USER")
pwd = os.getenv("INTERVISTA_PWD")
ids = [x.strip() for x in os.getenv("INTERVISTA_IDS").strip("[]").split(",")]

model = os.getenv("STACKIT_MODEL_ID")
key = os.getenv("STACKIT_KEY")
base = os.getenv("STACKIT_BASE")

data = None
catalog = None
forms = None


# ------- Static model definitions ----------------


class Address(BaseModel):
    typeId: int = Field(
        ...,
        description="Address type. -10 = Private, -11 = Business, -12 = Billing, -13 = Delivery",
    )
    street: str = Field(..., description="Street (without house number)")
    streetNumber: str = Field(..., description="House number")
    addition: Optional[str] = Field(
        None, description="Additional info (e.g. '2nd floor')"
    )
    zipcode: str = Field(..., description="Postal code")
    city: str = Field(..., description="City name")
    countryCode: str = Field(
        ..., min_length=2, max_length=2, description="2-char ISO country code (e.g. DE)"
    )


class ContactInfo(BaseModel):
    typeId: int = Field(
        ..., description="Contact type. -21=Mobile, -20=Landline, -23=Email, -22=Fax"
    )
    value: str = Field(..., description="Contact value: phone/email/etc.")


class Person(BaseModel):
    typeId: int = Field(..., description="Person type: -2101 main, -30 company")
    salutationId: Optional[int] = Field(
        None, description="Salutation ID, required for individuals"
    )
    title: Optional[str] = Field(None, description="Title such as Dr., Prof., etc.")
    firstName: Optional[str] = Field(None, description="First name. Not for companies.")
    lastName: str = Field(..., description="Last name or company name")
    dateOfBirth: Optional[str] = Field(
        None, description="Date of birth (YYYY-MM-DD)", pattern="^\\d{4}-\\d{2}-\\d{2}$"
    )
    addresses: List[Address] = Field(
        default_factory=list, description="Addresses of the person"
    )
    contactInformations: List[ContactInfo] = Field(
        default_factory=list, description="Contact info (phone, email, etc.)"
    )

    @model_validator(mode="before")
    @classmethod
    def check_lists(cls, data):
        errors = []
        if "addresses" in data and not isinstance(data["addresses"], list):
            errors.append(
                f"addresses must be a list, got {type(data['addresses']).__name__} ({data['addresses']!r})"
            )
        if "contactInformations" in data and not isinstance(
            data["contactInformations"], list
        ):
            errors.append(
                f"contactInformations must be a list, got {type(data['contactInformations']).__name__} ({data['contactInformations']!r})"
            )
        if errors:
            raise TypeError("; ".join(errors))
        return data

    @model_validator(mode="after")
    def check_fields(self):
        # Company or person?
        if self.typeId == -30:  # Company
            # Only lastName (company name) required
            if not self.lastName:
                raise ValueError("lastName (company name) required for companies.")
        else:
            # Person: salutationId, firstName, lastName required
            missing = []
            if not self.salutationId:
                missing.append("salutationId")
            if not self.firstName:
                missing.append("firstName")
            if not self.lastName:
                missing.append("lastName")
            if missing:
                raise ValueError(
                    f"Missing required fields for person: {', '.join(missing)}"
                )
        return self


class PaymentProvider(BaseModel):
    providerType: int = Field(
        ...,
        description="Type of payment service provider. Possible values: -160 = Credit card, -161 = Bank, -162 = Amazon Pay, -163 = PayPal, -165 = Google Pay",
    )
    providerNumber: Optional[str] = Field(
        None, description="BIC of the bank, necessary for intern'ational accounts only"
    )
    providerName: Optional[str] = Field(
        None, description="Possible values: Name of bank, 'VISA', 'AMEX', ..."
    )
    accountNumber: Optional[str] = Field(
        None,
        description="IBAN, credit card number or payment id (associated email for online payment services)",
    )
    accountOwner: Optional[str] = Field(None, description="Name of account owner")
    expirationDate: Optional[str] = Field(
        None, description="Needed when providerType=-160"
    )

    @model_validator(mode="after")
    def check_fields(self):
        ptype = getattr(self, "providerType")
        if ptype == -160:
            required = [
                "providerName",
                "accountNumber",
                "accountOwner",
                "expirationDate",
            ]
        elif ptype == -161:
            required = ["providerNumber", "accountNumber", "accountOwner"]
        elif ptype < -161 and ptype > -166:
            required = ["accountNumber"]
        else:
            raise ValueError(
                f"ProviderType {ptype} unknown. Possible values: -160 = Credit card, -161 = Bank, -162 = Amazon Pay, -163 = PayPal, -165 = Google Pay"
            )

        missing = [f for f in required if not getattr(self, f)]
        if missing:
            raise ValueError(
                f"Missing required fields for providerType {ptype}: {', '.join(missing)}"
            )
        return self


# ------- Handling the interaction with Intervista ----------------


async def _get_product_data(pv_id: str):
    logging.info(f"user: {user}, pwd: {pwd}")
    async with AsyncClient(auth=(user, pwd)) as client:
        response = await client.get(f"{url}{pv_id}{path}")
        response.raise_for_status()  # Will raise an error if the HTTP code was not 2xx
        raw_data = response.json()["result"][0]
        return raw_data


async def _reset():
    global data
    global catalog
    global forms

    data = {}
    catalog = {}
    forms = {}

    for id in ids:
        raw_data = await _get_product_data(id)
        data[id] = raw_data
        catalog[id] = _extract_product_spec(raw_data)
        forms[id] = _get_form_schema(id, raw_data)


# ------- Functions needed to create the global product data ----------------


def _extract_product_spec(raw_data: str):
    product = {}
    pv_master = raw_data["product"]
    at_master = raw_data["attributecategorie"]

    product["id"] = pv_master["productVariantId"]
    product["displayName"] = pv_master["displayName"]
    product["tag"] = pv_master["tag"]

    attribute = []
    for cat in at_master:
        for item in cat["attribute"]:
            temp = {}
            if item["inputField"] == True:
                if item["type"] == "Liste" and len(item["listelement"]) == 1:
                    temp["displayName"] = item["displayName"]
                    temp["displayDescription"] = item["displayDescription"]
                    temp["value"] = item["listelement"][0]["name"]
                    temp["visible"] = not item["hide"]
                    attribute.append(temp)
            else:
                temp["displayName"] = item["displayName"]
                temp["displayDescription"] = item["displayDescription"]
                temp["value"] = item["value"]
                temp["visible"] = not item["hide"]
                attribute.append(temp)

    product["attribute"] = attribute
    return product


def _extract_attribute_fields(model_name: str, raw_data: str):
    fields = {}
    attributes = [
        attr for cat in raw_data["attributecategorie"] for attr in cat["attribute"]
    ]

    for attr in attributes:
        if not (attr.get("inputField") is True and attr.get("hide") is False):
            continue

        # Valid Python field name
        field_name = f'attribute_{attr["attributeId"]}'

        title = attr.get("displayName", "")
        description = ""
        attr_type = attr.get("type")
        required = attr.get("required", False)
        default = attr.get("value", None)

        # Choose field type and constraints
        if attr_type == "Liste":
            allowed = tuple(val["name"] for val in attr.get("listelement", []))
            field_type = Literal[allowed] if allowed else str
            description += f" (allowed: {', '.join(allowed)})"
        elif attr_type == "Zahl":
            field_type = int
            try:
                if default is not None:
                    default = int(default)
            except:
                pass
            description += " (must be integer)"
        elif attr_type == "Datum":
            field_type = constr(pattern=r"^\d{4}-\d{2}-\d{2}$")
            description += " (YYYY-MM-DD)"
        elif attr_type == "Text":
            field_type = str
        elif attr_type == "Auswahl":
            field_type = bool
            try:
                if default is not None and isinstance(default, str):
                    default = default.lower() == "true"
            except:
                pass
        else:
            field_type = str

        if not required:
            field_type = Optional[field_type]
            # Default to None if not set
            default = default if default is not None else None
        else:
            default = default if default is not None else ...

        fields[field_name] = (
            field_type,
            Field(title=title, description=description, default=default),
        )
    return create_model(model_name, **fields)


def _get_form_schema(pv_id: str, raw_data: str):

    # create attribute schema
    Attributes = _extract_attribute_fields("Attributes", raw_data)

    # create module schema
    module_data = {}
    module_data["attributecategorie"] = []
    for sub_prod in raw_data["subproduct"]:
        ppv_id = sub_prod["partProductVariantId"]
        if not ppv_id in [19, 3112, 3091]:
            module_data["attributecategorie"].extend(
                [cat for cat in sub_prod["productTreeRenderData"]["attributecategorie"]]
            )
    Modules = _extract_attribute_fields("Modules", module_data)

    # compose form schema
    form_fields = {}
    form_fields["Person"] = (Person, Field(...))
    form_fields["Payment"] = (PaymentProvider, Field(...))
    form_fields["Attributes"] = (Attributes, Field(...))
    form_fields["Modules"] = (Modules, Field(...))
    Form = create_model(f"Form_{pv_id}", **form_fields)

    return Form


# ------- Helper functions needed to handle errors ----------------


# navigates to the (missing) schema field
def _get_schema_for_loc(schema, loc):
    """
    Traverse schema following the err['loc'], resolving refs as needed.
    Returns the field schema dict (where 'description', 'title', etc. live).
    """
    node = schema
    for i, part in enumerate(loc):
        # If $ref exists at this level, resolve it before going further
        if "$ref" in node.get("properties", {}).get(part, {}):
            node = _resolve_ref(schema, node["properties"][part]["$ref"])
        elif i < len(loc) - 1:
            # Go to next level down (should be an object property)
            node = node["properties"][part]
            # If this node is a $ref (and not directly on .properties), resolve
            if "$ref" in node:
                node = _resolve_ref(schema, node["$ref"])
        else:
            # Last part is the field itself
            node = node["properties"].get(part, {})
    return node


# resolves a reference for a schema node
def _resolve_ref(schema, ref):
    """Given a schema and a JSON pointer, resolve and return the referenced schema."""
    parts = ref.lstrip("#/").split("/")
    target = schema
    for part in parts:
        target = target[part]
    return target


# extracts information for (missing) schema field
def _extract_field_info(schema, loc):
    """Returns title, description, enum for the field at loc in schema."""
    field_schema = _get_schema_for_loc(schema.model_json_schema(), loc)
    return (
        field_schema.get("title", loc[-1]),
        field_schema.get("description", ""),
        field_schema.get("enum", []),
    )


# used to parse string input to json
def _parse_json(data):
    global url, key, model

    j_object = None

    if isinstance(data, dict):
        return data

    if not data or len(data) < 5:
        return None

    js = data
    for attempt in range(3):
        # Find JSON braces
        s = js.find("{")
        e = js.rfind("}") + 1
        if s == -1 or e <= s:
            print(f"Could not find valid JSON braces in: {js}")
            js = data  # reset for LLM
        else:
            try:
                j_object = json.loads(js[s:e])
                return j_object
            except json.JSONDecodeError:
                print(f"Attempt {attempt+1}: JSON decode failed.")

        # Try LLM repair
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a JSON expert function that accepts malformed and broken JSON strings. "
                        "You reply with the corrected version that requires the least changes and fully retains the content and structure. "
                        "Do not comment or explain, just return the corrected JSON."
                    ),
                },
                {"role": "user", "content": js},
            ]
            client = OpenAI(base_url=base, api_key=key)
            response = client.chat.completions.create(model=model, messages=messages)
            js = response.choices[0].message.content
        except Exception as e:
            print(f"An error occurred with LLM: {str(e)}")
            continue

    print(
        f"Failed to parse/fix JSON after 3 attempts. Original: {data}, Last attempt: {js}"
    )
    return None


# ------- MCP Tool Definitions ----------------

mcp = FastMCP(name="simple")


@mcp.tool(
    title="Reload Data",
    description="Triggers a refresh of data retrieved from the backend.",
    tags={"admin", "reset"},
)
async def reset():
    await _reset()


@mcp.tool(
    title="Get Products",
    description="Retrieves all available product specifications.",
    tags={"product", "catalog"},
)
async def get_catalog() -> str:
    global catalog

    request: Request = get_http_request()
    mantix_session_header = request.headers.get("x-mantix-session")

    if not catalog:
        await _reset()

    return json.dumps(catalog)


@mcp.tool(
    title="Get order form",
    description="Retrieves the order form for a specific product either by id or name.",
    tags={"product", "form", "order", "schema"},
)
async def get_form(id: Optional[str] = None, name: Optional[str] = None) -> str:
    global catalog
    global forms

    request: Request = get_http_request()
    mantix_session_header = request.headers.get("x-mantix-session")

    if not catalog:
        await _reset()

    # provide a
    if id and forms.get(id):
        result = forms.get(id).model_json_schema()
        return json.dumps(result, ensure_ascii=False)

    products = [{"id": v["id"], "name": v["displayName"]} for v in catalog.values()]
    options = []
    if name and name.strip():
        options = [p for p in products if name.lower() in p["name"].lower()]

    if len(options) > 0:
        id = options[-1]["id"]
        result = forms.get(str(id)).model_json_schema()
    # elif len(options) > 1:
    #     result = {
    #         "message": "product name ambiguous, which option did you mean?",
    #         "options": options,
    #     }
    else:
        result = {
            "message": "product unknown, here is the list of options",
            "options": products,
        }
    return json.dumps(result, ensure_ascii=False)

    # if not forms:
    #     await _reset()
    # form = forms.get(id)
    # if not form:
    #     return f"Error - Unknown Product / Form ID: {id}"
    # return json.dumps(form.model_json_schema())


@mcp.tool(
    title="Submit Order Form",
    description="Submits the order form for a specific product.",
    tags={"product", "form", "order", "schema"},
)
async def submit_form(id: str, data: str) -> str:
    request: Request = get_http_request()
    mantix_session_header = request.headers.get("x-mantix-session")

    if not forms:
        await _reset()
    form = forms.get(id)
    if not form:
        products = [{"id": v["id"], "name": v["displayName"]} for v in catalog.values()]
        result = {
            "message": "product unknown, here is the list of options",
            "options": products,
        }
        return json.dumps(result, ensure_ascii=False)

    try:
        json_data = _parse_json(data)
        form = form(**json_data)
        return f"successfully submitted {form.model_dump_json()} for session {mantix_session_header}."
    except ValidationError as e:
        suggestions = []
        for err in e.errors():
            loc = err["loc"]
            title, description, choices = _extract_field_info(form, loc)
            suggestion = f"Please provide a value for '{loc[-1]} ({title})'"
            if description:
                suggestion += f" {description}"
            if choices:
                suggestion += f". Possible values: {', '.join(choices)}"
            suggestions.append(suggestion)
        return (
            "submission failed:\n"
            + "\n".join(suggestions)
            + f"\nThis is the full schema: {form.model_json_schema()}"
        )


# ------- Server Configurations ----------------

app = FastAPI()
mcp_app = mcp.http_app(path="/")
app = FastAPI(lifespan=mcp_app.lifespan)
app.mount("/intervista", mcp_app)

logging.basicConfig(level=logging.INFO)


@app.middleware("http")
async def log_incoming_request(request: Request, call_next):
    body = await request.body()
    logging.info(f"INCOMING: {request.method} {request.url}")
    logging.info(f"Headers: {dict(request.headers)}")
    logging.info(f"Body: {body.decode('utf-8') if body else '(empty)'}")
    # Instead of forwarding here, let endpoints handle forwarding.
    response = await call_next(request)
    return response


# ------- Legacy Tools ----------------

# @mcp.tool()
# def get_address_form() -> str:
#     return json.dumps(Address.model_json_schema())


# @mcp.tool()
# def submit_address_form(data: str) -> str:
#     request: Request = get_http_request()
#     mantix_session_header = request.headers.get("x-mantix-session")

#     try:
#         json_data = _parse_json(data)
#         address = Address(**json_data)
#         # Valid!
#         return f"successfully processed {data} for session {mantix_session_header}."
#     except ValidationError as e:
#         # Not valid! Print or log the validation errors
#         print(e.json())
#         return e.json()
