from fastmcp import FastMCP, Context
import requests
from requests.auth import HTTPBasicAuth
import os
import copy
from openai import OpenAI
from typing import Dict, List
import json
import datetime
import logging
from dotenv import load_dotenv
import models
from pydantic import ValidationError


load_dotenv()


base = os.getenv("INTERVISTA_URL")
path = os.getenv("INTERVISTA_PATH")
user = os.getenv("INTERVISTA_USER")
pwd = os.getenv("INTERVISTA_PWD")


customers = {
    "Schutzgarant": [
        "64",
        "3415",
        "3424",
        "3436",
        "3448",
        "3571",
        "3610",
    ],
    "Test": [
        "64",
    ],
}

raw_data = {}
products = {}
configurations = {}
sessions: Dict[str, Dict] = {}


class MCPSession:
    def __init__(self, session_id: str, user_id: str, token: str, client_id: str):
        self.user_id = user_id
        self.session_id = session_id
        self.token = token
        self.client_id = client_id
        self.created_at = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()

        self.models: Dict[str, Dict] = {}
        self.forms: Dict[str, Dict] = {}

    async def get_model(self, pv_id: str) -> Dict:
        model = self.models.get(pv_id)
        if model:
            return model
        else:
            model = await self._retrieve_model(pv_id)
            self.models[pv_id] = model
            self.forms[pv_id] = {}
            return model

    async def submit_payment(self, pv_id: str, payment_data) -> str:
        payment_model = (await self.get_model(pv_id))["Payment"]
        try:
            payment_form = payment_model(payment_data)
        except (ValidationError, TypeError) as e:
            return str(e)
        self.forms.setdefault(pv_id, {})["Payment"] = payment_form
        return json.dumps(payment_form.model_dump())

    async def submit_person(self, pv_id: str, person_data) -> str:
        person_model = (await self.get_model(pv_id))["Person"]
        try:
            person_form = person_model(person_data)
        except (ValidationError, TypeError) as e:
            return str(e)
        self.forms.setdefault(pv_id, {})["Person"] = person_form
        return json.dumps(person_form.model_dump())

    async def submit_attributes(self, pv_id: str, attribute_data) -> str:
        attribute_model = (await self.get_model(pv_id))["Attributes"]
        try:
            attribute_form = attribute_model(attribute_data)
        except (ValidationError, TypeError) as e:
            return str(e)
        self.forms.setdefault(pv_id, {})["Attributes"] = attribute_form
        return json.dumps(attribute_form.model_dump())

    async def submit_modules(self, pv_id: str, module_data) -> str:
        module_model = (await self.get_model(pv_id))["Modules"]
        try:
            module_form = module_model(module_data)
        except (ValidationError, TypeError) as e:
            return str(e)
        self.forms.setdefault(pv_id, {})["Modules"] = module_form
        return json.dumps(module_form.model_dump())

    async def missing_data(self, pv_id: str) -> List[Dict]:
        model = await self.get_model(pv_id)
        missing = ["Payment", "Person", "Attributes", "Modules"]
        missing_models = []
        form = self.forms.get(pv_id)
        if form:
            for key in form.keys():
                missing.remove(key)
        for key in missing:
            missing_models.append(model[key])
        return missing_models

    @staticmethod
    async def _retrieve_model(pv_id: str) -> Dict:
        model = {}

        url = f"{base}{pv_id}{path}"
        session = requests.Session()
        response = session.get(url, auth=HTTPBasicAuth(user, pwd))
        data = response.json()["result"][0]

        model["Payment"] = models.PaymentProvider
        model["Person"] = models.Person
        attributes = [
            attr for cat in data["attributecategorie"] for attr in cat["attribute"]
        ]
        model["Attributes"] = models.generate_attribute_model("Attributes", attributes)

        modules = []
        for sub_prod in data[1]["subproduct"]:
            ppv_id = sub_prod["partProductVariantId"]
            if not ppv_id in [19, 3112, 3091]:
                modules.extend(
                    [
                        attr
                        for cat in sub_prod["productTreeRenderData"][
                            "attributecategorie"
                        ]
                        for attr in cat["attribute"]
                    ]
                )

        model["Modules"] = models.generate_attribute_model("Modules", modules)

        return model


mcp = FastMCP(name="stateful-MCP")


@mcp.tool
async def select_product(product_id: str, context: Context):
    # print(f"Session: {context.session_id}")
    # logging.info(f"Context: {context}")
    # if context.state.get("step") not in (None, "choice"):
    #     return {"error": "You can only select a product at the start."}
    # context.state["selected_product_id"] = product_id
    # context.state["step"] = "confirmed"
    # context.set_state = "test"
    return {
        "ok": True,
        "msg": f"Product {product_id} selected for State {context.get("user_id")}.",
    }


@mcp.tool
async def reset(context: Context):
    # context.state.clear()
    # context.state["step"] = "choice"
    return {
        "ok": True,
        "msg": f"Session reset for State {context.get_http_request.__dict__}.",
    }


@mcp.tool
def show_selection(context: Context):

    return {"selected_product_id": context.state["selected_product_id"]}


app = mcp.http_app(path="/")


def _get_products(customer: str, context=None) -> dict[str, object]:
    global customers
    global raw_data
    global products
    print(f"Context retrieved: {context}")
    context.state["customer"] = customer
    response = {}

    if not customer in customers:
        pass
    else:

        pv_ids = customers[customer]
        session = requests.Session()
        for id in pv_ids:
            if id in products:
                response[id] = products[id]
            else:
                data = None
                if id in raw_data:
                    data = raw_data[id]
                else:
                    item = session.get(url + id + path, auth=HTTPBasicAuth(user, pwd))
                    data = item.json()["result"][0]
                    raw_data[id] = data

                product = {}
                pv_master = data["product"]
                at_master = data["attributecategorie"]

                product["productId"] = pv_master["productId"]
                product["productVariantId"] = pv_master["productVariantId"]
                product["displayName"] = pv_master["displayName"]
                product["tag"] = pv_master["tag"]

                attribute = []
                for cat in at_master:
                    for item in cat["attribute"]:
                        temp = {}
                        if item["inputField"] == True:
                            if (
                                item["type"] == "Liste"
                                and len(item["listelement"]) == 1
                            ):
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
                response[id] = product
                products[id] = product
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("statefull_mcp:app", host="0.0.0.0", port=8005, reload=True)
