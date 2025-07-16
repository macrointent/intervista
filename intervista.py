from fastmcp import FastMCP
import requests
from requests.auth import HTTPBasicAuth
import os
from dotenv import load_dotenv

load_dotenv()
mcp = FastMCP(name="Intervista-MCP")

url = os.getenv("INTERVISTA_URL")
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


catalog = {}


@mcp.tool()
def get_products(customer: str) -> dict[str, object]:
    global customers
    global catalog
    response = {}

    if not customer in customers:
        pass
    else:

        pv_ids = customers[customer]
        session = requests.Session()
        for id in pv_ids:
            if id in catalog:
                response[id] = catalog[id]
            else:
                item = session.get(url + id + path, auth=HTTPBasicAuth(user, pwd))
                data = item.json()["result"][0]

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
                catalog[id] = product
    return response


app = mcp.http_app(path="/")
