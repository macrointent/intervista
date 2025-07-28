from fastmcp import FastMCP
import requests
from requests.auth import HTTPBasicAuth
import os
import copy
from openai import OpenAI
import json
import datetime
from dotenv import load_dotenv

load_dotenv()
mcp = FastMCP(name="Intervista-MCP")

url = os.getenv("INTERVISTA_URL")
path = os.getenv("INTERVISTA_PATH")
user = os.getenv("INTERVISTA_USER")
pwd = os.getenv("INTERVISTA_PWD")

model = os.getenv("STACKIT_MODEL_ID")
key = os.getenv("STACKIT_KEY")
base = os.getenv("STACKIT_BASE")

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


@mcp.tool()
def get_products(customer: str) -> dict[str, object]:
    return _get_products(customer)


@mcp.tool()
def get_form(
    customer: str,
    displayName: str,
) -> dict[str, object]:

    global raw_data
    global configurations

    # retrieve pv_id
    pv_id = None
    data = None
    for product in products.values():
        if displayName in product["displayName"]:
            pv_id = product["productVariantId"]
            data = raw_data.get(pv_id)
            break

    # if pv_id in configurations:
    #     return configurations[pv_id]
    if not pv_id:
        response = _get_products(customer)
        for product in response.values():
            if displayName in product["displayName"]:
                pv_id = product["productVariantId"]
                data = raw_data.get(pv_id)
                break

    if not pv_id:
        return {"Error:", "Product name unknown, use the correct displayName."}

    if data is None:
        session = requests.Session()
        item = session.get(url + str(pv_id) + path, auth=HTTPBasicAuth(user, pwd))
        data = item.json()["result"][0]
        raw_data[pv_id] = data
    # if pv_id in raw_data:
    #     data = raw_data[pv_id]
    # else:
    #     session = requests.Session()
    #     item = session.get(url + pv_id + path, auth=HTTPBasicAuth(user, pwd))
    #     data = item.json()["result"][0]
    #     raw_data[pv_id] = data

    configuration = create_configuration(pv_id=pv_id, data=data)
    # print(configuration)
    # configurations[pv_id] = configuration
    return configuration


@mcp.tool()
def submit_form(data):

    json_data = _parse_json(data)
    url = os.getenv("SCHUTZGARANT_URL")
    params = {"dryRun": "true", "async": "true"}
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    session = requests.Session()
    answer = session.post(
        url,
        auth=HTTPBasicAuth(user, pwd),
        headers=headers,
        params=params,
        json=json_data,
    )

    try:
        content = answer.json()
    except Exception:
        content = answer.text
    response = {"status_code": answer.status_code, "content": content}
    return response


app = mcp.http_app(path="/")


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


def _get_products(customer: str) -> dict[str, object]:
    global customers
    global raw_data
    global products

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


# ------------- definition of generic building blocks -----------------
generic_payment = {
    "__type": "object",
    "__properties": {
        "providerType": {
            "__type": "integer",
            "__enum": [-160, -161, -162, -163, -164, -165],
            "__description": "Type of payment service provider. Possible values: -160 = Credit card, -161 = Bank, -162 = Amazon Pay, -163 = PayPal, -165 = Google Pay",
        },
        "providerNumber": {"__type": "string", "__description": "e.g. BIC"},
        "providerName": {
            "__type": "string",
            "__description": "Name of bank, VISA, AMEX, ...",
        },
        "accountNumber": {
            "__type": "string",
            "__description": "IBAN or credit card number",
        },
        "expirationDate": {
            "__type": "string",
            "__description": "expiration date of credit card number   ",
        },
        "accountOwner": {
            "__type": "string",
            "__description": "Different account/credit card holder",
        },
    },
    "__description": "Data of a payment service provider. It can be a bank, a credit card institution, or an online payment service.",
    "__required": [
        "providerType",
        "providerNumber",
        "providerName",
        "accountNumber",
        "accountOwner",
    ],
}


generic_person = {
    "__type": "object",
    "__properties": {
        "typeId": {
            "__type": "integer",
            "__description": "Type of the person. The main person has the ID -2101.",
        },
        "salutationId": {
            "__type": "integer",
            "__enum": [-31, -32, -33, -34, -35, -36, -37],
            "__description": "Possible values: -37 = Life Partnership, -36 = Shared Apartment, -35 = Community of Heirs, -34 = Ms. and Mr., -33 = Mr. and Ms., -32 = Ms., -31 = Mr., -30 = Company",
        },
        "title": {
            "__type": "string",
            "__description": "Title such as Dr., Prof., etc.",
            "__required": False,
        },
        "firstName": {
            "__type": "string",
            "__description": "First name. Not applicable for companies.",
        },
        "lastName": {"__type": "string", "__description": "Last name or company name."},
        "dateOfBirth": {
            "__type": "string",
            "__format": "date",
            "__pattern": "^\\d{4}-\\d{2}-\\d{2}$",
            "__description": "Date of birth for natural persons.",
        },
        "addresses": {
            "__type": "array",
            "__description": "Postal addresses of the person.",
            "__items": {
                "__type": "object",
                "__properties": {
                    "typeId": {
                        "__type": "integer",
                        "__enum": [-10, -11, -12, -13],
                        "__description": "Type of address. Possible values: -10 = Private Address, -11 = Business Address, -12 = Billing Address, -13 = Delivery Address.",
                    },
                    "street": {
                        "__type": "string",
                        "__description": "Street (without house number).",
                    },
                    "streetNumber": {
                        "__type": "string",
                        "__description": "House number.",
                    },
                    "addition": {
                        "__type": "string",
                        "__description": "Additional information such as '2nd floor'.",
                        "__required": False,
                    },
                    "zipcode": {"__type": "string", "__description": "Postal code."},
                    "city": {"__type": "string", "__description": "City name"},
                    "countryCode": {
                        "__type": "string",
                        "__maxLength": 2,
                        "__description": "2-character ISO country code (e.g., DE for Germany).",
                    },
                },
            },
        },
        "contactInformations": {
            "__type": "array",
            "__description": "Phone numbers or email addresses of the person.",
            "__items": {
                "__type": "object",
                "__properties": {
                    "typeId": {
                        "__type": "integer",
                        "__enum": [-20, -21, -22, -23],
                        "__description": "Type of contact method. Possible values: -21 = Mobile Phone, -20 = Landline Phone, -23 = Email, -22 = Fax.",
                    },
                    "value": {
                        "__type": "string",
                        "__description": "Phone number or email address.",
                    },
                },
                "__required": ["typeId", "value"],
            },
        },
    },
    "__description": "holds the personal information of a natural person or company",
    "__required": ["typeId"],
}

# ------------- helpers to create the template structures ------------


def extract_payment_information(pv_id, data):

    payment = {}
    print(f"Calling extract_payment_information {pv_id}")

    attributes = data["attributecategorie"]

    for attr in attributes:
        if attr["name"] == "Zahlungsinformationen":
            for pay_attr in attr["attribute"]:
                if pay_attr["attributeId"] == 376:
                    if len(pay_attr["listelement"]) == 1:
                        payment["typeId"] = pay_attr["listelement"][0]["listelementId"]
                        if payment["typeId"] == -5902:
                            payment["provider"] = {
                                "providerType": -161,
                                "providerNumber": {
                                    "__type": "string",
                                    "__description": "BIC of the bank, necessary for international accounts only",
                                    "__mandatory": False,
                                },
                                "providerName": {
                                    "__type": "string",
                                    "__description": "Name of the bank",
                                    "__mandatory": False,
                                },
                                "accountNumber": {
                                    "__type": "string",
                                    "__description": "IBAN number of the bank account",
                                    "__mandatory": True,
                                },
                                "accountOwner": {
                                    "__type": "string",
                                    "__description": "Name of account owner",
                                    "__mandatory": True,
                                },
                            }
                        else:
                            payment["provider"] = generic_payment
                    else:
                        typeIds = []
                        for elem in pay_attr["listelement"]:
                            typeIds.append(elem["listelementId"])
                        payment["typeId"] = {
                            "__type": "integer",
                            "__enum": intervalIds,
                            "__description": "payment method: Payment interval, where -5902 = direct debit, -5901 = bank transfer",
                        }
                        payment["provider"] = generic_payment
                elif pay_attr["attributeId"] == 379:
                    if len(pay_attr["listelement"]) == 1:
                        payment["intervalId"] = pay_attr["listelement"][0][
                            "listelementId"
                        ]
                    else:
                        intervalIds = []
                        for elem in pay_attr["listelement"]:
                            intervalIds.append(elem["listelementId"])
                        payment["intervalId"] = {
                            "__type": "integer",
                            "__enum": intervalIds,
                            "__description": "payment interval: Payment interval, where -5105 = one-time, -5101 = monthly, -5102 = quarterly, -5103 = semi-annually, -5104 = annually",
                        }
            return payment


def extract_attribute_information(pv_id, data):
    print(f"Calling extract_attribute_information {pv_id}")

    attributes = []
    for ac in data["attributecategorie"]:
        for a in ac["attribute"]:
            if a["inputField"] == True and a["hide"] == False:
                if a["type"] == "Liste":
                    values = []
                    for v in a["listelement"]:
                        values.append(v["name"])
                    attribute = {
                        "attributeId": a["attributeId"],
                        "value": {
                            "__type": "string",
                            "__enum": values,
                            "__description": a["displayName"],
                            "__default": a["value"],
                            "__mandatory": a["required"],
                        },
                    }
                    attributes.append(attribute)
                elif a["type"] == "Zahl":
                    attribute = {
                        "attributeId": a["attributeId"],
                        "value": {
                            "__type": "integer",
                            "__description": a["displayName"],
                            "__default": a["value"],
                            "__mandatory": a["required"],
                        },
                    }
                    attributes.append(attribute)
                elif a["type"] == "Datum":
                    attribute = {
                        "attributeId": a["attributeId"],
                        "value": {
                            "__type": "string",
                            "__format": "date",
                            "__pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                            "__description": a["displayName"],
                            "__default": a["value"],
                            "mandatory": a["required"],
                        },
                    }
                    attributes.append(attribute)
                elif a["type"] == "Text":
                    attribute = {
                        "attributeId": a["attributeId"],
                        "value": {
                            "__type": "string",
                            "__description": a["displayName"],
                            "__default": a["value"],
                            "__mandatory": a["required"],
                        },
                    }
                    attributes.append(attribute)
                elif a["type"] == "Auswahl":
                    attribute = {
                        "attributeId": a["attributeId"],
                        "value": {
                            "__type": "bool",
                            "__description": a["displayName"],
                            "__default": a["value"],
                            "__mandatory": a["required"],
                        },
                    }
                    attributes.append(attribute)
    return attributes


def extract_contactPersons_information(pv_id, data):
    contactPersons = []
    print(f"Calling extract_contactPersons_information {pv_id}")

    for sub_prod in data["subproduct"]:
        ppv_id = sub_prod["partProductVariantId"]
        if ppv_id == 19:
            person = copy.deepcopy(generic_person)
            person["__properties"]["typeId"] = -2101
            person["__properties"]["salutationId"]["__enum"] = [
                -30,
                -31,
                -32,
                -39,
            ]
            person["__properties"]["addresses"] = [
                {
                    "typeId": -11,
                    "street": {
                        "__type": "string",
                        "__description": "Street (without house number).",
                    },
                    "streetNumber": {
                        "__type": "string",
                        "__description": "House number.",
                    },
                    "addition": {
                        "__type": "string",
                        "__description": "Additional information such as '2nd floor'.",
                    },
                    "zipcode": {
                        "__type": "string",
                        "__description": "Postal code.",
                    },
                    "city": {"__type": "string", "__description": "City name"},
                    "countryCode": {
                        "__type": "string",
                        "__description": "2-character ISO country code (e.g., DE for Germany).",
                    },
                }
            ]
            del person["__properties"]["dateOfBirth"]
            person["__properties"]["contactInformations"] = [
                {
                    "typeId": -23,
                    "value": {
                        "__type": "string",
                        "__description": "corporate /business email",
                    },
                },
                {
                    "typeId": -20,
                    "value": {
                        "__type": "string",
                        "__description": "corporate /business phone number",
                    },
                },
            ]
            person["__description"] = (
                "main customer, mainly a company. can be a person but needs to provide business-related information (address, mail...)"
            )
            person["__required"] = True
            contactPersons.append(person)
        elif ppv_id == 3112:
            person = copy.deepcopy(generic_person)
            person["__properties"]["typeId"] = -101007
            del person["__properties"]["salutationId"]
            del person["__properties"]["dateOfBirth"]
            del person["__properties"]["addresses"]
            person["__properties"]["contactInformations"] = [
                {
                    "typeId": -23,
                    "value": {
                        "__type": "string",
                        "__description": "corporate /business email",
                    },
                },
                {
                    "typeId": -20,
                    "value": {
                        "__type": "string",
                        "__description": "corporate /business phone number",
                    },
                },
            ]
            person["__description"] = (
                "main contact person at customer. only allowed if customer is a company"
            )
            person["__required"] = False
            contactPersons.append(person)
        elif ppv_id == 3091:
            person = copy.deepcopy(generic_person)
            person["__properties"]["typeId"] = -2101
            person["__properties"]["salutationId"]["__enum"] = [-31, -32, -39]
            person["__properties"]["addresses"] = [
                {
                    "typeId": -10,
                    "street": {
                        "__type": "string",
                        "__description": "Street (without house number).",
                    },
                    "streetNumber": {
                        "__type": "string",
                        "__description": "House number.",
                    },
                    "addition": {
                        "__type": "string",
                        "__description": "Additional information such as '2nd floor'.",
                    },
                    "zipcode": {
                        "__type": "string",
                        "__description": "Postal code.",
                    },
                    "city": {"__type": "string", "__description": "City name"},
                    "countryCode": {
                        "__type": "string",
                        "__description": "2-character ISO country code (e.g., DE for Germany).",
                    },
                }
            ]
            person["__properties"]["contactInformations"] = [
                {
                    "typeId": -23,
                    "value": {
                        "__type": "string",
                        "__description": "private email",
                    },
                },
                {
                    "typeId": -20,
                    "value": {
                        "__type": "string",
                        "__description": "private phone number",
                    },
                },
            ]
            person["__description"] = "private customer / contractor."
            person["__required"] = True
            contactPersons.append(person)
    return contactPersons


def extract_modules_information(pv_id, data):
    modules = []
    print(f"Calling extract_modules_information {pv_id}")
    for sub_prod in data["subproduct"]:
        ppv_id = sub_prod["partProductVariantId"]
        if not ppv_id in [19, 3112, 3091]:
            modules.extend(
                extract_attribute_information(ppv_id, sub_prod["productTreeRenderData"])
            )
    return modules


def create_configuration(pv_id, data):
    configuration = {
        "voNumber": "C001006389",
        "productVariantId": pv_id,
        "contractCode": "Mantix",
        "desiredActivationDate": datetime.datetime.now().strftime("%Y-%m-%d"),
    }

    configuration["payment"] = extract_payment_information(pv_id=pv_id, data=data)
    configuration["contactPersons"] = extract_contactPersons_information(
        pv_id=pv_id, data=data
    )
    configuration["attributes"] = extract_attribute_information(pv_id=pv_id, data=data)
    configuration["modules"] = extract_modules_information(pv_id=pv_id, data=data)
    return configuration


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("intervista:app", host="0.0.0.0", port=8002, reload=True)
