"""Tools that read and write CRM / banking data."""
from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel, Field

from crm_agent.data.repository import Repository
from crm_agent.guardrails.rules import check_consent, check_message
from crm_agent.scoring.features import enrich, to_records
from crm_agent.tools.registry import ToolError, tool


@lru_cache(maxsize=1)
def repo() -> Repository:
    return Repository()


class FindCustomersArgs(BaseModel):
    filters: dict = Field(default_factory=dict,
                          description="Whitelisted filters: city, segment, occupation_type, preferred_language, "
                                      "min_age, max_age, customer_ids")


@tool("find_customers", "List the RM's customers matching structured filters.", FindCustomersArgs)
def find_customers(filters: dict) -> list[dict]:
    return [{"customer_id": c["customer_id"], "name": c["name"], "city": c["city"], "segment": c["segment"]}
            for c in repo().find_customers(filters)]


class FeatureArgs(BaseModel):
    customer_ids: list[str] | None = Field(None, description="Subset to return; None = whole book")


@tool("get_customer_features",
      "Aggregate 6 months of transactions, accounts, digital events and campaign history into features.",
      FeatureArgs)
def get_customer_features(customer_ids: list[str] | None = None) -> dict[str, dict]:
    return to_records(enrich(repo().customer_features(customer_ids)))


class LookupArgs(BaseModel):
    query: str = Field(..., description="Customer id (C0123) or (part of) a name")


@tool("lookup_customer", "Find a customer by id or name.", LookupArgs)
def lookup_customer(query: str) -> list[dict]:
    return [{"customer_id": c["customer_id"], "name": c["name"], "city": c["city"]}
            for c in repo().search_customer(query)]


class CatalogueArgs(BaseModel):
    product_type: str | None = None


@tool("get_product_catalogue", "Return product variants and their policy parameters.", CatalogueArgs)
def get_product_catalogue(product_type: str | None = None) -> list[dict]:
    return repo().products(product_type)


class ConsentArgs(BaseModel):
    customer: dict
    channel: str = "whatsapp"


@tool("check_contact_consent", "Check channel opt-in and DND before any message is drafted.", ConsentArgs)
def check_contact_consent(customer: dict, channel: str = "whatsapp") -> dict:
    v = check_consent(customer, channel)
    return {"allowed": v is None, "reason": v.message if v else None, "rule_id": v.rule_id if v else None}


class SendArgs(BaseModel):
    customer: dict
    body: str
    offer: dict
    approved_by: str
    thread_id: str
    channel: str = "whatsapp"


@tool("send_whatsapp_message",
      "Queue an APPROVED message on the WhatsApp Business provider (mock). Re-validates consent and content.",
      SendArgs, side_effect=True)
def send_whatsapp_message(customer: dict, body: str, offer: dict, approved_by: str, thread_id: str,
                          channel: str = "whatsapp") -> dict:
    # Defence in depth: the send path re-checks even though the graph already did.
    if (v := check_consent(customer, channel)) is not None:
        raise ToolError(f"Blocked at send: {v.message}")
    if violations := check_message(body, {"offer": offer, "first_name": customer.get("first_name")}):
        raise ToolError(f"Blocked at send: {violations[0].message}")
    if repo().already_sent(customer["customer_id"], body):
        return {"status": "duplicate_skipped"}
    message_id = repo().enqueue_message(customer["customer_id"], channel, body, approved_by, thread_id)
    repo().audit(thread_id, approved_by, "message_queued",
                 {"customer_id": customer["customer_id"], "message_id": message_id})
    return {"status": "queued", "message_id": message_id}
