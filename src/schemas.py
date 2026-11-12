"""
Event contracts (schema v1) shared by producer, consumer and API.

Pydantic models give us:
* strict validation at the edge – bad events go to the dead-letter queue
  instead of crashing the consumer or poisoning features;
* one definition reused for the FastAPI request/response schemas (OpenAPI docs
  for free).
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = 1


class TxnType(str, Enum):
    POS = "POS"                 # card-present purchase at a terminal
    ECOM = "ECOM"               # online purchase (card-not-present)
    ATM = "ATM"                 # cash withdrawal
    TRANSFER = "TRANSFER"       # account-to-account transfer
    P2P = "P2P"                 # peer-to-peer wallet payment


class Channel(str, Enum):
    CARD_PRESENT = "card_present"
    CARD_NOT_PRESENT = "card_not_present"


class Transaction(BaseModel):
    """A single authorisation request as it arrives from the payment switch."""

    model_config = ConfigDict(extra="ignore", frozen=False)

    schema_version: int = SCHEMA_VERSION
    txn_id: str = Field(min_length=8, max_length=64)
    event_time: datetime
    card_id: str = Field(min_length=4)
    merchant_id: str = Field(min_length=2)
    merchant_name: str
    merchant_category: str
    txn_type: TxnType
    channel: Channel
    amount: float = Field(gt=0)                       # local currency (KRW/NGN can exceed 1M)
    currency: str = Field(min_length=3, max_length=3)
    amount_usd: float = Field(gt=0, lt=1_000_000)     # sanity cap in a common currency
    country: str = Field(min_length=3, max_length=3)
    city: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    device_id: str
    ip_country: str = Field(min_length=3, max_length=3)

    @field_validator("currency", "country", "ip_country")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()


class FraudLabel(BaseModel):
    """Delayed ground truth (chargeback / analyst confirmation)."""

    txn_id: str
    card_id: str
    is_fraud: bool
    fraud_type: str
    label_time: datetime


class Decision(str, Enum):
    APPROVE = "approve"
    REVIEW = "review"
    DECLINE = "decline"


class ScoreResponse(BaseModel):
    txn_id: str
    risk_score: float = Field(ge=0, le=100)
    decision: Decision
    if_score: float
    rule_points: float
    rules_triggered: list[str]
    top_features: list[dict]
    model_version: str
    latency_ms: float
