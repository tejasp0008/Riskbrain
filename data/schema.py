"""Dispute and transaction schemas shared across the pipeline.

Reason codes are grounded in real Visa and Mastercard dispute taxonomy
categories (not generic labels). Each entry carries the network,
category, a representment deadline in days, and the evidence the network
expects for a representment. This table is used by both the synthetic
data generator (Phase 2) and the dispute agent's retrieval/decision
steps (Phase 3-4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ReasonCodeCategory(str, Enum):
    FRAUD = "fraud"
    AUTHORIZATION = "authorization"
    PROCESSING_ERROR = "processing_error"
    CONSUMER_DISPUTE = "consumer_dispute"


@dataclass(frozen=True)
class ReasonCode:
    network: str  # "visa" or "mastercard"
    code: str
    category: ReasonCodeCategory
    description: str
    representment_deadline_days: int
    evidence_required: tuple[str, ...]


REASON_CODES: dict[str, ReasonCode] = {
    "visa_10.4": ReasonCode(
        network="visa",
        code="10.4",
        category=ReasonCodeCategory.FRAUD,
        description="Other Fraud - Card Absent Environment",
        representment_deadline_days=30,
        evidence_required=(
            "AVS match result",
            "CVV match result",
            "device fingerprint / IP geolocation",
            "prior undisputed transaction history from same device or shipping address",
        ),
    ),
    "visa_13.1": ReasonCode(
        network="visa",
        code="13.1",
        category=ReasonCodeCategory.CONSUMER_DISPUTE,
        description="Merchandise / Services Not Received",
        representment_deadline_days=30,
        evidence_required=(
            "proof of delivery with signature or tracking",
            "delivery date within expected window",
            "description matching order",
        ),
    ),
    "visa_12.6.1": ReasonCode(
        network="visa",
        code="12.6.1",
        category=ReasonCodeCategory.PROCESSING_ERROR,
        description="Duplicate Processing",
        representment_deadline_days=30,
        evidence_required=(
            "proof the two transactions are for distinct goods/services or dates",
            "original transaction receipts",
        ),
    ),
    "mc_4837": ReasonCode(
        network="mastercard",
        code="4837",
        category=ReasonCodeCategory.FRAUD,
        description="No Cardholder Authorization",
        representment_deadline_days=45,
        evidence_required=(
            "AVS/CVV match result",
            "3-D Secure authentication result",
            "device fingerprint / IP geolocation",
        ),
    ),
    "mc_4853": ReasonCode(
        network="mastercard",
        code="4853",
        category=ReasonCodeCategory.CONSUMER_DISPUTE,
        description="Cardholder Dispute - Not as Described or Defective",
        representment_deadline_days=45,
        evidence_required=(
            "product description at time of sale",
            "communication with cardholder",
            "return/refund policy shown at checkout",
        ),
    ),
    "mc_4831": ReasonCode(
        network="mastercard",
        code="4831",
        category=ReasonCodeCategory.PROCESSING_ERROR,
        description="Transaction Amount Differs",
        representment_deadline_days=45,
        evidence_required=(
            "itemized receipt matching charged amount",
            "currency conversion evidence if applicable",
        ),
    ),
}


@dataclass
class Transaction:
    transaction_id: str
    amount: float
    currency: str
    timestamp: datetime
    device_id: str
    device_is_new: bool
    ip_country: str
    bin_country: str
    mcc: str
    is_card_present: bool
    velocity_1h: int
    velocity_24h: int


@dataclass
class Dispute:
    dispute_id: str
    transaction_id: str
    network: str
    reason_code: str
    filed_at: datetime
    amount: float
    narrative_text: str = ""
    merchant_evidence: list[str] = field(default_factory=list)

    @property
    def reason(self) -> ReasonCode:
        key = f"{self.network}_{self.reason_code}"
        return REASON_CODES[key]
