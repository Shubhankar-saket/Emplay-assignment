"""The 20 output fields: label, extraction group, what to extract, and how to search for it."""
from __future__ import annotations

from dataclasses import dataclass

from core.models import DocType


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str  # name in the output JSON
    group: str  # extraction agent: dates | commercial | product | report
    description: str
    queries: tuple[str, ...] = ()
    doc_types: tuple[DocType, ...] | None = None  # extra retrieval pass restricted to these types
    addendum_sensitive: bool = False  # also search addenda (changes override the RFP)


FIELDS: list[FieldSpec] = [
    # --- dates ---
    FieldSpec("due_date", "Due Date", "dates",
              "Final deadline for submitting bids/proposals: date, time and time zone, after any addendum changes.",
              ("proposal due date and time", "submission deadline closing date"), ("bid_page",), True),
    FieldSpec("pre_bid_meeting", "Pre-Bid Meeting", "dates",
              "Pre-bid / pre-proposal conference: date, time, location or link, and whether attendance is mandatory.",
              ("pre-bid pre-proposal meeting conference",), ("bid_page",), True),
    FieldSpec("delivery_date", "Delivery Date", "dates",
              "Required delivery date or delivery timeframe (e.g. within N days of award).",
              ("delivery date timeframe after award", "shipping and delivery requirements"), None, True),
    FieldSpec("term", "Term", "dates",
              "Contract term / duration and renewal options.",
              ("contract term duration renewal options", "contract period years")),
    # --- commercial / legal ---
    FieldSpec("bid_number", "Bid Number", "commercial",
              "All solicitation identifiers (bid / RFP / solicitation / sourcing / PORFP numbers) from the RFP and "
              "the bid page. List every distinct one. Exclude the bid portal's own 'Reference Number'.",
              ("solicitation number RFP number bid number", "RFP # issued title page"), ("bid_page", "rfp")),
    FieldSpec("title", "Title", "commercial", "Official solicitation title.",
              ("solicitation title",), ("bid_page",)),
    FieldSpec("company", "Company", "commercial",
              "Issuing organization / agency (the buyer, not the vendors).",
              ("issuing organization agency district",), ("bid_page",)),
    FieldSpec("contact", "Contact", "commercial",
              "Procurement contact for this solicitation: name, email, phone.",
              ("buyer procurement contact name email phone", "Buyer name Email", "agency point of contact POC"), ("bid_page",)),
    FieldSpec("submission_type", "Submission Type", "commercial",
              "How responses must be submitted (online portal, email, sealed hard copy, number of copies).",
              ("how to submit proposal electronic submission portal email sealed",)),
    FieldSpec("bid_bond", "Bid Bond", "commercial",
              "Bid bond / bid security / proposal guarantee requirement and amount.",
              ("bid bond bid security proposal guarantee surety",)),
    FieldSpec("payment_terms", "Payment Terms", "commercial",
              "Payment and invoicing terms (net days, invoicing procedure, prompt payment).",
              ("payment terms invoice net days", "invoicing procedure payment")),
    FieldSpec("required_documents", "Required Documents", "commercial",
              "Every form, certificate and document bidders must submit with the response, as a list (e.g. W-9, "
              "conflict of interest questionnaire, HUB / subcontracting forms, affidavits, signed attachments, "
              "price sheets). Forms included in the bid package count.",
              ("required forms documents to submit with proposal", "affidavit certification forms must be completed"),
              ("affidavit", "form")),
    FieldSpec("mfg_registration", "MFG Registration", "commercial",
              "Manufacturer authorization, OEM certification or reseller/partner registration required of bidders.",
              ("manufacturer authorized reseller partner certification OEM",)),
    FieldSpec("cooperative_contract", "Cooperative Contract", "commercial",
              "Whether other entities may use the contract (cooperative / piggyback / interlocal) and any named program.",
              ("cooperative purchasing piggyback interlocal other entities",), ("bid_page",)),
    FieldSpec("installation", "Installation", "commercial",
              "Installation, deployment, imaging, configuration, etching / asset tagging or setup services required.",
              ("installation deployment imaging configuration services", "asset tagging etching setup"), None, True),
    # --- product ---
    FieldSpec("product", "Product", "product", "Products / items requested, with quantities.",
              ("items requested quantity laptops devices", "price schedule line items"), ("bid_page",)),
    FieldSpec("model_no", "Model_no", "product",
              "Manufacturer model number(s) / name(s) of requested products (e.g. 'Latitude 5550'). List all.",
              ("model number", "product model specifications"), ("specs",)),
    FieldSpec("part_no", "Part_no", "product",
              "Manufacturer part numbers / SKUs (e.g. '210-BLYZ'). List the relevant ones.",
              ("SKU part number",), ("specs",)),
    FieldSpec("product_specs", "Product Specs", "product",
              "Key technical specifications per product (processor, memory, storage, display, OS, warranty, ...).",
              ("technical specifications processor memory storage display", "minimum hardware requirements"),
              ("specs",), True),
    # --- written by the report agent from the other fields ---
    FieldSpec("summary", "Summary", "report", "Short summary of the solicitation."),
]

SPECS: dict[str, FieldSpec] = {f.key: f for f in FIELDS}
GROUPS: dict[str, list[str]] = {}
for _f in FIELDS:
    if _f.group != "report":
        GROUPS.setdefault(_f.group, []).append(_f.key)
DATE_FIELDS = ("due_date", "pre_bid_meeting", "delivery_date")
