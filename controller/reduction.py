"""Typed cross-document operations retain aligned inputs and leaf provenance."""
from __future__ import annotations

from decimal import Decimal
import json
import re
import unicodedata

from .context_schema import EvidenceClaim


# Factors use canonical SI units, with time quantities canonicalized to seconds.
UNIT_FACTORS = {
    "": ("", Decimal(1)), "1": ("", Decimal(1)),
    "kg": ("g", Decimal(1000)), "g": ("g", Decimal(1)),
    "mg": ("g", Decimal("0.001")),
    "km": ("m", Decimal(1000)), "m": ("m", Decimal(1)),
    "cm": ("m", Decimal("0.01")),
    "h": ("s", Decimal(3600)), "min": ("s", Decimal(60)),
    "s": ("s", Decimal(1)),
    "美元": ("USD", Decimal(1)), "USD": ("USD", Decimal(1)),
    "元": ("CNY", Decimal(1)), "CNY": ("CNY", Decimal(1)),
    "万元": ("CNY", Decimal(10000)),
    "credit": ("credit", Decimal(1)), "credits": ("credit", Decimal(1)),
    "Credit": ("credit", Decimal(1)), "Credits": ("credit", Decimal(1)),
}


def entity_key(claim):
    return unicodedata.normalize("NFKC", claim.entity or claim.text).strip().casefold()


def aligned_claims(claims):
    groups = {}
    for claim in claims:
        key = (entity_key(claim), claim.conditions, claim.version)
        groups.setdefault(key, []).append(claim)
    return tuple(tuple(group) for group in groups.values())


def reduce_claims(operation, claims, *, claim_id, need_id):
    """Count unique entities and sum one aligned value per entity/condition/version."""
    claims = tuple(claim for claim in claims if claim.serves(need_id))
    groups = aligned_claims(claims)
    parents = tuple(claim.claim_id for claim in claims)
    spans = {span.span_id: span for claim in claims for span in claim.source_spans}
    unit = ""
    if operation == "count":
        value = len(groups)
    elif operation == "sum":
        total = Decimal(0)
        units = set()
        for group in groups:
            converted = []
            for claim in group:
                canonical, factor = UNIT_FACTORS.get(claim.unit, (claim.unit, Decimal(1)))
                literal = str(claim.value)
                if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", literal):
                    literal = literal.replace(",", "")
                converted.append((canonical, Decimal(literal) * factor))
            if len(set(converted)) != 1:
                raise ValueError(f"conflicting quantities for {group[0].entity}")
            canonical, number = converted[0]
            units.add(canonical)
            total += number
        if len(units) > 1:
            raise ValueError(f"sum requires compatible units: {sorted(units)}")
        unit = next(iter(units), "")
        value = str(total)
    elif operation in {"union", "group"}:
        def facts(group):
            unique = {}
            for claim in group:
                record = {"entity": claim.entity, "relation": claim.relation,
                    "value": claim.value, "unit": claim.unit, "text": claim.text,
                    "conditions": claim.conditions, "version": claim.version}
                identity = {key: item for key, item in record.items() if key != "text"}
                unique.setdefault(json.dumps(identity, sort_keys=True, ensure_ascii=False), record)
            return tuple(unique.values())
        if operation == "union":
            value = [fact for group in groups for fact in facts(group)]
        else:
            value = [{"entity": group[0].entity, "conditions": group[0].conditions,
                      "version": group[0].version, "facts": facts(group)} for group in groups]
    else:
        raise ValueError(f"model reduction required for {operation}")
    text = f"{operation} of {len(value)} entries" if operation in {"union", "group"} else str(value)
    return EvidenceClaim(claim_id, need_id, "derived", text=text, value=value,
                         unit=unit, source_spans=tuple(spans.values()), operation=operation,
                         parent_claim_ids=parents)
