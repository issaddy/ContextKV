"""Exact numeric conversion shared by document projections and aggregation."""
from __future__ import annotations

from decimal import Decimal
import re


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


def canonical_quantity(value, unit):
    canonical, factor = UNIT_FACTORS.get(unit, (unit, Decimal(1)))
    literal = str(value)
    if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", literal):
        literal = literal.replace(",", "")
    return canonical, Decimal(literal) * factor
