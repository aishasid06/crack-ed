"""Prices in their own currency (never converted)."""

from __future__ import annotations

SYMBOL = {"GBP": "£", "EUR": "€", "USD": "$"}


def money(v: float | None, currency: str | None) -> str:
    """£12.00, €86.99, $5.00, otherwise "CHF 12.00"; GBP when the currency is unknown."""
    if v is None:
        return "unknown"
    cur = (currency or "GBP").upper()
    return f"{SYMBOL[cur]}{v:,.2f}" if cur in SYMBOL else f"{cur} {v:,.2f}"
