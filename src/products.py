"""
products.py — the set of insurance product categories we support and the
mapping from PDF filename to category.

WHY this is its own module:
- The product category becomes metadata on every chunk in the vector store.
- The retrieval pipeline filters by category so the agent only ever looks
  at the right vilkår for the customer's actual product.
- Keeping the mapping in one place means adding a new vilkår PDF is a
  one-line change here, not a hunt across the codebase.
"""

from __future__ import annotations

from typing import Literal

Product = Literal["Bil", "Innbo", "Hus", "Hytte", "Reise", "Helse", "Person"]

PRODUCTS: tuple[Product, ...] = (
    "Bil", "Innbo", "Hus", "Hytte", "Reise", "Helse", "Person",
)

# Human-readable Norwegian labels for the UI.
LABELS: dict[Product, str] = {
    "Bil": "🚗 Bil (motor)",
    "Innbo": "🛋️  Innbo",
    "Hus": "🏠 Hus",
    "Hytte": "🏞️  Hytte",
    "Reise": "✈️  Reise",
    "Helse": "🏥 Helse / behandling",
    "Person": "❤️  Person (liv, ulykke, uføre)",
}


def from_filename(name: str) -> Product:
    """Derive the product category from the PDF filename.

    The Gjensidige vilkår files follow predictable name patterns — this
    function just normalises them to our category enum. If we get an
    unrecognised name we default to 'Person' (the catch-all for personal
    insurance) and log nothing — the alternative would be silently
    dropping the document, which is worse.
    """
    n = name.lower()
    if n.startswith("bil"):
        return "Bil"
    if n.startswith("innbo"):
        return "Innbo"
    if n.startswith("hus"):
        return "Hus"
    if n.startswith("hytte"):
        return "Hytte"
    if n.startswith("reise"):
        return "Reise"
    if any(n.startswith(prefix) for prefix in ("helse", "behandling", "alvorlig")):
        return "Helse"
    if any(n.startswith(prefix) for prefix in ("liv", "ufore", "ulykke")):
        return "Person"
    return "Person"  # safe default for anything personal/health-shaped
