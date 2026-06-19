"""
products.py — exact list of supported insurance products + filename mapping.

I went with per-product filtering instead of per-category because of how
Gjensidige actually sells these — a customer has "Bil Pluss" or "Bil
Kasko", not just "bil insurance". The vilkår for Bil Pluss and Bil Kasko
are different documents and the egenandel rules differ. Letting the
agent retrieve from "any Bil document" leads to the wrong clause being
cited.

Adding a new vilkår PDF is a one-line change in PRODUCTS below.
"""

from __future__ import annotations

# (display label, filename, category) — order here is the order shown in the UI.
PRODUCTS: tuple[tuple[str, str, str], ...] = (
    # 🚗 Bil
    ("Bil Ansvar",           "Bil Ansvar 1.7.25.pdf",                             "Bil"),
    ("Bil Delkasko",         "Bil-Delkasko-alminnelige-vilkar.pdf",               "Bil"),
    ("Bil Kasko",            "Bil-Kasko-alminnelige-vilkar.pdf",                  "Bil"),
    ("Bil Pluss",            "Bil-Pluss-alminnelige-vilkar.pdf",                  "Bil"),
    # 🛋️  Innbo
    ("Innbo Standard",       "Innbo-Standard-alminnelige vilkar.pdf",             "Innbo"),
    ("Innbo Pluss",          "Innbo-Pluss-alminnelige-vilkar.pdf",                "Innbo"),
    ("Innbo Ung",            "Innbo-Ung-alminnelige-vilkar.pdf",                  "Innbo"),
    # 🏠 Hus
    ("Hus Standard",         "Hus-Standard-alminnelige-vilkar.pdf",               "Hus"),
    ("Hus Pluss",            "Hus-Pluss-alminnelige-vilkar.pdf",                  "Hus"),
    # 🏞️  Hytte
    ("Hytte Standard",       "Hytte-Standard-alminnelige-vilkar.pdf",             "Hytte"),
    ("Hytte Pluss",          "Hytte-Pluss-alminnelige-vilkar.pdf",                "Hytte"),
    # ✈️  Reise
    ("Reise",                "Reise-alminnelige-vilkar.pdf",                      "Reise"),
    ("Reise Pluss",          "Reise-Pluss-alminnelige-vilkar.pdf",                "Reise"),
    ("Reise Student",        "Reise-Student-i-utland-alminnelige-vilkar.pdf",     "Reise"),
    # 🏥 Helse / behandling
    ("Behandlingsforsikring", "behandlingsforsikring-alminnelige-vilkar.pdf",     "Helse"),
    ("Helse 55",             "helse-55-alminnelige-vilkar.pdf",                   "Helse"),
    ("Alvorlig sykdom",      "alvorlig-sykdom-alminnelige-vilkar.pdf",            "Helse"),
    # ❤️  Person
    ("Livsforsikring",       "livsforsikring-alminnelige-vilkar.pdf",             "Person"),
    ("Uføre pensjon",        "uforepensjon-alminnelige-vilkar.pdf",               "Person"),
    ("Uføre kapital",        "uforekapital-med-forskudd-alminnelige-vilkar.pdf",  "Person"),
    ("Ulykkesforsikring",    "ulykkesforsikring-alminnelige-vilkar.pdf",          "Person"),
)

CATEGORY_EMOJI = {
    "Bil": "🚗",
    "Innbo": "🛋️",
    "Hus": "🏠",
    "Hytte": "🏞️",
    "Reise": "✈️",
    "Helse": "🏥",
    "Person": "❤️",
}

LABEL_TO_FILE = {label: filename for label, filename, _ in PRODUCTS}
FILE_TO_LABEL = {filename: label for label, filename, _ in PRODUCTS}


def categories() -> list[str]:
    """Unique categories in their first-appearance order."""
    seen: list[str] = []
    for _, _, cat in PRODUCTS:
        if cat not in seen:
            seen.append(cat)
    return seen


def products_in(category: str) -> list[tuple[str, str]]:
    """(label, filename) for everything in a category."""
    return [(label, fname) for label, fname, cat in PRODUCTS if cat == category]


def category_for(label: str) -> str:
    for lab, _, cat in PRODUCTS:
        if lab == label:
            return cat
    return ""
