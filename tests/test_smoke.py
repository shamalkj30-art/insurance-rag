"""Smoke tests — no network, no API calls. Just structural sanity."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def test_imports():
    import rag      # noqa: F401
    import api      # noqa: F401
    import ingest   # noqa: F401
    import products  # noqa: F401


def test_refusal_string_in_prompt():
    from rag import REFUSAL, SYSTEM_PROMPT
    assert REFUSAL in SYSTEM_PROMPT


def test_product_mapping():
    from products import PRODUCTS, from_filename
    assert from_filename("Bil-Pluss-alminnelige-vilkar.pdf") == "Bil"
    assert from_filename("Innbo-Standard-alminnelige vilkar.pdf") == "Innbo"
    assert from_filename("Hytte-Pluss-alminnelige-vilkar.pdf") == "Hytte"
    assert from_filename("Reise-alminnelige-vilkar.pdf") == "Reise"
    assert from_filename("livsforsikring-alminnelige-vilkar.pdf") == "Person"
    assert from_filename("helse-55-alminnelige-vilkar.pdf") == "Helse"
    assert from_filename("ulykkesforsikring-alminnelige-vilkar.pdf") == "Person"
    # Every product mapping should land in the enum
    for name in ["Bil x.pdf", "Innbo x.pdf", "Hus x.pdf", "Hytte x.pdf",
                 "Reise x.pdf", "helse x.pdf", "ulykke x.pdf"]:
        assert from_filename(name) in PRODUCTS
