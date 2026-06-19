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
    from products import (
        FILE_TO_LABEL, LABEL_TO_FILE, PRODUCTS,
        categories, category_for, products_in,
    )
    # Every product is uniquely labelled
    labels = [p[0] for p in PRODUCTS]
    assert len(labels) == len(set(labels)), "duplicate product labels"

    # Round-trip both maps
    for label, filename, _cat in PRODUCTS:
        assert LABEL_TO_FILE[label] == filename
        assert FILE_TO_LABEL[filename] == label

    # Categories are well-formed
    cats = categories()
    assert "Bil" in cats and "Reise" in cats
    for cat in cats:
        items = products_in(cat)
        assert len(items) > 0
        for label, _ in items:
            assert category_for(label) == cat
