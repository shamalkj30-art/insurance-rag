"""
app.py — conversational Streamlit chat for asking about your vilkår.

Two-step product selection (category → specific product) so the agent
always knows exactly which PDF to read from. Then a chat where the agent
talks like a knowledgeable colleague reading the policy alongside you.

Run:
    streamlit run src/app.py
"""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from products import (
    CATEGORY_EMOJI, categories, category_for, products_in,
)
from rag import answer

load_dotenv()

st.set_page_config(
    page_title="Forsikringsassistent — Gjensidige vilkår",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Forsikringsassistent")
st.caption(
    "Spør om Gjensidige-vilkårene dine på norsk. Svarene kommer rett fra "
    "vilkårsdokumentet for forsikringen du har — og hvis svaret ikke står "
    "der, sier assistenten det i stedet for å gjette."
)

# --- Sidebar ----------------------------------------------------------
with st.sidebar:
    st.markdown("### Hvordan det fungerer")
    st.markdown(
        "1. Velg kategori (Bil, Innbo, Reise...)\n"
        "2. Velg den eksakte forsikringen din (Bil Pluss, Reise Student...)\n"
        "3. Still spørsmålet ditt\n\n"
        "Assistenten leser KUN vilkårsdokumentet for det produktet du valgte. "
        "Hvis svaret ikke står der, blir du henvist videre — ingen gjetting."
    )
    st.markdown("---")
    st.markdown("**Modell:** Claude Sonnet 4.6")
    st.markdown("**Embeddings:** intfloat/multilingual-e5-base (lokal)")
    st.markdown("**Vektorindeks:** FAISS")
    if st.button("🔄 Start på nytt"):
        for k in ("messages", "category", "product"):
            st.session_state.pop(k, None)
        st.rerun()


# --- Step 1: category --------------------------------------------------
if "category" not in st.session_state:
    st.subheader("Hva slags forsikring gjelder spørsmålet?")
    cats = categories()
    cols = st.columns(len(cats))
    for i, cat in enumerate(cats):
        if cols[i].button(f"{CATEGORY_EMOJI[cat]} {cat}", use_container_width=True):
            st.session_state["category"] = cat
            st.rerun()
    st.stop()


# --- Step 2: specific product within category --------------------------
if "product" not in st.session_state:
    cat = st.session_state["category"]
    st.subheader(f"Hvilken {cat.lower()}-forsikring har du?")
    options = products_in(cat)
    cols = st.columns(min(len(options), 4))
    for i, (label, _filename) in enumerate(options):
        if cols[i % len(cols)].button(label, use_container_width=True):
            st.session_state["product"] = label
            st.session_state["messages"] = [
                {
                    "role": "assistant",
                    "content": (
                        f"Klart — jeg har vilkårene for **{label}** foran meg. "
                        "Hva lurer du på?"
                    ),
                }
            ]
            st.rerun()
    if st.button("← Tilbake"):
        st.session_state.pop("category", None)
        st.rerun()
    st.stop()


# --- Step 3: chat ------------------------------------------------------
product = st.session_state["product"]
cat = category_for(product)
header_col, change_col = st.columns([4, 1])
with header_col:
    st.markdown(f"**Forsikring:** {CATEGORY_EMOJI.get(cat, '')} {product}")
with change_col:
    if st.button("Bytt forsikring"):
        for k in ("messages", "category", "product"):
            st.session_state.pop(k, None)
        st.rerun()

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Kilder fra vilkårene"):
                for s in msg["sources"]:
                    st.markdown(f"- **{s['source']}**, side {s['page']}")

if prompt := st.chat_input("Skriv spørsmålet ditt..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Leser vilkårene..."):
            result = answer(prompt, product_label=product)
        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("Kilder fra vilkårene"):
                for s in result["sources"]:
                    st.markdown(f"- **{s['source']}**, side {s['page']}")

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        }
    )
