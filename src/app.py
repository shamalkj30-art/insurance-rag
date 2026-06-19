"""
app.py — conversational Streamlit interface.

Flow:
  1. The user picks which insurance product they have (radio chips).
     This pins the retrieval filter for the whole conversation.
  2. The user asks questions in natural Norwegian.
  3. Each answer is grounded in the chosen product's vilkår, with citations.
     If the answer isn't in the vilkår, the agent says so honestly.

Run:
    streamlit run src/app.py
"""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from products import LABELS, PRODUCTS, Product
from rag import answer

load_dotenv()

st.set_page_config(
    page_title="Forsikringsassistent — Gjensidige vilkår",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Forsikringsassistent")
st.caption(
    "Spør om dine Gjensidige-vilkår på norsk. Svarene er hentet direkte fra "
    "vilkårsdokumentene — og hvis svaret ikke står der, sier assistenten det "
    "i stedet for å gjette."
)

# --- Sidebar -------------------------------------------------------------
with st.sidebar:
    st.markdown("### Hvordan det fungerer")
    st.markdown(
        "1. Velg hvilken forsikring spørsmålet gjelder.\n"
        "2. Still spørsmålet ditt på norsk.\n"
        "3. Assistenten henter kun fra vilkårene for **din** forsikring "
        "og siterer kildene.\n"
        "4. Hvis svaret ikke står i vilkårene, blir du henvist videre — "
        "ingen gjetting."
    )
    st.markdown("---")
    st.markdown("**Modell:** Claude Sonnet 4.6")
    st.markdown("**Embeddings:** intfloat/multilingual-e5-base (lokalt)")
    st.markdown("**Vektor-database:** FAISS")
    if st.button("🔄 Nullstill samtale"):
        st.session_state.pop("messages", None)
        st.session_state.pop("product", None)
        st.rerun()


# --- Product selection ---------------------------------------------------
if "product" not in st.session_state:
    st.subheader("👋 Hei! Hvilken forsikring gjelder spørsmålet?")
    cols = st.columns(len(PRODUCTS))
    for i, p in enumerate(PRODUCTS):
        if cols[i].button(LABELS[p], use_container_width=True):
            st.session_state["product"] = p
            st.session_state["messages"] = [
                {
                    "role": "assistant",
                    "content": (
                        f"Flott! Jeg hjelper deg med spørsmål om "
                        f"**{LABELS[p].split(maxsplit=1)[-1]}**-vilkårene. "
                        "Hva lurer du på?"
                    ),
                }
            ]
            st.rerun()
    st.stop()

# --- Chat ---------------------------------------------------------------
product: Product = st.session_state["product"]
st.markdown(f"**Forsikring:** {LABELS[product]}")

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Kilder"):
                for s in msg["sources"]:
                    st.markdown(f"- **{s['source']}**, side {s['page']}")

if prompt := st.chat_input("Skriv spørsmålet ditt..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Søker i vilkårene..."):
            result = answer(prompt, product=product)
        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("Kilder"):
                for s in result["sources"]:
                    st.markdown(f"- **{s['source']}**, side {s['page']}")

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        }
    )
