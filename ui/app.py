import streamlit as st
import requests
import os

st.set_page_config(page_title="Wikipedia RAG (LangChain)", page_icon="📚")

api_url = os.getenv("API_URL", "http://127.0.0.1:8000")

st.title("📚 Wikipedia RAG (LangChain)")
st.caption("Dense retrieval → cross-encoder rerank → grounded generation with citations")

q = st.text_input("Ask a question:")
k = st.slider("Top-K (final contexts)", 1, 20, 8)
rr = st.slider("Rerank pool (retrieve this many, then rerank)", 1, 200, 20)

if st.button("Ask") and q:
    with st.spinner("Retrieving + generating..."):
        try:
            resp = requests.post(f"{api_url}/ask", json={"question": q, "k": k, "rerank_top": rr})
            resp.raise_for_status()
            data = resp.json()
            st.markdown("### Answer")
            st.write(data["answer"])
            st.markdown("---")
            st.markdown("### Sources")
            for i, c in enumerate(data["contexts"], start=1):
                title = c.get("title","Untitled")
                url = c.get("url","")
                st.markdown(f"**[{i}] {title}**")
                if url:
                    st.write(url)
                st.write(c.get("text",""))
                st.write("---")
        except Exception as e:
            st.error(f"API error: {e}")
