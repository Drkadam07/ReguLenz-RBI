import streamlit as st
import httpx

API_BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="RBI Regulatory RAG Assistant", layout="wide")
st.title("🏛️ RBI Regulatory Compliance RAG Assistant")

with st.sidebar:
    st.header("Document Ingestion")
    uploaded_file = st.file_uploader("Upload RBI Circular (PDF)", type=["pdf"])
    if uploaded_file and st.button("Index Document"):
        with st.spinner("Parsing, embedding, and indexing..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            try:
                res = httpx.post(f"{API_BASE_URL}/ingest", files=files, timeout=120.0)
                if res.status_code == 200:
                    st.success(res.json()["message"])
                else:
                    st.error(f"Error: {res.text}")
            except Exception as e:
                st.error(f"Failed to connect to API: {e}")

    st.markdown("---")
    if st.button("Run Retrieval Benchmark"):
        with st.spinner("Computing Recall@K and MRR..."):
            try:
                res = httpx.get(f"{API_BASE_URL}/evaluate", timeout=60.0)
                if res.status_code == 200:
                    st.json(res.json()["metrics"])
                else:
                    st.error(res.text)
            except Exception as e:
                st.error(str(e))

query = st.text_input("Ask a regulatory compliance question:", placeholder="e.g., What are the KYC updation rules for dormant accounts?")
top_k = st.slider("Citations / Chunks to use:", min_value=1, max_value=5, value=3)

if st.button("Submit Query", type="primary") and query:
    with st.spinner("Executing hybrid retrieval and local generation..."):
        try:
            response = httpx.post(
                f"{API_BASE_URL}/query", 
                json={"question": query, "top_k": top_k}, 
                timeout=60.0
            )
            if response.status_code == 200:
                data = response.json()
                st.subheader("Grounded Answer")
                st.markdown(data["answer"])

                st.subheader("Verified Citations & Sources")
                for s in data["sources"]:
                    with st.expander(f"📄 {s['document_id']} | Page: {s['page']} (Score: {s['rerank_score']:.3f})"):
                        st.caption(f"Chunk ID: `{s['chunk_id']}`")
                        st.write(s["snippet"])
            else:
                st.error(f"Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Backend API unreachable: {e}")