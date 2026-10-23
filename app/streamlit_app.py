import streamlit as st
import httpx
import time

API_BASE_URL = "http://localhost:8000"
# Configure 5-minute timeout for slow local CPU inference
HTTP_TIMEOUT = httpx.Timeout(300.0, connect=10.0)

st.set_page_config(
    page_title="ReguLenz RBI - Compliance Assistant", page_icon="🏛️", layout="wide"
)

# ----------------- SIDEBAR: INGESTION & CONTROLS -----------------
with st.sidebar:
    st.title("⚙️ Control Panel")

    # 1. Health Status Card
    st.subheader("System Status")
    try:
        health_res = httpx.get(f"{API_BASE_URL}/health", timeout=3.0)
        if health_res.status_code == 200:
            hdata = health_res.json()
            if hdata["index_loaded"]:
                st.success(f"● Index Online ({hdata['total_chunks_indexed']} chunks)")
            else:
                st.warning("▲ Index Not Built (Ingestion Required)")
        else:
            st.error("● Backend error")
    except Exception:
        st.error("● Backend Offline (Check port 8000)")

    st.markdown("---")

    # 2. Bulk Indexing of data/raw/
    st.subheader("Bulk Indexing (`data/raw/`)")
    st.caption(
        "Parses all existing PDFs in `data/raw/` without manually uploading each file."
    )

    if st.button("⚡ Build / Rebuild Full Index", use_container_width=True):
        try:
            res = httpx.post(f"{API_BASE_URL}/build-index-all", timeout=HTTP_TIMEOUT)
            if res.status_code in (200, 202):
                with st.status(
                    "Executing bulk indexing in background...", expanded=True
                ) as status_box:
                    while True:
                        time.sleep(1.5)
                        poll = httpx.get(
                            f"{API_BASE_URL}/index-status", timeout=5.0
                        ).json()
                        status_box.write(f"Status: {poll['status_message']}")
                        if not poll["is_indexing"]:
                            if poll["index_ready"]:
                                status_box.update(
                                    label="Index built successfully!", state="complete"
                                )
                                st.rerun()
                            else:
                                status_box.update(
                                    label="Indexing failed.", state="error"
                                )
                            break
            else:
                st.error(res.json().get("detail", "Error starting job"))
        except Exception as e:
            st.error(f"Failed to communicate with API: {e}")

    st.markdown("---")

    # 3. Single File Upload
    st.subheader("Upload Single Circular")
    uploaded_file = st.file_uploader("Upload RBI Circular (.pdf)", type=["pdf"])
    if uploaded_file and st.button("Upload & Index PDF", use_container_width=True):
        with st.spinner("Parsing and embedding file..."):
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "application/pdf",
                )
            }
            try:
                res = httpx.post(
                    f"{API_BASE_URL}/ingest", files=files, timeout=HTTP_TIMEOUT
                )
                if res.status_code == 200:
                    st.success(res.json()["message"])
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"Error {res.status_code}: {res.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

    st.markdown("---")

    # 4. Evaluation Benchmark
    st.subheader("Benchmark Evaluation")
    if st.button("Run IR Evaluation Benchmark", use_container_width=True):
        with st.spinner("Calculating Recall@k and MRR across test dataset..."):
            try:
                res = httpx.get(f"{API_BASE_URL}/evaluate", timeout=HTTP_TIMEOUT)
                if res.status_code == 200:
                    metrics = res.json()["metrics"]
                    st.write("**Evaluation Metrics:**")
                    st.json(metrics)
                else:
                    st.error(res.text)
            except Exception as e:
                st.error(f"Evaluation error: {e}")

# ----------------- MAIN QUERY INTERFACE -----------------
st.title("🏛️ ReguLenz RBI: Regulatory Assistant")
st.markdown(
    "Ask precise compliance and regulatory questions with ground-truth source citations from official RBI Master Directions."
)

col_query, col_k = st.columns([4, 1])
with col_query:
    query = st.text_input(
        "Enter compliance query:",
        placeholder="e.g., What are the customer liability limits for unauthorized electronic banking transactions?",
    )
with col_k:
    top_k = st.number_input("Evidence Chunks", min_value=1, max_value=8, value=3)

if st.button("Search & Generate Grounded Answer", type="primary") and query:
    start_time = time.time()

    with st.spinner(
        "Retrieving relevant circular paragraphs and querying local LLM..."
    ):
        try:
            payload = {"question": query, "top_k": top_k}
            response = httpx.post(
                f"{API_BASE_URL}/query", json=payload, timeout=HTTP_TIMEOUT
            )

            elapsed = time.time() - start_time

            if response.status_code == 200:
                data = response.json()

                st.markdown(f"⏱️ *Generated in {elapsed:.2f}s*")

                # Answer Box
                st.subheader("Grounded Answer")
                st.info(data["answer"])

                # Source Citations Section
                st.subheader("Supporting Evidentiary Sources")
                if not data["sources"]:
                    st.warning("No explicit circular chunks supported this query.")
                else:
                    for i, s in enumerate(data["sources"], start=1):
                        with st.expander(
                            f"Citation [{i}] — {s['document_id']} (Page {s['page']}) | Rerank Score: {s['rerank_score']:.4f}"
                        ):
                            st.markdown(f"**Chunk ID:** `{s['chunk_id']}`")
                            st.write(s["snippet"])
            elif response.status_code == 503:
                st.error(
                    "Index not initialized. Use the sidebar to upload a document or click 'Build / Rebuild Full Index'."
                )
            else:
                st.error(
                    f"Server returned status {response.status_code}: {response.text}"
                )
        except httpx.ReadTimeout:
            st.error(
                "Request timed out. The local LLM is taking too long to generate on your hardware. Consider using a smaller model like `llama3.2:3b` or `qwen2.5:3b-instruct`."
            )
        except Exception as e:
            st.error(f"Failed to connect to API backend: {str(e)}")
