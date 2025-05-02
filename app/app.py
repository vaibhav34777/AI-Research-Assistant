import os
import streamlit as st
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

@st.cache_resource(show_spinner=False)
def load_index_and_data():
    index = faiss.read_index("faiss_index.idx")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    with open("metadatas.pkl", "rb") as f:
        metadatas = pickle.load(f)
    return index, chunks, metadatas

index, chunks, metadatas = load_index_and_data()

# Auto-tag topics if not present
for md in metadatas:
    src = md.get("source", "").lower()
    if "ml" in src or "machine" in src:
        md["topic"] = "Machine Learning"
    elif "elec" in src or "circuit" in src:
        md["topic"] = "Electronics"
    else:
        md["topic"] = "Other"

@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

embedder = load_embedder()

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    st.error("HF_TOKEN environment variable not set!")
    st.stop()

client = InferenceClient(token=hf_token)

def retrieve_top_k(query: str, k: int = 5):
    q_emb = embedder.encode([query])
    q_emb = q_emb.astype("float32")
    _, indices = index.search(q_emb, k)
    results = []
    for idx in indices[0]:
        md = metadatas[idx]
        results.append({
            "text": chunks[idx],
            "source": md.get("source", ""),
            "page": md.get("page", None),
        })
    return results

def generate_via_hf(context: str, question: str) -> str:
    prompt = f"{context}\n\nQ: {question}\nA:"
    resp = client.text_generation(
        model="google/flan-t5-large",
        prompt=prompt,
        max_new_tokens=240,
        temperature=0.7
    )
    return resp.strip()

# ---------------- UI ------------------

st.set_page_config(page_title="AI Research Assistant", layout="wide")
st.title("📚 AI Research Assistant")

st.markdown(
    """
    Type a question based on the indexed ML and electronics research papers
    """
)

# Sidebar: Show available papers
with st.sidebar:
    st.subheader("Indexed Papers")
    
    grouped = {"Machine Learning": set(), "Electronics": set(), "Other": set()}
    for md in metadatas:
        src_full = md.get("source", "")
        filename = os.path.basename(src_full)

        # Simple keyword-based topic tagging
        if any(kw in filename.lower() for kw in ["ml", "machine", "gpt", "neural", "learning"]):
            topic = "Machine Learning"
        elif any(kw in filename.lower() for kw in ["circuit", "vlsi", "electronic", "semiconductor", "fpga"]):
            topic = "Electronics"
        else:
            topic = "Other"

        grouped[topic].add(filename)

    for topic in ["Machine Learning", "Electronics", "Other"]:
        papers = sorted(grouped[topic])
        if papers:
            st.markdown(f"**{topic}**")
            for paper in papers:
                st.markdown(f"- {paper}")


# Main input and output
question = st.text_input("Ask a question about the indexed papers:")

if st.button("Get Answer") and question:
    with st.spinner("Retrieving relevant passages…"):
        docs = retrieve_top_k(question, k=5)
    context = ""
    for doc in docs:
        context += f"[{doc['source']}, page {doc['page']}]\n{doc['text']}\n\n"
    with st.spinner("Generating answer…"):
        answer = generate_via_hf(context, question)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Cited Passages")
    for doc in docs:
        st.markdown(f"- **{doc['source']}** (page {doc['page']}):  \n  {doc['text'][:200]}…")

st.markdown("---")
st.caption("Built with FAISS · Sentence-Transformers · Hugging Face Inference · Streamlit")
