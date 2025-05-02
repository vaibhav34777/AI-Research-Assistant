import streamlit as st
import faiss, pickle
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

@st.cache_resource(show_spinner=False)
def load_index_and_data():
    index = faiss.read_index("faiss_index.idx")
    with open("chunks.pkl",    "rb") as f: chunks    = pickle.load(f)
    with open("metadatas.pkl", "rb") as f: metadatas = pickle.load(f)
    return index, chunks, metadatas

index, chunks, metadatas = load_index_and_data()

# Load local embedder (CPU)
@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

embedder = load_embedder()

#Initialize HF Inference client
client = InferenceClient(token="YOUR_HF_TOKEN")

#Retrieval + Generation functions
def retrieve_top_k(query: str, k: int = 5):
    # embed query
    q_emb = embedder.encode([query])
    q_emb = q_emb.astype("float32")
    # search FAISS
    D, I = index.search(q_emb, k)
    results = []
    for idx in I[0]:
        md = metadatas[idx]
        results.append({
            "text":   chunks[idx],
            "source": md.get("source", ""),
            "page":   md.get("page", None),
        })
    return results

def generate_via_hf(context: str, question: str) -> str:
    prompt = f"{context}\n\nQ: {question}\nA:"
    resp = client.text_generation(
        model="tiiuae/falcon-7b-instruct",
        inputs=prompt,
        parameters={"max_new_tokens": 256, "temperature": 0.7},
    )
    # strip off the prompt portion
    return resp.generated_text[len(prompt):].strip()

# ————————————————————————————————————————————————
# 5) Streamlit UI
# ————————————————————————————————————————————————
st.set_page_config(page_title="AI Research Assistant", layout="wide")
st.title("📚 AI-Powered Research Assistant")

st.markdown(
    """
    Upload your own PDF or pick from the pre-loaded corpus (20 papers).
    Then type a question and get a grounded, source-backed answer!
    """
)

question = st.text_input("Ask a question about the indexed papers:")

if st.button("Get Answer") and question:
    with st.spinner("Retrieving relevant passages…"):
        docs = retrieve_top_k(question, k=5)
    # build a single context string
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

# Footer
st.markdown("---")
st.caption("Built with FAISS · Sentence-Transformers · Falcon-7B-Instruct · Streamlit")
