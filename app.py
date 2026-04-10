"""
SlideForge — FastAPI Backend
Exact pipeline from document.ipynb:
  1. Load docs from Wikipedia + arXiv (LangChain loaders)
  2. Chunk with RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
  3. Embed with all-MiniLM-L6-v2 (SentenceTransformer)
  4. Store in ChromaDB (separate collections per source)
  5. RAG retrieval per slide heading
  6. Score + deduplicate sentences → bullets
  7. Return structured JSON to frontend
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uuid
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import WikipediaLoader, ArxivLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ─── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="SlideForge API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global Model (loaded once on startup) ────────────────────────────────────
print("Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.")

# ─── Request / Response Schemas ───────────────────────────────────────────────
class GenerateRequest(BaseModel):
    topic: str
    num_slides: int = 8
    max_bullets: int = 5
    source: str = "both"   # "wikipedia" | "arxiv" | "both"

class SlideOut(BaseModel):
    title: str
    bullets: List[str]

class GenerateResponse(BaseModel):
    topic: str
    topic_type: str
    slides: List[SlideOut]

# ─── 1. Document Loaders ──────────────────────────────────────────────────────
def load_wiki_docs(topic: str):
    clean = topic.replace(".", "").replace("-", " ")
    loader = WikipediaLoader(query=clean, load_max_docs=3)
    docs = loader.load()
    for d in docs:
        d.metadata["source"] = "Wikipedia"
    print(f"Loaded {len(docs)} Wikipedia docs")
    return docs

def load_arxiv_docs(topic: str):
    try:
        loader = ArxivLoader(query=topic, load_max_docs=3)
        docs = loader.load()
        if not docs:
            print("No arXiv docs found")
            return []
        for d in docs:
            d.metadata["source"] = "arXiv"
        print(f"Loaded {len(docs)} arXiv docs")
        return docs
    except Exception as e:
        print(f"arXiv FAILED: {e}")
        return []

# ─── 2. Chunker ───────────────────────────────────────────────────────────────
def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} docs → {len(chunks)} chunks")
    return chunks

# ─── 3. Embedding Manager ─────────────────────────────────────────────────────
def generate_embeddings(texts: list) -> np.ndarray:
    print(f"Generating embeddings for {len(texts)} texts")
    embeddings = EMBED_MODEL.encode(texts, show_progress_bar=False)
    print(f"Embedding shape: {embeddings.shape}")
    return embeddings

# ─── 4. VectorStore ───────────────────────────────────────────────────────────
class VectorStore:
    def __init__(self, collection_name: str, chroma_path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=chroma_path)
        # Always start fresh per request
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass
        self.collection = self.client.create_collection(collection_name)
        print(f"VectorStore ready: {collection_name}")

    def add_documents(self, documents, embeddings: np.ndarray):
        if not documents:
            return
        ids, docs_text, metadatas, embed_list = [], [], [], []
        for doc, embedding in zip(documents, embeddings):
            ids.append(str(uuid.uuid4()))
            docs_text.append(doc.page_content)
            metadatas.append({k: str(v) for k, v in doc.metadata.items()})
            embed_list.append(embedding.tolist())
        self.collection.add(
            ids=ids,
            embeddings=embed_list,
            documents=docs_text,
            metadatas=metadatas
        )
        print(f"Added {len(documents)} docs to collection")

    def query(self, query_embedding: np.ndarray, k: int) -> list:
        if self.collection.count() == 0:
            return []
        k = min(k, self.collection.count())
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k
        )
        docs = []
        if results["documents"] and results["documents"][0]:
            for doc_id, document, metadata, distance in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                similarity_score = 1 / (1 + distance)
                docs.append({
                    "id": doc_id,
                    "content": document,
                    "metadata": metadata,
                    "similarity_score": similarity_score,
                    "weighted_score": similarity_score * 0.5
                })
        return docs

# ─── 5. RAG Retriever ─────────────────────────────────────────────────────────
class RAGRetriever:
    def __init__(self, arxiv_store: VectorStore, wiki_store: VectorStore):
        self.arxiv_store = arxiv_store
        self.wiki_store = wiki_store

    def retrieve(self, query: str, top_k: int = 20, skip_arxiv: bool = False) -> list:
        print(f"Retrieving for: '{query}'")
        k_each = top_k // 2
        query_embedding = generate_embeddings([query])[0]

        arxiv_docs = [] if skip_arxiv else self.arxiv_store.query(query_embedding, k_each)
        wiki_docs = self.wiki_store.query(query_embedding, k_each)

        combined = arxiv_docs + wiki_docs
        combined = sorted(combined, key=lambda x: x["weighted_score"], reverse=True)
        for i, d in enumerate(combined):
            d["rank"] = i + 1
        print(f"  → {len(arxiv_docs)} arXiv + {len(wiki_docs)} Wiki docs")
        return combined

# ─── 6. Topic Detection ───────────────────────────────────────────────────────
def detect_topic_type(rag_text: str) -> str:
    academic_signals = ["neural", "algorithm", "equation", "research", "paper",
                        "proposed", "model", "training", "dataset", "published"]
    pop_signals = ["season", "episode", "character", "aired", "cast", "series",
                   "album", "band", "film", "award", "director", "starring"]
    text_lower = rag_text.lower()
    academic_score = sum(1 for w in academic_signals if w in text_lower)
    pop_score = sum(1 for w in pop_signals if w in text_lower)
    return "academic" if academic_score >= pop_score else "pop_culture"

# ─── 7. Headings & Keywords ───────────────────────────────────────────────────
def generate_headings(topic: str, num_slides: int, topic_type: str) -> list:
    if topic_type == "academic":
        base = [topic, "Introduction", "History & Background", "How It Works",
                "Types & Categories", "Key Components", "Applications",
                "Advantages & Benefits", "Challenges & Limitations",
                "Future Scope", "Conclusion", "References"]
    else:
        base = [topic, "Introduction", "History & Background", "Key People",
                "Plot & Themes", "Reception", "Legacy & Impact",
                "Notable Episodes", "Cultural Impact",
                "Fun Facts", "Conclusion", "References"]
    return base[:num_slides]

ACADEMIC_KEYWORDS = {
    "Introduction":             ["overview", "definition", "what is", "refers to", "known as", "is a"],
    "History & Background":     ["history", "origin", "evolution", "introduced", "developed", "founded", "first"],
    "How It Works":             ["works", "process", "mechanism", "computes", "forward", "backpropagation", "weight", "activation"],
    "Types & Categories":       ["types", "kinds", "categories", "feedforward", "recurrent", "convolutional", "LSTM", "RNN", "CNN"],
    "Key Components":           ["neuron", "layer", "hidden", "input", "output", "bias", "weight", "node"],
    "Applications":             ["application", "used for", "applied", "image", "speech", "classification", "prediction", "recognition"],
    "Advantages & Benefits":    ["advantage", "benefit", "accurate", "faster", "efficient", "outperform", "better"],
    "Challenges & Limitations": ["challenge", "limitation", "problem", "drawback", "vanishing", "exploding", "difficult", "slow"],
    "Future Scope":             ["future", "research", "upcoming", "trend", "potential", "direction", "next"],
    "Conclusion":               ["summary", "conclude", "overall", "in summary", "this paper", "demonstrate"],
    "References":               []
}

POP_KEYWORDS = {
    "Introduction":         ["is a", "known as", "refers to", "american", "sitcom", "show", "television", "series", "film"],
    "History & Background": ["created", "premiered", "first aired", "developed", "founded", "began", "started", "origin"],
    "Key People":           ["cast", "starring", "actor", "actress", "character", "played by", "creator", "writer", "director"],
    "Plot & Themes":        ["story", "plot", "theme", "episode", "season", "follows", "centers", "about"],
    "Reception":            ["award", "emmy", "rating", "critics", "audience", "popular", "acclaim", "won", "nominated"],
    "Legacy & Impact":      ["influence", "legacy", "iconic", "cultural", "impact", "inspired", "remembered"],
    "Notable Episodes":     ["episode", "season", "memorable", "famous", "notable", "best", "iconic scene"],
    "Cultural Impact":      ["culture", "reference", "meme", "phrase", "catchphrase", "phenomenon", "generation"],
    "Fun Facts":            ["fact", "trivia", "behind the scenes", "originally", "almost", "nearly", "first choice"],
    "Conclusion":           ["summary", "overall", "one of", "considered", "remains", "still"],
    "References":           []
}

# ─── 8. Slide Builder ─────────────────────────────────────────────────────────
def build_slides(topic, num_slides, max_bullets, retriever, topic_type, arxiv_chunks):
    keywords = ACADEMIC_KEYWORDS if topic_type == "academic" else POP_KEYWORDS
    headings = generate_headings(topic, num_slides, topic_type)
    topic_words = set(topic.lower().split())
    final_slides = []
    used_sentences = set()

    for i, heading in enumerate(headings):
        if i == 0:
            final_slides.append({"title": topic, "bullets": []})
            continue

        # Per-slide RAG retrieval
        slide_docs = retriever.retrieve(
            f"{topic} {heading}",
            top_k=20,
            skip_arxiv=(len(arxiv_chunks) == 0)
        )

        slide_sentences = []
        for d in slide_docs:
            # Use Wikipedia for pop culture, both for academic
            if topic_type == "pop_culture" and d["metadata"].get("source") == "arXiv":
                continue
            for sent in d["content"].replace(". ", ".|").split("|"):
                sent = sent.strip()
                if len(sent) > 20:
                    slide_sentences.append(sent)

        # Score sentences
        scored = []
        for sent in slide_sentences:
            topic_score = sum(1 for w in topic_words if w in sent.lower())
            key_list = keywords.get(heading, [])
            keyword_score = sum(1 for kw in key_list if kw.lower() in sent.lower())
            total_score = topic_score + (keyword_score * 3)
            if total_score > 0:
                scored.append((total_score, sent))

        scored = sorted(scored, key=lambda x: x[0], reverse=True)

        # Deduplicate globally
        bullets = []
        for _, sent in scored:
            norm = sent.strip().lower()
            if norm not in used_sentences:
                used_sentences.add(norm)
                bullets.append(sent.strip())
            if len(bullets) == max_bullets:
                break

        # Fallback if no scored bullets
        if not bullets:
            bullets = [s for s in slide_sentences if len(s) > 20][:max_bullets]

        final_slides.append({"title": heading, "bullets": bullets})

    return final_slides

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "SlideForge API is running"}

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    topic = req.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Topic cannot be empty")

    source = req.source  # "wikipedia" | "arxiv" | "both"

    # ── Step 1: Load documents ──────────────────────────────────────────
    wiki_docs, arxiv_docs = [], []
    if source in ("wikipedia", "both"):
        wiki_docs = load_wiki_docs(topic)
    if source in ("arxiv", "both"):
        arxiv_docs = load_arxiv_docs(topic)

    all_docs = arxiv_docs + wiki_docs
    if not all_docs:
        raise HTTPException(status_code=404, detail="No documents found for this topic")

    # ── Step 2: Chunk ───────────────────────────────────────────────────
    chunks = split_documents(all_docs)
    arxiv_chunks = [c for c in chunks if c.metadata.get("source") == "arXiv"]
    wiki_chunks  = [c for c in chunks if c.metadata.get("source") == "Wikipedia"]

    # ── Step 3 & 4: Embed + Store ────────────────────────────────────────
    run_id = str(uuid.uuid4())[:8]  # unique per request to avoid collisions
    arxiv_store = VectorStore(f"arxiv_{run_id}")
    wiki_store  = VectorStore(f"wiki_{run_id}")

    if arxiv_chunks:
        arxiv_emb = generate_embeddings([c.page_content for c in arxiv_chunks])
        arxiv_store.add_documents(arxiv_chunks, arxiv_emb)

    if wiki_chunks:
        wiki_emb = generate_embeddings([c.page_content for c in wiki_chunks])
        wiki_store.add_documents(wiki_chunks, wiki_emb)

    # ── Step 5: RAG Retriever ────────────────────────────────────────────
    retriever = RAGRetriever(arxiv_store, wiki_store)

    # ── Step 6: Detect topic type ────────────────────────────────────────
    rag_text = " ".join(c.page_content for c in chunks[:30])
    topic_type = detect_topic_type(rag_text)
    print(f"Detected topic type: {topic_type}")

    # ── Step 7: Build slides ─────────────────────────────────────────────
    slides = build_slides(
        topic=topic,
        num_slides=req.num_slides,
        max_bullets=req.max_bullets,
        retriever=retriever,
        topic_type=topic_type,
        arxiv_chunks=arxiv_chunks
    )

    return GenerateResponse(
        topic=topic,
        topic_type=topic_type,
        slides=[SlideOut(**s) for s in slides]
    )

# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
