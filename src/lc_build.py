import argparse, os, json
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


def load_chunks(path: str) -> List[Document]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            meta = {k: obj.get(k) for k in ("id","title","url","lang","source","chunk_id")}
            docs.append(Document(page_content=obj["text"], metadata=meta))
    return docs

def build_faiss(chunks_jsonl: str, out_dir: str, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
    os.makedirs(out_dir, exist_ok=True)
    docs = load_chunks(chunks_jsonl)
    emb = HuggingFaceEmbeddings(model_name=embed_model)
    vs = FAISS.from_documents(docs, emb)
    vs.save_local(out_dir, index_name="lc_faiss")
    print(f"Saved LangChain FAISS to {out_dir} (docs={len(docs)})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--vs_dir", default="artifacts/lc_faiss")
    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()
    build_faiss(args.chunks, args.vs_dir, args.embed_model)

if __name__ == "__main__":
    main()
