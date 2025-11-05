import argparse, json, tqdm
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def ingest_titles(titles, out_path: str, lang: str = "en", chunk_size: int = 1000, chunk_overlap: int = 200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    rows = []
    for t in tqdm.tqdm(titles, desc="Fetching & chunking"):
        loader = WikipediaLoader(query=t, load_max_docs=1, lang=lang, doc_content_chars_max=200000)
        docs = loader.load()
        for d in docs:
            for i, ch in enumerate(splitter.split_text(d.page_content)):
                rows.append({
                    "id": f"{d.metadata.get('title', t)}::chunk_{i}",
                    "title": d.metadata.get("title", t),
                    "chunk_id": i,
                    "text": ch,
                    "source": "wikipedia_api",
                    "lang": lang,
                    "url": d.metadata.get("source", ""),
                })
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--titles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--chunk_size", type=int, default=1000)
    ap.add_argument("--chunk_overlap", type=int, default=200)
    args = ap.parse_args()
    with open(args.titles, "r", encoding="utf-8") as f:
        titles = [line.strip() for line in f if line.strip()]
    ingest_titles(titles, args.out, lang=args.lang, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

if __name__ == "__main__":
    main()
