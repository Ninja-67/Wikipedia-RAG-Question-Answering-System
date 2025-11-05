import argparse, json
from .lc_ingest import ingest_titles
from .lc_build import build_faiss

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    ping = sub.add_parser("ingest")
    ping.add_argument("--titles", required=True)
    ping.add_argument("--out", required=True)
    ping.add_argument("--lang", default="en")
    ping.add_argument("--chunk_size", type=int, default=1000)
    ping.add_argument("--chunk_overlap", type=int, default=200)

    pbuild = sub.add_parser("build")
    pbuild.add_argument("--chunks", required=True)
    pbuild.add_argument("--vs_dir", default="artifacts/lc_faiss")
    pbuild.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")

    pask = sub.add_parser("ask")
    pask.add_argument("question")
    pask.add_argument("--vs_dir", default="artifacts/lc_faiss")
    pask.add_argument("--k", type=int, default=8)
    pask.add_argument("--rerank_top", type=int, default=20)
    pask.add_argument("--provider", choices=["openai","hf"], default="openai")
    pask.add_argument("--openai_model", default="gpt-4o-mini")
    pask.add_argument("--hf_model", default=None)
    pask.add_argument("--max_tokens", type=int, default=256)
    pask.add_argument("--temperature", type=float, default=0.2)
    pask.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")

    args = ap.parse_args()
    if args.cmd == "ingest":
        with open(args.titles, "r", encoding="utf-8") as f:
            titles = [line.strip() for line in f if line.strip()]
        ingest_titles(titles, args.out, lang=args.lang, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    elif args.cmd == "build":
        build_faiss(args.chunks, args.vs_dir, args.embed_model)

    elif args.cmd == "ask":
        # Lazy import so ingest/build don't require retriever deps
        from .lc_chain import ask as lc_ask
        out = lc_ask(
            args.vs_dir, args.question, k=args.k, rerank_top=args.rerank_top,
            provider=args.provider, openai_model=args.openai_model, hf_model=args.hf_model,
            max_tokens=args.max_tokens, temperature=args.temperature, embed_model=args.embed_model
        )
        print(json.dumps(out, ensure_ascii=False, indent=2))

    else:
        ap.print_help()

if __name__ == "__main__":
    main()
