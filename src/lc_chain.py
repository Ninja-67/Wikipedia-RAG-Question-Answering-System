from typing import List, Dict, Optional
import re
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline
from langchain_openai import ChatOpenAI
from transformers import pipeline

from .lc_rerank import CrossEncoderReranker

DEFAULT_EMBED = "sentence-transformers/all-MiniLM-L6-v2"

def load_vectorstore(vs_dir: str, embed_model: str = DEFAULT_EMBED) -> FAISS:
    emb = HuggingFaceEmbeddings(model_name=embed_model)
    # keep allow_dangerous_deserialization on LOAD (not on save)
    return FAISS.load_local(vs_dir, emb, index_name="lc_faiss", allow_dangerous_deserialization=True)

def retrieve_and_rerank(vs: FAISS, query: str, rerank_top: int = 20, k: int = 8) -> List[Document]:
    # LC 0.2 retriever is Runnable -> use invoke()
    retriever = vs.as_retriever(search_kwargs={"k": rerank_top})
    pool = retriever.invoke(query)
    reranker = CrossEncoderReranker()
    reranked = reranker.compress_documents(pool, query)
    return reranked[:k]

PROMPT_TMPL = """Write only the answer sentence(s) using ONLY the provided context.
- If the answer is not in the context, reply exactly: "I don't know from the provided context."
- Include inline citations like [[n]] where n is the context index (1-based).
- Do not repeat the question. Do not include headings.

Context:
{context}

Question: {question}
Answer:
"""

def format_context(docs: List[Document], k: int) -> str:
    docs = docs[:k]
    lines = []
    for i, d in enumerate(docs, start=1):
        t = d.metadata.get("title", "Unknown")
        u = d.metadata.get("url", "")
        lines.append(f"[{i}] Title: {t}\nURL: {u}\nText: {d.page_content}")
    return "\n\n".join(lines)

def get_llm(provider: str = "openai", openai_model: str = "gpt-4o-mini",
            hf_model: str | None = None, max_tokens: int = 256, temperature: float = 0.2):
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=openai_model, temperature=temperature, max_tokens=max_tokens)

    mdl = hf_model or "HuggingFaceTB/SmolLM2-360M-Instruct"
    text_pipe = pipeline(
        "text-generation",
        model=mdl,
        device=-1,
        max_new_tokens=max_tokens,
        do_sample=temperature > 0.0,
        temperature=temperature,
        return_full_text=False,   
    )
    return HuggingFacePipeline(pipeline=text_pipe)


_CLEAN_RE = re.compile(r"(?:<im_start>.*?<im_end>\s*)|(?:\[INST\].*?\[/INST\]\s*)",
                       flags=re.DOTALL | re.IGNORECASE)

def _clean_answer(text: str) -> str:
    return _CLEAN_RE.sub("", text).strip()

def _postprocess_answer(text: str, have_contexts: bool) -> str:
    # Remove prompt echoes
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    lines = [ln for ln in lines if not re.match(r"^(context:|question:|answer\b)", ln, re.I)]
    # Dedupe while preserving order
    seen, out = set(), []
    for ln in lines:
        if ln not in seen:
            seen.add(ln); out.append(ln)
    ans = "\n".join(out).strip()
    # If no citation present and we have at least one context, add [[1]]
    if have_contexts and "[[" not in ans:
        ans = (ans + " [[1]]").strip()
    return ans
def ask(vs_dir: str, question: str, k: int = 8, rerank_top: int = 20,
        provider: str = "openai", openai_model: str = "gpt-4o-mini",
        hf_model: Optional[str] = None, max_tokens: int = 256, temperature: float = 0.2,
        embed_model: str = DEFAULT_EMBED) -> Dict:
    vs = load_vectorstore(vs_dir, embed_model=embed_model)
    docs = retrieve_and_rerank(vs, question, rerank_top=rerank_top, k=k)
    ctx = format_context(docs, k=k)

    prompt = PromptTemplate.from_template(PROMPT_TMPL)
    llm = get_llm(provider=provider, openai_model=openai_model, hf_model=hf_model,
                  max_tokens=max_tokens, temperature=temperature)
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"context": ctx, "question": question})
    answer = _postprocess_answer(_clean_answer(raw), have_contexts=bool(docs))

    contexts = [{
        "title": d.metadata.get("title"),
        "url": d.metadata.get("url"),
        "text": d.page_content,
        "score": d.metadata.get("rerank_score"),
    } for d in docs]
    return {"answer": answer, "contexts": contexts}
