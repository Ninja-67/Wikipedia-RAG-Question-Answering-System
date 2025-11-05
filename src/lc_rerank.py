from typing import List
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
import numpy as np

class CrossEncoderReranker:
    """Simple cross-encoder reranker (no LangChain base class)."""
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = None):
        self.model = CrossEncoder(model_name, device=device)

    def compress_documents(self, documents: List[Document], query: str) -> List[Document]:
        if not documents:
            return []
        pairs = [[query, d.page_content] for d in documents]
        scores = self.model.predict(pairs)
        order = np.argsort(-scores)

        out: List[Document] = []
        for rank, idx in enumerate(order):
            d = documents[int(idx)]
            d.metadata = {**d.metadata, "rerank_score": float(scores[int(idx)])}
            out.append(d)
        return out
