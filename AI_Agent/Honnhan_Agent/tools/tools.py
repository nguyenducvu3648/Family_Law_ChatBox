import re
import torch
from typing import List
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from models.models import rerank_model, rerank_tokenizer
from utils.utils import _safe_truncate

def rerank_with_baai(query, docs, top_k=15):
    if not docs:
        return docs

    pairs = [(query, d["content"]) for d in docs]
    inputs = rerank_tokenizer(
        [p[0] for p in pairs],
        [p[1] for p in pairs],
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    )

    with torch.no_grad():
        scores = rerank_model(**inputs).logits.view(-1).float()

    # Gắn lại score vào docs
    for d, s in zip(docs, scores):
        d["baai_score"] = float(s)

    reranked = sorted(docs, key=lambda x: x["baai_score"], reverse=True)
    return reranked[:top_k]

def tokenize(text):
    return re.findall(r'\w+', text.lower())

def _build_filter(query_text: str) -> Filter or None:
    """
    Build Qdrant filter từ query text.
    Sử dụng metadata.* path cho các trường nested.
    """
    conds: List[FieldCondition] = []
    
    # Tìm Điều
    m = re.search(r"(?i)\bđiều\s*(\d+)\b", query_text)
    if m:
        conds.append(FieldCondition(key="metadata.article_no", match=MatchValue(value=int(m.group(1)))))
    
    # Tìm Khoản
    m = re.search(r"(?i)\bkhoản\s*(\d+)\b", query_text)
    if m:
        conds.append(FieldCondition(key="metadata.clause_no", match=MatchValue(value=int(m.group(1)))))
    
    # Tìm Điểm
    m = re.search(r"(?i)\bđiểm\s*([a-z])\b", query_text)
    if m:
        conds.append(FieldCondition(key="metadata.point_letter", match=MatchValue(value=m.group(1).lower())))
    
    # Tìm Chương
    m = re.search(r"(?i)\bchương\s*(\d+)\b", query_text)
    if m:
        conds.append(FieldCondition(key="metadata.chapter_number", match=MatchValue(value=int(m.group(1)))))
    
    return Filter(must=conds) if conds else None