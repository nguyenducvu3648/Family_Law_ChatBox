from rank_bm25 import BM25Okapi
from tools.tools import tokenize
from models.models import client
from core.config import COLLECTION_NAME

def load_all_docs():
    docs = []
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            with_payload=True,
            offset=offset
        )
        for point in points:
            p = point.payload or {}
            docs.append({
                "citation": p.get("exact_citation", ""),
                "chapter_number": p.get("chapter_number", ""),
                "article_no": p.get("article_no", ""),
                "article_title": p.get("article_title", ""),
                "clause_no": p.get("clause_no", ""),
                "point_letter": p.get("point_letter", ""),
                "content": (p.get("content") or "").strip(),
            })
        offset = next_offset
        if offset is None:
            break
    return docs

all_docs = load_all_docs()
tokenized_corpus = [tokenize(d['content']) for d in all_docs]
bm25_global = BM25Okapi(tokenized_corpus)
