from rank_bm25 import BM25Okapi
import re

documents = []
bm25 = None


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def build_bm25(chunks):

    global bm25
    global documents

    documents = chunks

    tokenized = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized)


def keyword_search(query, top_k=7):

    global bm25
    global documents

    if bm25 is None:
        return []

    scores = bm25.get_scores(
        tokenize(query)
    )

    ranked = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        x[0]
        for x in ranked[:top_k]
    ]