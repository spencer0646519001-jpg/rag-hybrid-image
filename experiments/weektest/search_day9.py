# search_day9.py
from __future__ import annotations
import re
from typing import List, Dict, Tuple
import numpy as np

# ---------- 基本清洗 ----------
def normalize_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def tokenize(s: str) -> List[str]:
    s = normalize_text(s)
    # 只保留 a-z 的簡易斷詞；要支援中日文可換別的 tokenizer
    return re.findall(r"[a-z]+", s)

# ---------- 準備一點示例資料（title / snippet / tags） ----------
DOCS: List[Dict] = [
    {"title":"Opera Cake",
     "snippet":"French coffee buttercream and chocolate glaze cake.",
     "tags":["coffee","almond","buttercream"]},
    {"title":"Tiramisu",
     "snippet":"Italian dessert with espresso and mascarpone.",
     "tags":["coffee","cocoa"]},
    {"title":"Matcha Mousse",
     "snippet":"Light Japanese dessert with matcha green tea.",
     "tags":["matcha","tea","mousse"]},
    {"title":"Mont Blanc",
     "snippet":"Chestnut cream dessert, often with meringue.",
     "tags":["chestnut","cream"]},
    {"title":"Chocolate Tart",
     "snippet":"Rich chocolate ganache with crisp tart shell.",
     "tags":["chocolate","ganache"]},
]

# ---------- 建 vocab（用所有文件的文字） ----------
def build_vocab(docs: List[Dict]) -> Tuple[List[str], Dict[str,int]]:
    bag = []
    for d in docs:
        bag += tokenize(d["title"])
        bag += tokenize(d["snippet"])
        bag += [normalize_text(t) for t in d.get("tags", [])]
    vocab = sorted(set(bag))
    token2id = {tok:i for i, tok in enumerate(vocab)}
    return vocab, token2id

VOCAB, TOK2ID = build_vocab(DOCS)

# ---------- 假 embedding：Bag-of-Words → L2 規一化 ----------
def vectorize(text: str, tok2id: Dict[str,int]) -> np.ndarray:
    vec = np.zeros(len(tok2id), dtype=float)
    for tok in tokenize(text):
        if tok in tok2id:
            vec[tok2id[tok]] += 1.0            # 簡單詞頻 TF
    # L2 normalization：只比方向
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm

def doc_to_text(d: Dict) -> str:
    return f'{d["title"]} {d["snippet"]} {" ".join(d.get("tags", []))}'

def build_doc_matrix(docs: List[Dict]) -> np.ndarray:
    mat = []
    for d in docs:
        mat.append(vectorize(doc_to_text(d), TOK2ID))
    return np.vstack(mat) if mat else np.zeros((0, len(TOK2ID)))

DOC_MATRIX = build_doc_matrix(DOCS)   # shape: (n_docs, |V|)

# ---------- 核心：search(query) ----------
def search(query: str, top_k: int = 3) -> List[Tuple[int, float]]:
    qv = vectorize(query, TOK2ID)            # (|V|,)
    # 餘弦相似度（因為都 L2 規一化，內積就是 cosθ）
    sims = DOC_MATRIX @ qv                   # (n_docs,)
    top_idx = np.argsort(-sims)[:top_k]      # 取分數高的前 k
    return [(int(i), float(sims[i])) for i in top_idx]

# ---------- Demo ----------
if __name__ == "__main__":
    q = "coffee cake"
    results = search(q, top_k=3)
    print(f'🔎 Query: "{q}"')
    for rank, (i, s) in enumerate(results, 1):
        d = DOCS[i]
        print(f"{rank}. {d['title']:15s}  score={s:.3f}  | {d['snippet']}")

if __name__ == "__main__":
    q = "coffee cake"
    results = search(q, top_k=3)
    print(f'🔎 Query: "{q}"')
    for rank, (i, s) in enumerate(results, 1):
        d = DOCS[i]
        print(f"{rank}. {d['title']:15s}  score={s:.3f}  | {d['snippet']}")
