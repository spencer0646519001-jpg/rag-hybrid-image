from __future__ import annotations
from pathlib import Path
import os, json, re
from typing import List, Dict, Optional, Tuple
import numpy as np
from rapidfuzz import fuzz

# ========= 路徑 =========
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DATA_PATH = DATA_DIR / "desserts.json"
EMB_NPY = DATA_DIR / "embeddings.npy"
EMB_META = DATA_DIR / "embeddings.meta.json"

# ========= 文字正規化 =========
import unicodedata

def normalize(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)  # 把全形/半形統一
    s = s.casefold()                      # 比 lower() 更多語言友好
    s = " ".join(s.split())               # 把多個空白縮成一個
    return s


# ========= 載入資料 =========
def load_docs() -> List[Dict]:
    if not DATA_PATH.exists():
        return []
    text = DATA_PATH.read_text(encoding="utf-8").strip()
    if not text:
        return []
    items = json.loads(text)
    if not isinstance(items, list):
        raise ValueError("desserts.json 應為 list[dict]")
    for d in items:
        title = d.get("title", "")
        snippet = d.get("snippet", "")
        tags = " ".join(d.get("tags", []))
        d["_hay"] = normalize(" ".join([title, snippet, tags]))
    return items

def topk_indices_desc(scores: np.ndarray, k: int, stable: bool = True) -> np.ndarray:
    """
    由大到小回傳 Top-K 的索引；stable=True 同分時保留原順序。
    """
    k = max(0, min(int(k), scores.shape[-1]))
    if k == 0:
        return np.empty((0,), dtype=int)
    kind = 'stable' if stable else 'quicksort'
    return np.argsort(-scores, kind=kind)[:k]


# ========= 關鍵字打分（模糊） =========
def _best_for_term(term: str, hay: str) -> float:
    term = normalize(term)
    if not term or not hay:
        return 0.0
    if term in hay:               # 精確包含給高分
        return 120.0
    return max(
        float(fuzz.partial_ratio(term, hay)),
        float(fuzz.token_set_ratio(term, hay)),
        float(fuzz.ratio(term, hay)),
    )

def score_doc_keywords(q: str, d: dict, agg: str = "max") -> float:
    # agg: "max" 或 "avg"
    terms = [x for x in re.split(r"[,，、；;\s]+", q.strip()) if x]
    if not terms:
        return 0.0
    hay = d["_hay"]
    scores = [_best_for_term(t, hay) for t in terms]
    return max(scores) if agg == "max" else (sum(scores) / len(scores))

# ========= 文字向量（Embedding） =========
# 兩種路徑：
# 1) 本地 SentenceTransformers（預設）
# 2) 設置 USE_OPENAI_EMBEDDINGS=1 時，改用 OpenAI Embeddings（需 OPENAI_API_KEY）
_USE_OPENAI = os.getenv("USE_OPENAI_EMBEDDINGS", "0") == "1"

_TEXT_MODEL = None
_OPENAI_CLIENT = None
_TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_OPENAI_EMB_MODEL = os.getenv("OPENAI_EMB_MODEL", "text-embedding-3-small")

def _ensure_text_model():
    global _TEXT_MODEL, _OPENAI_CLIENT
    if _USE_OPENAI:
        if _OPENAI_CLIENT is None:
            from openai import OpenAI
            _OPENAI_CLIENT = OpenAI()
        return
    if _TEXT_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _TEXT_MODEL = SentenceTransformer(_TEXT_MODEL_NAME)

def embed_texts(texts: List[str]) -> np.ndarray:
    _ensure_text_model()
    if _USE_OPENAI:
        # ---- 建議（可選）批次：官方支援一次多輸入 ----
        # resp = _OPENAI_CLIENT.embeddings.create(model=_OPENAI_EMB_MODEL, input=[t or "" for t in texts])
        # arr = np.array([d.embedding for d in resp.data], dtype=np.float32)
        # return arr

        # ---- 現況：逐筆（保留你的寫法，但維度自動化）----
        vecs = []
        dim = None
        for t in texts:
            t = t or ""
            emb = _OPENAI_CLIENT.embeddings.create(model=_OPENAI_EMB_MODEL, input=t)
            v = np.array(emb.data[0].embedding, dtype=np.float32)
            if dim is None:
                dim = v.shape[0]
            vecs.append(v)
        if vecs:
            return np.vstack(vecs)
        # 若沒有資料，依模型名稱給出合理的空陣列形狀
        fallback_dim = 3072 if "large" in _OPENAI_EMB_MODEL else 1536
        return np.zeros((0, fallback_dim), dtype=np.float32)
    else:
        arr = _TEXT_MODEL.encode(texts, convert_to_numpy=True, normalize_embeddings=True)  # L2-normalized
        return arr.astype(np.float32)


def ensure_embeddings() -> Tuple[np.ndarray, List[str]]:
    ids = [d["id"] for d in DOCS]
    want_dim = None
    # 用一筆文字試探目前 encoder 維度
    if DOCS:
        probe = embed_texts([normalize(DOCS[0].get("title","") + " " + DOCS[0].get("snippet",""))])
        want_dim = int(probe.shape[1])

    if EMB_NPY.exists() and EMB_META.exists():
        meta = json.loads(EMB_META.read_text(encoding="utf-8"))
        if meta.get("ids") == ids:
            arr = np.load(EMB_NPY)
            if (want_dim is None) or (arr.shape[1] == want_dim):
                return arr, ids
            # 維度不符 → 重新建立
    # 重建
    texts = [normalize(d.get("title","") + " " + d.get("snippet","")) for d in DOCS]
    mat = embed_texts(texts)
    np.save(EMB_NPY, mat)
    EMB_META.write_text(json.dumps({"ids": ids}, ensure_ascii=False, indent=2), encoding="utf-8")
    return mat, ids

def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: (n, d)  b: (m, d) → (n, m)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    if not np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-2):
        a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
    if not np.allclose(np.linalg.norm(b, axis=1), 1.0, atol=1e-2):
        b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-9)
    return a @ b.T

def vector_search(q: str, top_k: int = 10) -> List[Tuple[int, float]]:
    mat, _ = ensure_embeddings()
    qv = embed_texts([normalize(q)])
    sims = cosine_sim_matrix(qv, mat)[0]
    idx = topk_indices_desc(sims, top_k, stable=True)
    return [(int(i), float(sims[i])) for i in idx]

def hybrid_hits(q: str, w_text: float = 0.5, w_vec: float = 0.5, top_k: int = 20) -> List[Tuple[int, float, float, float]]:
    kw_scores = np.array([score_doc_keywords(q, d) for d in DOCS], dtype=np.float32)
    kw_norm = (kw_scores / 120.0).clip(0, 1)
    mat, _ = ensure_embeddings()
    qv = embed_texts([normalize(q)])
    vec_scores = cosine_sim_matrix(qv, mat)[0]
    final = w_text * kw_norm + w_vec * vec_scores
    idx = topk_indices_desc(final, top_k, stable=True)
    return [(int(i), float(final[i]), float(kw_norm[i]), float(vec_scores[i])) for i in idx]

# ========= 封裝給 API 用（維持舊版介面） =========
def score_doc(q: str, d: dict) -> dict:
    # 同時保留 max/avg 欄位（和先前前端相容）
    s_max = score_doc_keywords(q, d)
    s_avg = s_max  # 這裡先用同一套（之後要改 AVG 再說）
    return {
        "id": d["id"],
        "title": d["title"],
        "snippet": d.get("snippet",""),
        "tags": d.get("tags", []),
        "lang": d.get("lang", "unk"),
        "score": float(s_max),
        "score_max": float(s_max),
        "score_avg": float(s_avg),
        "matched_on": "hybrid-ready",
    }
# ========= 讀入全域文件 =========
DOCS: List[Dict] = load_docs()

def refresh_docs() -> None:
    """當你更新了 data/desserts.json 後可呼叫，並自動刷新 embeddings"""
    global DOCS
    DOCS = load_docs()
    # 可選：強制清除舊 embeddings，下次 search 會自動重建
    if EMB_NPY.exists():
        EMB_NPY.unlink(missing_ok=True)
    if EMB_META.exists():
        EMB_META.unlink(missing_ok=True)

def search(
    q: str,
    mode: str = "hybrid",            # "vector" or "hybrid"
    w_text: float = 0.5,
    w_vec: float = 0.5,
    top_k: int = 10,
    stable: bool = True,
) -> List[Dict]:
    """
    統一搜尋介面：
    - mode="vector": 只用向量相似度
    - mode="hybrid": 文字模糊 + 向量相似度加權
    回傳每筆含文件欄位與得分細節，方便前端/CLI 使用。
    """
    if not DOCS:
        return []

    if mode not in {"vector", "hybrid"}:
        raise ValueError("mode 必須是 'vector' 或 'hybrid'")

    # 關鍵字分數：0~120 → 正規化到 0~1
    kw_scores = np.array([score_doc_keywords(q, d) for d in DOCS], dtype=np.float32)
    kw_norm = (kw_scores / 120.0).clip(0, 1)

    # 向量分數
    mat, ids = ensure_embeddings()               # (n_docs, dim), 與 ids 對齊 DOCS
    qv = embed_texts([normalize(q)])
    vec_scores = cosine_sim_matrix(qv, mat)[0]   # 0~1

    if mode == "vector":
        final = vec_scores
    else:
        final = (w_text * kw_norm) + (w_vec * vec_scores)

    # 排序
    idx = topk_indices_desc(final, top_k, stable=stable)

    # 組合回傳
    out: List[Dict] = []
    for i in idx:
        d = DOCS[i]
        out.append({
            "id": d["id"],
            "title": d.get("title", ""),
            "snippet": d.get("snippet", ""),
            "tags": d.get("tags", []),
            "lang": d.get("lang", "unk"),
            "score_final": float(final[i]),
            "score_text": float(kw_norm[i]),
            "score_vector": float(vec_scores[i]),
        })
    return out

  


if __name__ == "__main__":
    q = input("Query: ").strip()
    mode = os.getenv("SEARCH_MODE", "hybrid")  # hybrid / vector
    hits = search(q, mode=mode, w_text=0.5, w_vec=0.5, top_k=5)
    for r in hits:
        print(f"{r['id']:>4} | {r['score_final']:.3f} | txt={r['score_text']:.3f} vec={r['score_vector']:.3f} | {r['title']}")
