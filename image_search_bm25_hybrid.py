# ==============================
# image_search_bm25_hybrid.py
# 自含式 Hybrid 排序（BM25 × Cosine）模組
# 使用方式：
# 1) pip install rank-bm25 scikit-learn (若僅用 BM25：只需 rank-bm25)
# 2) 在你的主程式中：
#    from image_search_bm25_hybrid import HybridRanker
#    ranker = HybridRanker(embeddings=EMB, metadata=META, text_embed_fn=text_embed,
#                          kw_mode="bm25", w_vec=0.7, w_text=0.3)
#    results = ranker.rank("檸檬塔", topk=20)
# 3) 若無法安裝外掛，kw_mode 改用 "hits"（純關鍵字命中），或 "tfidf"（需要 scikit-learn）
# ==============================

from __future__ import annotations
import os, json, math
from typing import List, Dict, Callable, Optional
import numpy as np

# 可選外掛：優先用 BM25；其次 TF-IDF；否則退化為 hits
try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except Exception:
    _HAS_BM25 = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    _HAS_TFIDF = True
except Exception:
    _HAS_TFIDF = False


def _safe_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x) + 1e-9)


def _cosine_sim(q: np.ndarray, M: np.ndarray) -> np.ndarray:
    qn = q / _safe_norm(q)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    return Mn @ qn  # [-1, 1]


class HybridRanker:
    """
    Hybrid 排序器：score = w_vec * cosine01 + w_text * keyword
    - embeddings: (N, D) 之向量矩陣
    - metadata: List[dict]，每個元素至少有 {"name", "pretty_name", "tags"}
    - text_embed_fn: callable(str) -> np.ndarray(D,)
    - kw_mode: "bm25" | "tfidf" | "hits"
    """
    def __init__(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict],
        text_embed_fn: Callable[[str], np.ndarray],
        kw_mode: str = "bm25",
        w_vec: float = 0.7,
        w_text: float = 0.3,
    ) -> None:
        assert isinstance(embeddings, np.ndarray) and embeddings.ndim == 2, "embeddings 應為 (N,D) 矩陣"
        assert isinstance(metadata, list) and len(metadata) == embeddings.shape[0], "metadata 與 embeddings 筆數需一致"
        self.EMB = embeddings.astype(np.float32, copy=False)
        self.META = metadata
        self.text_embed = text_embed_fn
        self.w_vec = float(w_vec)
        self.w_text = float(w_text)
        self.kw_mode = kw_mode.lower()

        # 建索引（供關鍵字分數用）
        self._build_keyword_index()

    # ------------------ 基礎文本處理 ------------------
    @staticmethod
    def _doc_text(item: Dict) -> str:
        name = (item.get("name") or "")
        pretty = (item.get("pretty_name") or "")
        tags = item.get("tags") or []
        return f"{name} {pretty} {' '.join(tags)}".strip().lower()

    @staticmethod
    def _tokenize_zh_en_ja(s: str) -> List[str]:
        s = (s or "").lower().strip()
        # 空白切詞
        parts = [p for p in s.replace("　", " ").split(" ") if p]
        # 逐字 fallback（對中日文）
        chars = list(s.replace(" ", ""))
        # 去重保序
        seen = set()
        toks = []
        for t in parts + chars:
            if t and (t not in seen):
                seen.add(t)
                toks.append(t)
        return toks

    def _build_keyword_index(self) -> None:
        self._corpus_texts: List[str] = [self._doc_text(it) for it in self.META]
        if self.kw_mode == "bm25" and _HAS_BM25:
            tokenized = [self._tokenize_zh_en_ja(t) for t in self._corpus_texts]
            self._bm25 = BM25Okapi(tokenized)
            self._tfidf = None
        elif self.kw_mode == "tfidf" and _HAS_TFIDF:
            # 自訂 tokenizer：直接注入我們的切詞器以支援中日文
            self._tfidf = TfidfVectorizer(tokenizer=self._tokenize_zh_en_ja, preprocessor=lambda x: x)
            self._tfidf_mat = self._tfidf.fit_transform(self._corpus_texts)  # (N, V)
            self._bm25 = None
        else:
            # 純 hits 模式，不建索引
            self._bm25 = None
            self._tfidf = None

    # ------------------ 關鍵字分數 ------------------
    def _kw_scores(self, query: str) -> np.ndarray:
        qtokens = self._tokenize_zh_en_ja(query)
        N = len(self.META)
        if self.kw_mode == "bm25" and getattr(self, "_bm25", None) is not None:
            arr = np.array(self._bm25.get_scores(qtokens), dtype=np.float32)
            m = float(arr.max())
            return arr / (m + 1e-9) if m > 0 else np.zeros(N, dtype=np.float32)
        elif self.kw_mode == "tfidf" and getattr(self, "_tfidf", None) is not None:
            q = " ".join(qtokens)
            qvec = self._tfidf.transform([q])  # (1, V)
            # 將查詢與文件矩陣做點積，得到對每一文件的 tf-idf 相似度
            sims = (self._tfidf_mat @ qvec.T).toarray().reshape(-1)
            m = float(sims.max())
            return sims.astype(np.float32) / (m + 1e-9) if m > 0 else np.zeros(N, dtype=np.float32)
        else:
            # hits：命中比例
            scores = []
            for text in self._corpus_texts:
                hits = sum(1 for t in qtokens if t in text)
                denom = max(1, len(set(qtokens)))
                scores.append(hits / denom)
            return np.array(scores, dtype=np.float32)

    # ------------------ 主排序 ------------------
    def rank(self, query: str, topk: int = 20, vec_temp: float = 1.0) -> List[Dict]:
        # 向量分數（cosine → [0,1]）
        qvec = self.text_embed(query)  # (D,)
        cos = _cosine_sim(qvec, self.EMB)  # [-1,1]
        cos01 = (cos + 1.0) / 2.0
        if vec_temp != 1.0:
            cos01 = np.power(np.clip(cos01, 0.0, 1.0), vec_temp)

        # 關鍵字分數 [0,1]
        kw = self._kw_scores(query)

        # 混合
        score = self.w_vec * cos01 + self.w_text * kw
        order = np.argsort(-score)[:topk]

        results = []
        for i, idx in enumerate(order, start=1):
            item = self.META[idx]
            results.append({
                "rank": i,
                "name": item.get("name"),
                "pretty_name": item.get("pretty_name"),
                "score_vec": float(cos01[idx]),
                "score_kw": float(kw[idx]),
                "score_hybrid": float(score[idx]),
            })
        return results


# ==============================
# experiments/hybrid/hybrid_eval_bm25.py
# 權重掃描 + 指標計算（支援 kw_mode & 溫度）
# 執行：
# python experiments/hybrid/hybrid_eval_bm25.py --eval experiments/hybrid/eval_set.json \
#    --kw bm25 --sweep --vec-temp 0.9 --out experiments/hybrid/hybrid_results.csv
# ==============================

if __name__ == "__main__" and False:
    # 此區塊僅避免在匯入時執行；真正檔案請存為 experiments/hybrid/hybrid_eval_bm25.py
    pass

# 另存以下為 experiments/hybrid/hybrid_eval_bm25.py
