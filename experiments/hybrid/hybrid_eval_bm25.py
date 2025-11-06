# experiments/hybrid/hybrid_eval_bm25.py
from __future__ import annotations
import os, sys, json, math, argparse, re, inspect
from pathlib import Path
from typing import List, Dict, Callable, Optional, Any
import numpy as np
from rank_bm25 import BM25Okapi

# ====== 1) 專案根目錄自動定位 ======
ROOT = Path(__file__).resolve().parents[2]  # experiments/hybrid/ 上去兩層 = 專案根目錄
print(f"[DEBUG] ROOT: {ROOT}")

# 預設資料路徑
EMB_PATH  = ROOT / "data" / "img_embeddings.npy"
META_PATH = ROOT / "data" / "image_index.json"

print(f"[DEBUG] EMB path:  {EMB_PATH}")
print(f"[DEBUG] META path: {META_PATH}")

# ====== 2) 讀 embeddings ======
EMB = np.load(EMB_PATH)

# ====== 3) 讀 raw metadata，並「正規化」成 name/title/tags ======
with open(META_PATH, "r", encoding="utf-8") as f:
    RAW_META: List[Dict[str, Any]] = json.load(f)

def _stem_from_any(m: Dict[str, Any]) -> str:
    """從各種可能欄位名取得檔名（最後一段）。"""
    for k in ["name", "file", "filename", "path", "image", "img", "src"]:
        if k in m and m[k]:
            return Path(str(m[k])).name
    # 萬一真的都沒有，就給一個 id-based 名字
    if "id" in m and m["id"]:
        return f"img_{m['id']}"
    return "unknown.jpg"

def _title_from(m: Dict[str, Any], name: str) -> str:
    """優先用原本 title，沒有就從檔名推一個（底線/連字號轉空白）。"""
    t = m.get("title")
    if isinstance(t, str) and t.strip():
        return t.strip()
    stem = Path(name).stem
    stem = stem.replace("_", " ").replace("-", " ").strip()
    return stem

def _tags_from(m: Dict[str, Any], title: str) -> List[str]:
    """tags 可為 list 或逗號/空白分隔字串；若都沒有，從 title 拆字。"""
    tags = m.get("tags")
    if isinstance(tags, list):
        return [str(x).strip() for x in tags if str(x).strip()]
    if isinstance(tags, str) and tags.strip():
        # 允許用逗號或空白切
        parts = re.split(r"[,\s]+", tags.strip())
        return [p for p in parts if p]
    # 從 title 抽詞：英數字、漢字、假名
    words = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]|[\u3040-\u30ff]", title.lower())
    return [w for w in words if len(w) >= 2 or re.match(r"[\u4e00-\u9fff\u3040-\u30ff]", w)]

# 產生我們需要的結構：必定有 name/title/tags
META: List[Dict[str, Any]] = []
for m in RAW_META:
    name  = _stem_from_any(m)
    title = _title_from(m, name)
    tags  = _tags_from(m, title)
    META.append({"name": name, "title": title, "tags": tags, **m})

print(f"[DEBUG] META len : {len(META)}")
print(f"[DEBUG] ex name  : {META[0].get('name')}")
print(f"[DEBUG] ex title : {META[0].get('title')}")
print(f"[DEBUG] ex tags  : {META[0].get('tags')[:5]}")

# ====== 4) 文字嵌入函式載入 ======
# 將專案根目錄加入 sys.path，好 import image_search
sys.path.append(str(ROOT))
import importlib
IS = importlib.import_module("image_search")

_text_embed: Optional[Callable[[str], np.ndarray]] = None
for cand in ["text_embed", "embed_text", "encode_text", "text_to_vec", "query_to_vec"]:
    if hasattr(IS, cand):
        _text_embed = getattr(IS, cand)
        print(f"[INFO] 使用 image_search.{cand} 當作文字嵌入函式")
        break
if _text_embed is None:
    raise ImportError("找不到可用的文字嵌入函式（請確認 image_search.py 內有 text_embed 或類似函式）")

# ====== 5) 評測指標 ======
def precision_at_k(names: List[str], positives: List[str], k: int = 5) -> float:
    topk = names[:k]
    pos = set(positives)
    hits = sum(1 for n in topk if n in pos)
    return hits / max(1, k)

def mrr(names: List[str], positives: List[str]) -> float:
    pos = set(positives)
    for i, n in enumerate(names, 1):
        if n in pos:
            return 1.0 / i
    return 0.0

def ndcg_at_k(names: List[str], positives: List[str], k: int = 5) -> float:
    pos = set(positives)
    dcg = 0.0
    for i, n in enumerate(names[:k], 1):
        if n in pos:
            dcg += 1.0 / math.log2(i + 1)
    # IDCG：理想情況（把所有正解都排最前面）
    ideal_hits = min(k, len(pos))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return (dcg / idcg) if idcg > 0 else 0.0

# ====== 6) Hybrid Ranker（向量 + BM25） ======
def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if x.ndim == 1:
        x = x.reshape(1, -1)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms

def _cosine_sim(q: np.ndarray, m: np.ndarray) -> np.ndarray:
    qn = _l2_normalize_rows(q).reshape(1, -1)
    mn = _l2_normalize_rows(m)
    return (qn @ mn.T).ravel()

def _scale01(a: np.ndarray) -> np.ndarray:
    if a.size == 0:
        return a
    amin, amax = float(np.min(a)), float(np.max(a))
    if amax - amin < 1e-12:
        return np.zeros_like(a)
    return (a - amin) / (amax - amin)

class HybridRanker:
    def __init__(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
        text_embed_fn: Callable[[str], np.ndarray],
        kw_mode: str = "bm25",
        w_vec: float = 0.7,
        w_text: float = 0.3,
    ):
        assert embeddings.ndim == 2
        assert len(metadata) == embeddings.shape[0], "metadata 與 embeddings 筆數需一致"

        self.emb = embeddings.astype(np.float32, copy=False)
        self.meta = metadata
        self.text_embed = text_embed_fn
        self.kw_mode = kw_mode
        self.w_vec = w_vec
        self.w_text = w_text

        self._bm25: Optional[BM25Okapi] = None
        self._kw_empty: bool = True
        self._build_keyword_index()

    # ---- BM25 ----
    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        return re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]|[\u3040-\u30ff]", text.lower())

    def _build_keyword_index(self):
        docs: List[List[str]] = []
        for m in self.meta:
            parts: List[str] = []
            # 盡量湊到有內容：title + tags + name；都沒有再用 path 檔名
            t = m.get("title") or ""
            if t:
                parts.append(t)
            tg = m.get("tags") or []
            if isinstance(tg, list):
                parts.extend([str(x) for x in tg])
            elif isinstance(tg, str) and tg.strip():
                parts.extend(re.split(r"[,\s]+", tg.strip()))
            nm = m.get("name") or m.get("path") or ""
            if nm:
                parts.append(str(nm))
            if not parts and m.get("path"):
                parts.append(Path(m["path"]).stem)

            toks = self._tokenize(" ".join(parts))
            if not toks and m.get("path"):
                toks = self._tokenize(Path(m["path"]).stem)
            docs.append(toks)

        non_empty = sum(1 for d in docs if d) > 0
        uniq_vocab = len({tok for d in docs for tok in d})
        print(f"[DEBUG] keyword docs: {len(docs)}  uniq tokens: {uniq_vocab}  sample: {docs[0][:10] if docs else []}")

        if self.kw_mode == "bm25" and non_empty and uniq_vocab > 0:
            self._bm25 = BM25Okapi(docs)
            self._kw_empty = False
        else:
            self._bm25 = None
            self._kw_empty = True

    # ---- 向量分數 ----
    def _score_by_vector(self, query: str) -> np.ndarray:
        qvec = self.text_embed(query)
        if qvec.ndim == 2 and qvec.shape[0] == 1:
            qvec = qvec.reshape(-1)
        return _cosine_sim(qvec, self.emb)

    # ---- 排名 ----
    def rank(self, query: str, topk: int = 20) -> List[Dict[str, Any]]:
        vec_scores = self._score_by_vector(query)

        if self.kw_mode == "bm25" and not self._kw_empty and self._bm25 is not None:
            q_tokens = self._tokenize(query)
            kw_scores = np.array(self._bm25.get_scores(q_tokens), dtype=float)
        else:
            kw_scores = np.zeros(len(self.meta), dtype=float)

        vec_s = _scale01(vec_scores)
        kw_s = _scale01(kw_scores)
        scores = self.w_vec * vec_s + self.w_text * kw_s
        order = np.argsort(-scores)

        results: List[Dict[str, Any]] = []
        for idx in order[:topk]:
            m = self.meta[idx]
            nm = m.get("name") or m.get("file") or m.get("filename") or Path(str(m.get("path",""))).name or f"img_{idx}"
            results.append({
                "id": m.get("id"),
                "name": str(nm),
                "title": m.get("title"),
                "tags": m.get("tags"),
                "score": float(scores[idx]),
                "index": int(idx),
            })
        return results

# ====== 7) 載入評測集（支援多個可能檔名；找不到就給預設） ======
def _resolve_eval_path() -> Path:
    candidates = [
        ROOT / "data" / "eval_set.json",
        ROOT / "eval_set.json",
        ROOT / "data" / "eval_set",   # 沒副檔名的情況
        ROOT / "eval_set",            # 沒副檔名的情況
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]  # 預設放 data/eval_set.json

EVAL_PATH = _resolve_eval_path()

DEFAULT_EVAL = [
    {"query": "檸檬塔",                "positives": ["lemone_tarte.jpg", "lemone_tarte1.jpg"]},
    {"query": "抹茶捲",                "positives": ["matcha_cream_roll.jpg"]},
    {"query": "草莓蛋糕",              "positives": ["strawberry_shortcake.jpg"]},
    {"query": "可麗露,canel",         "positives": ["canele.jpg"]},
    {"query": "巧克力塔,chocolate tarte","positives": ["chocolate_tart.jpg","chocolate_tarte.jpg"]},
    {"query": "financier,費南雪",     "positives": ["financier.jpg"]},
    {"query": "cheese_cake,起司蛋糕",  "positives": ["cheese_cake.jpg"]},
]

if EVAL_PATH.exists():
    try:
        with open(EVAL_PATH, "r", encoding="utf-8") as f:
            EVAL = json.load(f)
    except Exception as e:
        print(f"[WARN] 讀取 {EVAL_PATH.name} 失敗：{e}，改用預設測資。")
        EVAL = DEFAULT_EVAL
else:
    print(f"[WARN] 找不到 {EVAL_PATH.name}，改用預設測資。")
    EVAL = DEFAULT_EVAL

# 檢查 positives 是否都存在於 metadata.name
meta_names = {m.get("name") for m in META}
missing: List[str] = []
for e in EVAL:
    for p in e["positives"]:
        if p not in meta_names:
            missing.append(f"query={e['query']}  missing={p}")
if missing:
    print("[WARN] 以下評測檔名在 metadata 找不到對應 name：")
    for line in missing:
        print("  -", line)

# ====== 8) CLI 參數 & 主程式 ======
def build_text_embed_with_temp(base_fn: Callable[[str], np.ndarray], vec_temp: float) -> Callable[[str], np.ndarray]:
    """如果 text_embed 支援 temperature/vec_temp 參數就帶入，否則直接呼叫。"""
    sig = inspect.signature(base_fn)
    param_names = list(sig.parameters.keys())
    def _wrap(q: str) -> np.ndarray:
        kwargs = {}
        if "temperature" in param_names:
            kwargs["temperature"] = vec_temp
        elif "vec_temp" in param_names:
            kwargs["vec_temp"] = vec_temp
        return base_fn(q, **kwargs) if kwargs else base_fn(q)
    return _wrap

def main():
    ap = argparse.ArgumentParser(description="使用向量+BM25 的影像檢索評測")
    ap.add_argument("--kw", choices=["bm25", "hits"], default="bm25", help="bm25=開啟關鍵字檢索; hits=只用向量")
    ap.add_argument("--vec-temp", type=float, default=0.9, help="傳給 text_embed 的 temperature（若支援）")
    ap.add_argument("--w_vec", type=float, default=0.7, help="向量分數權重")
    ap.add_argument("--w_text", type=float, default=0.3, help="關鍵字分數權重")
    ap.add_argument("--out", type=str, default="", help="把每筆結果輸出成 CSV")
    args = ap.parse_args()

    # 關鍵字模式選擇
    kw_mode = "bm25" if args.kw == "bm25" else "hits"

    # 包一層，讓不支援 temperature 也能用
    text_embed = build_text_embed_with_temp(_text_embed, args.vec_temp)

    # 建 Ranker
    ranker = HybridRanker(
        embeddings=EMB,
        metadata=META,
        text_embed_fn=text_embed,
        kw_mode=kw_mode,
        w_vec=args.w_vec,
        w_text=args.w_text,
    )

    # 跑評測
    results = []
    for e in EVAL:
        q = e["query"]
        positives = e["positives"]
        hits = ranker.rank(q, topk=20)
        names = [h["name"] for h in hits]

        p5 = precision_at_k(names, positives, k=5)
        r1 = mrr(names, positives)
        nd = ndcg_at_k(names, positives, k=5)
        results.append((q, p5, r1, nd))

        print(f"[Q] {q}")
        print(f"   top5: {[n for n in names[:5]]}")
        print(f"   P@5={p5:.2f}  MRR={r1:.2f}  nDCG@5={nd:.2f}\n")

    avg_p5  = float(np.mean([r[1] for r in results])) if results else 0.0
    avg_mrr = float(np.mean([r[2] for r in results])) if results else 0.0
    avg_nd  = float(np.mean([r[3] for r in results])) if results else 0.0

    print("====== 平均指標 ======")
    print(f"Precision@5: {avg_p5:.3f}")
    print(f"MRR:         {avg_mrr:.3f}")
    print(f"nDCG@5:      {avg_nd:.3f}")
    print("=====================")
    print(f"[DEBUG] EMB shape: {EMB.shape}")
    print(f"[DEBUG] META len : {len(META)}")
    print(f"[DEBUG] ex name  : {META[0].get('name')}")

if __name__ == "__main__":
    main()
