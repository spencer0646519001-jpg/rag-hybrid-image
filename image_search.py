# image_search.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

# 依你的專案結構：資料都在專案根目錄下的 data/
DATA_DIR = Path("data")
INDEX_JSON = DATA_DIR / "image_index.json"
EMB_NPY = DATA_DIR / "img_embeddings.npy"

# 使用與建索引時相同的模型名稱（前面 build_image_index.py 用 clip-ViT-B-32）
MODEL_NAME = "clip-ViT-B-32"

# ---- Lazy-loaded singletons ----
_MODEL = None          # SentenceTransformer
_INDEX = None          # List[dict]
_EMB_MATRIX = None     # np.ndarray, shape (n_images, dim)


# ========== Utilities ==========
def _ensure_model():
    """Lazy 取得/載入 CLIP 模型（sentence-transformers 版本）。"""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(MODEL_NAME)
    return _MODEL


def _load_index() -> List[dict]:
    """載入 image_index.json"""
    global _INDEX
    if _INDEX is None:
        with INDEX_JSON.open("r", encoding="utf-8") as f:
            _INDEX = json.load(f)
    return _INDEX


def _load_embeddings() -> np.ndarray:
    """載入 img_embeddings.npy，回傳 (n_images, dim)"""
    global _EMB_MATRIX
    if _EMB_MATRIX is None:
        _EMB_MATRIX = np.load(EMB_NPY)
        # 以防萬一，再做一次 L2 正規化（與建檔時 normalize_embeddings=True 一致）
        norms = np.linalg.norm(_EMB_MATRIX, axis=1, keepdims=True) + 1e-12
        _EMB_MATRIX = _EMB_MATRIX / norms
    return _EMB_MATRIX


def _pretty_name(info: dict) -> str:
    """
    從 index 的單筆 dict 取一個可顯示的名稱。
    相容三種欄位：filename / path / id；若是路徑則取尾端檔名。
    """
    name = info.get("filename") or info.get("path") or info.get("id") or ""
    try:
        # 若有可能是路徑，把檔名切出來
        if "/" in name or "\\" in name:
            name = Path(name).name
    except Exception:
        pass
    return str(name)


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """回傳分數由高到低的 top-k 索引。"""
    k = max(1, min(k, scores.shape[0]))
    # argsort 從小到大，取負號可達到由大到小
    return np.argsort(-scores)[:k]


# ========== Search APIs ==========
def search_image_by_image(query_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
    """
    輸入圖片路徑，回傳 (name, score) 排序結果。
    """
    model = _ensure_model()
    index = _load_index()
    mat = _load_embeddings()  # (n, dim)

    # 讀圖並轉 embedding（CLIP image encoder）
    img = Image.open(query_path).convert("RGB")
    q_vec = model.encode([img], convert_to_tensor=False, normalize_embeddings=True)[0]
    q_vec = np.asarray(q_vec, dtype=np.float32)

    # 餘弦相似度（因為皆已 L2 正規化，可直接點積）
    sims = mat @ q_vec  # (n,)

    top_idx = _topk_indices(sims, top_k)
    results: List[Tuple[str, float]] = []
    for i in top_idx:
        info = index[i]
        name = _pretty_name(info)
        results.append((name, float(sims[i])))

    return results


def search_image_by_text(query_text: str, top_k: int = 5) -> List[Tuple[str, float]]:
    """
    輸入文字描述（中文/英文皆可），用 CLIP 文本編碼器對圖片做相似度搜尋。
    """
    model = _ensure_model()
    index = _load_index()
    mat = _load_embeddings()  # (n, dim)

    # 文本轉 embedding（CLIP text encoder）
    q_vec = model.encode([query_text], convert_to_tensor=False, normalize_embeddings=True)[0]
    q_vec = np.asarray(q_vec, dtype=np.float32)

    sims = mat @ q_vec  # (n,)
    top_idx = _topk_indices(sims, top_k)

    results: List[Tuple[str, float]] = []
    for i in top_idx:
        info = index[i]
        name = _pretty_name(info)
        results.append((name, float(sims[i])))

    return results


# ========== CLI ==========
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Image search (by image or by text) over prebuilt CLIP embeddings."
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--image", type=str, help="查詢圖片檔案路徑")
    g.add_argument("--text", type=str, help="查詢文字")
    p.add_argument("query", nargs="?", help="不加旗標時：若像是檔案就當作 image，否則當作 text")
    p.add_argument("--top-k", type=int, default=5)
    return p


def main():
    parser = _build_argparser()
    args = parser.parse_args()

    if args.image:
        hits = search_image_by_image(args.image, top_k=args.top_k)
    elif args.text:
        hits = search_image_by_text(args.text, top_k=args.top_k)
    else:
        # 嘗試自動判斷：提供的位置像檔案就視為 image，否則視為 text
        if not args.query:
            parser.print_help()
            return
        q = args.query
        if Path(q).exists():
            hits = search_image_by_image(q, top_k=args.top_k)
        else:
            hits = search_image_by_text(q, top_k=args.top_k)

    print("\n=== Top results ===")
    for name, score in hits:
        print(f"{name:>24s}  |  {score:.4f}")


if __name__ == "__main__":
    main()

from sentence_transformers import SentenceTransformer
import numpy as np

# 初始化文字嵌入模型（與圖片索引相同）
_text_model = SentenceTransformer("clip-ViT-B-32")

def text_embed(text: str) -> np.ndarray:
    """
    將文字查詢轉換成向量，用於 Hybrid 檢索。
    輸入: text (str)
    回傳: np.ndarray 向量
    """
    emb = _text_model.encode([text], normalize_embeddings=True)[0]
    return np.asarray(emb, dtype=np.float32)
