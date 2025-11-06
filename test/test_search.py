import os
import numpy as np
import pytest
from main import search, refresh_docs, DOCS

def test_search_top1_has_strawberry_in_title_or_tags():
    # 重新載入並重建向量（你的 refresh_docs 會清掉舊 npy/metadata）
    refresh_docs()

    # 沒有草莓相關就跳過（避免資料更動時硬 fail）
    has_strawberry = any(
        ("草莓" in d.get("title","")) or
        ("草莓" in " ".join(d.get("tags", []))) or
        ("Strawberry" in d.get("title","")) or
        ("Strawberry" in " ".join(d.get("tags", [])))
        for d in DOCS
    )
    if not has_strawberry:
        pytest.skip("資料中沒有草莓相關甜點，跳過此測試。")

    # 查詢只用「草莓」避免不在資料內的詞影響（奶油/蛋糕）
    q = "草莓"
    # 拉高文字權重，確保關鍵字佔主導
    results = search(q, mode="hybrid", w_text=0.8, w_vec=0.2, top_k=3)
    assert len(results) > 0

    top1 = results[0]
    title = top1["title"]
    tags = top1["tags"]
    print("Top1:", title, "| tags:", tags)

    assert (
        ("草莓" in title) or ("草莓" in " ".join(tags)) or
        ("Strawberry" in title) or ("Strawberry" in " ".join(tags))
    ), "Top1 應該與草莓相關（標題或標籤）"


def test_vector_search_stability():
    q = "抹茶"
    res1 = search(q, mode="vector", top_k=3)
    res2 = search(q, mode="vector", top_k=3)
    titles1 = [r["title"] for r in res1]
    titles2 = [r["title"] for r in res2]
    assert titles1 == titles2, "向量搜尋結果應該穩定"
