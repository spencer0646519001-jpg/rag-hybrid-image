#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自動從 data/image_index.json 產生 data/eval_set.json（或指定路徑）。

Heuristics：
1) 以 title / tags / name 萃取查詢字串（支援中/英/日）。
2) 以「名稱規範化」與「關鍵字反向索引」推估 positives（正解清單）。
3) 支援抽樣數量、最小正解數、策略（by_tag/by_title/mixed）、隨機種子。

使用方式：
  python tools/generate_eval_set.py \
    --meta data/image_index.json \
    --out  data/eval_set.json \
    --n 20 \
    --strategy mixed \
    --min-positives 1 \
    --seed 42

產出格式（eval_set.json）：
[
  {"query": "檸檬塔", "positives": ["lemon_tart.jpg", "lemon_tarte.jpg"]},
  {"query": "strawberry shortcake", "positives": ["strawberry_shortcake.jpg"]}
]
"""
from __future__ import annotations
import json, re, argparse, random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set

TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]|[\u3040-\u30ff]")

# -------- utils --------

def norm_name(name: str) -> str:
    """把檔名/標題做一個寬鬆正規化：
    - 小寫
    - 去副檔名
    - 把非字母數字/中日文假名的字元視為分隔，移除空白與連字號底線
    - 移除連續數字尾碼（_01, -2 等）
    """
    p = Path(name).name
    stem = p.rsplit(".", 1)[0]
    s = stem.lower()
    s = re.sub(r"[_\-]+", " ", s)
    toks = TOKEN_RE.findall(s)
    base = "".join(toks)
    # 去掉尾端連號數字（e.g., tart01 -> tart）
    base = re.sub(r"(\d+)$", "", base)
    return base

def tokenize_text(text: str) -> List[str]:
    if not text:
        return []
    return TOKEN_RE.findall(text.lower())

# -------- core --------

def load_meta(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    # 產生穩定欄位：name/title/tags
    fixed = []
    for m in meta:
        name = m.get("name") or m.get("file") or m.get("filename") or m.get("image") or m.get("img") or m.get("src") or m.get("path") or "unknown.jpg"
        name = Path(str(name)).name
        title = m.get("title") or Path(name).stem.replace("_", " ").replace("-", " ")
        tags = m.get("tags")
        if isinstance(tags, str):
            tags = [t for t in re.split(r"[\s,]+", tags) if t]
        if not isinstance(tags, list):
            tags = []
        fixed.append({"name": name, "title": str(title), "tags": [str(t) for t in tags], **m})
    return fixed

def build_inverted_index(meta: List[Dict[str, Any]]) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """建立兩種反向索引：
    - root_index：以檔名規範化 root 對應 names（e.g., lemon_tart.jpg, lemon-tarte1.jpg → "lemontart"）
    - token_index：以 token（來自 title/tags/name）對應 names
    """
    root_index: Dict[str, Set[str]] = {}
    token_index: Dict[str, Set[str]] = {}
    for m in meta:
        name = m["name"]
        root = norm_name(name)
        root_index.setdefault(root, set()).add(name)
        parts: List[str] = []
        parts.append(m.get("title", ""))
        parts.extend(m.get("tags", []))
        parts.append(Path(name).stem)
        toks = tokenize_text(" ".join(parts))
        for t in toks:
            token_index.setdefault(t, set()).add(name)
    return root_index, token_index


def guess_query_candidates(m: Dict[str, Any]) -> List[str]:
    """從單筆 meta 推測可用的 query 候選字串。優先順序：
    tags（<=2 個字的 token 可略過）→ title → name 的人類化版本
    """
    out: List[str] = []
    # 從 tags
    tags = m.get("tags", [])
    tag_tokens = []
    for t in tags:
        # 切成 token 後再組回人類可讀字串（英文保留、中文/假名單字也可）
        toks = tokenize_text(str(t))
        token_str = " ".join(tok for tok in toks if len(tok) >= 2 or re.match(r"[\u4e00-\u9fff\u3040-\u30ff]", tok))
        if token_str:
            tag_tokens.append(token_str)
    # 選擇性加入 1~2 個標籤
    out.extend(tag_tokens[:2])

    # 從 title
    title = (m.get("title") or "").strip()
    if title:
        out.append(title)

    # 從 name 的 stem
    name = m.get("name", "")
    stem = Path(name).stem.replace("_", " ").replace("-", " ")
    if stem and stem not in out:
        out.append(stem)

    # 去重，保序
    seen = set()
    uniq: List[str] = []
    for q in out:
        if q and q not in seen:
            seen.add(q)
            uniq.append(q)
    return uniq


def make_eval_items(meta: List[Dict[str, Any]], n: int, strategy: str,
                    min_pos: int, seed: int) -> List[Dict[str, Any]]:
    random.seed(seed)
    root_index, token_index = build_inverted_index(meta)
    names = [m["name"] for m in meta]

    # 候選母集：每筆對應若干 query 候選
    pool: List[Tuple[str, List[str]]] = []  # (name, [q1,q2,...])
    for m in meta:
        qs = guess_query_candidates(m)
        if qs:
            pool.append((m["name"], qs))

    # 依策略做抽樣
    random.shuffle(pool)
    eval_items: List[Dict[str, Any]] = []

    def positives_for_query(q: str, base_name: str) -> List[str]:
        pos: Set[str] = set()
        # 1) root 相同者（常見於多版本同甜點）
        base_root = norm_name(base_name)
        pos |= root_index.get(base_root, set())
        # 2) token 命中的擴充
        for tok in tokenize_text(q):
            pos |= token_index.get(tok, set())
        # 3) 自身一定要在 positives（確保至少 1）
        pos.add(base_name)
        return sorted(pos)

    # 逐一取樣，直到滿足 n 筆或池用盡
    for base_name, qs in pool:
        if len(eval_items) >= n:
            break
        # 按策略選一個 query
        cand_qs: List[str] = []
        if strategy == "by_tag":
            cand_qs = [q for q in qs if any(q == t or q in t for t in (meta_by_name(meta, base_name).get("tags", [])))]
        elif strategy == "by_title":
            t = meta_by_name(meta, base_name).get("title", "")
            cand_qs = [q for q in qs if q == t or q in t or t in q]
        else:  # mixed
            cand_qs = qs
        if not cand_qs:
            continue
        q = random.choice(cand_qs)

        pos = positives_for_query(q, base_name)
        if len(pos) < min_pos:
            continue
        eval_items.append({"query": q, "positives": pos})

    return eval_items


def meta_by_name(meta: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
    for m in meta:
        if m.get("name") == name:
            return m
    return {}

# -------- CLI --------

def main():
    ap = argparse.ArgumentParser(description="Generate eval_set.json from image_index.json")
    ap.add_argument("--meta", type=Path, default=Path("data/image_index.json"))
    ap.add_argument("--out", type=Path, default=Path("data/eval_set.json"))
    ap.add_argument("--n", type=int, default=20, help="產生的評測筆數上限")
    ap.add_argument("--strategy", choices=["by_tag","by_title","mixed"], default="mixed")
    ap.add_argument("--min-positives", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    meta = load_meta(args.meta)
    items = make_eval_items(meta, n=args.n, strategy=args.strategy,
                            min_pos=args.min_positives, seed=args.seed)

    # 去重（同 query 合併 positives）
    merged: Dict[str, Set[str]] = {}
    for it in items:
        q = it["query"].strip()
        merged.setdefault(q, set()).update(it["positives"])
    final = [{"query": q, "positives": sorted(list(ps))} for q, ps in merged.items()]

    # 輸出
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    # 統計
    total = len(final)
    avg_pos = sum(len(it["positives"]) for it in final) / total if total else 0.0
    print(f"[OK] Wrote {total} items → {args.out}")
    print(f"     avg #positives = {avg_pos:.2f}")
    # 顯示前 3 筆預覽
    for it in final[:3]:
        print("  -", it)

if __name__ == "__main__":
    main()
