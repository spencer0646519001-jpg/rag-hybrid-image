from pathlib import Path
import json, re
from typing import List, Dict, Optional
from rapidfuzz import fuzz

# --- 路徑設定 & 檢查 ---
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "desserts.json"
print("__file__ in script:", __file__)
print("cwd:", Path().resolve())
print("BASE_DIR:", BASE_DIR)
print("DATA_PATH exists?", DATA_PATH.exists())

# --- util ---
def normalize(s: str) -> str:
    # 去頭尾空白 + 小寫 + 壓縮多空白（不移除中日文）
    s = str(s) if s is not None else ""
    s = s.strip().lower()
    s = " ".join(s.split())
    return s

# --- data ---
def load_docs() -> List[Dict]:
    items: List[Dict] = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    for d in items:
        d["_hay"] = normalize(" ".join([d["title"], d["snippet"], " ".join(d["tags"])]))
    return items

DOCS = load_docs()
print("資料筆數:", len(DOCS))

# --- 打分：MAX/AVG 兩種 ---
def _best_for_term(term: str, hay: str) -> float:
    term = normalize(term)
    return max(
        float(fuzz.partial_ratio(term, hay)),
        float(fuzz.token_set_ratio(term, hay)),
        float(fuzz.ratio(term, hay)),
    )

def score_doc_max(q: str, d: dict) -> float:
    terms = [x for x in re.split(r"[,，、；;\s]+", q.strip()) if x]
    if not terms:
        return 0.0
    hay = d["_hay"]
    return max(_best_for_term(t, hay) for t in terms)

def score_doc_avg(q: str, d: dict) -> float:
    terms = [x for x in re.split(r"[,，、；;\s]+", q.strip()) if x]
    if not terms:
        return 0.0
    hay = d["_hay"]
    scores = [_best_for_term(t, hay) for t in terms]
    return sum(scores) / len(scores)

# 提供給後端 API 的統一打分（目前採用 MAX）
def score_doc(q: str, d: dict) -> dict:
    s = score_doc_max(q, d)  # 想切換成 AVG 就改這行
    return {
        "id": d["id"], "title": d["title"], "snippet": d["snippet"],
        "tags": d["tags"], "lang": d["lang"],
        "score": s, "matched_on": "all"
    }

# --- 並排對照：MAX vs AVG ---
def compare_side_by_side(q: str, top: int = 3):
    scored_max = []
    scored_avg = []
    for d in DOCS:
        title = d["title"]
        scored_max.append((score_doc_max(q, d), title))
        scored_avg.append((score_doc_avg(q, d), title))

    scored_max.sort(key=lambda x: x[0], reverse=True)
    scored_avg.sort(key=lambda x: x[0], reverse=True)

    left  = scored_max[:top]
    right = scored_avg[:top]
    if len(left)  < top: left  += [(0.0, "")] * (top - len(left))
    if len(right) < top: right += [(0.0, "")] * (top - len(right))

    sep = "-" * 96
    print(f"\n查詢關鍵字：{q}")
    print(sep)
    print(f"{'MAX 排名':<6} {'分數':>6}  標題（MAX）{'':<28} │ {'AVG 排名':<6} {'分數':>6}  標題（AVG）")
    print(sep)
    for i in range(top):
        sL, tL = left[i]
        sR, tR = right[i]
        tL_show = (tL[:34] + "…") if len(tL) > 35 else tL
        tR_show = (tR[:34] + "…") if len(tR) > 35 else tR
        print(f"{i+1:<6} {sL:6.1f}  {tL_show:<36} │ {i+1:<6} {sR:6.1f}  {tR_show}")
    print(sep)

def search_with(q: str,
                scorer,           # 傳入 score_doc_max 或 score_doc_avg
                k: int = 5,
                min_score: int = 10,
                lang: str | None = None,
                tag: str  | None = None):
    # 過濾資料池
    pool = DOCS
    if lang:
        pool = [d for d in pool if d.get("lang") == lang]
    if tag:
        pool = [d for d in pool if tag in (d.get("tags") or [])]

    # 打分
    hits = []
    for d in pool:
        s = scorer(q, d)  # <-- 關鍵：改用外面傳進來的打分函式
        hits.append({
            "id": d["id"], "title": d["title"], "snippet": d["snippet"],
            "tags": d["tags"], "lang": d["lang"],
            "score": s, "matched_on": "all"
        })

    hits.sort(key=lambda x: x["score"], reverse=True)

    # 有標籤的 debug
    label = "MAX" if scorer is score_doc_max else "AVG"
    print(f"---- debug top-3 ({label}) ----")
    for h in hits[:3]:
        print(f"{h['score']:>5.1f} | {h['title']}")
    print("--------------------------------")

    return [h for h in hits if h["score"] >= min_score][:k]


def show_results(label: str, results: list[dict]):
    print(f"\n[{label}]")
    if not results:
        print("（沒有命中）")
        return
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['score']:.0f}] {r['title']} 〈{r['matched_on']}〉")

# --- 搜尋（含語言/標籤過濾） ---
def search(q: str,
           k: int = 5,
           min_score: int = 10,
           lang: Optional[str] = None,
           tag: Optional[str] = None):
    pool = DOCS
    if lang:
        pool = [d for d in pool if d.get("lang") == lang]
    if tag:
        pool = [d for d in pool if tag in (d.get("tags") or [])]

    hits = [score_doc(q, d) for d in pool]
    hits.sort(key=lambda x: x["score"], reverse=True)

    print("---- debug top-3 ----")
    for h in hits[:3]:
        print(f"{h['score']:>5.1f} | {h['title']}")
    print("---------------------")

    return [h for h in hits if h["score"] >= min_score][:k]

# --- CLI ---
if __name__ == "__main__":
    print("小提示：輸入 q 退出；支援中/英/日，逗號可分多詞（例：macaron, 抹茶）")
    while True:
        q = input("\n搜尋關鍵字：").strip()
        if q.lower() in {"q", "quit", "exit"}:
            break

        # 並排表格（同一個 q 的 MAX vs AVG）
        compare_side_by_side(q, top=3)

        # 同一個 q，各跑一次 MAX 與 AVG（而且有明確標籤）
        res_max = search_with(q, score_doc_max, k=5, min_score=30)
        show_results("search (MAX)", res_max)

        res_avg = search_with(q, score_doc_avg, k=5, min_score=30)
        show_results("search (AVG)", res_avg)
