from pathlib import Path
import json, re
from typing import List, Dict, Optional
from rapidfuzz import fuzz

VERSION = "A1.4"
print("RUNNING:", __file__, VERSION)

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "desserts.json"

# --- util ---
def normalize(s: str) -> str:
    # 只做去頭尾空白＋變小寫＋壓縮多空白，不移除任何字元（保留中日文）
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

# --- scoring ---
def best_match_score(q: str, t: str) -> float:
    if not q or not t:
        return 0.0
    if q in t:
        return 100.0
    return max(
        float(fuzz.partial_ratio(q, t)),
        float(fuzz.token_set_ratio(q, t)),
        float(fuzz.WRatio(q, t)),
    )

FIELD_WEIGHTS = {"title": 1.0, "tags": 0.8, "snippet": 0.6}

def score_doc(q: str, d: dict) -> dict:
    terms = [x for x in re.split(r"[,\s]+", q.strip()) if x]
    hay = d["_hay"]

    def s_one(term: str) -> float:
        term = normalize(term)
        return max(
            float(fuzz.partial_ratio(term, hay)),
            float(fuzz.token_set_ratio(term, hay)),
            float(fuzz.ratio(term, hay)),
        )

    scores = [s_one(t) for t in terms]
    s_avg = sum(scores) / len(scores) if scores else 0.0
    s_max = max(scores) if scores else 0.0

    return {
        "id": d["id"], "title": d["title"], "snippet": d["snippet"],
        "tags": d["tags"], "lang": d["lang"],
        "score_avg": s_avg, "score_max": s_max, "matched_on": "all"
    }



# --- 補上 search()，讓 CLI 可以運作 ---
def search(q: str,
           k: int = 5,
           min_score: int = 5,
           lang: Optional[str] = None,
           tag: Optional[str] = None):
    """
    q: 查詢字串（可含多詞、逗號/空白分隔）
    lang: 過濾語言（'zh'/'en'/'ja'；None 表示不過濾）
    tag:  單一標籤過濾；None 表示不過濾
    """
    # 先過濾資料池
    pool = DOCS
    if lang:
        pool = [d for d in pool if d.get("lang") == lang]
    if tag:
        # and：所有 tags 都要在文件中
     pool= [d for d in pool if all(t in (d.get("tags") or []) for t in tags)]
# or：至少一個 tag 在文件中
     pool = [d for d in pool if any(t in (d.get("tags") or []) for t in tags)]

    
    print(f"pool size after filters: {len(pool)}")

    # 打分 + 排序
    hits = [score_doc(q, d) for d in pool]
    hits.sort(key=lambda x: x["score_avg"], reverse=True)

    # debug top-3
    print("---- debug top-3 ----")
    for h in hits[:3]:
        print(f"{h['score_avg']:>5.1f} | {h['title']}")
    print("---------------------")
    print("min_score =", min_score, "type:", type(min_score))


    # 門檻 + 截取 k 筆
    return [h for h in hits if h["score_avg"] >= min_score][:k]

# --- CLI ---
print(">>> using search from", __file__)

if __name__ == "__main__":
    print("小提示：輸入 q 退出；支援中/英/日，逗號可分多詞（例：macaron, 抹茶）")
    while True:
        q = input("\n搜尋關鍵字：").strip()
        if q.lower() in {"q", "quit", "exit"}:
            break
        results = search(q, k=5, min_score=70)
        if not results:
            print("（沒有命中）")
            continue
        for i, r in enumerate(results, 1):
         print(f"{i}. [avg={r['score_avg']:.0f} | max={r['score_max']:.0f}] {r['title']}")

