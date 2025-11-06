# dessert_search.py
import json
import re
from pathlib import Path

# ---- fuzzy matching backend: prefer rapidfuzz, fallback to fuzzywuzzy ----
try:
    from rapidfuzz import process as fuzz_process
except Exception:
    from fuzzywuzzy import process as fuzz_process  # type: ignore

# 預設用程式同資料夾的 dessert_data.json
DATA_PATH = Path(__file__).parent / "dessert_data.json"

def load_knowledge(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        # 若檔案不存在，建立一個空的
        path.write_text("{}", encoding="utf-8")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # 確保 value 都是 list[str]
    fixed = {}
    for k, v in data.items():
        if isinstance(v, list):
            fixed[k] = [str(x).strip().lower() for x in v if str(x).strip()]
        else:
            fixed[k] = [str(v).strip().lower()] if str(v).strip() else []
    return fixed

def save_knowledge(path: Path, data: dict[str, list[str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def parse_keywords(s: str) -> list[str]:
    # 逗號、頓號、"和"、空白都當作分隔
    parts = re.split(r"[,，、\s和]+", s.lower())
    return [p.strip() for p in parts if p.strip()]

def merge_tags(old_tags: list[str], new_tags: list[str]) -> list[str]:
    # 新標籤放前面，舊標籤去重後接上
    existed = set(t.lower() for t in new_tags)
    rest = [t for t in old_tags if t.lower() not in existed]
    return [*new_tags, *rest]

def fuzzy_name_hits(key: str, names: list[str], limit: int = 3, score_cut: int = 80) -> list[str]:
    hits = []
    for name, score, *_ in fuzz_process.extract(key, names, limit=limit):
        # rapidfuzz 回傳 (name, score, idx)；fuzzywuzzy 回傳 (name, score)
        if score >= score_cut:
            hits.append(name)
    return hits

def search(kwds: list[str], kb: dict[str, list[str]]) -> set[str]:
    results: set[str] = set()
    names = list(kb.keys())

    # 1) 比對名稱（模糊）
    for k in kwds:
        results.update(fuzzy_name_hits(k, names, limit=3, score_cut=80))

    # 2) 比對標籤（關鍵字包含即可）
    for k in kwds:
        for name, tags in kb.items():
            if any(k in tag for tag in tags):
                results.add(name)
    return results

def main():
    kb = load_knowledge(DATA_PATH)

    user_input = input("請輸入你想搜尋的甜點關鍵字（可輸入多個）：")
    keywords = parse_keywords(user_input)

    results = search(keywords, kb)

    if results:
        print("\n🔎 你可能在找這些甜點：\n")
        for name in sorted(results):
            print(f"- {name}：{kb.get(name, [])}")
    else:
        print("😢 查無相關甜點")
        choice = input("❓是否要新增新的甜點資料？(y/n)：").strip().lower()
        if choice == "y":
            name = input("請輸入甜點名稱：").strip()
            if name in kb:
                print(f"⚠️『{name}』已存在，目前標籤為：{kb[name]}")
            else:
                tag_input = input("請輸入標籤（用逗號、空白或『和』分開）：").strip().lower()
                new_tags = parse_keywords(tag_input)
                kb[name] = new_tags
                print(f"✅ 已新增「{name}」標籤為：{kb[name]}")
                save_knowledge(DATA_PATH, kb)

    # 不論剛才是否新增，都提供一次編輯機會
    edit_choice = input("\n✏️ 是否要編輯某個甜點的標籤？(y/n)：").strip().lower()
    if edit_choice == "y":
        name = input("請輸入要修改標籤的甜點名稱（請照上面顯示的名稱輸入）：").strip()
        tag_input = input("請輸入標籤（用逗號、空白或『和』分開）：").strip().lower()
        new_tags = parse_keywords(tag_input)

        if name in kb:
            kb[name] = merge_tags(kb[name], new_tags)
            print(f"✅ 已更新「{name}」標籤為：{kb[name]}")
        else:
            kb[name] = new_tags
            print(f"✅ 已新增「{name}」標籤為：{kb[name]}")
        save_knowledge(DATA_PATH, kb)

if __name__ == "__main__":
    main()
