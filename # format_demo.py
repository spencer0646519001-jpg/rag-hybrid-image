# format_demo.py

data = [
    {"title": "抹茶草莓塔的穩定出品參數", "score_avg": 43.25, "score_max": 100},
    {"title": "抹茶ガナッシュの乳化溫度", "score_avg": 7.3, "score_max": 50},
    {"title": "Raspberry macaron shell cracks", "score_max": 60},  # 故意漏掉 score_avg
]

print("=== 原始 dict 內容 ===")
for d in data:
    print(d)

print("\n=== 直接取值 (h.get) ===")
for d in data:
    # 沒有 key 時，給預設值 0
    avg = d.get("score_avg", 0.0)
    mx = d.get("score_max", 0.0)
    print(f"avg={avg}, max={mx}")

print("\n=== 格式化輸出 (排版用) ===")
for d in data:
    avg = d.get("score_avg", 0.0)
    mx = d.get("score_max", 0.0)
    # :>5.1f → 右對齊，5字寬，小數1位
    print(f"[avg={avg:>5.1f} | max={mx:>5.1f}] {d['title']}")
