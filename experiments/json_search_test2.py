import json
import re

# 先讀取原始資料（如果有的話）
try:
    with open("dessert_data.json", "r", encoding="utf-8") as f:
        dessert_knowledge = json.load(f)
except FileNotFoundError:
    dessert_knowledge = {}

# 使用者輸入甜點名稱
name = input("請輸入甜點名稱：").strip()

# 檢查是否已存在
if name in dessert_knowledge:
    print(f"⚠️「{name}」已存在，目前標籤為：{dessert_knowledge[name]}")
else:
    # 輸入標籤（可用逗號、空格、和 連接）
    tag_input = input("請輸入標籤（用逗號、空格或 和 分隔）：").strip().lower()
    tags = re.split(r"[,，、 和 ]+", tag_input)
    tags = [tag for tag in tags if tag]  # 避免空字串

    # 加入資料
    dessert_knowledge[name] = tags

    # 儲存回 JSON 檔
    with open(r"C:\Users\Spencer\Desktop\python_learning\dessert_data.json", "w", encoding="utf-8") as f:
        json.dump(dessert_knowledge, f, ensure_ascii=False, indent=2)

    print(f"✅ 已新增「{name}」，標籤為：{tags}")
