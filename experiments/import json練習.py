import json

dessert_knowledge = {
    "舒芙蕾": ["法式", "經典", "蛋", "蓬鬆"],
    "蒙布朗": ["栗子", "法式", "山型", "甜點"],
    "提拉米蘇": ["義式", "咖啡", "起司", "經典"],
    "千層派": ["法式", "酥皮", "奶油"],
    "布朗尼": ["美式", "巧克力", "濃郁"]
}

# 將資料存成 dessert_data.json 檔案
with open(r"C:\Users\Spencer\Desktop\python_learning\dessert_data.json", "w", encoding="utf-8") as f:
    json.dump(dessert_knowledge, f, ensure_ascii=False, indent=2)


print("✅ 資料已儲存為 JSON 檔案！")
