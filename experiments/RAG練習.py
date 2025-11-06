from fuzzywuzzy import process
dessert_knowledge = {
    "舒芙蕾": ["法式", "經典", "蛋", "蓬鬆"],
    "蒙布朗": ["栗子", "法式", "山型", "甜點"],
    "提拉米蘇": ["義式", "咖啡", "起司", "經典"],
    "千層派": ["法式", "酥皮", "奶油"],
    "布朗尼": ["美式", "巧克力", "濃郁"],
}

user_input = input("請輸入你想搜尋的甜點關鍵字：").lower()
import re
keywords = re.split('[,，、 和 ]+', user_input)
found = False

print(keywords)

results = set()

for keyword in keywords:
    if not keyword:
        continue
    matches = process.extract(keyword, dessert_knowledge.keys(), limit=3)
for name, score in matches:
    results.add(name)




# 搜尋標籤
for keyword in keywords:
    if not keyword:
        continue
    for name, tags in dessert_knowledge.items():
        if any(keyword in tag for tag in tags):
            results.add(name)
              
# 排序輸出
if results:
    print("你可能在找這些甜點：")
    for name in sorted(results):
        print(f"- {name} 標籤: {dessert_knowledge[name]}")
else:
    print("查無相關甜點")