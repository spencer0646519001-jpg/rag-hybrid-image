from fuzzywuzzy import process
import difflib
import random
import string
import time

# 建立模擬甜點資料庫（1000 筆隨機甜點名稱）
def random_word(length):
    return ''.join(random.choices(string.ascii_lowercase, k=length))

dessert_db = {f"dessert_{i}_{random_word(5)}": ["sweet", "cold", "fancy"] for i in range(1000)}

# 加入幾筆實際甜點名稱
dessert_db.update({
    "souffle": ["french", "egg", "fluffy"],
    "montblanc": ["chestnut", "french", "classic"],
    "tiramisu": ["italian", "coffee", "cheese"]
})

# 模擬使用者輸入（錯拼）
user_input = "sufurei"

# difflib 測試
start_difflib = time.time()
difflib_matches = difflib.get_close_matches(user_input, dessert_db.keys(), n=3, cutoff=0.4)
end_difflib = time.time()

# fuzzywuzzy 測試
start_fuzzy = time.time()
fuzzy_matches = process.extract(user_input, dessert_db.keys(), limit=3)
end_fuzzy = time.time()

# 顯示結果
print("=== difflib 結果 ===")
print("執行時間：", round(end_difflib - start_difflib, 5), "秒")
for name in difflib_matches:
    print("-", name)

print("\n=== fuzzywuzzy 結果 ===")
print("執行時間：", round(end_fuzzy - start_fuzzy, 5), "秒")
for name, score in fuzzy_matches:
    print(f"- {name}（相似度：{score}）")
