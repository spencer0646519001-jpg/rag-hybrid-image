# ===========================================
# Python 基礎概念綜合練習：()、[]、{}、索引、set操作
# ===========================================

print("=== 1. 三種括號的用法對比 ===")

# () - tuple（元組）：不可更改、有順序
tuple_example = ("蘋果", "香蕉", "橘子")
print(f"Tuple: {tuple_example}")
print(f"第一個水果: {tuple_example[0]}")  # 用索引取值
# tuple_example[0] = "西瓜"  # 這行會報錯！tuple不能更改

# [] - list（清單）：可更改、有順序  
list_example = ["蘋果", "香蕉", "橘子"]
print(f"List: {list_example}")
print(f"第二個水果: {list_example[1]}")   # 用索引取值
list_example[0] = "西瓜"  # 可以更改
list_example.append("葡萄")  # 可以新增
print(f"更改後的List: {list_example}")

# {} - dict（字典）：key-value配對
dict_example = {"name": "小明", "age": 40, "city": "台北"}
print(f"Dict: {dict_example}")
print(f"姓名: {dict_example['name']}")    # 用key取值，不是索引！

# {} - set（集合）：不重複、無順序
set_example = {"蘋果", "香蕉", "橘子", "蘋果"}  # 重複的"蘋果"會被自動去除
print(f"Set: {set_example}")

print("\n" + "="*50)

print("=== 2. 索引(Index)詳細解釋 ===")

# 索引：序列型資料的位置編號，從0開始
fruits = ["蘋果", "香蕉", "橘子", "葡萄", "西瓜"]
print(f"水果清單: {fruits}")
print(f"索引0: {fruits[0]}")   # 第1個
print(f"索引1: {fruits[1]}")   # 第2個  
print(f"索引2: {fruits[2]}")   # 第3個
print(f"負索引-1: {fruits[-1]}")  # 最後一個
print(f"負索引-2: {fruits[-2]}")  # 倒數第二個

# 切片：取一段範圍
print(f"切片[1:3]: {fruits[1:3]}")   # 從索引1到2（不含3）
print(f"切片[:2]: {fruits[:2]}")     # 從開頭到索引1
print(f"切片[2:]: {fruits[2:]}")     # 從索引2到結尾

# 字串也有索引！
text = "Hello"
print(f"字串: {text}")
print(f"字串索引0: {text[0]}")       # 'H'
print(f"字串索引-1: {text[-1]}")     # 'o'

print("\n" + "="*50)

print("=== 3. set（集合）常用方法詳解 ===")

# 建立set的幾種方式
dessert_set = set()  # 空集合
favorite_desserts = {"蛋糕", "布丁", "冰淇淋"}
print(f"我的最愛甜點: {favorite_desserts}")

# add() - 加入一個元素
print("\n--- add()方法 ---")
dessert_set.add("舒芙蕾")
dessert_set.add("蒙布朗")
dessert_set.add("舒芙蕾")  # 重複加入，不會有兩個
print(f"加入後: {dessert_set}")

# update() - 一次加入多個元素
print("\n--- update()方法 ---")
new_desserts = ["提拉米蘇", "千層派", "布朗尼"]
dessert_set.update(new_desserts)
print(f"批量加入後: {dessert_set}")

# 也可以update另一個set
more_desserts = {"馬卡龍", "泡芙"}
dessert_set.update(more_desserts)
print(f"加入更多後: {dessert_set}")

# remove() vs discard() - 移除元素
print("\n--- remove() vs discard() ---")
print(f"移除前: {dessert_set}")

# remove() - 如果元素不存在會報錯
dessert_set.remove("泡芙")
print(f"remove泡芙後: {dessert_set}")

# discard() - 如果元素不存在不會報錯
dessert_set.discard("不存在的甜點")  # 不會報錯
dessert_set.discard("布朗尼")
print(f"discard後: {dessert_set}")

# 其他常用set方法
print("\n--- 其他set方法 ---")
set1 = {"A", "B", "C"}
set2 = {"B", "C", "D"}

print(f"set1: {set1}")
print(f"set2: {set2}")
print(f"交集（共同元素）: {set1 & set2}")          # {'B', 'C'}
print(f"聯集（所有元素）: {set1 | set2}")          # {'A', 'B', 'C', 'D'}
print(f"差集（set1有但set2沒有）: {set1 - set2}")   # {'A'}

print("\n" + "="*50)

print("=== 4. 在RAG系統中的實際應用 ===")

# 模擬你的甜點RAG系統，展示三種資料結構的搭配使用
dessert_knowledge = {
    "舒芙蕾": "法式經典甜點，蛋白打發製成",
    "蒙布朗": "栗子泥和鮮奶油製作", 
    "提拉米蘇": "義大利甜點，馬斯卡彭起司製作",
    "千層派": "多層酥皮製作",
    "布朗尼": "濃郁巧克力蛋糕"
}

# 模擬用戶查詢
user_queries = ["舒芙蕾,布朗尼", "派", "不存在的甜點"]

for query in user_queries:
    print(f"\n查詢: '{query}'")
    
    # 使用list儲存關鍵字（保持順序）
    keywords = query.split(",")
    print(f"關鍵字list: {keywords}")
    
    # 使用set儲存結果（自動去重）
    results = set()
    
    for keyword in keywords:
        keyword = keyword.strip()  # 去除空白
        for dessert_name in dessert_knowledge.keys():
            if keyword in dessert_name:
                results.add(dessert_name)  # set自動去重
    
    # 顯示結果
    if results:
        print(f"找到 {len(results)} 個結果:")
        for dessert in results:
            print(f"  {dessert}: {dessert_knowledge[dessert]}")
    else:
        print("  沒有找到相關甜點")

print("\n" + "="*50)
print("=== 練習總結 ===")
print("• list: 用[]，可重複，有順序，用append()新增")
print("• set: 用{}，不重複，無順序，用add()新增") 
print("• dict: 用{}，key-value配對，用[]取值")
print("• 索引: 從0開始的位置編號，只適用於有序資料")