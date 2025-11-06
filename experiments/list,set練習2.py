
# 顧客輸入紀錄：可重複、要保留順序
customer_inputs = []

while True:
    name = input("請輸入甜點名稱（輸入 exit 結束）：")
    if name == "exit":
        break
    customer_inputs.append(name)

print("所有顧客輸入順序紀錄：", customer_inputs)

# 甜點總目錄：不重複、不在意順序
dessert_catalog = set()

while True:
    name = input("請輸入甜點名稱（輸入 exit 結束）：")
    if name == "exit":
        break
    dessert_catalog.add(name)

print("甜點總目錄（不重複）：", dessert_catalog)
