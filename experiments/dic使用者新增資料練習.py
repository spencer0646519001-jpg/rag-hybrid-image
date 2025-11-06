

while True:
    name = input("請輸入甜點名稱(exit結束)：")
    if name == "exit":
        break
    tags = input("請輸入標籤（用逗號）：")
    dessert_db[name] = [t.strip() for t in tags.split(",")]

print(dessert_db)


new_name = input("請輸入新甜點名稱：")
new_tags = input("請輸入標籤（用逗號分隔）：")
tags_list = [tag.strip() for tag in new_tags.split(",")]  # 去除標籤空白
dessert_db[new_name] = tags_list