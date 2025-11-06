import random

for day in range(1, 4):
    print(f"\nDay {day}")
    for item in ["麵粉", "蛋", "牛奶", "巧克力"]:
        stock = random.randint(1, 10)
        if stock < 3:
            print(f"{item} 存量不足（剩下 {stock}），立刻警示並跳過今天剩下檢查！")
            break  # 跳出今天剩下的食材檢查，進到下一天
        else:
            print(f"{item} 剩下 {stock}")
    else:
        print("今日所有食材存量檢查完畢。")
