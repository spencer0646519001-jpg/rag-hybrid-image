for day in range(1, 4):  # 3天
    print(f"\nDay {day}")
    for shift in ["早班", "晚班"]:
        print(f"  Shift: {shift}")
        for worker in [1, 2, 3]:
            if worker == 2:
                print(f"    Worker {worker}: 請假（跳過）")
                continue
            print(f"    Worker {worker}: 正常出勤")
            if shift == "晚班":
                print("      晚班需加班")



            # 判斷 ID=2 是否請假
            # 晚班加班提示
