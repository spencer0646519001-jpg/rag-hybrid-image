from pathlib import Path

# 引入剛剛的 normalize
def normalize(s: str) -> str:
    s = str(s) if s is not None else ""
    s = s.strip().lower()
    s = " ".join(s.split())
    return s

# 測試案例
tests = [
    None,
    123,
    "   Hello   World   ",
    "HELLO",
    "   Mixed   CASE   Letters ",
    "   多   空   白   ",
    "  Hello    世界   ",
    "  python\tRocks   \n\nGreat  "
]

print("=== normalize 測試 ===")
for i, t in enumerate(tests, 1):
    print(f"{i}. 原始：{repr(t)}")
    print(f"   處理後：{repr(normalize(t))}")
    print("-" * 40)
