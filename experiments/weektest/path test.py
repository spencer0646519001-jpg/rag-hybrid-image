from pathlib import Path

p = Path("tmp/hello.txt")

print("exists?", p.exists())   # 先看檔案是否存在
print("is_file?", p.is_file()) # 是否是檔案
print("is_dir?", p.is_dir())   # 是否是資料夾

# 建立資料夾
p.parent.mkdir(exist_ok=True)

# 建立檔案
p.write_text("hi", encoding="utf-8")

print("exists?", p.exists())   # 檔案已存在
print("is_file?", p.is_file()) # True
print("is_dir?", p.is_dir())   # False
