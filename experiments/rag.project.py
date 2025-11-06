from pathlib import Path
import json

BASE_DIR = Path(__file__).parent              # 這個 .py 檔所在資料夾
DATA_PATH = BASE_DIR / "desserts.json" 

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print("loaded items:", len(data))
