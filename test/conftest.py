# test/conftest.py
from pathlib import Path
import sys

# 讓測試能 import 專案根目錄的 main.py
ROOT = Path(__file__).resolve().parents[1]  # test/ → 專案根
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def make_doc(title: str, snippet: str = "", tags=None):
    """測試用的輕量 doc factory（測試專用，不放產品碼）"""
    tags = tags or []
    d = {"id": "x", "title": title, "snippet": snippet, "tags": tags}
    d["_hay"] = " ".join([title, snippet, " ".join(tags)]).strip().lower()
    return d
