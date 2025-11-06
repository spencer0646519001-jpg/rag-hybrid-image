# test/weektest/test_day5.py
from pathlib import Path
import sys
import pytest

# ---- 找到專案根目錄（有 main.py 的那層）----
THIS = Path(__file__).resolve()
ROOT = next(p for p in THIS.parents if (p / "main.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import score_doc_keywords


def make_doc(title, snippet="", tags=None):
    tags = tags or []
    d = {"id": "x", "title": title, "snippet": snippet, "tags": tags}
    d["_hay"] = " ".join([title, snippet, " ".join(tags)]).strip().lower()
    return d


def test_invalid_int():
    with pytest.raises(ValueError):
        int("abc")


@pytest.mark.parametrize(
    "query, expected",
    [
        ("cake", True),       # 命中
        ("tiramisu", True),   # 命中
        ("pizza", False),     # 不命中
    ],
)
def test_keywords(query, expected):
    doc = make_doc("Tiramisu cake", "Italian dessert", ["coffee"])
    score = score_doc_keywords(query, doc)
    print(f"{query=}, {score=}")
    assert (score > 60) == expected
