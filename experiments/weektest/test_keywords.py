# test/weektest/test_keywords.p
from main import score_doc_keywords   # 只引入產品功能
from conftest import make_doc         # 引入測試 helper（或直接用 pytest 自動可見）

def test_partial_ratio_higher_for_substring():
    d = make_doc("tiramisu classic cake")
    s1 = score_doc_keywords("tira", d, agg="max")
    s2 = score_doc_keywords("cake", d, agg="max")
    assert s1 >= s2

def test_exact_match_gets_120():
    d = make_doc("tiramisu")
    assert score_doc_keywords("tiramisu", d) >= 120.0

def test_avg_is_stricter_than_max():
    d = make_doc("new york cheesecake baked")
    q = "new cheesecake"
    s_max = score_doc_keywords(q, d, agg="max")
    s_avg = score_doc_keywords(q, d, agg="avg")
    assert s_max >= s_avg
