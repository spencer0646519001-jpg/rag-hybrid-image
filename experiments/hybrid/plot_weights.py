# experiments/hybrid/plot_weights.py
import csv, glob, os
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "experiments" / "hybrid"

# 讀取所有掃描 CSV（包含你先前的 results_1106.csv 也一起納入）
pattern = str(OUT_DIR / "results_*.csv")
files = sorted(glob.glob(pattern))

if not files:
    raise SystemExit(f"[ERROR] 找不到任何 CSV：{pattern}")

def _to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0

rows: List[Dict] = []
for fp in files:
    with open(fp, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("query") == "__MEAN__":
                row["__file__"] = os.path.basename(fp)
                rows.append(row)

if not rows:
    raise SystemExit("[ERROR] CSV 中找不到 __MEAN__ 列，請確認先跑過評測。")

# 取資料欄位
ws, hit1, hit5, p5, mrr, ndcg, rec = [], [], [], [], [], [], []
labels = []  # 用於圖例註記

for r in rows:
    w_vec = _to_float(r.get("w_vec", "0"))
    ws.append(w_vec)
    hit1.append(_to_float(r.get("hit@1", "0")))
    hit5.append(_to_float(r.get("hit@5", "0")))
    p5.append(_to_float(r.get("precision@5", "0")))
    mrr.append(_to_float(r.get("mrr", "0")))
    ndcg.append(_to_float(r.get("ndcg@5", "0")))
    rec.append(_to_float(r.get("recall@5", "0")))
    labels.append(f"{r.get('kw_mode','?')}-{w_vec:.1f}")

# 依 w_vec 排序
order = sorted(range(len(ws)), key=lambda i: ws[i])
ws     = [ws[i] for i in order]
hit1   = [hit1[i] for i in order]
hit5   = [hit5[i] for i in order]
p5     = [p5[i] for i in order]
mrr    = [mrr[i] for i in order]
ndcg   = [ndcg[i] for i in order]
rec    = [rec[i] for i in order]

def _best_idx(vals: List[float]) -> int:
    best = max(vals)
    return vals.index(best)

# 找出各指標最佳 w_vec
best_hit1 = ws[_best_idx(hit1)]
best_ndcg = ws[_best_idx(ndcg)]
best_mrr  = ws[_best_idx(mrr)]

print("=== Best Weights (by metric) ===")
print(f"Hit@1 best w_vec: {best_hit1}")
print(f"nDCG@5 best w_vec: {best_ndcg}")
print(f"MRR best w_vec: {best_mrr}")

# 存一份總表
summary_path = OUT_DIR / "sweep_summary.csv"
with open(summary_path, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["w_vec", "hit@1", "hit@5", "recall@5", "precision@5", "mrr", "ndcg@5"])
    for i in range(len(ws)):
        w.writerow([ws[i], hit1[i], hit5[i], rec[i], p5[i], mrr[i], ndcg[i]])
print(f"[OK] Summary written → {summary_path}")

# 畫圖（每張圖一個指標；不指定顏色/樣式）
plt.figure()
plt.plot(ws, hit1, marker="o")
plt.title("Hit@1 vs w_vec")
plt.xlabel("w_vec")
plt.ylabel("Hit@1")
plt.grid(True)
plt.savefig(OUT_DIR / "weight_hit1.png", dpi=160, bbox_inches="tight")

plt.figure()
plt.plot(ws, ndcg, marker="o")
plt.title("nDCG@5 vs w_vec")
plt.xlabel("w_vec")
plt.ylabel("nDCG@5")
plt.grid(True)
plt.savefig(OUT_DIR / "weight_ndcg.png", dpi=160, bbox_inches="tight")

plt.figure()
plt.plot(ws, mrr, marker="o")
plt.title("MRR vs w_vec")
plt.xlabel("w_vec")
plt.ylabel("MRR")
plt.grid(True)
plt.savefig(OUT_DIR / "weight_mrr.png", dpi=160, bbox_inches="tight")

print("[OK] Plots saved → weight_hit1.png / weight_ndcg.png / weight_mrr.png")
