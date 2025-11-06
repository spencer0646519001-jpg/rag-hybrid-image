import os
import sys
import matplotlib.pyplot as plt

# ✅ 自動尋找並加入 main.py 所在的根路徑
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))  # 回到 rag.project 根目錄
if project_root not in sys.path:
    sys.path.append(project_root)

from main import search, refresh_docs

# ✅ 確保載入最新資料與 embeddings
refresh_docs()

# 查詢關鍵字
query = "草莓"

# 不同權重組合 (w_text, w_vec)
weights = [
    (1.0, 0.0),
    (0.8, 0.2),
    (0.5, 0.5),
    (0.2, 0.8),
    (0.0, 1.0)
]

titles, finals, text_scores, vec_scores = [], [], [], []

# 🔍 逐一測試不同權重下的搜尋結果
for w_text, w_vec in weights:
    results = search(query, mode="hybrid", w_text=w_text, w_vec=w_vec, top_k=3)
    top = results[0]
    print(f"w_text={w_text:.1f}, w_vec={w_vec:.1f} → Top1: {top['title']} (score={top['score_final']:.3f})")

    titles.append(top["title"])
    finals.append(top["score_final"])
    text_scores.append(top.get("score_text", 0.0))
    vec_scores.append(top.get("score_vector", 0.0))


# 🧭 顯示文字結果
print("\n=== Hybrid 搜尋結果對照 ===")
for i, (w_t, w_v) in enumerate(weights):
    print(f"{i+1}. w_text={w_t}, w_vec={w_v} → {titles[i]}")

# 📊 視覺化折線圖
x_labels = [f"{w_t}/{w_v}" for w_t, w_v in weights]

plt.figure(figsize=(8, 5))
plt.plot(x_labels, text_scores, "o--", label="Text score")
plt.plot(x_labels, vec_scores, "s--", label="Vector score")
plt.plot(x_labels, finals, "^-", label="Final (Hybrid)")
plt.title("Hybrid 搜尋權重變化實驗")
plt.xlabel("w_text / w_vec")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
