import numpy as np
from pathlib import Path
import sys


# ---- 找到專案根目錄（有 main.py 的那層）----
THIS = Path(__file__).resolve()
ROOT = next(p for p in THIS.parents if (p / "main.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 建立兩個向量（假設是 embedding）
A = np.array([0.9, 0.1, 0.3])
B = np.array([0.8, 0.2, 0.4])
C = np.array([-0.9, -0.1, -0.3])  # 反方向的例子

# 餘弦相似度函數
def cosine_similarity(a, b):
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot / (norm_a * norm_b)

print("A vs B:", cosine_similarity(A, B))
print("A vs C:", cosine_similarity(A, C))
