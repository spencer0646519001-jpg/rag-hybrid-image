import numpy as np

def normalize(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    norm[norm == 0] = 1
    return v / norm

# 文件向量（每列一個文件）
docs = np.array([
    [1, 0, 0],
    [0, 1, 1],
    [1, 1, 0]
], dtype=float)

# 查詢向量（一次輸入兩個）
queries = np.array([
    [1, 0, 1],
    [0, 1, 0]
], dtype=float)

# 步驟1：正規化
docs_norm = normalize(docs)
queries_norm = normalize(queries)

# 步驟2：計算所有相似度矩陣
similarity_matrix = np.dot(queries_norm, docs_norm.T)
print("相似度矩陣：\n", similarity_matrix)

# 步驟3：每筆查詢取Top-K
k = 2
topk_idx = np.argsort(-similarity_matrix, axis=1)[:, :k]

# 印結果
for q_idx, top_docs in enumerate(topk_idx):
    print(f"\n查詢 {q_idx} 的Top-{k}:")
    for doc_idx in top_docs:
        score = similarity_matrix[q_idx, doc_idx]
        print(f"  Doc {doc_idx}: 相似度 = {score:.3f}")
