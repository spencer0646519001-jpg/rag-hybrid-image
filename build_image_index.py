# build_image_index.py
from pathlib import Path
import json
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# === 路徑設定 ===
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
IMG_DIR = DATA_DIR / "images"
INDEX_PATH = DATA_DIR / "image_index.json"
EMB_PATH = DATA_DIR / "img_embeddings.npy"

# === 主要函式 ===
def build_image_index():
    """掃描 data/images，生成 image_index.json + img_embeddings.npy"""
    IMG_DIR.mkdir(exist_ok=True)
    files = sorted([p for p in IMG_DIR.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    if not files:
        print("⚠️ 沒找到圖片，請放入 data/images/")
        return

    # 建立索引
    index = [{"id": f"img{i+1}", "path": str(p.name)} for i, p in enumerate(files)]
    INDEX_PATH.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ 已建立索引：{INDEX_PATH.name} ({len(index)} 張圖)")

    # 載入 CLIP 模型
    model = SentenceTransformer("clip-ViT-B-32")

    # 生成圖片向量
    embeddings = []
    for p in tqdm(files, desc="Embedding 圖片中"):
        img = Image.open(p).convert("RGB")
        emb = model.encode(img, convert_to_numpy=True, normalize_embeddings=True)
        embeddings.append(emb)

    arr = np.vstack(embeddings).astype(np.float32)
    np.save(EMB_PATH, arr)
    print(f"✅ 已儲存向量檔：{EMB_PATH.name}  shape={arr.shape}")

if __name__ == "__main__":
    build_image_index()
