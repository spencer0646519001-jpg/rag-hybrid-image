# server.py
from __future__ import annotations

from typing import List, Optional, Literal, Tuple, Dict, Any
from collections import Counter
from pathlib import Path
import json, os, re

from fastapi import FastAPI, Query, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from PIL import Image
import imagehash

# -------- env & optional OpenAI --------
from dotenv import load_dotenv
load_dotenv()

try:
    from openai import OpenAI as _OpenAI
except Exception:  # 若 openai 套件不存在，也不影響其餘 API
    _OpenAI = None

_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
_OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# -------- sentence split for extractive fallback --------
_SENT_SPLIT = re.compile(r"[。\.！!?！？]\s*")

def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text or "") if s.strip()]

# -------- project core imports --------
# 這些來自你的 main.py（不改名）
from main import (
    DOCS, normalize, score_doc,
    vector_search, hybrid_hits, ensure_embeddings, embed_texts
)

# -------- FastAPI app --------
app = FastAPI(title="RAG MVP API", version="0.9")

# CORS：前端常見本機埠
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500", "http://localhost:5500",
        "http://127.0.0.1:3000", "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Pydantic Models ----------
class SearchHit(BaseModel):
    id: str
    title: str
    snippet: str
    tags: List[str]
    lang: str
    score: float
    matched_on: str

class SearchResponse(BaseModel):
    total: int
    items: List[SearchHit]

class LangCount(BaseModel):
    lang: str
    count: int

class TagCount(BaseModel):
    tag: str
    count: int

# ---------- utils ----------
def parse_multi(s: Optional[str]) -> List[str]:
    if not s:
        return []
    s = s.replace("，", ",")
    return [x.strip() for x in s.split(",") if x.strip()]

def pick_score_for_sort(h: dict, order: Literal["max","avg"]) -> float:
    return float(h.get("score_max" if order == "max" else "score_avg",
                       h.get("score", 0.0)))

def filter_pool(lang: Optional[str], tags: List[str], mode: Literal["and","or"]) -> List[dict]:
    pool = DOCS
    if lang:
        pool = [d for d in pool if d.get("lang") == lang]
    if tags:
        if mode == "and":
            pool = [d for d in pool if all(t in (d.get("tags") or []) for t in tags)]
        else:
            pool = [d for d in pool if any(t in (d.get("tags") or []) for t in tags)]
    return pool

# ---------- health / embeddings ----------
@app.get("/health")
def health():
    ok_emb = False
    try:
        _, _ = ensure_embeddings()
        ok_emb = True
    except Exception:
        ok_emb = False
    return {"ok": True, "docs": len(DOCS), "embeddings_ready": ok_emb}

@app.post("/embed/rebuild")
def embed_rebuild():
    ensure_embeddings()
    return {"ok": True, "message": "embeddings rebuilt"}

# ---------- stats ----------
@app.get("/languages", response_model=List[LangCount])
def languages():
    c = Counter(d.get("lang","unk") for d in DOCS)
    items = sorted(c.items(), key=lambda x:(-x[1], x[0]))
    return [{"lang":k, "count":v} for k,v in items]

@app.get("/tags", response_model=List[TagCount])
def tags(lang: Optional[str] = Query(None)):
    pool = DOCS if not lang else [d for d in DOCS if d.get("lang")==lang]
    c = Counter(t for d in pool for t in (d.get("tags") or []))
    items = sorted(c.items(), key=lambda x:(-x[1], x[0]))
    return [{"tag":k, "count":v} for k,v in items]

# ---------- search ----------
@app.get("/search", response_model=SearchResponse)
def search_api(
    q: str = Query(..., min_length=1),
    min_score: int = Query(20, ge=0, le=120),
    lang: Optional[str] = None,
    tag: Optional[str] = None,
    tag_mode: Literal["and","or"] = "or",
    offset: int = Query(0, ge=0),
    limit: int = Query(5, ge=1, le=50),
    order: Literal["max","avg"] = "max",
    direction: Literal["desc","asc"] = "desc",
    mode: Literal["text","vector","hybrid"] = "hybrid",
    w_text: float = Query(0.5, ge=0.0, le=1.0),
    w_vec:  float = Query(0.5, ge=0.0, le=1.0),
):
    tags = parse_multi(tag)
    pool = filter_pool(lang, tags, tag_mode)

    if mode == "text":
        raw = [score_doc(q, d) for d in pool]
        for h in raw:
            h["score_use"] = pick_score_for_sort(h, order)
        reverse = (direction == "desc")
        hits_sorted = sorted(raw, key=lambda x:x["score_use"], reverse=reverse)
        hits_filtered = [h for h in hits_sorted if h["score_use"] >= float(min_score)]
        page = hits_filtered[offset: offset+limit]
        items = [SearchHit(**{
            "id":h["id"], "title":h["title"], "snippet":h.get("snippet",""),
            "tags":h.get("tags",[]), "lang":h.get("lang","unk"),
            "score":float(h["score_use"]), "matched_on":h.get("matched_on","all")
        }) for h in page]
        return SearchResponse(total=len(hits_filtered), items=items)

    if mode == "vector":
        mat, ids = ensure_embeddings()
        idx_map = {id_: i for i, id_ in enumerate(ids)}
        pool_rows = [idx_map[d["id"]] for d in pool if d["id"] in idx_map]
        if not pool_rows:
            return SearchResponse(total=0, items=[])
        qv = embed_texts([normalize(q)])
        sims = (qv @ mat.T)[0]             # cosine (已正規化)
        vec_pairs = [(r, float(sims[r])) for r in pool_rows]
        vec_pairs.sort(key=lambda x: x[1], reverse=True)
        page_pairs = vec_pairs[offset: offset+limit]
        items = []
        for row, sc in page_pairs:
            d = DOCS[row]
            items.append(SearchHit(
                id=d["id"], title=d["title"], snippet=d.get("snippet",""),
                tags=d.get("tags",[]), lang=d.get("lang","unk"),
                score=float(sc), matched_on="vector"
            ))
        return SearchResponse(total=len(pool_rows), items=items)

    # hybrid
    hits = hybrid_hits(q, w_text=w_text, w_vec=w_vec, top_k=len(DOCS))
    pool_ids = {d["id"] for d in pool}
    picks = [(i, f, kw, ve) for (i,f,kw,ve) in hits if DOCS[i]["id"] in pool_ids]
    picks = [p for p in picks if p[1] >= float(min_score)/120.0]
    total = len(picks)
    page = picks[offset: offset+limit]
    items = []
    for i, f, kw, ve in page:
        d = DOCS[i]
        items.append(SearchHit(
            id=d["id"], title=d["title"], snippet=d.get("snippet",""),
            tags=d.get("tags",[]), lang=d.get("lang","unk"),
            score=float(f*120.0),
            matched_on=f"hybrid(kw={kw:.2f},vec={ve:.2f})"
        ))
    return SearchResponse(total=total, items=items)

# ---------- OpenAI helper ----------
def _llm_generate_answer(question: str, contexts: List[str], max_chars: int,
                         allow_openbook: bool = False, debug: bool = False) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    有金鑰就用 OpenAI 生成；沒有金鑰則回傳 (None, dbg) 讓上層走備援。
    如果 allow_openbook=False 且 contexts 為空，會請模型回答「不知道」。
    """
    dbg: Dict[str, Any] = {"ctx_count": len(contexts)}
    if not (_OpenAI and _OPENAI_API_KEY):
        return None, dbg

    try:
        client = _OpenAI(api_key=_OPENAI_API_KEY)

        # 組合 context 區塊
        ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
        # 嚴格 / 開卷提示
        if contexts or not allow_openbook:
            system_txt = (
                "You are a concise retrieval QA assistant.\n"
                "Answer ONLY using the provided context. "
                "If the answer is not in the context, say you don't know."
            )
        else:
            system_txt = (
                "You are a concise general-knowledge assistant. "
                "If relevant context is empty, you may answer from your general knowledge."
            )

        prompt = (
            f"Question: {question}\n\n"
            f"Context:\n{ctx if contexts else '(empty)'}\n\n"
            f"Write the answer in the same language as the question. "
            f"Keep it under {max_chars} characters."
        )
        if debug:
            dbg["prompt_head"] = system_txt
            dbg["prompt_tail"] = prompt

        resp = client.chat.completions.create(
            model=_OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_txt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        txt = (resp.choices[0].message.content or "").strip()
        if debug:
            dbg["model"] = getattr(resp, "model", _OPENAI_MODEL)
            dbg["finish_reason"] = getattr(getattr(resp.choices[0], "finish_reason", None), "value", None)
        return (txt or None), dbg

    except Exception as e:
        if debug:
            dbg["llm_error"] = str(e)
        return None, dbg

# ---------- /answer ----------
@app.get("/answer")
def answer_api(
    q: str = Query(..., min_length=1, description="問題/查詢"),
    lang: str = Query("zh", description="語料語言(zh/en/ja)"),
    top_k: int = Query(3, ge=1, le=10),
    max_chars: int = Query(300, ge=50, le=1200),
    order: Literal["max", "avg"] = Query("max", description="取證據時的排序依據"),
    use_llm: bool = Query(True, description="有金鑰時是否使用 LLM 生成"),
    openbook: bool = Query(False, description="找不到內容時，是否允許用一般知識回答"),
    debug: bool = Query(False, description="是否回傳除錯資訊"),
):
    # 1) 打分 → 取證據
    pool = [d for d in DOCS if (not lang or d.get("lang") == lang)]
    hits = [score_doc(q, d) for d in pool]
    for h in hits:
        h["score_use"] = pick_score_for_sort(h, order)
    hits.sort(key=lambda x: x["score_use"], reverse=True)
    sel = hits[:top_k]

    # 2) 準備 context（title + snippet）
    contexts: List[str] = []
    for h in sel:
        title = h.get("title", "").strip()
        snip  = h.get("snippet", "").strip()
        combined = f"{title}\n{snip}".strip()
        if combined:
            contexts.append(combined)

    # 打分資訊（debug 用）
    best = round(float(sel[0]["score_use"]), 2) if sel else 0.0
    score_gate = best >= 60.0  # 你可以調整門檻
    openbook_used = False

    # 3) LLM
    answer_text = None
    dbg: Dict[str, Any] = {}
    if use_llm:
        # 若門檻太低、或真的沒有 context，且 openbook=True，就允許一般知識
        allow_openbook = openbook and (not contexts or not score_gate)
        if allow_openbook:
            openbook_used = True
        answer_text, dbg = _llm_generate_answer(
            question=q, contexts=contexts if not allow_openbook else [],
            max_chars=max_chars, allow_openbook=allow_openbook, debug=debug
        )

    # 4) 抽取式備援
    if not answer_text:
        q_norm = normalize(q)
        terms = [t for t in re.split(r"[,\s]+", q_norm) if t]
        picked: List[str] = []
        for h in sel:
            for s in _split_sentences(h.get("snippet", "")):
                s_norm = normalize(s)
                if s and any(t in s_norm for t in terms):
                    if s not in picked:
                        picked.append(s)
        answer_text = "。".join(picked).strip() or "(no answer)"
        if len(answer_text) > max_chars:
            answer_text = answer_text[:max_chars].rstrip("，、。；;,. ") + "…"

    # 5) citations
    cits = [
        {
            "id": h["id"],
            "title": h["title"],
            "score": round(float(h.get("score_use", h.get("score", 0.0))), 1),
        }
        for h in sel
    ]

    # 6) 組回應（debug 可選）
    base = {
        "answer": answer_text,
        "citations": cits,
        "total_candidates": len(hits),
        "used_llm": bool(answer_text and use_llm and _OpenAI and _OPENAI_API_KEY),
    }
    if debug:
        base["debug"] = {
            "contexts_count": len(contexts),
            "contexts_preview": contexts[:1],
            "llm": dbg if dbg else None,
            "openbook": openbook_used,
            "best_score": best,
            "score_gate": score_gate,
        }
    return base

# ---------- 圖片搜尋（pHash） ----------
IMAGES_DIR = (Path(__file__).parent / "images")
IMAGES_DIR.mkdir(exist_ok=True)
_hash_cache: Dict[Path, Any] = {}

def _img_hash(path: Path):
    if path in _hash_cache:
        return _hash_cache[path]
    try:
        h = imagehash.phash(Image.open(path))
        _hash_cache[path] = h
        return h
    except Exception:
        return None

@app.post("/image/search")
async def image_search(file: UploadFile = File(...), limit: int = 8):
    up = Image.open(file.file).convert("RGB")
    hq = imagehash.phash(up)
    candidates = []
    for p in IMAGES_DIR.glob("*.*"):
        hp = _img_hash(p)
        if hp is None:
            continue
        dist = hq - hp
        candidates.append({"path": p.name, "distance": int(dist)})
    candidates.sort(key=lambda x: x["distance"])
    return {"items": candidates[:limit]}

# ---------- 資料管理 ----------
@app.post("/docs/upload")
async def docs_upload(file: UploadFile = File(...)):
    """
    上傳 JSON（list[dict]，含 id/title/snippet/tags/lang）覆蓋 data/desserts.json 並重建向量。
    """
    try:
        content = await file.read()
        items = json.loads(content.decode("utf-8"))
        if not isinstance(items, list):
            raise HTTPException(400, "JSON 應為陣列")
        need = {"id","title","snippet","tags","lang"}
        for i, d in enumerate(items):
            if not need.issubset(d.keys()):
                raise HTTPException(400, f"第 {i} 筆缺欄位（需含 {need}）")
        (Path(__file__).parent / "data" / "desserts.json").write_text(
            json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        ensure_embeddings()
        return {"ok": True, "count": len(items)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"處理失敗：{e}")
