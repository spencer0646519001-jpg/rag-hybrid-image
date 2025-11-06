import React, { useMemo, useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Search, Image as ImageIcon, Upload, Wand2, Sparkles, RefreshCw, Link2, FileJson, Camera, TextSearch } from "lucide-react";

// ultra‑clean primitives
const Button = ({ className = "", children, ...props }) => (
  <button className={`px-3 py-2 rounded-xl border border-black/10 hover:border-black/20 active:scale-[.99] transition ${className}`} {...props}>{children}</button>
);
const Input = ({ className = "", ...props }) => (
  <input className={`w-full px-4 py-2 rounded-xl bg-white/70 backdrop-blur border border-black/10 focus:outline-none focus:ring-2 focus:ring-amber-200 ${className}`} {...props} />
);
const Chip = ({ children }) => (
  <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs bg-black/5">{children}</span>
);

// mock data
const DOCS = [
  { id: "d1", title: "抹茶草莓塔的穩定出品參數", url: "https://example.com/matcha-tart", tags: ["塔皮","抹茶","草莓","乳化","溫度曲線"], snippet: "塔皮水活性控制在 0.55–0.60；甘納許乳化最佳溫域 34–36°C。若草莓含水高，建議刷可可脂薄層降低滲透。", score: 0.93 },
  { id: "d2", title: "焦糖布丁孔洞的根因分析", url: "https://example.com/creme-caramel", tags: ["布丁","蒸烤","氣泡","孔洞","低溫長時"], snippet: "孔洞多半來自過度攪拌或過高溫蒸烤。建議 150°C、45 分鐘，水浴水面 80–85°C，液體過篩兩次。", score: 0.88 },
  { id: "d3", title: "覆盆子馬卡龍殼裂的監測清單", url: "https://example.com/macarons", tags: ["馬卡龍","龜裂","乾燥時間","烘烤曲線"], snippet: "環境濕度 > 60% 時需延長表皮乾燥到 30–45 分鐘；使用 135–150°C 上下火，視顯著足修正。", score: 0.82 }
];
const IMAGES = [
  { id: "i1", alt: "抹茶塔表面微裂紋", url: "https://images.unsplash.com/photo-1546549039-49eade7bfe4a?q=80&w=1200&auto=format&fit=crop" },
  { id: "i2", alt: "蛋奶凍氣泡孔洞", url: "https://images.unsplash.com/photo-1541782814452-d5f8c3dfc750?q=80&w=1200&auto=format&fit=crop" },
  { id: "i3", alt: "馬卡龍殼裂紋", url: "https://images.unsplash.com/photo-1551022370-0eb3d664b9dc?q=80&w=1200&auto=format&fit=crop" }
];

export default function App() {
  const [deg, setDeg] = useState(32);
  const [tab, setTab] = useState("text");
  const [query, setQuery] = useState("");
  const [minScore, setMinScore] = useState(0);
  const [tags, setTags] = useState([]);
  const [images, setImages] = useState([]);
  const [uploadedUrl, setUploadedUrl] = useState(null);

  useEffect(() => {
    const id = setInterval(() => setDeg(d => (d + 0.3) % 360), 50);
    return () => clearInterval(id);
  }, []);

  const TAG_POOL = useMemo(() => Array.from(new Set(DOCS.flatMap(d => d.tags))).sort(), []);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    let arr = DOCS.filter(d => d.score >= minScore);
    if (tags.length) arr = arr.filter(d => tags.every(t => d.tags.includes(t)));
    if (!q) return arr;
    return arr.filter(d => d.title.toLowerCase().includes(q) || d.snippet.toLowerCase().includes(q) || d.tags.join(" ").toLowerCase().includes(q));
  }, [query, minScore, tags]);

  const onUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith("image/")) return alert("僅支援圖片");
    if (file.size > 8 * 1024 * 1024) return alert("圖片過大（>8MB）");
    const url = URL.createObjectURL(file);
    setUploadedUrl(prev => { if (prev) URL.revokeObjectURL(prev); return url; });
    setImages(IMAGES);
    setTab("image");
  };

  return (
    <div style={{minHeight:'100vh', background:`linear-gradient(${deg}deg, #FFF7D1 0%, #FFD66B 40%, #FFBD4A 100%)`}} className="transition-colors">
      <div className="max-w-5xl mx-auto px-4 py-8">
        <header className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-3">
            <motion.div initial={{opacity:0, y:-6}} animate={{opacity:1, y:0}} className="p-2 rounded-xl bg-black text-white">
              <Sparkles className="w-5 h-5" />
            </motion.div>
            <h1 className="text-2xl font-semibold tracking-tight">RAG 甜點知識庫</h1>
          </div>
          <div className="text-xs text-black/60">MVP · 文字×圖像檢索</div>
        </header>

        {/* search bar */}
        <div className="rounded-2xl bg-white/70 backdrop-blur shadow-sm p-3 flex items-center gap-2">
          <Search className="w-5 h-5 text-black/60"/>
          <Input value={query} onChange={e=>setQuery(e.target.value)} placeholder="輸入關鍵字（例：抹茶塔 裂痕 乳化）"/>
          <Button onClick={()=>{ /* mock */ }} className="bg-black text-white border-black">檢索</Button>
        </div>

        {/* tabs */}
        <div className="mt-6 flex items-center gap-2">
          {[
            {k:'text', label:'文字檢索', icon:<TextSearch className="w-4 h-4"/>},
            {k:'image', label:'圖像檢索', icon:<ImageIcon className="w-4 h-4"/>},
            {k:'data', label:'來源', icon:<FileJson className="w-4 h-4"/>},
          ].map(t => (
            <button key={t.k} onClick={()=>setTab(t.k)} className={`px-3 py-1.5 rounded-xl border border-black/10 bg-white/60 backdrop-blur flex items-center gap-2 text-sm ${tab===t.k? 'outline outline-2 outline-black':'hover:border-black/20'}`}>
              {t.icon}{t.label}
            </button>
          ))}
        </div>

        {/* simple filters */}
        {tab==='text' && (
          <div className="mt-4 grid gap-3 md:grid-cols-[1fr,auto]">
            <div className="flex flex-wrap gap-2">
              {TAG_POOL.map(t => (
                <button key={t} onClick={()=>setTags(prev=> prev.includes(t)? prev.filter(x=>x!==t): [...prev, t])} className={`px-2.5 py-1 rounded-full text-xs border ${tags.includes(t)?'bg-black text-white border-black':'bg-white/70 border-black/10 hover:border-black/20'}`}>{t}</button>
              ))}
            </div>
            <div className="flex items-center gap-2 text-sm">
              <span className="text-black/60">分數</span>
              <input type="range" min={0} max={1} step={0.01} value={minScore} onChange={e=>setMinScore(Number(e.target.value))} />
              <span className="text-black/60 w-10 tabular-nums text-right">{minScore.toFixed(2)}</span>
            </div>
          </div>
        )}

        {/* content */}
        <div className="mt-6">
          {tab==='text' && (
            <div className="grid md:grid-cols-2 gap-4">
              {filtered.map(d => (
                <div key={d.id} className="rounded-2xl bg-white/80 backdrop-blur p-4 shadow-sm">
                  <div className="text-sm text-black/60">score {d.score.toFixed(2)}</div>
                  <div className="mt-1 font-medium leading-tight">{d.title}</div>
                  <p className="mt-2 text-sm text-black/80 leading-relaxed">{d.snippet}</p>
                  <div className="mt-3 flex flex-wrap gap-2">
                    {d.tags.map(t => <Chip key={t}>{t}</Chip>)}
                  </div>
                  <div className="mt-3 flex items-center gap-2 text-sm">
                    <a className="inline-flex items-center gap-1 text-black/80 underline decoration-black/30 underline-offset-4" href={d.url} target="_blank" rel="noreferrer"><Link2 className="w-3 h-3"/>來源</a>
                    <Button className="text-xs">加入上下文</Button>
                  </div>
                </div>
              ))}
            </div>
          )}

          {tab==='image' && (
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <label className="inline-flex items-center gap-2 px-3 py-2 rounded-xl border border-black/10 bg-white/70 backdrop-blur cursor-pointer"><Upload className="w-4 h-4"/>上傳圖片
                  <input type="file" accept="image/*" onChange={onUpload} className="hidden"/>
                </label>
                <Button onClick={()=>setImages(IMAGES)} className="bg-white/70">使用範例影像</Button>
              </div>

              {uploadedUrl && (
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="rounded-2xl bg-white/80 p-4 backdrop-blur">
                    <div className="text-sm font-medium mb-2 inline-flex items-center gap-2"><Camera className="w-4 h-4"/>你的上傳</div>
                    <img src={uploadedUrl} alt="uploaded" className="w-full rounded-xl"/>
                  </div>
                  <div className="rounded-2xl bg-white/80 p-4 backdrop-blur">
                    <div className="text-sm font-medium mb-2 inline-flex items-center gap-2"><Wand2 className="w-4 h-4"/>系統判讀（mock）</div>
                    <div className="text-sm text-black/80 space-y-1">
                      <div>偵測到：<Chip>裂紋</Chip> <Chip>鏡面不均</Chip></div>
                      <div>建議檢索詞：<Chip>馬卡龍 殼裂</Chip> <Chip>釉 鏡面</Chip></div>
                      <div className="text-black/60">信心：0.79</div>
                    </div>
                  </div>
                </div>
              )}

              <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {images.map(img => (
                  <div key={img.id} className="rounded-2xl bg-white/80 p-3 backdrop-blur">
                    <img src={img.url} alt={img.alt} className="w-full h-40 object-cover rounded-xl"/>
                    <div className="pt-2 text-sm text-black/80">{img.alt}</div>
                    <div className="pt-2 flex gap-2">
                      <Button className="text-xs">以圖搜文</Button>
                      <Button className="text-xs">加入上下文</Button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {tab==='data' && (
            <div className="rounded-2xl bg-white/80 p-4 backdrop-blur">
              <div className="text-sm font-medium mb-2 inline-flex items-center gap-2"><FileJson className="w-4 h-4"/>暫存的 RAG 上下文（mock）</div>
              <ul className="list-disc pl-5 text-sm text-black/80 space-y-1">
                <li>抹茶塔乳化與草莓含水處理（d1）</li>
                <li>蛋奶凍孔洞根因（d2）</li>
              </ul>
              <div className="mt-3 flex gap-2">
                <Button className="text-xs"><RefreshCw className="w-3 h-3 mr-1 inline"/>重新生成</Button>
                <Button className="text-xs bg-black text-white border-black"><Sparkles className="w-3 h-3 mr-1 inline"/>導出 SOP</Button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
