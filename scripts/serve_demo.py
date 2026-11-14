"""Local-only demo: live template inference and explicitly labelled saved DINO scores."""
import argparse
import base64
import csv
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import sys
import tempfile
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PAGE = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>工业产品异常检测演示</title>
<style>*{box-sizing:border-box}body{font:16px system-ui;background:#f1f5f9;color:#0f172a;margin:0}main{max-width:1180px;margin:auto;padding:28px}h1{margin-bottom:6px}.note{color:#475569;line-height:1.6}section{background:white;padding:22px;border-radius:14px;margin-top:18px;box-shadow:0 2px 8px #0f172a0a}label{display:inline-block;margin-right:16px;margin-bottom:14px}select,input,button{font:inherit;padding:9px;border:1px solid #cbd5e1;border-radius:7px;max-width:100%}button{background:#1d4ed8;color:white;cursor:pointer}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:15px}img{width:100%;object-fit:contain;max-height:370px;background:#eee;border-radius:8px}.scores{display:flex;gap:24px;flex-wrap:wrap}.scores>div{padding:16px;background:#eff6ff;border-radius:10px;flex:1;min-width:270px}strong{font-size:21px}#message{padding:10px;color:#b45309}small{color:#475569}@media(max-width:700px){.grid{grid-template-columns:1fr}}</style>
<main><h1>工业产品异常检测</h1><div class="note">Part 1 实时模板差分 · Part 2 已保存结果对照。两种方法的分数单位不同。图像标签不参与实时预测。</div>
<section><label>产品类别 <select id="category"></select></label><label>正常参考数 <select id="shot"><option>1</option><option>2</option><option>4</option><option>8</option></select></label><label>参考组 <select id="seed"><option>0</option><option>1</option><option>2</option></select></label><br>
<label>测试样例 <select id="sample"></select></label><button id="run">检测样例</button><hr><label>或上传相同拍摄规格的产品图 <input id="upload" type="file" accept="image/*"></label><button id="uploaded">检测上传图片</button><div id="message"></div></section>
<section id="result" hidden><div class="scores"><div><b>Part 1 · 本次实时推理</b><p><strong id="a"></strong></p><div id="ascore"></div></div><div><b>Part 2 · 历史结果回放</b><p><strong id="b"></strong></p><div id="bscore"></div><small>这里没有实时调用 DINOv2；任意上传图片没有对应历史结果。</small></div></div>
<div class="grid"><div><h3>输入图片</h3><img id="input"></div><div><h3>Part 1 异常热图</h3><img id="heat"></div><div><h3>Part 1 叠加图</h3><img id="overlay"></div></div><p class="note">颜色仅用于显示，不改变异常分数或评估数据。OK/NG 由独立正常校准图的 95% 分位阈值决定。</p></section></main>
<script>const $=id=>document.getElementById(id);let info;async function samples(){let r=await fetch('/samples?category='+encodeURIComponent($('category').value));let x=await r.json();$('sample').replaceChildren(...x.map(p=>new Option(p,p)))}
async function predict(upload){$('message').textContent='正在检测；首次配置需要建立参考模板并计算正常校准阈值…';$('run').disabled=$('uploaded').disabled=true;try{let p={category:$('category').value,shot:+$('shot').value,seed:+$('seed').value};if(upload){let f=$('upload').files[0];if(!f)throw Error('请先选择图片');p.image=await new Promise((ok,no)=>{let r=new FileReader();r.onload=()=>ok(r.result.split(',')[1]);r.onerror=no;r.readAsDataURL(f)})}else p.image_path=$('sample').value;
let r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)});let x=await r.json();if(!r.ok)throw Error(x.error);$('result').hidden=false;$('a').textContent=x.part1.decision;$('ascore').textContent=`异常分数 ${x.part1.score.toFixed(5)} · 阈值 ${x.part1.threshold.toFixed(5)}`;$('b').textContent=x.part2?x.part2.decision:'无对应历史结果';$('bscore').textContent=x.part2?`已保存分数 ${x.part2.score.toFixed(5)} · 阈值 ${x.part2.threshold.toFixed(5)}`:'';for(let k of ['input','heat','overlay'])$(k).src='data:image/png;base64,'+x[k];$('message').textContent='检测完成';}catch(e){$('message').textContent=e.message}finally{$('run').disabled=$('uploaded').disabled=false}}
fetch('/info').then(r=>r.json()).then(x=>{info=x;$('category').replaceChildren(...x.categories.map(c=>new Option(c,c)));$('category').value='cashew';samples()});$('category').onchange=samples;$('run').onclick=()=>predict(false);$('uploaded').onclick=()=>predict(true);</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    import cv2
    import numpy as np
    from PIL import Image
    from src.data.visa import VisAReader
    from src.part1.inspector import TemplateInspector
    cv2.setNumThreads(2)
    manifest = json.loads((ROOT / "manifests/visa_1cls.json").read_text())
    reader = VisAReader(args.source, manifest)
    with (ROOT / "results/tables/part2_gpu_predictions.csv").open(newline="") as handle:
        b_rows = {(r["category"], int(r["shot"]), int(r["seed"]), r["image_path"]): r for r in csv.DictReader(handle)}
    cached = {}
    def png(array):
        buffer = io.BytesIO(); Image.fromarray(array).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    class Handler(BaseHTTPRequestHandler):
        def send(self, data, status=200, html=False):
            body = data.encode() if html else json.dumps(data).encode()
            self.send_response(status); self.send_header("Content-Type", "text/html; charset=utf-8" if html else "application/json")
            self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)
        def do_GET(self):
            route = urlparse(self.path)
            if route.path == "/":
                self.send(PAGE, html=True)
            elif route.path == "/info":
                self.send({"categories": list(manifest["categories"]), "part2_mode": "saved score replay"})
            elif route.path == "/samples":
                category = parse_qs(route.query).get("category", [""])[0]
                self.send([r["image_path"] for r in manifest["categories"].get(category, {}).get("tests", [])])
            else:
                self.send({"error": "Not found"}, 404)
        def do_POST(self):
            try:
                if self.path != "/predict":
                    raise ValueError("Unknown route")
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16*1024**2:
                    raise ValueError("图片请求过大")
                request = json.loads(self.rfile.read(size))
                c, k, s = request["category"], int(request["shot"]), int(request["seed"])
                if c not in manifest["categories"] or k not in manifest["shots"] or s not in manifest["seeds"]:
                    raise ValueError("未知产品配置")
                key = (c,k,s)
                if key not in cached:
                    if len(cached) >= 2:
                        cached.pop(next(iter(cached)))
                    entry = manifest["categories"][c]
                    inspector = TemplateInspector()
                    inspector.fit_references([reader.path(p) for p in entry["references"][str(k)][str(s)]], s)
                    threshold = float(np.quantile([inspector.predict(reader.path(p),False)["score"] for p in entry["calibration"]],.95,method="linear"))
                    cached[key] = inspector, threshold
                inspector, threshold = cached[key]
                relative = request.get("image_path")
                with tempfile.TemporaryDirectory(prefix="cv_demo_") as temporary:
                    if "image" in request:
                        if relative:
                            raise ValueError("Choose sample OR upload")
                        raw = base64.b64decode(request["image"], validate=True)
                        path = Path(temporary) / "upload.png"
                        with Image.open(io.BytesIO(raw)) as image:
                            if image.width*image.height > 16_000_000:
                                raise ValueError("图片像素过多")
                            image.convert("RGB").save(path)
                    else:
                        if relative not in {r["image_path"] for r in manifest["categories"][c]["tests"]}:
                            raise ValueError("Unknown sample")
                        path = reader.path(relative)
                    prediction = inspector.predict(path)
                    with Image.open(path) as image:
                        rgb = np.asarray(image.convert("RGB"))
                heat = prediction["anomaly_map"]
                display = np.clip((heat-heat.min())/(float(np.ptp(heat)) or 1)*255,0,255).astype(np.uint8)
                colored = cv2.cvtColor(cv2.applyColorMap(display,cv2.COLORMAP_INFERNO),cv2.COLOR_BGR2RGB)
                b = b_rows.get((c,k,s,relative)) if relative else None
                self.send({"part1":{"score":prediction["score"],"threshold":threshold,"decision":"NG" if prediction["score"]>threshold else "OK"},
                           "part2":{"score":float(b["score"]),"threshold":float(b["threshold"]),"decision":b["decision"],"mode":"saved score replay"} if b else None,
                           "input":png(rgb),"heat":png(colored),"overlay":png((.6*rgb+.4*colored).astype(np.uint8))})
            except Exception as exc:
                self.send({"error": str(exc)},400)
    print(f"DEMO http://127.0.0.1:{args.port} | live Part 1; saved Part 2 scores only", flush=True)
    HTTPServer(("127.0.0.1",args.port),Handler).serve_forever()


if __name__ == "__main__":
    main()
