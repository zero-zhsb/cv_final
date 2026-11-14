"""Create offline viewer, planned narration/subtitles, timeline and a source ZIP."""
from pathlib import Path
import csv
import hashlib
import json
import re
import shutil
import textwrap
import zipfile

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'presentation/midterm_video'


def stamp(seconds, srt=False):
    millis=round(seconds*1000);minutes,tail=divmod(millis,60000);secs,ms=divmod(tail,1000)
    return f'00:{minutes:02}:{secs:02},{ms:03}' if srt else f'{minutes:02}:{secs:02}'


def main():
    segs=json.loads((BASE/'script_segments.json').read_text())
    assert segs[0]['start']==0 and segs[-1]['end']==280
    assert all(a['end']==b['start'] for a,b in zip(segs,segs[1:]))
    total=0;md=['# English narration — planned 4:40 cut','',
        'Timing is an editing plan, not a measured recording. Rehearse and retime subtitles after recording. Choose this OR the Chinese narration, not both.','']
    srt=[];counter=0
    with (BASE/'timeline.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['start','end','duration_seconds','speaker','asset','section']);writer.writeheader()
        for s in segs:
            words=s['text'].split();total+=len(words)
            writer.writerow({'start':stamp(s['start']),'end':stamp(s['end']),
                'duration_seconds':s['end']-s['start'],'speaker':s['speaker'],'asset':'assets/'+s['asset'],'section':s['title']})
            md.extend([f"## {s['speaker']} | {stamp(s['start'])}–{stamp(s['end'])} | {s['title']}",'',s['text'],''])
            # Balanced short chunks; exact alignment must follow actual spoken audio.
            chunks=[words[i:i+15] for i in range(0,len(words),15)]
            cursor=s['start'];rate=(s['end']-s['start'])/len(words)
            for chunk in chunks:
                end=cursor+len(chunk)*rate;counter+=1
                srt.extend([str(counter),f'{stamp(cursor,True)} --> {stamp(end-.08,True)}',
                    '\n'.join(textwrap.wrap(' '.join(chunk),width=49)), ''])
                cursor=end
    md.extend([f'Word count: **{total}**. Average over 4:40: **{total/280*60:.1f} words/minute**, including visual pauses. Actual speech and screen capture are still to be recorded.'])
    (BASE/'script_en.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    (BASE/'subtitles_en.srt').write_text('\n'.join(srt).rstrip()+'\n',encoding='utf-8')
    (BASE/'data').mkdir(exist_ok=True)
    for source,target in [('part1_part2_cashew_milestone.csv','preliminary_comparison.csv'),
                          ('part2_gpu_overall_by_shot.csv','part2_overall_by_shot.csv')]:
        shutil.copyfile(ROOT/'results/tables'/source,BASE/'data'/target)
    cards=[]
    for s in segs:
        if cards and cards[-1]['src']==s['asset']:cards[-1]['seconds']+=s['end']-s['start']
        else:cards.append({'src':s['asset'],'seconds':s['end']-s['start'],'title':s['title']})
    html='''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Midterm video materials · 4:40</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#090f1b;color:#eef4ff;font:15px system-ui,sans-serif;overflow:hidden}
main{height:100vh;display:flex;align-items:center;justify-content:center}img{max-width:100vw;max-height:100vh;object-fit:contain}
nav{position:fixed;bottom:12px;left:50%;transform:translateX(-50%);display:flex;gap:10px;align-items:center;background:#091323ed;padding:10px 16px;border:1px solid #496383;border-radius:12px;white-space:nowrap}
button{background:#203b61;color:white;border:1px solid #6a86aa;border-radius:6px;padding:8px 12px;cursor:pointer}
:fullscreen nav{opacity:0;transition:opacity .2s}:fullscreen nav:hover, :fullscreen nav:focus-within{opacity:1}
#note{position:fixed;top:10px;left:12px;background:#101a2ddd;padding:8px;border-radius:7px;font-size:12px}:fullscreen #note{display:none}
</style>
<main><img id="card" alt="Midterm scientific visualization"></main>
<div id="note">静态真实结果素材 · 没有配音或新推理 · ←/→ 翻页，空格计时，F 全屏，R 重置<br>全屏时控制条自动隐藏；鼠标移到底部可显示。</div>
<nav><button id="prev">上一页</button><button id="play">播放时间线</button><button id="next">下一页</button><button id="full">全屏 F</button><button id="normal">正常样例 08</button><span id="state"></span></nav>
<script>
const cards=CARDS_JSON;
let i=0,elapsed=0,playing=false,last=performance.now(),optional=false;
const img=document.querySelector('#card'),state=document.querySelector('#state'),play=document.querySelector('#play');
for(const c of cards){const p=new Image();p.src='assets/'+c.src}
function refresh(){img.src='assets/'+(optional?'08_optional_normal_check.png':cards[i].src);state.textContent=optional?'08 正常样例（时间线已暂停）':`${i+1}/7 · ${cards[i].title} · ${Math.floor(elapsed)} / ${cards[i].seconds}s`;play.textContent=playing?'暂停':'播放时间线'}
function move(d){optional=false;i=Math.max(0,Math.min(cards.length-1,i+d));elapsed=0;last=performance.now();refresh()}
function toggle(){optional=false;playing=!playing;last=performance.now();refresh()}
function fullscreen(){if(document.fullscreenElement)document.exitFullscreen();else document.documentElement.requestFullscreen().catch(()=>{})}
document.querySelector('#prev').onclick=()=>move(-1);document.querySelector('#next').onclick=()=>move(1);play.onclick=toggle;document.querySelector('#full').onclick=fullscreen;
document.querySelector('#normal').onclick=()=>{playing=false;optional=!optional;refresh()};
addEventListener('keydown',e=>{if(['ArrowRight','ArrowLeft',' ','f','F','r','R'].includes(e.key))e.preventDefault();if(e.key==='ArrowRight')move(1);if(e.key==='ArrowLeft')move(-1);if(e.key===' ')toggle();if(e.key.toLowerCase()==='f')fullscreen();if(e.key.toLowerCase()==='r'){i=0;elapsed=0;playing=false;optional=false;refresh()}});
setInterval(()=>{const now=performance.now();if(playing){elapsed+=(now-last)/1000;while(elapsed>=cards[i].seconds){elapsed-=cards[i].seconds;if(i<cards.length-1)i++;else{playing=false;elapsed=cards[i].seconds;break}}refresh()}last=now},200);refresh();
</script></html>'''.replace('CARDS_JSON',json.dumps(cards))
    (BASE/'index.html').write_text(html,encoding='utf-8')
    # Portable package includes small selected raw arrays, not the full dataset/model.
    deliver=ROOT.parent/'deliverables';deliver.mkdir(exist_ok=True)
    target=deliver/'cv_midterm_video_materials.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(BASE.rglob('*')):
            if p.is_file() and p.name!='.gitignore':z.write(p,'midterm_video/'+p.relative_to(BASE).as_posix())
        for path in ['scripts/build_midterm_video_assets.py','scripts/package_midterm_video.py']:
            z.write(ROOT/path,'repository_helpers/'+path)
    print(json.dumps({'english_words':total,'planned_seconds':280,'subtitle_cues':counter,'zip':str(target),'zip_bytes':target.stat().st_size},indent=2))


if __name__=='__main__':main()
