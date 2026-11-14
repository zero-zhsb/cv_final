/** Render the authoritative deck to the video cards, without running a model. */
import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import crypto from 'node:crypto';
const modules=process.env.RUNTIME_NODE_MODULES ?? process.env.CODEX_PRIMARY_RUNTIME_NODE_MODULES;
if(!modules)throw new Error('Set RUNTIME_NODE_MODULES to the installed workspace Node packages directory');
const require=createRequire(path.join(modules,'cv_midterm_resolver.cjs'));
const {PresentationFile,FileBlob}=await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const root=process.cwd(),base=path.join(root,'presentation/midterm_video');
const deck=path.join(root,'presentation/midterm_ppt/CV_Midterm_FullComparison_Final.pptx');
const segs=JSON.parse(await fs.readFile(path.join(base,'script_segments.json'),'utf8'));
const cards=[...new Set(segs.map(s=>s.asset))];
const P=await PresentationFile.importPptx(await FileBlob.load(deck));
if(P.slides.items.length!==cards.length)throw new Error('Deck/timeline slide counts differ');
for(const [i,s] of P.slides.items.entries()){
 const b=await P.export({slide:s,format:'png',scale:1.5});
 await fs.writeFile(path.join(base,'assets',cards[i]),new Uint8Array(await b.arrayBuffer()));
}
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
const provenance=JSON.parse(await fs.readFile(path.join(base,'provenance.json'),'utf8'));
provenance.version='full-part1-and-part2-comparison';
provenance.authoritative_deck='presentation/midterm_ppt/CV_Midterm_FullComparison_Final.pptx';
provenance.deck_sha256=hash(await fs.readFile(deck));
provenance.part1_full_source_commit='616701a98692ad0be3cecf904d1c1149dd4ec962';
provenance.part1_full_run='part1_full_stream_cuda_v1';
provenance.part1_full_audit=JSON.parse(await fs.readFile(path.join(root,'results/tables/part1_full/audit.json'),'utf8'));
provenance.full_comparison_csv='results/tables/part1_full/comparison.csv';
provenance.full_comparison_csv_sha256=hash(await fs.readFile(path.join(root,provenance.full_comparison_csv)));
provenance.qualitative_note='The cashew 1-shot example remains an illustrative case; the main table and trend use complete six-category results. Both methods miss cashew bad/075.JPG at 8-shot seed 0.';
provenance.raw_sha256={};
for(const n of (await fs.readdir(path.join(base,'raw'))).sort()){
 const b=await fs.readFile(path.join(base,'raw',n));provenance.raw_sha256[n]=hash(b);
}
await fs.writeFile(path.join(base,'provenance.json'),JSON.stringify(provenance,null,2)+'\n');
console.log('Rendered',cards.length,'authoritative 1920x1080 cards and updated provenance');
