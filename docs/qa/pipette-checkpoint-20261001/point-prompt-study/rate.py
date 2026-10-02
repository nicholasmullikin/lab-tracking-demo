"""Explicit qualitative ratings from direct review of all 28 candidate sheets.
5=clean approximate visible pixels; 4=mostly correct, small errors/uncertain tip;
3=notable leakage or missing part; 2=major non-target inclusion; 1=wrong/failed.
These are single AI-reviewer ordinal judgments, not IoU or human annotations.
"""
import json,csv
from pathlib import Path
R=Path(__file__).resolve().parent;m=json.loads((R/'manifest.json').read_text());names=list(next(iter(m['cases'].values()))['conditions'])
# Each entry: selected candidate index, best-of-four visual grade, model-top grade.
ratings={
'f000-T2':[(0,5,5),(1,5,5),(0,5,5),(3,5,5),(0,5,5),(0,5,5),(1,5,5),(1,5,4),(0,5,5),(0,5,5),(0,5,5),(1,5,4),(3,5,4),(1,5,3),(0,5,5),(1,5,3)],
'f000-T1':[(0,3,3),(1,4,3),(0,4,3),(3,3,3),(0,4,4),(1,4,3),(1,4,3),(1,3,2),(1,4,4),(0,4,4),(1,4,3),(1,3,3),(1,3,3),(3,3,3),(0,4,3),(3,3,3)],
'f000-fpv':[(0,5,5),(3,5,5),(0,5,5),(0,5,5),(0,5,5),(0,5,5),(0,5,5),(3,5,4),(3,5,5),(3,5,5),(3,5,5),(3,5,3),(0,5,5),(3,5,5),(3,5,5),(0,4,4)],
'f100-T2':[(0,3,3),(1,4,3),(0,3,3),(0,3,3),(1,3,2),(1,3,3),(1,3,2),(1,3,2),(1,3,3),(0,3,3),(0,3,3),(1,3,3),(1,3,2),(1,3,3),(0,3,3),(1,3,2)],
'f200-T4':[(1,3,2),(1,3,2),(0,3,2),(1,3,2),(1,3,2),(1,3,2),(1,3,3),(1,3,3),(0,4,3),(0,4,4),(0,4,4),(0,4,4),(1,3,2),(0,4,4),(0,4,4),(1,3,2)],
'f300-T3':[(1,4,2),(1,4,4),(1,4,4),(1,3,2),(0,4,4),(1,4,4),(1,3,3),(1,3,2),(0,4,3),(1,4,3),(1,3,3),(1,3,2),(1,3,2),(1,3,3),(1,3,3),(1,3,2)],
'f500-T1':[(1,4,2),(1,4,2),(1,4,2),(1,3,2),(1,4,2),(1,3,2),(1,2,2),(1,2,2),(1,4,2),(1,4,2),(1,3,2),(1,3,2),(1,3,2),(1,3,3),(1,4,3),(1,2,2)]}
extra={
'f000-T2':[(0,5,5),(0,5,5),(0,5,5)],
'f000-T1':[(1,4,3),(0,4,4),(0,4,3)],
'f000-fpv':[(3,5,5),(0,5,5),(3,5,5)],
'f100-T2':[(1,3,3),(1,3,3),(0,3,3)],
'f200-T4':[(0,4,4),(0,4,4),(0,4,4)],
'f300-T3':[(1,4,4),(1,4,4),(1,4,4)],
'f500-T1':[(0,4,4),(1,4,4),(1,4,3)]}
for cid,rows in extra.items():ratings[cid].extend(rows)
notes={
'f000-T2':'Good box already works. Eight positives and a false positive make the top-ranked mask leak into neighbours; a better candidate still exists.',
'f000-T1':'Body+shaft is more useful than cap alone. Repeated positives increase neighbouring-pipette spill. None fully resolves neighbour contamination.',
'f000-fpv':'Corrected box alone is strong; many positives add holes. This case supports fixing the box before adding clicks.',
'f100-T2':'Rack/shaft ambiguity remains. Body-only has the cleanest available candidate; additional positives can expand the glove or omit the separate shaft.',
'f200-T4':'Targeted negatives reduce rack spill. Three positives plus one or three negatives produce the strongest reviewed candidates, with residual errors.',
'f300-T3':'Body or shaft alone, and body+shaft, control glove spill. Cap/hook-only and numerous positives are worse. Distal transparent tip remains uncertain.',
'f500-T1':'Model top rank frequently selects the glove. Removing the cap click and using one body/shaft positive pair plus a glove negative greatly reduces that error. Tube separation remains unresolved.'}
result={'ground_truth':False,'reviewer':'single AI visual reviewer','rating_scale':{5:'clean approximate visible pixels',4:'mostly correct with small errors or uncertain tip',3:'notable leakage or missing part',2:'major non-target inclusion',1:'wrong object or failed'},'decoder_calls':133,'candidate_masks_inspected':532,'cases':{},'summary':[]};csvrows=[]
for cid,rows in ratings.items():
 result['cases'][cid]={'note':notes[cid],'conditions':{}}
 for name,(idx,best,top) in zip(names,rows):
  original=m['cases'][cid]['conditions'][name];topidx=original['result']['deterministic_best_candidate_index'];assert best>=top
  o={'selected_candidate_index':idx,'best_of_four_visual_grade':best,'model_top_candidate_index':topidx,'model_top_visual_grade':top};result['cases'][cid]['conditions'][name]=o;csvrows.append({'case':cid,'condition':name,**o})
for name in names:
 scores=[x['conditions'][name] for x in result['cases'].values()];prompt=next(iter(m['cases'].values()))['conditions'][name]['prompt'];row={'condition':name,'positive_points':len(prompt['fg']),'negative_points':len(prompt['bg']),'box_prompt':not prompt['point_only'],'mostly_correct_available':sum(x['best_of_four_visual_grade']>=4 for x in scores),'mostly_correct_model_top':sum(x['model_top_visual_grade']>=4 for x in scores),'major_spill_model_top':sum(x['model_top_visual_grade']<=2 for x in scores)};result['summary'].append(row);print(row)
(R/'ratings.json').write_text(json.dumps(result,indent=2))
with (R/'ratings.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
