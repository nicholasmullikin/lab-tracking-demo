"""Export explicitly selected image-only blue-pipette masks and review artifacts."""
import json,shutil
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parent;views=['T1','T2','T3','T4','T5','fpv'];base=json.loads((R/'round1-prompts.json').read_text())
selected=json.loads((R/'selections.json').read_text());before=json.loads((R/'before-refinement/labels.json').read_text())
choices={int(f):[(d[v]['round'],d[v]['candidate_index']) for v in views] for f,d in selected.items()}
clean={int(f):[v for v,o in d.items() if o['quality']=='usable-approximate-draft'] for f,d in selected.items()}
notes={f:'; '.join(v+': '+selected[str(f)][v]['review_note'] for v in views if before['frames'][str(f)]['views'][v]['quality']=='needs-review') for f in choices}
counts={str(rd):sum(len(x) for x in json.loads((R/f'round{rd}-prompts.json').read_text())['views'].values()) for rd in range(1,6)}
calls=sum(counts.values());usable=sum(map(len,clean.values()));new_calls=sum(counts[str(rd)] for rd in [3,4,5])
result={'source_frames':list(choices),'step':100,'physical_id':'blue','ground_truth':False,'video_propagation_run':False,'production_start_frame_unchanged':600,'decoder_calls':calls,'candidate_masks':calls*4,'decodes_by_round':counts,'recommended_start':0,'refinement':{'targeted_images':27,'decoder_calls':new_calls,'candidate_masks':new_calls*4,'usable_before':15,'usable_after':usable,'needs_review_after':42-usable},'review_method':'Directly viewed originals, enlarged coordinate crops, all five passes of candidate sheets and all 27 original/before/after comparisons. Explicit visual selection, retaining earlier masks when revisions regressed. Qualitative judgments, not ground truth or measured tracking accuracy.','frames':{}}
gif=[];comparison=[]
for f,selections in choices.items():
 d=R/str(f);(d/'final/masks').mkdir(parents=True,exist_ok=True);data={'source_frame':f,'source_time_seconds':f/(30000/1001),'state':'resting' if f==0 else 'held','note':notes[f],'views':{}};tiles=[];crop_tiles=[]
 for v,(rd,idx) in zip(views,selections):
  mpath=d/f'round{rd}/results/masks/{v}_blue-held_candidate-{idx:02d}.png';m=cv2.imread(str(mpath),0);im=cv2.imread(str(d/f'{v}-original.png'));h,w=im.shape[:2];assert m.shape==(h,w) and set(np.unique(m))<={0,255};shutil.copyfile(mpath,d/f'final/masks/{v}-blue.png');mask=m>0;over=im.copy();over[mask]=(over[mask]*.5+np.array([255,170,20])*.5).astype('uint8');cv2.imwrite(str(d/f'final/{v}-selected.png'),over)
  chosen_prompt=json.loads((R/f'round{rd}-prompts.json').read_text())['views'][str(f)][v];x,y,x2,y2=chosen_prompt['box'];x=max(0,x-35);y=max(0,y-35);x2=min(w,x2+35);y2=min(h,y2+35);a=im[y:y2,x:x2];b=over[y:y2,x:x2];pair=[]
  quality='usable-approximate-draft' if v in clean[f] else 'needs-review';note=selected[str(f)][v]['review_note']
  for title,img in [('Original',a),('Selected',b)]:
   sc=min(320/img.shape[1],370/img.shape[0]);img=cv2.resize(img,None,fx=sc,fy=sc);tile=np.full((410,320,3),24,np.uint8);tile[35:35+img.shape[0],:img.shape[1]]=img;cv2.putText(tile,f'{v} {title}',(5,24),0,.6,(255,255,255),1);pair.append(tile)
  crop_tiles.append(np.hstack(pair));cv2.imwrite(str(d/f'final/{v}-crop-pair.jpg'),crop_tiles[-1])
  sc=min(640/w,480/h);small=cv2.resize(over,None,fx=sc,fy=sc);tile=np.full((515,640,3),24,np.uint8);tile[35:35+small.shape[0],:small.shape[1]]=small;cv2.putText(tile,f'{v} raw {f}: '+('draft' if v in clean[f] else 'REVIEW'),(5,24),0,.6,(255,255,255),1);tiles.append(tile)
  data['views'][v]={'physical_id':'blue','state':data['state'],'image_size_wh':[w,h],'selection':{'round':rd,'candidate_index':idx},'mask_uri':f'{f}/final/masks/{v}-blue.png','quality':quality,'review_note':note,'prompt_box_xyxy':chosen_prompt['box'],'mask_pixels':int(mask.sum())}
 nativegrid=np.vstack([np.hstack(tiles[:3]),np.hstack(tiles[3:])]);cv2.imwrite(str(d/'all-selected.jpg'),nativegrid);croppedgrid=np.vstack([np.hstack(crop_tiles[:3]),np.hstack(crop_tiles[3:])]);cv2.imwrite(str(d/'all-crop-pairs.jpg'),croppedgrid)
 # GIF has seven samples, not continuous footage: T2/T3 originals and selected masks.
 canvas=Image.new('RGB',(1280,470),(24,24,24));draw=ImageDraw.Draw(canvas);draw.text((10,10),f'Original frame {f} / sampled every 100 frames / blue pipette only',fill='white')
 for i,v in enumerate(['T2','T3']):canvas.paste(Image.open(d/f'final/{v}-crop-pair.jpg'),(i*640,45))
 gif.append(canvas);result['frames'][str(f)]=data
 comparison.append(cv2.resize(croppedgrid,(960,410)))
(R/'labels.json').write_text(json.dumps(result,indent=2));cv2.imwrite(str(R/'all-frames-crop-comparison.jpg'),np.vstack(comparison))
palette=Image.fromarray(np.vstack([np.asarray(im) for im in gif])).quantize(colors=256);gif=[im.quantize(palette=palette,dither=Image.Dither.NONE) for im in gif];gif[0].save(R/'T2-T3-every100.gif',save_all=True,append_images=gif[1:],duration=1600,loop=0)
# Compact animation of the two clearest gains, before versus after.
examples=[]
for f,v in [(0,'fpv'),(100,'T4')]:
 row=Image.open(R/str(f)/f'{v}-before-after.jpg');canvas=Image.new('RGB',(1260,450),(24,24,24));canvas.paste(row,(0,40));ImageDraw.Draw(canvas).text((10,10),f'Blue pipette refinement: original frame {f}, camera {v}; sparse examples',fill='white');examples.append(canvas)
examples[0].save(R/'refinement-examples.gif',save_all=True,append_images=examples[1:],duration=2200,loop=0)
body=[f'<h1>Blue pipette: every 100 original frames, 0–600</h1><p>Retried all 27 flagged images: {new_calls} extra decodes / {new_calls*4} new candidates directly inspected. Five passes total: {calls} decodes / {calls*4} candidates. Approximate usable drafts: 15→{usable} of 42; {42-usable} still require review. These are visual judgments, not measured accuracy. Blue only. No propagation or tracking rerun. Best tested starting candidate: frame 0, especially T2–T5.</p><p>Before/after examples:</p><img style="max-width:100%" src="refinement-examples.gif"><h2>Sampled T2/T3 views</h2><img style="max-width:100%" src="T2-T3-every100.gif">']
for f in choices:
 body.append(f'<h2>Original frame {f}</h2><p>{notes[f]}</p>')
 panels=[('Refinement: originals / before / after',f'{f}/refinement-before-after.jpg'),('All originals',f'{f}/all-originals.jpg'),('Original versus selected crops',f'{f}/all-crop-pairs.jpg'),('Full selected masks',f'{f}/all-selected.jpg'),('Pass 1 candidates',f'{f}/all-candidates.jpg')]+[(f'Pass {rd} candidates',f'{f}/round{rd}/all.jpg') for rd in range(2,6)]
 for title,path in panels:
  if (R/path).exists():body.append(f'<details><summary>{title}</summary><img style="max-width:100%" src="{path}"></details>')
(R/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Every 100 frames: refinement</title><style>body{background:#171717;color:white;font:16px sans-serif;margin:24px}details{margin:12px}</style>'+''.join(body))
(R/'README.md').write_text(f"""# Blue pipette initialization sweep and refinement

Raw zero-based frames 0,100,200,300,400,500,600 in all six cameras. Blue only, physical_id blue; frame 0 resting, later held. The legacy candidate filenames say blue-held even at 0.

Five passes: {counts}; {calls} decodes / {calls*4} candidate masks. Retried all 27 previously flagged images in passes 3–5: {new_calls} decodes / {new_calls*4} new candidates. Pass 3 used cropped box plus points; pass 4 cropped points only; pass 5 whole-image 1280 encoding with corrected positives within the visually reviewed main pipette mask. Corrected misplaced prompt points after checking enlarged originals. All candidates and all 27 original/before/after comparisons directly inspected.

Approximate usable drafts: 15→{usable}/42; {42-usable} still need review. Qualitative visible-pixel judgment, not reference labels or measured tracking accuracy. Support-tube separation, neighbouring pipettes, gloves and thin shafts remain difficult. Earlier selections retained where new masks regressed. Explicit selections and notes are in selections.json, with full provenance and counts in labels.json. Binary masks remain source-sized.

No temporal propagation or production start change: existing tracking starts at 600. Frame 0, especially T2–T5, remains the best tested initialization candidate. before-refinement/ archives the prior review and selections. GIFs contain sparse examples, not continuous video.

Reproduce with model-environment Python: extract.py; run.py; run.py round2; run.py round3; run.py round4; run.py round5; export.py. Project .venv Python: export_rerun.py. Review review.html, refinement-examples.gif and the refinement-before-after.jpg sheets under each frame.
""")
print('Exported',len(choices)*6,'source-sized masks;',usable,'approximate drafts;',42-usable,'review')
