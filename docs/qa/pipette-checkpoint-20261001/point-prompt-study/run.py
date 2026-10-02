"""Decode each prompt independently; no iterative mask feedback or image recropping."""
import json,sys,importlib.util,hashlib
from pathlib import Path
import cv2,numpy as np,torch
R=Path(__file__).resolve().parent;S=R.parent/'every100-review';spec=importlib.util.spec_from_file_location('worker',R.parent/'frame-150-labeling/iterate.py');worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
from muggled_sam.make_sam import make_sam_from_state_dict
cfg=json.loads((R/'design.json').read_text());core=make_sam_from_state_dict(Path(cfg['checkpoint']));model=core.get_interactive_context().to(device='cuda',dtype=torch.bfloat16);extra='extra' in sys.argv;manifest={'ground_truth':False,'method':'fixed full image 1280; independent decodes; no mask feedback','checkpoint':cfg['checkpoint'],'cases':{}}
if extra:manifest=json.loads((R/'manifest.json').read_text())
with torch.inference_mode():
 for case in cfg['cases']:
  cid=case['id'];im=cv2.imread(str(S/str(case['source_frame'])/f"{case['view']}-original.png"));h,w=im.shape[:2];encoded=model.encode_image(im,cfg['max_side_length'],True);out=R/cid;(out/'results/masks').mkdir(parents=True,exist_ok=True);results=manifest['cases'].get(cid,{}).get('conditions',{}) if extra else {};rows=[]
  for prompt in case['conditions']:
   if extra and prompt['id'] not in ['p1-body-n1','p2-body-shaft-n1','p2-body-shaft-n3']:continue
   name=prompt['id'];result=worker._decode_rectangle(interactive_model=worker.PointOnly(model) if prompt['point_only'] else model,encoded_image=encoded,frame=im,prompt_box=dict(zip(['x1','y1','x2','y2'],prompt['box'])),target_label='blue',candidate_id=name,results_directory=out/'results',fg_points=[[x/w,y/h] for x,y in prompt['fg']],bg_points=[[x/w,y/h] for x,y in prompt['bg']],show_review=False);results[name]={'prompt':prompt,'result':result};tiles=[]
   for idx in [-1,0,1,2,3]:
    a=im.copy();title=name if idx==-1 else f"c{idx} est={result['candidates'][idx]['iou_score']:.3f}"+(' TOP' if idx==result['deterministic_best_candidate_index'] else '')
    if idx==-1:
     if not prompt['point_only']:cv2.rectangle(a,tuple(prompt['box'][:2]),tuple(prompt['box'][2:]),(255,170,20),1)
     for p in prompt['fg']:cv2.circle(a,tuple(p),3,(0,255,0),-1)
     for p in prompt['bg']:cv2.circle(a,tuple(p),3,(0,0,255),-1)
    else:
     m=cv2.imread(str(out/f'results/masks/{name}_candidate-{idx:02d}.png'),0)>0;a[m]=(a[m]*.5+np.array([255,170,20])*.5).astype('uint8')
    x,y,x2,y2=case['display_crop'];a=a[y:y2,x:x2];sc=min(340/a.shape[1],330/a.shape[0]);a=cv2.resize(a,None,fx=sc,fy=sc);t=np.full((380,340,3),24,np.uint8);t[40:40+a.shape[0],:a.shape[1]]=a;cv2.putText(t,title,(4,19),0,.45,(255,255,255),1)
    if idx==-1:cv2.putText(t,f"{len(prompt['fg'])}+ / {len(prompt['bg'])}-",(4,35),0,.4,(255,255,255),1)
    tiles.append(t)
   row=np.hstack(tiles);rows.append(row);cv2.imwrite(str(out/f'{name}.jpg'),row)
  if extra:cv2.imwrite(str(out/'candidates-extra.jpg'),np.vstack(rows))
  else:
   for chunk in range((len(rows)+3)//4):cv2.imwrite(str(out/f'candidates-{chunk+1}.jpg'),np.vstack(rows[chunk*4:chunk*4+4]))
  manifest['cases'][cid]={'case':case,'source_sha256':hashlib.sha256((S/str(case['source_frame'])/f"{case['view']}-original.png").read_bytes()).hexdigest(),'conditions':results};(R/'manifest.json').write_text(json.dumps(manifest,indent=2));del encoded;print(cid,len(rows),'decodes /',len(rows)*4,'candidates',flush=True)
print('COMPLETE')
