"""Seven deliberately chosen cases, fixed whole-image 1280 embeddings and boxes.
Points are authored on enlarged originals, independently of predicted masks.
Ordering: body, shaft, cap, body2, shaft2, body3, shaft3, body4.
"""
import json,cv2,numpy as np
from pathlib import Path
R=Path(__file__).resolve().parent;S=R.parent/'every100-review';base=json.loads((S/'round1-prompts.json').read_text());c=[]
rows=[
(0,'T2',[[620,476],[529,424],[704,531],[639,487],[513,413],[655,493],[550,436],[595,460]],[[604,544],[575,400],[727,501]],'clear resting control'),
(0,'T1',[[774,432],[821,389],[733,469],[763,442],[812,400],[789,418],[805,407],[754,450]],[[762,399],[728,430],[823,406]],'resting neighbours'),
(0,'fpv',[[1536,1004],[1550,1200],[1544,884],[1537,1050],[1549,1245],[1538,1090],[1551,1175],[1539,973]],[[1603,1000],[1630,1170],[1586,1100]],'resting FPV thin shaft'),
(100,'T2',[[405,493],[578,655],[362,463],[438,513],[597,672],[417,504],[563,643],[389,493]],[[476,600],[555,700],[564,541]],'hand and tip rack'),
(200,'T4',[[914,579],[813,546],[977,577],[892,582],[788,541],[934,580],[836,551],[904,579]],[[800,570],[876,600],[870,529]],'rack behind thin shaft'),
(300,'T3',[[1415,378],[1376,429],[1465,261],[1403,391],[1364,445],[1431,363],[1387,414],[1422,371]],[[1377,345],[1366,460],[1348,530]],'hand/background and transparent tip'),
(500,'T1',[[832,262],[867,304],[773,185],[821,249],[860,295],[840,274],[855,289],[847,282]],[[780,230],[833,305],[889,335]],'hand cable and support tube')]
conditions=[('box-only',0,0,False),('p1-body',1,0,False),('p1-shaft',1,0,False),('p1-cap',1,0,False),('p2-body-shaft',2,0,False),('p3-parts',3,0,False),('p5-spread',5,0,False),('p8-spread',8,0,False),('p3-n1',3,1,False),('p3-n3',3,3,False),('p5-n3',5,3,False),('p8-n3',8,3,False),('p3-far3',3,3,False),('points-only-p3',3,0,True),('points-only-p3-n3',3,3,True),('p3-wrong-positive',4,0,False),('p1-body-n1',1,1,False),('p2-body-shaft-n1',2,1,False),('p2-body-shaft-n3',2,3,False)]
for f,v,pos,neg,note in rows:
 cid=f'f{f:03d}-{v}';im=cv2.imread(str(S/str(f)/f'{v}-original.png'));h,w=im.shape[:2];box=base['views'][str(f)][v]['box']
 if (f,v)==(0,'fpv'):box=[1515,871,1595,1288]
 if (f,v)==(300,'T3'):box=[1275,225,1520,542]
 x,y,x2,y2=box;crop=[max(0,x-45),max(0,y-45),min(w,x2+45),min(h,y2+45)]
 far=[[max(0,x-160),max(0,y-130)],[min(w-1,x2+160),max(0,y-130)],[min(w-1,x2+160),min(h-1,y2+130)]]
 case={'id':cid,'source_frame':f,'view':v,'note':note,'box':box,'positive_pool':pos,'negative_pool':neg,'distant_negatives':far,'display_crop':crop,'conditions':[]}
 for name,npos,nneg,only in conditions:
  fg=pos[:npos];bg=neg[:nneg]
  if name=='p1-shaft':fg=[pos[1]]
  if name=='p1-cap':fg=[pos[2]]
  if name=='p3-far3':bg=far
  if name=='p3-wrong-positive':fg=pos[:3]+[neg[0]]
  case['conditions'].append({'id':name,'fg':fg,'bg':bg,'point_only':only,'box':box})
 c.append(case);audit=im.copy()
 for k,p in enumerate(pos):cv2.circle(audit,tuple(p),3,(0,255,0),-1);cv2.putText(audit,str(k+1),(p[0]+3,p[1]-3),0,.3,(0,255,0),1)
 for k,p in enumerate(neg):cv2.circle(audit,tuple(p),3,(0,0,255),-1);cv2.putText(audit,'N'+str(k+1),(p[0]+3,p[1]-3),0,.3,(0,0,255),1)
 x,y,x2,y2=crop;audit=audit[y:y2,x:x2];sc=min(650/audit.shape[1],800/audit.shape[0]);cv2.imwrite(str(R/f'{cid}-placement.png'),cv2.resize(audit,None,fx=sc,fy=sc))
(R/'design.json').write_text(json.dumps({'ground_truth':False,'max_side_length':1280,'checkpoint':'/home/nick/src/muggled_sam/model_weights/sam3.1_multiplex.pt','cases':c},indent=2));print(len(c),'cases;',len(conditions),'conditions;',len(c)*len(conditions),'decodes')
