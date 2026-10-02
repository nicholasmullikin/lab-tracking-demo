"""Self-contained Rerun review of originals, selected masks and all refinement passes."""
import json,cv2
from pathlib import Path
import rerun as rr,rerun.blueprint as b
R=Path(__file__).resolve().parent;labels=json.loads((R/'labels.json').read_text());views=['T1','T2','T3','T4','T5','fpv'];rr.init('finebio-every100-blue',recording_id='raw0-600-blue-refinement-five-passes');rr.save(R/'labels.rrd')
for fs,data in labels['frames'].items():
 f=int(fs);rr.set_time('source_frame',sequence=f)
 rr.log('readout',rr.TextDocument(f"""# Original frame {f}
Blue pipette only, state: {data['state']}.

Retried all 27 flagged images: 74 decodes / 296 new candidates directly inspected.
Approximate usable drafts: 15 → 26 of 42. 16 remain flagged.
Five passes total: 130 decodes / 520 candidates.

Frame 0 is still the strongest tested initialization, especially T2–T5. FPV at 0 improved substantially; T1 retains neighbour contamination.

{data['note']}

Qualitative judgments, not ground truth or measured tracking accuracy. No temporal propagation. Existing tracking start remains 600.

Before/after tab: retried views show original / previous / selected. Unchanged views show original / selected. Camera tabs show every available candidate pass; missing passes are marked explicitly.
source_frame switches sparse samples, not continuous footage.
""",media_type=rr.MediaType.MARKDOWN))
 for v,obj in data['views'].items():
  paths=[('original',R/fs/f'{v}-original.png'),('selected',R/fs/f'final/{v}-selected.png'),('crop_pair',R/fs/f'final/{v}-crop-pair.jpg'),('before_after',R/fs/f'{v}-before-after.jpg')]+[(f'pass{rd}',R/fs/f'round{rd}/{v}-candidates.jpg') for rd in range(1,6)]
  for kind,path in paths:
   if kind=='before_after' and not path.exists():path=R/fs/f'final/{v}-crop-pair.jpg'
   if path.exists():im=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
   else:
    import numpy as np
    im=np.full((180,600,3),24,np.uint8);cv2.putText(im,f'{v}: {kind} not run on frame {f}',(15,85),0,.6,(255,255,255),1)
   rr.log(f'{v}/{kind}',rr.Image(im).compress(jpeg_quality=92))
  rr.log(f'{v}/note',rr.TextDocument(f"{obj['quality']} / round {obj['selection']['round']}, candidate {obj['selection']['candidate_index']}\n\n{obj['review_note']}",media_type=rr.MediaType.MARKDOWN))
def spatial(v,kind):
 defaults=[b.VisualBounds2D(x_range=(0,1920),y_range=(0,1440 if v=='fpv' else 1080))] if kind in ['original','selected'] else []
 return b.Spatial2DView(origin=f'{v}/{kind}',name=f'{v} {kind}',defaults=defaults)
tabs=[b.Horizontal(b.Grid(*[spatial(v,'selected') for v in views],grid_columns=3),b.TextDocumentView(origin='readout',name='Review notes'),column_shares=[4,1],name='All cameras'),b.Grid(*[spatial(v,'crop_pair') for v in views],grid_columns=3,name='Original vs selected'),b.Horizontal(b.Grid(*[spatial(v,'before_after') for v in views],grid_columns=2),b.TextDocumentView(origin='readout'),column_shares=[5,1],name='Refinement before vs after')]
for v in views:
 tabs.append(b.Vertical(spatial(v,'before_after'),b.Tabs(*[spatial(v,f'pass{rd}') for rd in range(1,6)],active_tab=4,name='Candidate passes'),b.TextDocumentView(origin=f'{v}/note'),row_shares=[4,3,1],name=v))
bp=b.Blueprint(b.Tabs(*tabs,active_tab=2),b.TimePanel(timeline='source_frame',fps=100),auto_layout=False,auto_views=False);rr.send_blueprint(bp);bp.save('finebio-every100-blue',R/'labels.rbl');rr.disconnect();print(R/'labels.rrd')
