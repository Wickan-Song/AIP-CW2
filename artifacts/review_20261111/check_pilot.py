"""Read-only audit of the pilot; writes review evidence, never trains or edits checkpoints."""
import ast, contextlib, hashlib, io, json, os, random, re
from pathlib import Path
import numpy as np
import torch
PROJECT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT)
os.environ['MPLBACKEND'] = 'Agg'
notebook = json.loads((PROJECT / 'Predict.ipynb').read_text())
ns = {}
def cell(index): return ''.join(notebook['cells'][index]['source'])
def definitions(index):
    tree = ast.parse(cell(index))
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom))]
    exec(compile(tree, f'cell_{index}', 'exec'), ns)
def file_hash(p): return hashlib.sha256(p.read_bytes()).hexdigest()
protected = list((PROJECT/'artifacts').glob('*.pt')) + list((PROJECT/'artifacts').glob('*.json')) + [PROJECT/'Predict.ipynb']
before = {str(p.relative_to(PROJECT)): file_hash(p) for p in protected}
for c in notebook['cells']:
    if c['cell_type'] == 'code': compile(''.join(c['source']), 'notebook', 'exec')
assert not re.search(r'[\u4e00-\u9fff]', json.dumps(notebook, ensure_ascii=False))
assert not any(o.get('output_type') == 'error' for c in notebook['cells'] for o in c.get('outputs', []))
for i in [2,3,5,6]:
    with contextlib.redirect_stdout(io.StringIO()) as output: exec(cell(i),ns)
    print(f'Data cell {i} passed',flush=True)
    if i==5: audit_log=output.getvalue()
manifest = json.loads((PROJECT/'artifacts/option2_manifest.json').read_text())
from collections import Counter
counts={a:{f'{s}/{r}':v for (s,r),v in Counter((r['split'],r['role']) for r in rows).items()} for a,rows in manifest['samples'].items()}
for a, rows in manifest['samples'].items():
    roles={r['class_id']:r['role'] for r in manifest['class_roles'][a]}
    train={r['class_id'] for r in rows if r['split']=='train'}
    assert len(train)==25 and all(roles[c]=='known' for c in train)
    for r in rows:
        assert ('mask_path' in r) == (r['split']=='train')
        assert r['role'] in ns['allowed_roles'][r['split']]
definitions(8); definitions(10); definitions(15); definitions(19)
Dataset=ns['KnownImageDataset']; LesionDataset=ns['LesionTrainingDataset']; LesionModel=ns['LesionSupervisedResNet18']
rows=ns['manifests']['A']; train=[r for r in rows if r['split']=='train']; val=[r for r in rows if r['split']=='val']
# Independently verify paired initialization and multiple augmented batches.
def initial_pair(lesion):
    random.seed(42); torch.manual_seed(42)
    if lesion: model=LesionModel(25,weights=ns['WEIGHTS'])
    else:
        model=ns['resnet18'](weights=ns['WEIGHTS']); model.fc=torch.nn.Linear(512,25)
    loader=ns['DataLoader'](LesionDataset(train) if lesion else Dataset(train,training=True),batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(42),num_workers=0)
    it=iter(loader); batches=[next(it) for _ in range(3)]
    return model,batches
b,bb=initial_pair(False); l,lb=initial_pair(True)
assert all(torch.equal(v,l.classifier.state_dict()[k]) for k,v in b.state_dict().items())
for x,y in zip(bb,lb): assert torch.equal(x[0],y[0]) and torch.equal(x[1],y[1])
print('Paired initial weights and first three augmented batches match',flush=True)
# Isolate auxiliary loss to establish its effect on shared features.
torch.set_num_threads(4)
l.train(); logits,mask_logits=l(lb[0][0][:2],predict_lesions=True)
aux=ns['lesion_loss'](mask_logits,lb[0][2][:2],lb[0][3][:2]); aux.backward()
assert l.classifier.layer2[0].conv1.weight.grad.abs().sum()>0
assert l.classifier.fc.weight.grad is None
z=torch.randn(2,1,8,8,requires_grad=True); target=torch.randint(0,2,z.shape).float(); valid=torch.zeros_like(z);valid[:,:,2:6,1:7]=1
loss=ns['lesion_loss'](z,target,valid);loss.backward()
assert torch.all(z.grad[valid==0]==0) and z.grad[valid==1].abs().sum()>0
with torch.no_grad():
 changed=z.detach().clone(); changed[valid==0]=100
 assert torch.allclose(ns['lesion_loss'](changed,target,valid),loss.detach())
l.eval()
with torch.inference_mode():
 direct=l(lb[0][0][:2]); joint,_=l(lb[0][0][:2],predict_lesions=True)
 assert torch.allclose(direct,joint,atol=1e-6)
print('Auxiliary gradient, padding exclusion and RGB forward checks passed',flush=True)
# Threshold and ties: compare notebook areas with independent pair counting.
def independent(k,c,u):
    rank=(k[:,None]>u).astype(float)+0.5*(k[:,None]==u)
    return float((rank*c[:,None]).mean()),float(rank.mean())
for k,c,u in [(np.array([1.,1.,0.]),np.array([True,False,True]),np.array([1.,0.])),(np.ones(3),np.zeros(3,dtype=bool),np.ones(2)),(np.array([2.,3.]),np.ones(2,dtype=bool),np.array([0.,1.]))]:
    got=ns['open_set_curves'](k,c,u)[2:]; assert np.allclose(got,independent(k,c,u),atol=1e-12)
class RGBOnly(Dataset):
    def __getitem__(self,i):return super().__getitem__(i)[0]
loader=ns['DataLoader'](RGBOnly(val),batch_size=32,num_workers=0)
role=np.array([r['role'] for r in val]); known=role=='known'; y=np.array([r['target'] for r in val if r['role']=='known'])
recorded=json.loads((PROJECT/'artifacts/pilot_A_seed42_open_set_validation.json').read_text())
results={}; raw={}
for label,prefix,model in [('Baseline','baseline',b),('Lesion','lesion',l)]:
    path=PROJECT/f'artifacts/{prefix}_A_seed42_lr0.0003_epochs20.pt'
    saved=torch.load(path,map_location='cpu',weights_only=True)
    assert saved['manifest_sha256']==file_hash(PROJECT/'artifacts/option2_manifest.json')
    assert saved['known_class_order']==manifest['known_class_order']['A']
    assert saved['assignment']=='A' and saved['seed']==42
    assert len(saved['history'])==20
    best=max(saved['history'],key=lambda r:r['val_macro_f1'])
    assert best['epoch']==saved['selected_epoch']
    if label=='Lesion':assert saved['lesion_loss_weight']==0.25
    model.load_state_dict(saved['model_state_dict']); model.to(ns['DEVICE']).eval()
    with torch.inference_mode(): out=torch.cat([model(x.to(ns['DEVICE'])).cpu() for x in loader])
    raw[label]=out.numpy(); prediction=out[known].argmax(1).numpy(); correct=prediction==y
    f1=[]
    for cl in range(25):
        tp=np.sum((prediction==cl)&(y==cl));den=np.sum(prediction==cl)+np.sum(y==cl);f1.append(2*tp/den if den else 0)
    result={'selected_epoch':saved['selected_epoch'],'known_correct':int(correct.sum()),'accuracy':float(correct.mean()),'macro_f1':float(np.mean(f1)),'scores':{}}
    old=recorded['models'][label]
    assert abs(result['macro_f1']-old['known_macro_f1'])<1e-6
    assert result['accuracy']==old['known_accuracy']
    assert abs(result['macro_f1']-best['val_macro_f1'])<1e-6
    for score_name,s in ns['known_score_vectors'](out).items():
        k=s[known];thresh=sorted(k,reverse=True)[int(np.ceil(.95*len(k)))-1]
        ref=old['scores'][score_name];assert abs(float(thresh)-ref['threshold'])<1e-5
        accepted=k>=thresh; assert int(accepted.sum())==ref['known_accepted_count']
        score_results={}
        for group in ['familiar_dev','unfamiliar_dev']:
            u=s[role==group];area,auc=independent(k,correct,u)
            notebook_area=ns['open_set_curves'](k,correct,u)[2:]
            assert np.allclose(notebook_area,[area,auc],atol=1e-12)
            refgroup=ref['groups'][group]
            assert abs(area-refgroup['auoscr'])<1e-10 and abs(auc-refgroup['unknown_auroc'])<1e-10
            assert int((u>=thresh).sum())==refgroup['unknown_accepted_count']
            score_results[group]={'auoscr':area,'unknown_auroc':auc,'unknown_accepted':int((u>=thresh).sum()),'unknown_count':len(u)}
        result['scores'][score_name]={'threshold':float(thresh),'known_accepted':int(accepted.sum()),'groups':score_results}
    results[label]=result
    print(label,json.dumps(result),flush=True)
after={str(p.relative_to(PROJECT)):file_hash(p) for p in protected}
assert before==after
output_dir=Path(__file__).parent
np.savez_compressed(output_dir/'validation_logits.npz',**raw)
(output_dir/'data_audit.txt').write_text(audit_log)
evidence={'review_date':'2026-11-11','device':str(ns['DEVICE']),'source_sha256':before,'counts':counts,'checks':['English sources and outputs; code syntax; no saved error outputs','Fresh source-data audit and unchanged frozen manifest','A/B eligible roles; masks only in training manifest records','Shared initial weights and first three augmented batches identical','Auxiliary gradient reaches layer2; padding gradient zero','Classification forward paths equal in eval mode','Tied-score and all-wrong synthetic AUOSCR/AUROC checks','Saved checkpoint re-inference on 374 RGB validation images','Independent per-class F1 and pairwise AUOSCR/AUROC','All source notebooks, checkpoints and original JSON files unchanged'],'models':results}
(output_dir/'review_results.json').write_text(json.dumps(evidence,indent=2)+'\n')
print('ALL REVIEW CHECKS PASSED',flush=True)
