"""Stage 2: generation, baselines, CFG, novelty, diversity, validity, multi-target, distribution matching, Pareto."""
import numpy as np, pandas as pd, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from src.config import *
from src.data.dataset import get_records_df, PerovskiteDataset
from src.data.descriptors import composition_to_x0
from src.data.magpie import site_table
from src.models.denoiser import Denoiser
from src.models.scheduler import GaussianDiffusion
from src.models.surrogate import Surrogate
from src.decode import Decoder
from src.train import resolve_device
import json, random
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parent.parent
OUT = OUTPUT_DIR
FIG = OUT / "figures"
TAB = OUT / "tables"
cfg = DiffusionConfig()
device = resolve_device(cfg)
print(f"device={device}")
stats = np.load(STATS_NPZ)
df_train = get_records_df()
ds_train = PerovskiteDataset(cfg, split="train")
training_formulas = set(ds_train.df.apply(lambda r: f"{r['a']}{r['b']}{r['x']}3", axis=1))

             
model = Denoiser(d=cfg.d, cond_dim=cfg.cond_dim, hidden=cfg.hidden_dim, n_blocks=cfg.n_blocks, t_emb_dim=cfg.t_emb_dim, cond_emb_dim=cfg.cond_emb_dim, null_prob=0.0).to(device)
model.load_state_dict(torch.load(CKPT_DIR/"diffusion.pt", map_location=device))
model.eval()
surrogate = Surrogate(cfg.d).to(device)
surrogate.load_state_dict(torch.load(CKPT_DIR/"surrogate.pt", map_location=device))
surrogate.eval()
diffusion = GaussianDiffusion(T=cfg.T, s=cfg.s, device=str(device))
decoder = Decoder(cfg, stats)

                                
def generate_for_target(target_bg, target_ehull=0.0, w=2.0, n=2000, steps=50):
    cond = torch.as_tensor((np.array([target_bg,target_ehull],dtype=np.float32)-stats["prop_mean"])/stats["prop_std"], dtype=torch.float32, device=device).unsqueeze(0)
    cands=[]
    batch=min(512,n)
    for start in range(0,n,batch):
        bs=min(batch, n-start)
        c = cond.expand(bs,-1)
        x_gen = diffusion.ddim_sample(model, (bs,cfg.d), cond=c, steps=steps, eta=cfg.eta, guidance_w=w).detach().cpu().numpy()
        cands.extend(decoder.decode_batch(x_gen))
              
    valid = decoder.filter(cands, tol_min=0.8, tol_max=1.0, require_charge_balance=True, novelty=False)
    if valid:
        X=np.stack([composition_to_x0(c.a,c.b,c.x,cfg) for c in valid], dtype=np.float32)
        Xn=(X-stats["x_mean"])/stats["x_std"]
        with torch.no_grad():
            pred=surrogate(torch.as_tensor(Xn,device=device)).cpu().numpy()
        pred_bg=pred[:,0]*stats["prop_std"][0]+stats["prop_mean"][0]
        pred_hull=pred[:,1]*stats["prop_std"][1]+stats["prop_mean"][1]
        for c,pb,ph in zip(valid,pred_bg,pred_hull):
            c.pred_bandgap=float(pb); c.pred_ehull=float(ph)
    return cands, valid

                                                              
                                       
                                                              
print("\n=== RECONSTRUCTION & NOISE ===")
                                                           
                                                                                                                                                
                           
                     
from src.data.dataset import load_records
from src.data.descriptors import split_sites
rec = load_records()
x0_train = np.stack([composition_to_x0(r.a,r.b,r.x,cfg) for r in rec])
            
xn_train = (x0_train - stats["x_mean"])/stats["x_std"]
                  
cands = decoder.decode_batch(xn_train)
correct_a = sum(1 for r,c in zip(rec,cands) if r.a==c.a)
correct_b = sum(1 for r,c in zip(rec,cands) if r.b==c.b)
correct_x = sum(1 for r,c in zip(rec,cands) if r.x==c.x)
correct_all = sum(1 for r,c in zip(rec,cands) if r.a==c.a and r.b==c.b and r.x==c.x)
print(f"Reconstruction: A {correct_a}/{len(rec)}={correct_a/len(rec):.3f} B {correct_b/len(rec)} C {correct_x/len(rec)} All {correct_all/len(rec)}")
                                                                                 
def reencode(c):
    x = composition_to_x0(c.a,c.b,c.x,cfg)
    return (x-stats["x_mean"])/stats["x_std"]
xn_re = np.stack([reencode(c) for c in cands])
mse = ((xn_train - xn_re)**2).mean()
mae = np.abs(xn_train - xn_re).mean()
print(f"descriptor MSE {mse:.6f} MAE {mae:.6f}")

                                  
print("noise sweep...")
ts = [0,10,25,50,75,100,150,200,250,299]
                           
idx = np.random.RandomState(0).choice(len(xn_train), 200, replace=False)
x0_subset = torch.as_tensor(xn_train[idx], dtype=torch.float32, device=device)
errors=[]
for t in ts:
    t_b = torch.full((len(x0_subset),), t, device=device, dtype=torch.long)
    noise = torch.randn_like(x0_subset)
    x_t = diffusion.q_sample(x0_subset, t_b, noise)
    with torch.no_grad():
                                                                                     
                                                                                               
                                                                              
        cond = torch.zeros(len(x0_subset),2,device=device)
        eps_pred = model(x_t, t_b, cond)
        err = (eps_pred - noise).pow(2).mean().item()
                           
        x0_est = diffusion.predict_x0_from_eps(x_t, t_b, eps_pred)
        x0_err = (x0_est - x0_subset).pow(2).mean().item()
    errors.append((t, err, x0_err))
    print(f" t={t:3d} eps_mse={err:.4f} x0_mse={x0_err:.4f}")
pd.DataFrame(errors, columns=["t","eps_mse","x0_mse"]).to_csv(TAB/"denoising_vs_t.csv", index=False)
fig, ax = plt.subplots(figsize=(6,4))
ax.plot([e[0] for e in errors], [e[1] for e in errors], marker='o', label='eps MSE')
ax.plot([e[0] for e in errors], [e[2] for e in errors], marker='s', label='x0 MSE')
ax.set_xlabel("diffusion timestep t")
ax.set_ylabel("MSE")
ax.set_title("Denoising error vs noise level")
ax.legend()
ax.set_yscale('log')
fig.tight_layout()
fig.savefig(FIG/"denoising_vs_t.png", dpi=150)
plt.close(fig)

                                                              
                                     
                                                              
print("\n=== MULTI-TARGET ===")
targets = [0.5,1.0,1.4,2.0,3.0]
multi_rows=[]
for tg in targets:
    cands, valid = generate_for_target(tg, w=2.0, n=2000)
    if valid:
        pred_bg = np.array([c.pred_bandgap for c in valid])
        mean=np.mean(pred_bg); median=np.median(pred_bg)
        mae=float(np.mean(np.abs(pred_bg - tg)))
        medae=float(np.median(np.abs(pred_bg - tg)))
        within025=float(np.mean(np.abs(pred_bg-tg)<0.25))
        within05=float(np.mean(np.abs(pred_bg-tg)<0.5))
        within1=float(np.mean(np.abs(pred_bg-tg)<1.0))
                                            
        X=np.stack([composition_to_x0(c.a,c.b,c.x,cfg) for c in valid],dtype=np.float32)
        Xn=(X-stats["x_mean"])/stats["x_std"]
        if len(Xn)>500:
            idx2=np.random.RandomState(0).choice(len(Xn),500,replace=False)
            Xn=Xn[idx2]
        D=Xn[:,None,:]-Xn[None,:,:]
        div=float(np.sqrt((D**2).sum(-1))[np.triu_indices(len(Xn),1)].mean()) if len(Xn)>1 else 0
        valid_frac=len(valid)/2000
        novel_frac=np.mean([c.formula not in training_formulas for c in valid]) if valid else 0
        stable_frac=np.mean([c.pred_ehull<0.1 for c in valid]) if valid else 0
        w2=float(np.mean(wasserstein_distance(df_train['bandgap'].to_numpy(), pred_bg)) if len(pred_bg)>0 else np.nan)
    else:
        mean=median=mae=medae=within025=within05=within1=div=valid_frac=novel_frac=stable_frac=np.nan
        w2=np.nan
    multi_rows.append({"target":tg,"n_valid":len(valid) if valid else 0,"valid_frac":valid_frac,"mean_pred":mean,"median_pred":median,"MAE_ML":mae,"MedAE_ML":medae,"within0.25":within025,"within0.5":within05,"within1.0":within1,"diversity":div,"novel_frac":novel_frac,"stable_frac":stable_frac,"wass_dist":w2})
    print(multi_rows[-1])

pd.DataFrame(multi_rows).to_csv(TAB/"multi_target.csv", index=False)
                                       
fig, ax = plt.subplots(figsize=(5,5))
tg_arr=np.array([r["target"] for r in multi_rows])
mean_arr=np.array([r["mean_pred"] for r in multi_rows])
median_arr=np.array([r["median_pred"] for r in multi_rows])
ax.plot(tg_arr, tg_arr, 'k--', label='ideal y=x')
ax.errorbar(tg_arr, mean_arr, yerr=[np.abs(mean_arr-median_arr)], fmt='o-', capsize=4, label='mean ±median diff')
ax.scatter(tg_arr, median_arr, marker='s', color='orange', label='median')
for r in multi_rows:
    ax.text(r["target"], r["mean_pred"]+0.15, f"MAE_ML (pred) {r['MAE_ML']:.2f}", ha='center', fontsize=7)
ax.set_xlabel("target bandgap (eV)")
ax.set_ylabel("achieved pred bandgap (eV)")
ax.set_title("Target vs achieved bandgap (CFG w=2.0) — MAE_ML (pred)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG/"target_vs_achieved.png", dpi=150)
plt.close(fig)

                          
fig, axes = plt.subplots(1, len(targets), figsize=(14,3), sharey=True)
for ax, tg in zip(axes, targets):
    _, valid = generate_for_target(tg, w=2.0, n=2000)
    if valid:
        pb=np.array([c.pred_bandgap for c in valid])
        ax.hist(pb, bins=30, color="#4C72B0", alpha=0.8)
        ax.axvline(tg, color='r', linestyle='--')
        ax.set_title(f"target {tg} eV\nMAE_ML (pred) {np.mean(np.abs(pb-tg)):.2f}")
        ax.set_xlabel("pred bg (eV) — MAE_ML (pred)")
plt.tight_layout()
plt.savefig(FIG/"fidelity_hist_per_target.png", dpi=150)
plt.close(fig)

                                                              
           
                                                              
print("\n=== BASELINES ===")
                                       
def random_valid_sample(n):
    rng=np.random.RandomState(0)
    cands=[]
    attempts=0
    while len(cands)<n and attempts< n*10:
        a=rng.choice(A_SITES); b=rng.choice(B_SITES); x=rng.choice(X_SITES)
                                                                                                     
                                                                     
        from src.data.descriptors import tolerance_factor
        from src.decode import Decoder as Dec
        bal=Decoder.charge_balanced(a,b,x)
        tf=tolerance_factor(a,b,x)
        if bal and 0.8<=tf<=1.0:
                                                            
            cands.append((a,b,x,tf,bal))
        attempts+=1
                                  
    return cands

baseline_rows=[]
           
rand_tuples=random_valid_sample(5000)
if rand_tuples:
    X=np.stack([composition_to_x0(a,b,x,cfg) for a,b,x,_ ,_ in rand_tuples],dtype=np.float32)
    Xn=(X-stats["x_mean"])/stats["x_std"]
    with torch.no_grad():
        pred=surrogate(torch.as_tensor(Xn,device=device)).cpu().numpy()
    pred_bg=pred[:,0]*stats["prop_std"][0]+stats["prop_mean"][0]
    pred_hull=pred[:,1]*stats["prop_std"][1]+stats["prop_mean"][1]
    mae=float(np.mean(np.abs(pred_bg-1.4)))
    valid_frac=len(rand_tuples)/5000
                                       
    formulas=[f"{a}{b}{x}3" for a,b,x,_,_ in rand_tuples]
    novel_frac=np.mean([f not in training_formulas for f in formulas])
               
    if len(Xn)>500:
        idx2=np.random.RandomState(0).choice(len(Xn),500,replace=False)
        Xn2=Xn[idx2]
    else:
        Xn2=Xn
    D=Xn2[:,None,:]-Xn2[None,:,:]
    div=float(np.sqrt((D**2).sum(-1))[np.triu_indices(len(Xn2),1)].mean()) if len(Xn2)>1 else 0
    stable_frac=float(np.mean(pred_hull<0.1))
    baseline_rows.append({"baseline":"Random valid ABX3","n":len(rand_tuples),"valid_frac":valid_frac,"MAE_ML@1.4 (pred)":mae,"stable":stable_frac,"novel":novel_frac,"diversity":div})
    print(baseline_rows[-1])

                                                          
rng=np.random.RandomState(1)
tr_formulas = ds_train.df.sample(2000, replace=True, random_state=0)
X=np.stack([composition_to_x0(r.a,r.b,r.x,cfg) for _,r in tr_formulas.iterrows()],dtype=np.float32)
Xn=(X-stats["x_mean"])/stats["x_std"]
with torch.no_grad():
    pred=surrogate(torch.as_tensor(Xn,device=device)).cpu().numpy()
pred_bg=pred[:,0]*stats["prop_std"][0]+stats["prop_mean"][0]
pred_hull=pred[:,1]*stats["prop_std"][1]+stats["prop_mean"][1]
mae=float(np.mean(np.abs(pred_bg-1.4)))
                                                                            
                                                                          
true_bg=tr_formulas['bandgap'].to_numpy() if 'bandgap' in tr_formulas else df_train['bandgap'].sample(2000, random_state=0).to_numpy()
true_mae=float(np.mean(np.abs(true_bg-1.4)))
           
D=Xn[:,None,:]-Xn[None,:,:]
div=float(np.sqrt((D**2).sum(-1))[np.triu_indices(len(Xn),1)].mean()) if len(Xn)>1 else 0
baseline_rows.append({"baseline":"Training-distribution","n":len(tr_formulas),"valid_frac":1.0,"MAE_ML@1.4 (pred)":mae,"MAE_true@1.4 (DFT)":true_mae,"stable":float(np.mean(pred_hull<0.1)),"novel":0.0,"diversity":div})
print(baseline_rows[-1])

                                                             
df_train_with_dist = df_train.copy()
df_train_with_dist['abs_err']=(df_train_with_dist['bandgap']-1.4).abs()
nearest = df_train_with_dist.nsmallest(200, 'abs_err')
                                 
Xn_near=np.stack([composition_to_x0(r.a,r.b,r.x,cfg) for _,r in nearest.iterrows()],dtype=np.float32)
Xn_near=(Xn_near-stats["x_mean"])/stats["x_std"]
with torch.no_grad():
    pred=surrogate(torch.as_tensor(Xn_near,device=device)).cpu().numpy()
pred_bg=pred[:,0]*stats["prop_std"][0]+stats["prop_mean"][0]
baseline_rows.append({"baseline":"NN retrieval (200 closest)","n":len(nearest),"valid_frac":1.0,"MAE_true@1.4 (DFT)":float(nearest['abs_err'].mean()),"MAE_ML@1.4 (pred)":float(np.mean(np.abs(pred_bg-1.4))),"stable":float(np.mean(nearest['ehull']<0.1)),"novel":0.0,"diversity":0})
print(baseline_rows[-1])

                                                                                             
                                                                                                            
np.random.seed(0)
X_rand = np.random.randn(20000, cfg.d).astype(np.float32)
                                              
with torch.no_grad():
    pred=surrogate(torch.as_tensor(X_rand,device=device)).cpu().numpy()
pred_bg=pred[:,0]*stats["prop_std"][0]+stats["prop_mean"][0]
pred_hull=pred[:,1]*stats["prop_std"][1]+stats["prop_mean"][1]
score=np.abs(pred_bg-1.4)+0.5*np.clip(pred_hull,0,None)
best_idx=np.argsort(score)[:2000]
X_best=X_rand[best_idx]
             
from src.decode import Decoder
                   
cands_opt=[]
for i in range(0,len(X_best),512):
    cands_opt.extend(decoder.decode_batch(X_best[i:i+512]))
valid_opt=decoder.filter(cands_opt, tol_min=0.8, tol_max=1.0, require_charge_balance=True, novelty=False)
if valid_opt:
    Xopt=np.stack([composition_to_x0(c.a,c.b,c.x,cfg) for c in valid_opt],dtype=np.float32)
    Xn_opt=(Xopt-stats["x_mean"])/stats["x_std"]
    with torch.no_grad():
        pred=surrogate(torch.as_tensor(Xn_opt,device=device)).cpu().numpy()
    pred_bg=pred[:,0]*stats["prop_std"][0]+stats["prop_mean"][0]
    mae=float(np.mean(np.abs(pred_bg-1.4)))
    novel_frac=np.mean([c.formula not in training_formulas for c in valid_opt])
    D=Xn_opt[:,None,:]-Xn_opt[None,:,:]
    if len(Xn_opt)>500:
        idx2=np.random.RandomState(0).choice(len(Xn_opt),500,replace=False)
        Xn_opt2=Xn_opt[idx2]
        D=Xn_opt2[:,None,:]-Xn_opt2[None,:,:]
        div=float(np.sqrt((D**2).sum(-1))[np.triu_indices(len(Xn_opt2),1)].mean())
    else:
        div=float(np.sqrt((D**2).sum(-1))[np.triu_indices(len(Xn_opt),1)].mean()) if len(Xn_opt)>1 else 0
    baseline_rows.append({"baseline":"Random search over x0 (surrogate-opt)","n":len(valid_opt),"valid_frac":len(valid_opt)/2000,"MAE_ML@1.4 (pred)":mae,"novel":novel_frac,"diversity":div,"stable":float(np.mean([c.pred_bandgap for c in valid_opt] is not None))})
                                        
    pred_hull=pred[:,1]*stats["prop_std"][1]+stats["prop_mean"][1] if len(valid_opt)>0 else np.array([])
    stable=float(np.mean(pred_hull<0.1)) if len(pred_hull)>0 else 0
    baseline_rows[-1]["stable"]=stable
    print(baseline_rows[-1])

                                                      
_, valid_diff = generate_for_target(1.4, w=2.0, n=2000)
if valid_diff:
    pred_bg=np.array([c.pred_bandgap for c in valid_diff])
    mae=float(np.mean(np.abs(pred_bg-1.4)))
    novel_frac=np.mean([c.formula not in training_formulas for c in valid_diff])
    X=np.stack([composition_to_x0(c.a,c.b,c.x,cfg) for c in valid_diff],dtype=np.float32)
    Xn=(X-stats["x_mean"])/stats["x_std"]
    if len(Xn)>500:
        idx2=np.random.RandomState(0).choice(len(Xn),500,replace=False)
        Xn2=Xn[idx2]
        D=Xn2[:,None,:]-Xn2[None,:,:]
        div=float(np.sqrt((D**2).sum(-1))[np.triu_indices(len(Xn2),1)].mean())
    else:
        D=Xn[:,None,:]-Xn[None,:,:]
        div=float(np.sqrt((D**2).sum(-1))[np.triu_indices(len(Xn),1)].mean()) if len(Xn)>1 else 0
    pred_hull=np.array([c.pred_ehull for c in valid_diff])
    stable=float(np.mean(pred_hull<0.1))
    baseline_rows.append({"baseline":"Diffusion CFG 2.0 (ours)","n":len(valid_diff),"valid_frac":len(valid_diff)/2000,"MAE_ML@1.4 (pred)":mae,"stable":stable,"novel":novel_frac,"diversity":div})
    print(baseline_rows[-1])

pd.DataFrame(baseline_rows).to_csv(TAB/"baselines.csv", index=False)

                                                              
                                          
                                                              
print("\n=== CFG TRADEOFF ===")
cfg_vals=[1.0,1.5,2.0,2.5,3.0]
                                                                                       
                       
try:
    summ=pd.read_csv(OUT/"generation_summary.csv")
    print(summ)
except: summ=None

                                                        
cfg_rows=[]
for w in cfg_vals:
    _, valid = generate_for_target(1.4, w=w, n=2000)
    if valid:
        pb=np.array([c.pred_bandgap for c in valid])
        mae=float(np.mean(np.abs(pb-1.4)))
        medae=float(np.median(np.abs(pb-1.4)))
        within05=float(np.mean(np.abs(pb-1.4)<0.5))
        novel=float(np.mean([c.formula not in training_formulas for c in valid]))
                   
        X=np.stack([composition_to_x0(c.a,c.b,c.x,cfg) for c in valid],dtype=np.float32)
        Xn=(X-stats["x_mean"])/stats["x_std"]
        if len(Xn)>500:
            idx2=np.random.RandomState(0).choice(len(Xn),500,replace=False)
            Xn2=Xn[idx2]
            D=Xn2[:,None,:]-Xn2[None,:,:]
            div=float(np.sqrt((D**2).sum(-1))[np.triu_indices(len(Xn2),1)].mean())
        else:
            D=Xn[:,None,:]-Xn[None,:,:]
            div=float(np.sqrt((D**2).sum(-1))[np.triu_indices(len(Xn),1)].mean()) if len(Xn)>1 else 0
        stable=float(np.mean([c.pred_ehull<0.1 for c in valid]))
        cfg_rows.append({"w":w,"MAE_ML":mae,"MedAE_ML":medae,"within0.5":within05,"novel":novel,"diversity":div,"stable":stable,"n_valid":len(valid)})
    else:
        cfg_rows.append({"w":w,"MAE_ML":np.nan,"MedAE_ML":np.nan,"within0.5":0,"novel":0,"diversity":0,"stable":0,"n_valid":0})
pd.DataFrame(cfg_rows).to_csv(TAB/"cfg_tradeoff.csv", index=False)
fig, axes = plt.subplots(1,3, figsize=(12,4))
ax=axes[0]; ax.plot([r["w"] for r in cfg_rows],[r["MAE_ML"] for r in cfg_rows], marker='o', color='#4472C4'); ax.set_xlabel("CFG w"); ax.set_ylabel("Target MAE_ML (pred, eV)"); ax.set_title("Conditioning vs CFG (pred)")
ax2=axes[1]; ax2.plot([r["w"] for r in cfg_rows],[r["novel"] for r in cfg_rows], marker='o', color='#ED7D31'); ax2.set_xlabel("CFG w"); ax2.set_ylabel("Novelty"); ax2.set_title("Novelty vs CFG")
ax3=axes[2]; ax3.plot([r["w"] for r in cfg_rows],[r["diversity"] for r in cfg_rows], marker='o', color='#70AD47'); ax3.set_xlabel("CFG w"); ax3.set_ylabel("Diversity"); ax3.set_title("Diversity vs CFG")
fig.tight_layout()
fig.savefig(FIG/"cfg_tradeoff.png", dpi=150)
plt.close(fig)

                                                              
                                                      
                                                              
print("\n=== NOVELTY / MEMORIZATION / FUNNEL ===")
                                          
cands, valid = generate_for_target(1.4, w=2.0, n=5000)
                                          
                                                  
        
total=5000
                   
from src.data.descriptors import tolerance_factor
from src.decode import Decoder as Dec
                                                              
                                                               
                 
n_charge = sum(1 for c in cands if c.charge_balanced)
n_tol = sum(1 for c in cands if 0.8<=c.tol_factor<=1.0)
n_both = sum(1 for c in cands if c.charge_balanced and 0.8<=c.tol_factor<=1.0)
n_novel = sum(1 for c in cands if c.charge_balanced and 0.8<=c.tol_factor<=1.0 and c.formula not in training_formulas)
              
                                                                                                        
n_stable = sum(1 for c in valid if c.pred_ehull<0.1) if valid else 0
                                                     
funnel = pd.DataFrame([
    {"filter":"Generated","count":total,"fraction":1.0},
    {"filter":"Charge balanced","count":n_charge,"fraction":n_charge/total},
    {"filter":"Tolerance 0.8-1.0","count":n_tol,"fraction":n_tol/total},
    {"filter":"Valid (charge+tol)","count":n_both,"fraction":n_both/total},
    {"filter":"Novel (not in train)","count":n_novel,"fraction":n_novel/total},
    {"filter":"Stable* Ehull<0.1","count":n_stable,"fraction":n_stable/total if total else 0},
])
funnel.to_csv(TAB/"validity_funnel.csv", index=False)
print(funnel)

                  
                                                    
                                    
X_train = np.stack([composition_to_x0(r.a,r.b,r.x,cfg) for r in load_records()],dtype=np.float32)
Xn_train=(X_train-stats["x_mean"])/stats["x_std"]
                      
if valid:
    X_valid=np.stack([composition_to_x0(c.a,c.b,c.x,cfg) for c in valid],dtype=np.float32)
    Xn_valid=(X_valid-stats["x_mean"])/stats["x_std"]
                   
    from scipy.spatial.distance import cdist
    D=cdist(Xn_valid, Xn_train, metric='euclidean')
    d_min=D.min(axis=1)
                  
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(d_min, bins=40, color="#4472C4", alpha=0.8)
    ax.set_xlabel("d_min to training set (descriptor Euclidean)")
    ax.set_ylabel("count")
    ax.set_title("Memorization: nearest-neighbor distance (valid, w=2.0)")
    fig.tight_layout()
    fig.savefig(FIG/"memorization_dmin.png", dpi=150)
    plt.close(fig)
                                                                 
                                                        
                              
    idx_tr=np.random.RandomState(0).choice(len(Xn_train),500,replace=False)
    Xn_tr_sub=Xn_train[idx_tr]
    D_tr=np.sqrt(((Xn_tr_sub[:,None,:]-Xn_tr_sub[None,:,:])**2).sum(-1))
    d_tr=D_tr[np.triu_indices(len(Xn_tr_sub),1)]
                        
    idx_g=np.random.RandomState(0).choice(len(Xn_valid),500,replace=False) if len(Xn_valid)>500 else np.arange(len(Xn_valid))
    Xn_g_sub=Xn_valid[idx_g]
    D_g=np.sqrt(((Xn_g_sub[:,None,:]-Xn_g_sub[None,:,:])**2).sum(-1))
    d_g=D_g[np.triu_indices(len(Xn_g_sub),1)] if len(Xn_g_sub)>1 else np.array([0])
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(d_tr, bins=40, alpha=0.6, label="training pairwise", color="#70AD47", density=True)
    ax.hist(d_g, bins=40, alpha=0.6, label="generated pairwise", color="#ED7D31", density=True)
    ax.set_xlabel("pairwise distance")
    ax.set_ylabel("density")
    ax.legend()
    ax.set_title("Diversity: pairwise distance distribution")
    fig.tight_layout()
    fig.savefig(FIG/"pairwise_diversity.png", dpi=150)
    plt.close(fig)
                                                                      
    novel_flag=np.array([c.formula not in training_formulas for c in valid])
                                                        
    bins=np.linspace(d_min.min(), d_min.max(), 5)
    bin_novel=[]
    for i in range(len(bins)-1):
        mask=(d_min>=bins[i])&(d_min<bins[i+1])
        if mask.sum()>0:
            bin_novel.append((bins[i], bins[i+1], np.mean(novel_flag[mask]), mask.sum()))
    print("novelty vs d_min bins:", bin_novel)
              
    order=np.argsort(d_min)
                                                       
    examples=[]
    for idx_ex in [order[0], order[len(order)//2], order[-1]]:
        c=valid[idx_ex]
                            
        nearest_idx=np.argmin(D[idx_ex])
        tr_rec=load_records()[nearest_idx]
        examples.append({"generated":c.formula,"nearest":f"{tr_rec.a}{tr_rec.b}{tr_rec.x}3","d_min":float(d_min[idx_ex]),"novel": bool(novel_flag[idx_ex]),"dist_a":c.dist_a,"dist_b":c.dist_b,"dist_x":c.dist_x})
    pd.DataFrame(examples).to_csv(TAB/"memorization_examples.csv", index=False)
    print(examples)
                             
    pd.DataFrame({"d_min":d_min,"novel":novel_flag}).to_csv(TAB/"dmin_distribution.csv", index=False)
                              
    fig, ax = plt.subplots(figsize=(6,4))
            
    y_jitter=np.random.randn(len(d_min))*0.02 + novel_flag.astype(float)
    ax.scatter(d_min, y_jitter, alpha=0.4, s=10)
    ax.set_xlabel("d_min to train")
    ax.set_ylabel("novel (jittered)")
    ax.set_title("Novelty vs descriptor distance")
    fig.tight_layout()
    fig.savefig(FIG/"novelty_vs_dmin.png", dpi=150)
    plt.close(fig)
else:
    print("no valid for novelty")

                                                              
                       
                                                              
print("\n=== DISTRIBUTION MATCHING ===")
train_bg=df_train['bandgap'].to_numpy()
                                                                                                                  
                                                                                                        
                            
from scipy.spatial.distance import jensenshannon
def js_div(p,q, bins=40):
                              
    lo=min(p.min(), q.min()); hi=max(p.max(), q.max())
    bins_e=np.linspace(lo,hi, bins)
    hp,_=np.histogram(p,bins=bins_e, density=True)
    hq,_=np.histogram(q,bins=bins_e, density=True)
                        
    hp=hp/hp.sum() if hp.sum()>0 else hp
    hq=hq/hq.sum() if hq.sum()>0 else hq
                
    hp=np.clip(hp,1e-8,None); hq=np.clip(hq,1e-8,None)
    return jensenshannon(hp,hq)**2                 

                        
_, valid_14 = generate_for_target(1.4, w=2.0, n=2000)
pred_14=np.array([c.pred_bandgap for c in valid_14]) if valid_14 else np.array([])
                                                                                                    
                                            

dist_rows=[]
if len(pred_14)>0:
    wd=wasserstein_distance(train_bg, pred_14)
                                           
    js=js_div(train_bg, pred_14)
    dist_rows.append({"comparison":"train vs gen target 1.4 w2.0","wasserstein":wd,"js_div":js})
                                                                                                          
                                                                                                          
                                     
                                        
    batch=2000
    xs=[]
    for start in range(0,batch,512):
        bs=min(512,batch-start)
        null_cond=torch.zeros(bs,2,device=device)
        x_gen=diffusion.ddim_sample(model, (bs,cfg.d), cond=null_cond, steps=cfg.ddim_steps, eta=cfg.eta, guidance_w=1.0).detach().cpu().numpy()
        xs.extend(decoder.decode_batch(x_gen))
    valid_uncond=decoder.filter(xs, tol_min=0.8, tol_max=1.0, require_charge_balance=True, novelty=False)
    if valid_uncond:
        X=np.stack([composition_to_x0(c.a,c.b,c.x,cfg) for c in valid_uncond],dtype=np.float32)
        Xn=(X-stats["x_mean"])/stats["x_std"]
        with torch.no_grad():
            pred=surrogate(torch.as_tensor(Xn,device=device)).cpu().numpy()
        pred_uncond=pred[:,0]*stats["prop_std"][0]+stats["prop_mean"][0]
        wd_u=wasserstein_distance(train_bg, pred_uncond)
        js_u=js_div(train_bg, pred_uncond)
        dist_rows.append({"comparison":"train vs unconditional gen","wasserstein":wd_u,"js_div":js_u})
        wd_cond_vs_uncond=wasserstein_distance(pred_uncond, pred_14)
        js_cond_vs_uncond=js_div(pred_uncond, pred_14)
        dist_rows.append({"comparison":"uncond vs cond 1.4","wasserstein":wd_cond_vs_uncond,"js_div":js_cond_vs_uncond})
        print(dist_rows)
pd.DataFrame(dist_rows).to_csv(TAB/"distribution_matching.csv", index=False)

                                                              
                                        
                                                              
print("\n=== PARETO + CANDIDATE TABLE ===")
                       
if valid_14:
    fig, ax = plt.subplots(figsize=(6,5))
    ax.scatter(df_train['bandgap'], df_train['ehull'], alpha=0.3, s=8, color="#4C72B0", label="training (DFT)")
    ax.scatter([c.pred_bandgap for c in valid_14], [c.pred_ehull for c in valid_14], alpha=0.5, s=12, color="#ED7D31", label="generated 1.4w2 (pred)")
    ax.axvline(1.4, color='k', ls='--', lw=1)
    ax.axhline(0.1, color='k', ls=':', lw=1)
    ax.set_xlabel("bandgap (eV)")
    ax.set_ylabel("E_hull (eV/atom)")
    ax.set_title("Pareto: bandgap vs stability (pred)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG/"pareto.png", dpi=150)
    plt.close(fig)

                                                                                       
try:
    cand_df=pd.read_csv(OUT/"candidates.csv")
                                                     
                                               
    X_train=np.stack([composition_to_x0(r.a,r.b,r.x,cfg) for r in load_records()],dtype=np.float32)
    Xn_train=(X_train-stats["x_mean"])/stats["x_std"]
    from scipy.spatial.distance import cdist
    for idx, row in cand_df.iterrows():
        x=composition_to_x0(row['A'],row['B'],row['X'],cfg)
        xn=(x-stats["x_mean"])/stats["x_std"]
        dmin=cdist(xn[None,:], Xn_train).min()
        cand_df.at[idx,'d_min']=dmin
                               
        nearest_idx=np.argmin(cdist(xn[None,:], Xn_train))
        cand_df.at[idx,'nearest_train']=f"{load_records()[nearest_idx].a}{load_records()[nearest_idx].b}{load_records()[nearest_idx].x}3"
                      
    cand_df['novel']=~cand_df['formula'].isin([f"{r.a}{r.b}{r.x}3" for r in load_records()])
    cand_df.to_csv(TAB/"candidate_table_enriched.csv", index=False)
    print(cand_df.head().to_string(index=False))
except Exception as e:
    print("candidate table failed", e)

print("\n=== stage2 done ===")
