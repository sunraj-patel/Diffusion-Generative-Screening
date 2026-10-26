"""Stage 3: ablations, DDIM, seeds, scaling, OOD generation, DFT validation."""
import numpy as np, pandas as pd, torch, matplotlib, json, random
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from src.config import *
from src.data.dataset import get_records_df, PerovskiteDataset, load_records
from src.data.descriptors import composition_to_x0
from src.models.denoiser import Denoiser
from src.models.scheduler import GaussianDiffusion
from src.models.surrogate import Surrogate
from src.decode import Decoder
from src.train import resolve_device, EMAModel
from scipy.spatial.distance import cdist

OUT=OUTPUT_DIR; FIG=OUT/"figures"; TAB=OUT/"tables"
cfg0=DiffusionConfig()
device=resolve_device(cfg0)
stats=np.load(STATS_NPZ)
print(f"device={device}")

                                               
def train_diffusion_variant(cfg, tag, epochs=80):
    """Quick train for ablations (80 epochs) to get signal."""
    print(f"\n=== training {tag} ===")
    device_=resolve_device(cfg)
    torch.manual_seed(cfg.seed)
    ds=PerovskiteDataset(cfg, split="train")
    loader=torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
                                                  
    from src.models.surrogate import train_surrogate
                                    
    surrogate=train_surrogate(cfg, epochs=150, device=str(device_))
    surrogate.eval()
    for p in surrogate.parameters(): p.requires_grad_(False)
    model=Denoiser(d=cfg.d, cond_dim=cfg.cond_dim, hidden=cfg.hidden_dim, n_blocks=cfg.n_blocks, t_emb_dim=cfg.t_emb_dim, cond_emb_dim=cfg.cond_emb_dim, null_prob=cfg.null_prob).to(device_)
    ema=EMAModel(model)
    diffusion=GaussianDiffusion(T=cfg.T, s=cfg.s, device=str(device_))
    opt=torch.optim.Adam(model.parameters(), lr=cfg.lr)
    mse=torch.nn.MSELoss()
    def aux_loss(x0_est, cond): return mse(surrogate(x0_est), cond)
    for epoch in range(epochs):
        model.train()
        total=0; totaux=0; nb=0
        for x0, cond in loader:
            x0, cond = x0.to(device_), cond.to(device_)
            b=x0.shape[0]
            t=torch.randint(0,cfg.T,(b,),device=device_)
            noise=torch.randn_like(x0)
            x_t=diffusion.q_sample(x0,t,noise)
            opt.zero_grad()
            eps_pred=model(x_t,t,cond)
            loss=mse(eps_pred, noise)
            if cfg.aux_loss_weight>0:
                t_aux=torch.randint(0,cfg.T//4,(cfg.aux_timesteps*b,),device=device_)
                x_aux=diffusion.q_sample(x0.repeat(cfg.aux_timesteps,1), t_aux, torch.randn(cfg.aux_timesteps*b,cfg.d,device=device_))
                eps_aux=model(x_aux,t_aux,cond.repeat(cfg.aux_timesteps,1))
                x0_est=diffusion.predict_x0_from_eps(x_aux,t_aux,eps_aux)
                aux=aux_loss(x0_est, cond.repeat(cfg.aux_timesteps,1))
                w=diffusion.alpha_bar[t_aux]*cfg.aux_loss_weight
                loss=loss+(w*aux).mean()
                totaux+=(w*aux).mean().item()*b
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); ema.update(model)
            total+=mse(eps_pred,noise).item()*b
            nb+=b
        if epoch%20==0:
            print(f" {tag} epoch {epoch} eps {total/nb:.4f} aux {totaux/nb:.4f}")
    ema.apply_to(model)
          
    torch.save(model.state_dict(), CKPT_DIR/f"diffusion_{tag}.pt")
    torch.save(surrogate.state_dict(), CKPT_DIR/f"surrogate_{tag}.pt")
    return model, surrogate

                                                              
                                                          
                                                              
                      
cfg_noaux=DiffusionConfig(aux_loss_weight=0.0, epochs=80, seed=0)
model_noaux, sur_noaux = train_diffusion_variant(cfg_noaux, "noaux", epochs=80)

                                                                  
                                                                                                                      
                                                                                 

def train_no_struct(tag="nostruct", epochs=80):
    print(f"\n=== training {tag} ===")
    cfg=DiffusionConfig(n_structural=0, epochs=80, seed=0)
                                                                         
    device_=resolve_device(cfg)
    torch.manual_seed(cfg.seed)
    df=get_records_df()
    from src.data.magpie import element_features
    from src.data.descriptors import structural_features
    def comp_to_x0_nostruct(a,b,x,cfg_):
        return np.concatenate([element_features(a),element_features(b),element_features(x)]).astype(np.float32)
    x0=np.stack([comp_to_x0_nostruct(r.a,r.b,r.x,cfg) for r in load_records()]).astype(np.float32)
    props=df[["bandgap","ehull"]].to_numpy(dtype=np.float32)
                   
    x_mean=x0.mean(0); x_std=x0.std(0)+1e-8
    p_mean=props.mean(0); p_std=props.std(0)+1e-8
    xn=(x0-x_mean)/x_std; y=(props-p_mean)/p_std
                     
    rng=np.random.RandomState(cfg.seed)
    idx=rng.permutation(len(xn))[:int(0.9*len(xn))]
    xn_tr=xn[idx]; y_tr=y[idx]
               
    class SurrogateNoStruct(torch.nn.Module):
        def __init__(self,d): super().__init__(); self.net=torch.nn.Sequential(torch.nn.Linear(d,128), torch.nn.SiLU(), torch.nn.Linear(128,128), torch.nn.SiLU(), torch.nn.Linear(128,128), torch.nn.SiLU(), torch.nn.Linear(128,2))
        def forward(self,x): return self.net(x)
    sur=SurrogateNoStruct(cfg.d).to(device_)
    opt=torch.optim.Adam(sur.parameters(), lr=1e-3)
    Xt=torch.as_tensor(xn_tr,device=device_); yt=torch.as_tensor(y_tr,device=device_)
    for ep in range(150):
        perm=torch.randperm(len(Xt),device=device_)
        for i in range(0,len(Xt),64):
            id2=perm[i:i+64]
            opt.zero_grad()
            loss=torch.nn.MSELoss()(sur(Xt[id2]), yt[id2])
            loss.backward(); opt.step()
    sur.eval()
    for p in sur.parameters(): p.requires_grad_(False)
               
    model=Denoiser(d=cfg.d, cond_dim=2, hidden=256, n_blocks=5, t_emb_dim=64, cond_emb_dim=128, null_prob=0.15).to(device_)
    ema=EMAModel(model)
    diffusion=GaussianDiffusion(T=cfg.T, s=cfg.s, device=str(device_))
    opt=torch.optim.Adam(model.parameters(), lr=1e-3)
    mse=torch.nn.MSELoss()
    loader=torch.utils.data.DataLoader(list(zip(torch.as_tensor(xn_tr), torch.as_tensor(y_tr))), batch_size=64, shuffle=True, drop_last=True)
    for epoch in range(epochs):
        tot=0; nb=0
        for x0_, cond_ in loader:
            x0_,cond_=x0_.to(device_), cond_.to(device_)
            b=x0_.shape[0]
            t=torch.randint(0,cfg.T,(b,),device=device_)
            noise=torch.randn_like(x0_)
            x_t=diffusion.q_sample(x0_,t,noise)
            opt.zero_grad()
            eps_pred=model(x_t,t,cond_)
            loss=mse(eps_pred,noise)
                 
            t_aux=torch.randint(0,cfg.T//4,(3*b,),device=device_)
            x_aux=diffusion.q_sample(x0_.repeat(3,1), t_aux, torch.randn(3*b,cfg.d,device=device_))
            eps_aux=model(x_aux,t_aux,cond_.repeat(3,1))
            x0_est=diffusion.predict_x0_from_eps(x_aux,t_aux,eps_aux)
            aux=mse(sur(x0_est), cond_.repeat(3,1))
            w=diffusion.alpha_bar[t_aux]*cfg.aux_loss_weight
            loss=loss+(w*aux).mean()
            loss.backward(); opt.step(); ema.update(model)
            tot+=mse(eps_pred,noise).item()*b; nb+=b
        if epoch%20==0: print(f" {tag} epoch {epoch} eps {tot/nb:.4f}")
    ema.apply_to(model)
    torch.save(model.state_dict(), CKPT_DIR/f"diffusion_{tag}.pt")
                                 
    np.savez(CKPT_DIR/f"stats_{tag}.npz", x_mean=x_mean, x_std=x_std, prop_mean=p_mean, prop_std=p_std)
    return model, sur, (x_mean,x_std,p_mean,p_std)

model_nostruct, sur_nostruct, stats_nostruct = train_no_struct("nostruct", epochs=80)

                                                                                     
def eval_model_variant(model_path, surrogate_path, stats_path, d, w=2.0, n=1500, target=1.4):
    ckpt_model=torch.load(model_path, map_location=device)
                               
    m=Denoiser(d=d, cond_dim=2, hidden=256, n_blocks=5, t_emb_dim=64, cond_emb_dim=128, null_prob=0.0).to(device)
    m.load_state_dict(ckpt_model); m.eval()
    sur_ckpt=torch.load(surrogate_path, map_location=device)
                       
    if d==55:
        sur=Surrogate(d).to(device)
    else:
        class S(torch.nn.Module):
            def __init__(self,d): super().__init__(); self.net=torch.nn.Sequential(torch.nn.Linear(d,128), torch.nn.SiLU(), torch.nn.Linear(128,128), torch.nn.SiLU(), torch.nn.Linear(128,128), torch.nn.SiLU(), torch.nn.Linear(128,2))
            def forward(self,x): return self.net(x)
        sur=S(d).to(device)
    sur.load_state_dict(sur_ckpt); sur.eval()
    if stats_path.exists():
        st=np.load(stats_path)
        x_mean_,x_std_,p_mean_,p_std_ = st["x_mean"], st["x_std"], st["prop_mean"], st["prop_std"]
    else:
        x_mean_,x_std_,p_mean_,p_std_ = stats["x_mean"], stats["x_std"], stats["prop_mean"], stats["prop_std"]
    diffusion_=GaussianDiffusion(T=300, s=0.008, device=str(device))
                                                                       
                                                                                                                                                                                                                      
                                                                                                                                                                                                                               
                                                                                                                                                                                        
                                           
              
    cond=torch.as_tensor((np.array([target,0.0],dtype=np.float32)-p_mean_)/p_std_, dtype=torch.float32, device=device).unsqueeze(0)
    cands=[]
    for start in range(0,n,512):
        bs=min(512,n-start)
        c=cond.expand(bs,-1)
        x_gen=diffusion_.ddim_sample(m, (bs,d), cond=c, steps=50, eta=0.0, guidance_w=w).detach().cpu().numpy()
                                                                                                
        if d==45:
                                                                            
                                                                      
                                                                                                                                                                   
                                                         
                                                                
            from src.data.magpie import site_table
            from src.config import A_SITES,B_SITES,X_SITES
            A_table=(site_table(A_SITES)-x_mean_[:15])/x_std_[:15] if d==45 else None                                 
                                                                                                  
                                                                                                  
            pass
        else:
                             
            dec=Decoder(cfg0, {"x_mean":x_mean_,"x_std":x_std_}) if d==55 else None
                                                                                                                                       
            pass
                                                                                                                                          
                                                  
                                                              
                                                  
    cond_exp=cond.expand(n,-1)
               
    xs=[]
    for start in range(0,n,512):
        bs=min(512,n-start)
        c=cond.expand(bs,-1)
        x_gen=diffusion_.ddim_sample(m, (bs,d), cond=c, steps=50, eta=0.0, guidance_w=w).detach()
                                            
        with torch.no_grad():
            pred=sur(torch.as_tensor(x_gen)).cpu().numpy()
                             
        pb=pred[:,0]*p_std_[0]+p_mean_[0]
        xs.append(pb)
    pred_all=np.concatenate(xs)
    mae=float(np.mean(np.abs(pred_all-target)))
    return {"MAE_ML_raw (pred)":mae, "mean_pred":float(pred_all.mean())}

            
res={}
res["baseline (full aux)"]=eval_model_variant(CKPT_DIR/"diffusion.pt", CKPT_DIR/"surrogate.pt", Path("nonexist"), d=55, w=2.0)
res["no_aux"]=eval_model_variant(CKPT_DIR/"diffusion_noaux.pt", CKPT_DIR/"surrogate_noaux.pt", Path("nonexist"), d=55, w=2.0)
res["no_struct"]={"MAE_ML_raw (pred)": float("nan"), "mean_pred": float("nan")}                                                               
print(res)
pd.DataFrame(res).T.to_csv(TAB/"ablation_results.csv")

                                                              
                     
                                                              
print("\n=== DDIM STEPS ===")
model_full=Denoiser(d=cfg0.d, cond_dim=2, hidden=256, n_blocks=5, t_emb_dim=64, cond_emb_dim=128, null_prob=0.0).to(device)
model_full.load_state_dict(torch.load(CKPT_DIR/"diffusion.pt", map_location=device)); model_full.eval()
sur_full=Surrogate(cfg0.d).to(device); sur_full.load_state_dict(torch.load(CKPT_DIR/"surrogate.pt", map_location=device)); sur_full.eval()
diffusion_full=GaussianDiffusion(T=cfg0.T, s=cfg0.s, device=str(device))
decoder_full=Decoder(cfg0, stats)
ddim_rows=[]
for steps in [10,25,50,100]:
    import time
    t0=time.time()
    cond=torch.as_tensor((np.array([1.4,0.0],dtype=np.float32)-stats["prop_mean"])/stats["prop_std"], dtype=torch.float32, device=device).unsqueeze(0).expand(1500, -1)
           
    cands=[]
    for start in range(0,1500,512):
        bs=min(512,1500-start)
        c=cond[start:start+bs]
        x_gen=diffusion_full.ddim_sample(model_full, (bs,cfg0.d), cond=c, steps=steps, eta=0.0, guidance_w=2.0).detach().cpu().numpy()
        cands.extend(decoder_full.decode_batch(x_gen))
    valid=decoder_full.filter(cands, tol_min=0.8, tol_max=1.0, require_charge_balance=True, novelty=False)
    elapsed=time.time()-t0
    if valid:
        X=np.stack([composition_to_x0(c.a,c.b,c.x,cfg0) for c in valid],dtype=np.float32)
        Xn=(X-stats["x_mean"])/stats["x_std"]
        with torch.no_grad():
            pred=sur_full(torch.as_tensor(Xn,device=device)).cpu().numpy()
        pred_bg=pred[:,0]*stats["prop_std"][0]+stats["prop_mean"][0]
        mae=float(np.mean(np.abs(pred_bg-1.4)))
        valid_frac=len(valid)/1500
    else:
        mae=np.nan; valid_frac=0
    ddim_rows.append({"steps":steps,"MAE_ML":mae,"valid_frac":valid_frac,"time":elapsed,"n_valid":len(valid) if valid else 0})
    print(ddim_rows[-1])
pd.DataFrame(ddim_rows).to_csv(TAB/"ddim_ablation.csv", index=False)
fig, ax=plt.subplots(figsize=(6,4))
ax.plot([r["steps"] for r in ddim_rows],[r["MAE_ML"] for r in ddim_rows], marker='o')
ax.set_xlabel("DDIM steps"); ax.set_ylabel("Target MAE_ML (pred, eV)"); ax.set_title("Sampling steps vs quality (MAE_ML pred)")
fig.tight_layout(); fig.savefig(FIG/"ddim_ablation.png", dpi=150); plt.close(fig)

                                                              
                           
                                                              
print("\n=== SEED ROBUSTNESS ===")
                                                                                                                                                                    
seed_rows=[]
for s in [0,1,2]:
    torch.manual_seed(s); np.random.seed(s)
                   
    cond=torch.as_tensor((np.array([1.4,0.0],dtype=np.float32)-stats["prop_mean"])/stats["prop_std"], dtype=torch.float32, device=device).unsqueeze(0).expand(1500,-1)
    cands=[]
                                     
    for start in range(0,1500,512):
        bs=min(512,1500-start)
        c=cond[start:start+bs]
                                                             
        x_gen=diffusion_full.ddim_sample(model_full, (bs,cfg0.d), cond=c, steps=50, eta=0.0, guidance_w=2.0).detach().cpu().numpy()
        cands.extend(decoder_full.decode_batch(x_gen))
    valid=decoder_full.filter(cands, tol_min=0.8, tol_max=1.0, require_charge_balance=True, novelty=False)
    if valid:
        X=np.stack([composition_to_x0(c.a,c.b,c.x,cfg0) for c in valid],dtype=np.float32)
        Xn=(X-stats["x_mean"])/stats["x_std"]
        with torch.no_grad():
            pred=sur_full(torch.as_tensor(Xn,device=device)).cpu().numpy()
        pred_bg=pred[:,0]*stats["prop_std"][0]+stats["prop_mean"][0]
        mae=float(np.mean(np.abs(pred_bg-1.4)))
        novel=np.mean([c.formula not in set(get_records_df().apply(lambda r: f"{r['a']}{r['b']}{r['x']}3",axis=1)) for c in valid])
                   
        if len(Xn)>500:
            idx=np.random.RandomState(s).choice(len(Xn),500,replace=False)
            Xn2=Xn[idx]
            D=Xn2[:,None,:]-Xn2[None,:,:]
            div=float(np.sqrt((D**2).sum(-1))[np.triu_indices(len(Xn2),1)].mean())
        else:
            D=Xn[:,None,:]-Xn[None,:,:]; div=float(np.sqrt((D**2).sum(-1))[np.triu_indices(len(Xn),1)].mean()) if len(Xn)>1 else 0
        seed_rows.append({"seed":s,"MAE_ML":mae,"valid":len(valid)/1500,"novel":novel,"diversity":div})
    else:
        seed_rows.append({"seed":s,"MAE_ML":np.nan,"valid":0,"novel":0,"diversity":0})
    print(seed_rows[-1])
pd.DataFrame(seed_rows).to_csv(TAB/"seed_robustness.csv", index=False)

                                                              
                                      
                                                              
print("\n=== SCALING ===")
                                                                                                                                                              
df_all=get_records_df()
scaling_rows=[]
for frac in [0.25,0.5,0.75,1.0]:
    rng=np.random.RandomState(0)
    n=int(frac*len(df_all))
    idx=rng.choice(len(df_all), n, replace=False)
    df_sub=df_all.iloc[idx]
                    
    x0=np.stack([composition_to_x0(r.a,r.b,r.x,cfg0) for r in df_sub.itertuples()],dtype=np.float32)
    props=df_sub[["bandgap","ehull"]].to_numpy(dtype=np.float32)
                                      
    rng2=np.random.RandomState(1)
    n_test=int(0.2*len(df_sub))
    test_idx=rng2.choice(len(df_sub), n_test, replace=False)
    train_idx=np.array([i for i in range(len(df_sub)) if i not in set(test_idx)])
    x_mean, x_std = x0[train_idx].mean(0), x0[train_idx].std(0)+1e-8
    p_mean, p_std = props[train_idx].mean(0), props[train_idx].std(0)+1e-8
    Xn=(x0-x_mean)/x_std; y=(props-p_mean)/p_std
                                
    from src.models.surrogate import Surrogate
    sur=Surrogate(cfg0.d).to(device)
    opt=torch.optim.Adam(sur.parameters(), lr=1e-3)
    Xt=torch.as_tensor(Xn[train_idx],device=device); yt=torch.as_tensor(y[train_idx],device=device)
    for ep in range(300):
        perm=torch.randperm(len(Xt),device=device)
        for i in range(0,len(Xt),64):
            id2=perm[i:i+64]
            opt.zero_grad()
            loss=torch.nn.MSELoss()(sur(Xt[id2]), yt[id2])
            loss.backward(); opt.step()
    sur.eval()
    with torch.no_grad():
        pred=sur(torch.as_tensor(Xn[test_idx],device=device)).cpu().numpy()*p_std+p_mean
        true=props[test_idx]
        mae_bg=float(np.mean(np.abs(pred[:,0]-true[:,0])))
        mae_eh=float(np.mean(np.abs(pred[:,1]-true[:,1])))
    scaling_rows.append({"frac":frac,"n":n,"mae_bg":mae_bg,"mae_eh":mae_eh})
    print(scaling_rows[-1])
pd.DataFrame(scaling_rows).to_csv(TAB/"scaling.csv", index=False)
fig, ax=plt.subplots(figsize=(6,4))
ax.plot([r["frac"] for r in scaling_rows],[r["mae_bg"] for r in scaling_rows], marker='o', label='bandgap MAE')
ax.plot([r["frac"] for r in scaling_rows],[r["mae_eh"] for r in scaling_rows], marker='s', label='Ehull MAE')
ax.set_xlabel("fraction of data"); ax.set_ylabel("MAE"); ax.legend(); ax.set_title("Scaling: dataset size vs surrogate error")
fig.tight_layout(); fig.savefig(FIG/"scaling.png", dpi=150); plt.close(fig)

                                                              
                                             
                                                              
print("\n=== GENERATION UNDER OOD ===")
                                                                                     
                                                                                                                
                                                                                                                  
                                                                                                    
HELD = ["Y","La","Ce","Pr","Nd","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu"]
                       
train_b_counts=get_records_df()['b'].value_counts(normalize=True)
held_freq_train=sum(train_b_counts.get(b,0) for b in HELD)
print(f"held B freq in train {held_freq_train:.3f}")
                              
from collections import Counter
               
cond=torch.as_tensor((np.array([1.4,0.0],dtype=np.float32)-stats["prop_mean"])/stats["prop_std"], dtype=torch.float32, device=device).unsqueeze(0).expand(5000,-1)
cands=[]
for start in range(0,5000,512):
    bs=min(512,5000-start)
    x_gen=diffusion_full.ddim_sample(model_full, (bs,cfg0.d), cond=cond[start:start+bs], steps=50, eta=0.0, guidance_w=2.0).detach().cpu().numpy()
    cands.extend(decoder_full.decode_batch(x_gen))
valid=decoder_full.filter(cands, tol_min=0.8, tol_max=1.0, require_charge_balance=True, novelty=False)
if valid:
    cnt=Counter(c.b for c in valid)
    held_freq_gen=sum(cnt.get(b,0) for b in HELD)/len(valid) if valid else 0
    print(f"held freq in generated valid {held_freq_gen:.3f} ({sum(cnt.get(b,0) for b in HELD)}/{len(valid)})")
                                                                     
    ood_row={"held_train_freq":held_freq_train,"held_gen_freq":held_freq_gen,"n_valid":len(valid)}
    pd.DataFrame([ood_row]).to_csv(TAB/"ood_generation_proxy.csv", index=False)
                                                                                                                       
                                                                                                                                                  
                                                                                                                                      
                                                                            
    from src.evaluate_surrogate import split_train_test, build_matrices
    df_,x0_,props_,Xn_,y_,tr_,te_, (pm,ps)=split_train_test(cfg0,"hard_b")
                                                                                        
    x_mean_tr, x_std_tr = x0_[tr_].mean(0), x0_[tr_].std(0)+1e-8
    Xn_all=(x0_-x_mean_tr)/x_std_tr
    D=cdist(Xn_all[te_], Xn_all[tr_])
    d_min_test=D.min(axis=1)
                                     
    from src.evaluate_surrogate import fit_surrogate
    sur_hard=fit_surrogate(Xn_[tr_], y_[tr_], device, epochs=400)
    with torch.no_grad():
        pred=sur_hard(torch.as_tensor(Xn_[te_],device=device)).cpu().numpy()*ps+pm
    true=props_[te_]
    err_bg=np.abs(pred[:,0]-true[:,0])
                 
    corr=np.corrcoef(d_min_test, err_bg)[0,1]
    print(f"hard_b error vs d_min corr {corr:.3f}")
    fig, ax=plt.subplots(figsize=(5,4))
    ax.scatter(d_min_test, err_bg, alpha=0.5, s=12)
    ax.set_xlabel("d_min to train (OOD distance)")
    ax.set_ylabel("|bandgap error| (eV)")
    ax.set_title(f"Surrogate error vs OOD distance (hard_b)\ncorr={corr:.2f}")
    fig.tight_layout(); fig.savefig(FIG/"error_vs_ood_distance.png", dpi=150); plt.close(fig)
    pd.DataFrame({"d_min":d_min_test,"err_bg":err_bg}).to_csv(TAB/"error_vs_ood.csv", index=False)

                                                              
                                                                 
                                                              
print("\n=== DFT VALIDATION PREP ===")
                                                                                                                            
                                                                                                  
                                                                            
                                                      
try:
    top=pd.read_csv(OUT/"candidates.csv").head(20)
                                                          
    df_map=get_records_df().set_index("formula")                                              
                                         
    mp_dict={}
    for _,r in get_records_df().iterrows():
        f=f"{r['a']}{r['b']}{r['x']}3"
        mp_dict[f]=(r['bandgap'], r['ehull'])
    rows=[]
    for _,r in top.iterrows():
        f=r['formula']
        dft=mp_dict.get(f, (np.nan, np.nan))
        rows.append({"formula":f,"pred_bg":r['pred_bandgap'],"pred_ehull":r['pred_ehull'],"dft_bg":dft[0],"dft_ehull":dft[1],"error_bg":r['pred_bandgap']-dft[0] if not np.isnan(dft[0]) else np.nan,"NNd":r['dist_a']+r['dist_b']+r['dist_x']})
    pd.DataFrame(rows).to_csv(TAB/"dft_validation_top20.csv", index=False)
    print(pd.DataFrame(rows).head(10).to_string(index=False))
                                                                    
except Exception as e:
    print("dft prep failed", e)

print("\n=== stage3 done ===")
