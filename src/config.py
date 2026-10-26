import os
"""Global configuration for the perovskite diffusion project."""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
OUTPUT_DIR = ROOT / "outputs"
for _d in (DATA_DIR, CKPT_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

MP_API_KEY = os.environ.get("MP_API_KEY", "")                                               
MP_BASE_URL = "https://api.materialsproject.org"

RAW_JSONL = DATA_DIR / "abx3_raw.jsonl"
DATASET_CSV = DATA_DIR / "abx3_dataset.csv"
STATS_NPZ = DATA_DIR / "stats.npz"

                                                             
                                                                              
                                                                              
A_SITES = [
    "Li", "Na", "K", "Rb", "Cs", "Fr",
    "Be", "Mg", "Ca", "Sr", "Ba", "Ra",
    "Sc", "Y", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho",
    "Er", "Tm", "Yb", "Lu",
    "Ac", "Th", "Pa", "U", "Np", "Pu",
    "Tl", "In", "Bi", "Pb", "Sn", "Ag", "Au", "Cd", "Hg",
]
B_SITES = [
    "Sc", "Y", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho",
    "Er", "Tm", "Yb", "Lu",
    "Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W", "Mn", "Re",
    "Fe", "Ru", "Os", "Co", "Rh", "Ir", "Ni", "Pd", "Pt", "Cu", "Ag", "Au",
    "Zn", "Cd", "Hg", "Al", "Ga", "In", "Si", "Ge", "Sn", "Pb", "Mg", "Be",
    "B", "C", "N",
]
X_SITES = ["F", "Cl", "Br", "I", "O", "S", "Se", "Te", "N", "P"]

                                                                          
                                                                                        
HALIDES = {"F", "Cl", "Br", "I"}


@dataclass
class DiffusionConfig:
                      
    n_features_site: int = 15
    n_structural: int = 10
    property_dims: int = 2                   

                    
    T: int = 300
    beta_start: float = 1e-4
    beta_end: float = 0.02
    s: float = 0.008                          

                                      
    t_emb_dim: int = 128
    cond_emb_dim: int = 256
    hidden_dim: int = 512
    n_blocks: int = 6
    null_prob: float = 0.2                                                                    
    dropout: float = 0.1

                                                                              
    aux_loss_weight: float = 0.15
    aux_timesteps: int = 4                                                    

              
    lr: float = 5e-4
    batch_size: int = 64
    epochs: int = 400
    seed: int = 0
    device: str = "auto"
                                                           
    balanced_sampling: bool = True
                                                                                                        
    elemental_only: bool = True

              
    ddim_steps: int = 50
    eta: float = 0.0
    guidance_weights: tuple = (1.0, 1.5, 2.0, 2.5, 3.0)
    n_candidates_per_setting: int = 5000
    target_bandgap: float = 1.4
    target_ehull: float = 0.0

    @property
    def d(self):
        if getattr(self, "elemental_only", False):
            return 3 * self.n_features_site
        return 3 * self.n_features_site + self.n_structural

    @property
    def cond_dim(self):
        return self.property_dims
