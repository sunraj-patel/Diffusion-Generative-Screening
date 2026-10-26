"""Write work/arms.json: number of structures per arm and measured compute cost (GPU hours from Slurm accounting in work/gen_costs.json)."""
import gzip, json, os
n = lambda p: sum(1 for _ in gzip.open(p, "rt"))
C = json.load(open("work/gen_costs.json"))                                                      
legacy_unique = len(open("work/legacy_compositions.csv").read().strip().splitlines()) - 1
arms = {
 "gap14": dict(label="MatterGen, gap-conditioned (1.4 eV)", n_generated=n("work/pool_gap14.jsonl.gz"), cost=f"{C['gap14']['gpu_hours']:.1f} V100-h"),
 "base": dict(label="MatterGen, unconditional", n_generated=n("work/pool_base.jsonl.gz"), cost=f"{C['base']['gpu_hours']:.1f} V100-h"),
 "gap14g4": dict(label="MatterGen, gap-conditioned, guidance 4", n_generated=n("work/gen/gap14g4_s4/pool.jsonl.gz"), cost=f"{C['gap14g4']['gpu_hours']:.1f} V100-h"),
 "enum": dict(label="Prototype substitution", n_generated=n("work/pool_enum.jsonl.gz"), cost="CPU, minutes"),
 "legacy": dict(label="Legacy 45-D diffusion + prototypes", n_generated=legacy_unique, cost="CPU, minutes (GPU-free inference)"),
}
json.dump(arms, open("work/arms.json", "w"), indent=1); print(arms)
