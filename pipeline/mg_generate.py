"""Thin wrapper to call mattergen.scripts.generate.main with list-valued sampling_config_overrides (Fire cannot parse them
reliably through the shell). usage: python mg_generate.py out_dir model_dir batch_size num_batches seed guidance props_json overrides_json"""
import json, sys
from mattergen.scripts.generate import main

out, model, bs, nb, seed, guid, props, ovr = sys.argv[1:9]
main(output_path=out, model_path=model, batch_size=int(bs), num_batches=int(nb), seed=int(seed),
     diffusion_guidance_factor=float(guid), properties_to_condition_on=json.loads(props) or None,
     sampling_config_overrides=json.loads(ovr) or None, record_trajectories=False)
