"""MatterGen condition loader that draws a chemical system per sample from a list (one batch can mix many A-B-X systems).
Used through sampling_config_overrides:  condition_loader_partial._target_=mg_multisystem.get_loader
No change to MatterGen itself: it only wraps get_number_of_atoms_condition_loader with a custom per-sample transform."""
import random
import torch
from mattergen.common.data.condition_factory import get_number_of_atoms_condition_loader


class SetChemSys:
    def __init__(self, systems, energy_above_hull, seed, log):
        self.systems, self.eah, self.rng, self.log = systems, energy_above_hull, random.Random(seed), log

    def __call__(self, sample):
        cs = self.rng.choice(self.systems)
        if self.log: open(self.log, "a").write(cs + "\n")
        kw = {"chemical_system": cs}
        if self.eah is not None: kw["energy_above_hull"] = torch.tensor(float(self.eah))
        return sample.replace(**kw)


def get_loader(num_atoms_distribution, num_samples, batch_size, systems_file, energy_above_hull=None, seed=0, log=None, shuffle=False, properties=None):
    systems = [l.strip() for l in open(systems_file) if l.strip()]
    return get_number_of_atoms_condition_loader(num_atoms_distribution=num_atoms_distribution, num_samples=num_samples,
        batch_size=batch_size, shuffle=shuffle, transforms=[SetChemSys(systems, energy_above_hull, seed, log)])
