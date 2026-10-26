"""Builds the local, public-ready release directory release/abx3-perovskite-screening/ (nothing is pushed or published).
Copies code, lightweight tests, configs, aggregate results, candidate structures, DFT inputs/outputs, manuscript and poster sources; writes MANIFEST.sha256.
Excludes: raw MP/Alex downloads, trained ensemble weights (results/models), pseudopotential files, local credentials, caches and anything matching a secret pattern."""
import os, re, shutil, hashlib, sys, fnmatch
OUT = "release/abx3-perovskite-screening"
INCLUDE = [                                                    
    "README_RELEASE.md", "CODE_DATA_AVAILABILITY.md", "NOTICE.md", "requirements.txt", "environment", "pytest.ini", "configs", "pipeline", "src", "tests", "docs",
    "candidates", "reports", "dft/scripts", "dft/preflight.json", "dft/symmetrization_record.json", "dft/candidate_selection.md", "dft/pseudopotential_manifest.csv",
    "dft/structures", "dft/results", "dft/inputs", "paper/main.tex", "paper/main.pdf", "paper/refs.bib", "paper/sections", "paper/generated", "paper/figures", "paper/figures_src",
    "poster/poster.tex", "poster/poster.pdf", "poster/poster_preview.png", "poster/README.md", "poster/figures", "poster/figures_src",
    "results/*.csv", "results/*.json", "work/*.json", "work/*.csv", "work/cand", "work/gen_costs.json", "work/timing", "data/curated/curation_report.json",
]
EXCLUDE_DIRS = {"__pycache__", ".git", ".pytest_cache", "tmp", "unused", "pseudo", "m1", "models", ".ipynb_checkpoints"}
EXCLUDE_FILES = ["*.pyc", ".env*", "*.pem", "*.key", "check_pred*.csv", "*.UPF", "*backup*", "*.bak", ".DS_Store", "*.aux", "*.log", "tmp_*"]
SECRET = re.compile(r"(MP_API_KEY\s*[=:]\s*['\"][A-Za-z0-9]{20,}['\"])|(api[_-]?key['\"]?\s*[=:]\s*['\"][A-Za-z0-9]{20,}['\"])|(-----BEGIN [A-Z ]*PRIVATE KEY-----)|(sun@\d{4})", re.I)
import glob
files = []
for pat in INCLUDE:
    for p in glob.glob(pat):
        if os.path.isdir(p):
            for r, ds, fs in os.walk(p):
                ds[:] = [d for d in ds if d not in EXCLUDE_DIRS]
                files += [os.path.join(r, f) for f in fs]
        else: files.append(p)
files = sorted({f for f in files if not any(fnmatch.fnmatch(os.path.basename(f), x) for x in EXCLUDE_FILES)})
if os.path.exists(OUT): shutil.rmtree(OUT)
bad = []; total = 0
for f in files:
    if os.path.getsize(f) > 25e6: print("skip large", f); continue
    if f.endswith((".py", ".sh", ".sbatch", ".md", ".tex", ".yaml", ".yml", ".txt", ".csv", ".json", ".in", ".out", ".bib")):
        try:
            if SECRET.search(open(f, errors="ignore").read()): bad.append(f); continue
        except Exception: pass
    d = os.path.join(OUT, f); os.makedirs(os.path.dirname(d), exist_ok=True); shutil.copy2(f, d); total += os.path.getsize(f)
if bad: print("SECRET-LIKE CONTENT in (not copied):", bad); sys.exit(1)
with open(os.path.join(OUT, "MANIFEST.sha256"), "w") as fh:
    for r, _, fs in os.walk(OUT):
        for f in sorted(fs):
            if f == "MANIFEST.sha256": continue
            p = os.path.join(r, f); fh.write(f"{hashlib.sha256(open(p, 'rb').read()).hexdigest()}  {os.path.relpath(p, OUT)}\n")
print(f"release: {len(files)} files, {total/1e6:.1f} MB -> {OUT}")
