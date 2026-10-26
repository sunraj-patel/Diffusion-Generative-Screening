#!/bin/bash
cd /scratch/sunraj/diffusion_ext/run/CMB-15/final; export OMP_NUM_THREADS=1
mpirun --bind-to none -np 24 pw.x -nk 4 -in scf.in > scf.out 2> scf.err
mpirun --bind-to none -np 24 pw.x -nk 8 -in nscf.in > nscf.out 2> nscf.err
touch chain.done
