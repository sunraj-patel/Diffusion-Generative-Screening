#!/bin/bash
# Runs on the cluster head node (shared /scratch). launch.sh <node> <dir> <np> <nk> <input.in>
# Starts pw.x fully detached on <node>; writes <name>.out/.err/.time and <name>.done when finished.
node=$1; dir=$2; np=$3; nk=$4; f=$5; b=${f%.in}
cat > $dir/$b.job.sh <<EOF
#!/bin/bash
cd $dir; export OMP_NUM_THREADS=1; t0=\$(date +%s)
mpirun --bind-to none -np $np pw.x -nk $nk -in $f > $b.out 2> $b.err
echo \$((\$(date +%s)-t0)) > $b.time; touch $b.done
EOF
ssh -n -o BatchMode=yes $node "setsid nohup bash $dir/$b.job.sh </dev/null >/dev/null 2>&1 &"
