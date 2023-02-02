# Finite-state experiment
#
# Usage: run_finite_experiments.py [start=2] [stop=5] [num=20] [nx=21] [single=0] [device=/CPU:0] [mask=31]
#
# Mask:
# 1 = seq_bw
# 2 = par_bw
# 4 = seq_bwfw
# 8 = par_bwfw
# 16 = par_bwfw

#python run_finite_experiments.py start=2 stop=5 num=20 nx=5 single=1 device=/CPU:0 mask=5
#python run_finite_experiments.py start=2 stop=5 num=20 nx=5 single=0 device=/CPU:0 mask=26
#python run_finite_experiments.py start=2 stop=5 num=20 nx=11 single=1 device=/CPU:0 mask=5
#python run_finite_experiments.py start=2 stop=5 num=20 nx=11 single=0 device=/CPU:0 mask=26
#python run_finite_experiments.py start=2 stop=5 num=20 nx=21 single=1 device=/CPU:0 mask=5
#python run_finite_experiments.py start=2 stop=5 num=20 nx=21 single=0 device=/CPU:0 mask=26

python run_finite_experiments.py start=2 stop=5 num=20 nx=5 single=0 device=/CPU:0 mask=5
python run_finite_experiments.py start=2 stop=5 num=20 nx=11 single=0 device=/CPU:0 mask=5
python run_finite_experiments.py start=2 stop=5 num=20 nx=21 single=0 device=/CPU:0 mask=5
