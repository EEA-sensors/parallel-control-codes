# Nonlinear experiments-bellman
#
# Usage: run_nonlinear_experiments.py [start=2] [stop=5] [num=20] [single=0] [device=/CPU:0] [mask=7]
#
# Mask:
# 1 = seq
# 2 = par_1
# 4 = par_2

#python run_nonlinear_experiments.py start=2 stop=5 num=20 single=1 device=/CPU:0 mask=1
#python run_nonlinear_experiments.py start=2 stop=5 num=20 single=0 device=/CPU:0 mask=6

python run_nonlinear_experiments.py start=2 stop=5 num=20 single=0 device=/CPU:0 mask=1
