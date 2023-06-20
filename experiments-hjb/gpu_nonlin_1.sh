# Run HJB experiment on CPU and GPU
#
# Usage: run_nonlin_experiments.py [model=velocity] [start=2] [stop=3] [num=5] [steps=10] [n=20] [single=0] [device=/CPU:0] [mask=65535]
#
# Mask:
# 1 = seq_upwind
# 2 = seq_assoc
# 4 = par_assoc

python run_nonlin_experiments.py start=2 stop=4 num=20 n=20 device=/GPU:0 mask=7
#python run_nonlin_experiments.py start=2 stop=4 num=20 n=20 device=/CPU:0 mask=7
python run_nonlin_experiments.py start=2 stop=4 num=20 n=40 device=/GPU:0 mask=7
#python run_nonlin_experiments.py start=2 stop=4 num=20 n=40 device=/CPU:0 mask=7
python run_nonlin_experiments.py start=2 stop=4 num=20 n=60 device=/GPU:0 mask=7
#python run_nonlin_experiments.py start=2 stop=4 num=20 n=60 device=/CPU:0 mask=7
python run_nonlin_experiments.py start=2 stop=4 num=20 n=80 device=/GPU:0 mask=7
#python run_nonlin_experiments.py start=2 stop=4 num=20 n=80 device=/CPU:0 mask=7
python run_nonlin_experiments.py start=2 stop=4 num=20 n=100 device=/GPU:0 mask=7
#python run_nonlin_experiments.py start=2 stop=4 num=20 n=100 device=/CPU:0 mask=7
