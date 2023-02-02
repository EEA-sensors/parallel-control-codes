# Run basic GPU experiments-bellman for the mass model
#
# Usage: run_linear_experiments.py [model=tracking] [start=2] [stop=5] [num=20] [n=4] [nc=4] [single=0] [device=/CPU:0] [mask=65535]

# Mask:
# 1 = sequential_bw
# 2 = parallel_bw
# 4 = sequential_bwfw
# 8 = parallel_bwfw_1
# 16 = parallel_bwfw_2
# 32 = sequential_cond%d_bwfw
# 64 = parallel_cond%d_bwfw_1
# 128 = parallel_cond%d_bwfw_2

#python run_linear_experiments.py model=mass start=2 stop=4 num=20 single=0 n=2 device=/GPU:0 mask=31
#python run_linear_experiments.py model=mass start=2 stop=4 num=20 single=0 n=4  device=/GPU:0 mask=31
#python run_linear_experiments.py model=mass start=2 stop=4 num=20 single=0 n=8  device=/GPU:0 mask=31
#python run_linear_experiments.py model=mass start=2 stop=4 num=20 single=0 n=16 device=/GPU:0 mask=31
#python run_linear_experiments.py model=mass start=2 stop=4 num=20 single=0 n=32 device=/GPU:0 mask=31
#python run_linear_experiments.py model=mass start=2 stop=4 num=20 single=0 n=64 device=/GPU:0 mask=31
#python run_linear_experiments.py model=mass start=2 stop=4 num=20 single=0 n=128 device=/GPU:0 mask=31
python run_linear_experiments.py model=mass start=2 stop=4 num=20 single=0 n=256 device=/GPU:0 mask=31

#python run_linear_experiments.py model=mass start=2 stop=4 num=20 single=1 n=2 device=/GPU:0 mask=5
#python run_linear_experiments.py model=mass start=2 stop=4 num=20 single=1 n=4  device=/GPU:0 mask=5
#python run_linear_experiments.py model=mass start=2 stop=4 num=20 single=1 n=8  device=/GPU:0 mask=5
#python run_linear_experiments.py model=mass start=2 stop=4 num=20 single=1 n=16 device=/GPU:0 mask=5
#python run_linear_experiments.py model=mass start=2 stop=4 num=20 single=1 n=32 device=/GPU:0 mask=5
#python run_linear_experiments.py model=mass start=2 stop=4 num=20 single=1 n=64 device=/GPU:0 mask=5
#python run_linear_experiments.py model=mass start=2 stop=4 num=20 single=1 n=128 device=/GPU:0 mask=5
python run_linear_experiments.py model=mass start=2 stop=4 num=20 single=1 n=256 device=/GPU:0 mask=5
