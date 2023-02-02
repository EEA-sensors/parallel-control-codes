# Run partial condensing CPU experiments-bellman for the tracking model
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

python run_linear_experiments.py model=tracking nc=2 start=2 stop=5 num=20 single=0 device=/CPU:0 mask=224
python run_linear_experiments.py model=tracking nc=4 start=2 stop=5 num=20 single=0 device=/CPU:0 mask=224
python run_linear_experiments.py model=tracking nc=8 start=2 stop=5 num=20 single=0 device=/CPU:0 mask=224
python run_linear_experiments.py model=tracking nc=16 start=2 stop=5 num=20 single=0 device=/CPU:0 mask=224
python run_linear_experiments.py model=tracking nc=32 start=2 stop=5 num=20 single=0 device=/CPU:0 mask=224
python run_linear_experiments.py model=tracking nc=64 start=2 stop=5 num=20 single=0 device=/CPU:0 mask=224
python run_linear_experiments.py model=tracking nc=128 start=2 stop=5 num=20 single=0 device=/CPU:0 mask=224


