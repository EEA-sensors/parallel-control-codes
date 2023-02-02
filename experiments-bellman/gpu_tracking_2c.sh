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

python run_linear_experiments.py model=tracking nc=100 start=5 stop=5 num=1 single=0 device=/GPU:0 mask=224
python run_linear_experiments.py model=tracking nc=150 start=5 stop=5 num=1 single=0 device=/GPU:0 mask=224
python run_linear_experiments.py model=tracking nc=200 start=5 stop=5 num=1 single=0 device=/GPU:0 mask=224
python run_linear_experiments.py model=tracking nc=250 start=5 stop=5 num=1 single=0 device=/GPU:0 mask=224
python run_linear_experiments.py model=tracking nc=300 start=5 stop=5 num=1 single=0 device=/GPU:0 mask=224
python run_linear_experiments.py model=tracking nc=350 start=5 stop=5 num=1 single=0 device=/GPU:0 mask=224
python run_linear_experiments.py model=tracking nc=400 start=5 stop=5 num=1 single=0 device=/GPU:0 mask=224
python run_linear_experiments.py model=tracking nc=450 start=5 stop=5 num=1 single=0 device=/GPU:0 mask=224
python run_linear_experiments.py model=tracking nc=500 start=5 stop=5 num=1 single=0 device=/GPU:0 mask=224
