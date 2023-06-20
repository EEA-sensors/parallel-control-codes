# Run continuous tracking experiment on CPU and GPU
#
# Usage: run_contlin_experiments.py [model=tracking] [start=2] [stop=3] [num=5] [steps=100] [n=4] [niter=4] [single=0] [device=/CPU:0] [mask=65535]
#
# Mask:
# 1 = seq_bw
# 2 = seq_bw_fw
# 4 = par_bw
# 8 = par_bw_fw
# 16 = par_bw_fwbw
# 32 = parareal_%d_bw
# 64 = parareal_%d_bw_fw
# 128 = parareal_%d_bw_fwbw

# Only with limited range where parareal works and TF does not run out of memory:
python run_contlin_experiments.py start=2.5 stop=3 num=10 device=/CPU:0 mask=255
python run_contlin_experiments.py start=2.5 stop=3 num=10 device=/GPU:0 mask=255

