# Run Viterbi experiment on CPU and GPU
#
# Usage: run_viterbi_experiments.py [model=ge] [start=2] [stop=3] [num=5] [steps=10] [single=0] [device=/CPU:0] [mask=65535]
#
# Mask:
# 1 = viterbi_seq_bwfw
# 2 = viterbi_par_bwfw
# 4 = viterbi_seqpar_bwfw
# 8 = viterbi_par_bwfwbw

python run_viterbi_experiments.py stop=4 num=10 device=/CPU:0 mask=255
python run_viterbi_experiments.py stop=4 num=10 device=/GPU:0 mask=255

