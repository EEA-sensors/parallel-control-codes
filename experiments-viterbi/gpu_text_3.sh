# Run Viterbi text experiment on CPU and GPU
#
# Usage: run_text_experiments.py [order=1] [single=0] [device=/CPU:0] [mask=65535]
#
# Mask:
# 1 = viterbi_seq_bwfw
# 2 = viterbi_seqpar_bwfw
# 4 = viterbi_par_bwfw
# 8 = viterbi_par_bwfwbw

python run_text_experiments.py order=3 device=/GPU:0 mask=3
#python run_text_experiments.py order=3 device=/CPU:0 mask=3
