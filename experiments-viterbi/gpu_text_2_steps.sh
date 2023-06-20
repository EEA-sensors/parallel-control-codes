# Run Viterbi text experiment on CPU and GPU
#
# Usage: run_text_experiments_2.py [order=1] [single=0] [steps=-1] [device=/CPU:0] [mask=65535]
#
# Mask:
# 1 = viterbi_seq_bwfw
# 2 = viterbi_seqpar_bwfw
# 4 = viterbi_par_bwfw
# 8 = viterbi_par_bwfwbw


# 100 205 421 865 1774 3642 7474 15341 31485 64620
# 100 141 198 278 391 549 772 1085 1525 2144 3014 4237 5956 8373 11771 16547 23261 32699 45968 64620

for i in 100 141 198 278 391 549 772 1085 1525 2144 3014 4237 5956 8373 11771 16547 23261 32699 45968 64620
do
  python run_text_experiments_2.py order=2 steps=$i device=/GPU:0 mask=3
  python run_text_experiments_2.py order=2 steps=$i device=/CPU:0 mask=3
done

