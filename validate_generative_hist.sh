DATASETS="data/class1_hist.npy,data/class2_hist.npy,data/class3_hist.npy"
FRAC=0.8
D=2
python main.py --X $DATASETS --pre hist --model gen --d $D --frac $FRAC --permu unbalance

