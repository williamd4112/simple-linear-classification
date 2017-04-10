DATASETS="data/class1.npy,data/class2.npy,data/class3.npy"

FRAC=0.8
D=2
python main.py --X $DATASETS --pre lda --model gen --d $D --frac $FRAC --permu unbalance

