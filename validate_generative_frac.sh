DATASETS="data/class1.npy,data/class2.npy,data/class3.npy"

D=2
FRAC=$1
python main.py --X $DATASETS --model gen --d $D --frac $FRAC --permu balance

