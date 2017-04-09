DATASETS="data/class1.npy,data/class2.npy,data/class3.npy"
FRAC=1.0
D=2
OUTPUT=$1
python main.py --task train --output $OUTPUT --X $DATASETS --model gen --d $D --frac $FRAC --permu unbalance

