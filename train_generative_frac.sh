DATASETS="data/class1.npy,data/class2.npy,data/class3.npy"
D=2
OUTPUT=$1
FRAC=$2
python main.py --task train --output $OUTPUT --X $DATASETS --model gen --d $D --frac $FRAC --permu balance

