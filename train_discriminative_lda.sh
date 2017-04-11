DATASETS="data/class1.npy,data/class2.npy,data/class3.npy"
D=2
OUTPUT=$1
FRAC=$2

python main.py --task train --output $OUTPUT --X $DATASETS --pre lda --model dis --d $D --epoch 100 --batch_size 64 --lr 1.0 --frac $FRAC --permu unbalance --tolerance 0.00
