DATASETS="data/class1_hist.npy,data/class2_hist.npy,data/class3_hist.npy"

D=2
FRAC=0.8
OUTPUT=$1
python main.py --task train --pre hist --output $OUTPUT --X $DATASETS --model dis --d $D --epoch 20 --batch_size 64 --lr 1.0 --frac $FRAC --permu unbalance --tolerance 0.001
