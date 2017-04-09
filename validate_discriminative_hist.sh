DATASETS="data/class1_hist.npy,data/class2_hist.npy,data/class3_hist.npy"

D=2
FRAC=0.8
python main.py --task validate --pre hist --X $DATASETS --model dis --d $D --epoch 20 --batch_size 64 --lr 1.0 --frac $FRAC --permu balance --tolerance 0.001
