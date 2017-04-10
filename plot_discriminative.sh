DATASETS="data/class1.npy,data/class2.npy,data/class3.npy"
LOAD=$1
python main.py --task plot --X $DATASETS --load ${LOAD}.npy --basis ${LOAD}_basis.npy --std ${LOAD}_std.npy --model dis

