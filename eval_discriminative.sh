DATASETS="data/class1.npy,data/class2.npy,data/class3.npy"
D=2
LOAD=$1
python main.py --task eval --load ${LOAD}.npy --basis ${LOAD}_basis.npy --std ${LOAD}_std.npy --X $DATASETS --model dis

