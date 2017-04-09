LOAD=$1
python main.py --task plot --load ${LOAD}.npy --basis ${LOAD}_basis.npy --std ${LOAD}_std.npy --model gen

