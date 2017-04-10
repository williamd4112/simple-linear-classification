LOAD=$1
MODEL=$2
OUTPUT="./result/DemoTarget.csv"
DATASETS=./data/demo.npy

python dataset_converter.py ./Demo data/demo.npy && \
echo "Demo images convertion done" && \
python main.py --task test --load ${LOAD}.npy --basis ${LOAD}_basis.npy --std ${LOAD}_std.npy --output $OUTPUT --X $DATASETS --model $MODEL

