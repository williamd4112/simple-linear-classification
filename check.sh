LOAD=$1
MODEL=$2
OUTPUT=result/result-check.csv
DATASETS="data/class1_pca_check.npy,data/class2_pca_check.npy,data/class3_pca_check.npy"

python dataset_converter.py ./data/Data_Train/Class1 data/class1_pca_check.npy && \
python dataset_converter.py ./data/Data_Train/Class2 data/class2_pca_check.npy && \
python dataset_converter.py ./data/Data_Train/Class3 data/class3_pca_check.npy &&
echo "Checking images convertion done" && \
python main.py --task test --load ${LOAD}.npy --basis ${LOAD}_basis.npy --std ${LOAD}_std.npy --output $OUTPUT --X $DATASETS --model $MODEL && \
python score.py data/answer_train.csv result/result-check.csv
