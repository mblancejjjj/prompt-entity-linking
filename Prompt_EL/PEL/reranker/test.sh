# MODEL_NAME_OR_PATH=./tmp/biosyn-biobert-ncbi-disease
# OUTPUT_DIR=./tmp/biosyn-biobert-ncbi-disease

MODEL_NAME_OR_PATH=./best/biosyn-biobert-ncbi-disease
OUTPUT_DIR=./best/biosyn-biobert-ncbi-disease
DATA_DIR=../datasets/ncbi-disease

python eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --dictionary_path ${DATA_DIR}/test_dictionary.txt \
    --data_dir ${DATA_DIR}/processed_test \
    --output_dir ${OUTPUT_DIR} \
    --use_cuda \
    --topk 20 \
    --max_length 25 \
    --save_predictions