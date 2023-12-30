# MODEL_NAME_OR_PATH=./tmp/biosyn-biobert-ncbi-disease
# OUTPUT_DIR=./tmp/biosyn-biobert-ncbi-disease

#

MODEL_NAME_OR_PATH=./best/biosyn-biobert-ncbi-disease
OUTPUT_DIR=./best/biosyn-biobert-ncbi-disease
# DATA_DIR=../datasets/ncbi-disease
DATA_DIR=../NCBI_processed

python eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --dictionary_path ${DATA_DIR}/train_dictionary.txt \
    --data_dir ${DATA_DIR}/processed_train \
    --output_dir ${OUTPUT_DIR} \
    --use_cuda \
    --topk 30 \
    --max_length 25 \
    --save_predictions


MODEL_NAME_OR_PATH=./best/biosyn-biobert-ncbi-disease
OUTPUT_DIR=./best/biosyn-biobert-ncbi-disease
# DATA_DIR=../datasets/ncbi-disease
DATA_DIR=../NCBI_processed

python eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --dictionary_path ${DATA_DIR}/test_dictionary.txt \
    --data_dir ${DATA_DIR}/processed_test \
    --output_dir ${OUTPUT_DIR} \
    --use_cuda \
    --topk 20 \
    --max_length 25 \
    --save_predictions


# MODEL_NAME_OR_PATH=./tmp/biosyn-biobert-ncbi
# OUTPUT_DIR=./output
# DATA_DIR=../NCBI

# python eval.py \
#     --model_name_or_path ${MODEL_NAME_OR_PATH} \
#     --dictionary_path ${DATA_DIR}/train_dictionary.txt \
#     --data_dir ${DATA_DIR}/test_data.txt \
#     --output_dir ${OUTPUT_DIR} \
#     --use_cuda \
#     --topk 20 \
#     --max_length 25 \
#     --save_predictions

# python eval.py \
#     --model_name_or_path ${MODEL_NAME_OR_PATH} \
#     --dictionary_path ${DATA_DIR}/train_dictionary.txt \
#     --data_dir ${DATA_DIR}/train_data.txt \
#     --output_dir ${OUTPUT_DIR} \
#     --use_cuda \
#     --topk 20 \
#     --max_length 25 \
#     --save_predictions