MODEL_NAME_OR_PATH=dmis-lab/biobert-base-cased-v1.1
# MODEL_NAME_OR_PATH='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
# MODEL_NAME_OR_PATH=emilyalsentzer/Bio_ClinicalBERT
# MODEL_NAME_OR_PATH='sultan/BioM-ELECTRA-Large-Discriminator'
# MODEL_NAME_OR_PATH='michiyasunaga/BioLinkBERT-base'
# ps -ef|grep main.py|grep -v grep|cut -c 9-15|xargs kill -9
# conda activate wsd 

# OUTPUT_DIR=./tmp/biosyn-biobert-ncbi-disease
# DATA_DIR=../datasets/ncbi-disease

# python train.py \
#     --model_name_or_path ${MODEL_NAME_OR_PATH} \
#     --train_dictionary_path ${DATA_DIR}/train_dictionary.txt \
#     --train_dir ${DATA_DIR}/processed_traindev \
#     --output_dir ${OUTPUT_DIR} \
#     --use_cuda \
#     --topk 20 \
#     --epoch 10 \
#     --train_batch_size 16\
#     --learning_rate 1e-5 \
#     --max_length 25 \
#     --save_checkpoint_all \
#     --seed 42


OUTPUT_DIR=./tmp/biosyn-biobert-ncbi
DATA_DIR=../NCBI

python train.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --train_dictionary_path ${DATA_DIR}/train_dictionary.txt \
    --train_dir ${DATA_DIR}/train_data.txt \
    --output_dir ${OUTPUT_DIR} \
    --use_cuda \
    --topk 20 \
    --epoch 10 \
    --train_batch_size 16\
    --learning_rate 1e-5 \
    --max_length 25 \
    --save_checkpoint_all \
    --seed 42