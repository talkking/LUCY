#!/bin/bash
WORK_DIR=$(pwd)

ME=$(basename "$0")
ME=${ME%.*}
TIMESTAMP=$(date '+%m%d%y-%H%M%S')

OUTPUT_DIR=${WORK_DIR}/outputs/${ME}

MODEL_NAME_OR_PATH="outputs/s2p0"
AUDIO_ENCODER="models/audio-encoder-Qwen2-7B-instruct-weight-base-11wh-tunning"

COMMON_DIR="manifest/AudioQA-1M"

DATA_JSONS="$COMMON_DIR/train.json $COMMON_DIR/train.json $COMMON_DIR/train.json"
EVAL_DATA_JSONS="$COMMON_DIR/eval.json $COMMON_DIR/eval.json $COMMON_DIR/eval.json"
TASKS="AQACONVA RQACONVA RQACONV"

. $(dirname "$0")/parse_data_dir.sh


unset CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$WORK_DIR
TRAINING_SCRPT=vita/scripts/train.py
if [[ -z $DISTRIBUTED_ARGS ]]; then
	LAUNCH_CMD="deepspeed --include localhost:0,1,2,3,4,5,6,7 $TRAINING_SCRPT"
else
	LAUNCH_CMD="torchrun $DISTRIBUTED_ARGS $TRAINING_SCRPT"
fi
$LAUNCH_CMD \
	--deepspeed config/zero2.json\
    --model_type "qwen2" \
	--initialize_additional_modules False \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --audio_encoder $AUDIO_ENCODER \
	--audio_projector_type "identity" \
    --freeze_backbone False \
    --freeze_audio_encoder_adapter True \
    --freeze_audio_encoder True \
    --freeze_tts_adapter False \
    --freeze_embed_tokens False \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 12 \
    --add_codec_target True \
	--num_train_epochs 5 \
    --load_best_model_at_end True \
    --save_steps 400 \
    --save_total_limit 3 \
    --eval_strategy "steps" \
    --eval_steps 400 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 25 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --bf16 True \
    --model_hidden_size 1536 \
    --audio_encoder_hidden_size 1024 \
    --audio_projector_hidden_size 7168 \
    --audio_num_codebook 7 \
    --text_vocab_size 152064 \
    --text_special_tokens 64 \
    --audio_vocab_size 4096 \
    --audio_special_tokens 64 \
    --data_jsons $DATA_JSONS \
    --eval_data_jsons $EVAL_DATA_JSONS \
    --text_additional "EOT" "PAD_T" "BOT" "ANS_T" "TTS" "TQA" "TQAA" \
    --audio_additional "EOA" "PAD_A" "BOA" "ANS_A" "ASR" "AQA" "AQAA" "M29" "F10" "ER" \
    --asr_template /mnt/data/hetinggao/manifest/asr_prompts/asr_template.json \
    --tasks ${TASKS} \
    --output_dir ${OUTPUT_DIR} \
    --sample_rate 16000 \
    --audio_feature_rate 50 \
    --dataloader_num_workers 2 \
    --remove_unused_columns False \
    --max_keep_sample_size $((25*16000)) \
	--tune_text_embed True \
	--tie_word_embeddings True \
	--loss_reduction mean \
	--max_input_length 1500 \
	--use_last_turn_if_codec True \
