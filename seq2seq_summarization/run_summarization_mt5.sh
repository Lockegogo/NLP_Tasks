export OUTPUT_DIR=./summ_mt5_results/

python3 run_summarization_mt5.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=mT5 \
    --model_checkpoint=../models/mT5_multilingual_XLSum \
    --train_file=../data/lcsts/train.txt \
    --dev_file=../data/lcsts/dev.txt \
    --test_file=../data/lcsts/test.txt \
    --max_input_length=512 \
    --max_target_length=32 \
    --learning_rate=1e-5 \
    --num_train_epochs=3 \
    --batch_size=32 \
    --beam_search_size=4 \
    --no_repeat_ngram_size=2 \
    --do_train \
    --warmup_proportion=0. \
    --seed=42
