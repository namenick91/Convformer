# pyenv activate science-venv && cd Documents/VScodeProjects/LSTF-Transformers-paper/convformer && bash scripts/ETT_script/custom.sh

export CUDA_VISIBLE_DEVICES=0

# convformer (with modified convstem 2)

# 24
python -u run.py \
  --is_training 1 \
  --root_path ./data/raw/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_24 \
  --model Convformer \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 5 \
  --num_rand_features 256 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --use_torch_compile False \
  --des 'ablationStudy_convstem_24_A.3' \
  --itr 3

# 720
python -u run.py \
  --is_training 1 \
  --root_path ./data/raw/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_720 \
  --model Convformer \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 5 \
  --num_rand_features 256 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --use_torch_compile False \
  --des 'ablationStudy_convstem_720_A.3' \
  --itr 3
