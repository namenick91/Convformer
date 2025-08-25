# TERMINAL COMMAND:
# pyenv activate science-venv && cd Documents/VScodeProjects/LSTF-Transformers-paper/convformer && bash scripts/Ablations/ablation3.sh

# INFO:
# multi-scale trend/seasonality (Series Decomp)
#                               (Informer vs Informer_Decomp vs Autoformer)

# A model: 24, 48, 168, 336, 720 (original)
# B model: 24, 48, 168, 336, 720 (modified)

export CUDA_VISIBLE_DEVICES=0

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <root_path> <data_path> <data>" >&2
  exit 64
fi

ROOT_PATH="$1"
DATA_PATH="$2"
DATA_NAME="$3"

# Sweep prediction horizons
PRED_LENS=(24 48 168 336 720)

# Common arguments (dataset-specific are injected from positional args)
COMMON_ARGS=(
  --is_training 1
  --root_path "$ROOT_PATH"
  --data_path "$DATA_PATH"
  --data "$DATA_NAME"
  --features M
  --target OT
  --freq h
  --seq_len 96
  --label_len 48
  --e_layers 2
  --d_layers 1
  --factor 5 #
  --enc_in 7
  --dec_in 7
  --c_out 7
  --use_torch_compile False
  --itr 3
)

# MODEL A (Pure Informer)
model="Informer"
logdir="../logs/ablation3/A"
mkdir -p "$logdir"

for pl in "${PRED_LENS[@]}"; do
  model_id="Ablation3A_96_${pl}_${DATA_NAME}"
  des="ablation_decomp_${pl}_A"

  echo "==> Running ${des} (data=${DATA_NAME}, pred_len=${pl})"
  python -u ../run.py \
    "${COMMON_ARGS[@]}" \
    --model "$model" \
    --pred_len "$pl" \
    --model_id "$model_id" \
    --des "$des" \
    | tee "${logdir}/${model_id}.log"
done

# MODEL B (Informer with explicit series decomposition from Autoformer)
model="Informer_Decomp" 
logdir="../logs/ablation3/B"
mkdir -p "$logdir"

for pl in "${PRED_LENS[@]}"; do
  model_id="Ablation3B_96_${pl}_${DATA_NAME}"
  des="ablation_decomp_${pl}_B"

  echo "==> Running ${des} (data=${DATA_NAME}, pred_len=${pl})"
  python -u ../run.py \
    "${COMMON_ARGS[@]}" \
    --model "$model" \
    --pred_len "$pl" \
    --model_id "$model_id" \
    --des "$des" \
    | tee "${logdir}/${model_id}.log"
done

# MODEL C (Autoformer)
model="Autoformer" 
logdir="../logs/ablation3/C"
mkdir -p "$logdir"

for pl in "${PRED_LENS[@]}"; do
  model_id="Ablation3C_96_${pl}_${DATA_NAME}"
  des="ablation_decomp_${pl}_C"

  echo "==> Running ${des} (data=${DATA_NAME}, pred_len=${pl})"
  python -u ../run.py \
    "${COMMON_ARGS[@]}" \
    --model "$model" \
    --pred_len "$pl" \
    --model_id "$model_id" \
    --des "$des" \
    | tee "${logdir}/${model_id}.log"
done
