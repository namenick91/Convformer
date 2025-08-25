# Usage (when sourced):
#   run_convformer "<root_path>" "<data_path>" "<data>"
# Example:
#   run_convformer "./data/raw/ETT-small/" "ETTh1.csv" "ETTh1"

run_convformer() {
  local root_path="${1:?root_path required}"
  local data_path="${2:?data_path required}"
  local data="${3:?data required}"

  # Prediction horizons to sweep
  local -a PRED_LENS=(24 48 168 336 720)

  # Shared args (dataset-specific are injected via the function args above)
  local -a COMMON_ARGS=(
    --is_training 1
    --model Convformer
    --root_path "$root_path"
    --data_path "$data_path"
    --data "$data"
    --features M
    --target OT
    --freq h                
    --seq_len 96
    --label_len 48
    --e_layers 2
    --d_layers 1
    --factor 5
    --num_rand_features 256 
    --enc_in 7
    --dec_in 7
    --c_out 7
    --use_torch_compile False
    --itr 3
  )

  # MODEL
  for pl in "${PRED_LENS[@]}"; do
    local model_id="Convformer_96_${pl}_${data}"
    local des="convformer_${pl}"

    local logfile="../logs/convformer/${model_id}.log"
    mkdir -p "$(dirname "$logfile")"

    echo "==> Running ${des} (data=${data}, pred_len=${pl})"
    python -u ../run.py \
      "${COMMON_ARGS[@]}" \
      --pred_len "${pl}" \
      --model_id "${model_id}" \
      --des "${des}" \
      | tee "$logfile"
  done
}

# Allow running this file directly as well:
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <root_path> <data_path> <data>" >&2
    exit 64
  fi
  run_convformer "$1" "$2" "$3"
fi
