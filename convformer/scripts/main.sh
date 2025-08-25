# pyenv activate science-venv && cd Documents/VScodeProjects/LSTF-Transformers-paper/convformer/scripts && bash main.sh

# Resolve this script’s directory so relative paths are robust
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Path to ablation scripts
ABLAT1_SH="${SCRIPT_DIR}/Ablations/ablation1.sh"
ABLAT2_SH="${SCRIPT_DIR}/Ablations/ablation2.sh"
ABLAT3_SH="${SCRIPT_DIR}/Ablations/ablation3.sh"

source "${SCRIPT_DIR}/ETT_script/Autoformer.sh"
source "${SCRIPT_DIR}/ETT_script/Convformer.sh"
source "${SCRIPT_DIR}/ETT_script/Performer.sh"
source "${SCRIPT_DIR}/ETT_script/Informer.sh"

# -------------------------------------------------------------
# | ABLATION1 | ABLATION1 | ABLATION1 | ABLATION1 | ABLATION1 |
# -------------------------------------------------------------

# Custom
# bash "$ABLAT1_SH" "${SCRIPT_DIR}/../data/raw/synth/" "LOCAL_MOTIFS.csv" "custom"

# ETTh1 (hourly)
# bash "$ABLAT1_SH" "${SCRIPT_DIR}/../data/raw/ETT-small/" "ETTh1.csv" "ETTh1" # ~1h43m  № 1 ------------------------ DONE

# ETTh2 (hourly)
# bash "$ABLAT1_SH" "${SCRIPT_DIR}/../data/raw/ETT-small/" "ETTh2.csv" "ETTh2"

# -------------------------------------------------------------
# | ABLATION2 | ABLATION2 | ABLATION2 | ABLATION2 | ABLATION2 |
# -------------------------------------------------------------

# Custom
# bash "$ABLAT2_SH" "${SCRIPT_DIR}/../data/raw/synth/" "LOCAL_DELAY.csv" "custom"

# ETTh1 (hourly)
# bash "$ABLAT2_SH" "${SCRIPT_DIR}/../data/raw/ETT-small/" "ETTh1.csv" "ETTh1" # ~3h12m  № 2 ------------------------ DONE

# ETTh2 (hourly)
# bash "$ABLAT2_SH" "${SCRIPT_DIR}/../data/raw/ETT-small/" "ETTh2.csv" "ETTh2"

# -------------------------------------------------------------
# | ABLATION3 | ABLATION3 | ABLATION3 | ABLATION3 | ABLATION3 |
# -------------------------------------------------------------

# Custom
# bash "$ABLAT3_SH" "${SCRIPT_DIR}/../data/raw/synth/" "MULTISCALE.csv" "custom"  # ~4h55m № 3 ------------------------ DONE

# ETTh1 (hourly)
# bash "$ABLAT3_SH" "${SCRIPT_DIR}/../data/raw/ETT-small/" "ETTh1.csv" "ETTh1"  # ~4h55m № 4 ------------------------ DONE

# ETTh2 (hourly)
# bash "$ABLAT3_SH" "${SCRIPT_DIR}/../data/raw/ETT-small/" "ETTh2.csv" "ETTh2"

# --------------------------------------------------------
# | INFORMER | INFORMER | INFORMER | INFORMER | INFORMER |
# --------------------------------------------------------

# ETTh1 (hourly)
# run_informer "${SCRIPT_DIR}/../data/raw/ETT-small/" "ETTh1.csv" "ETTh1"  # ~1h43m

# ETTh2 (hourly)
# run_informer "${SCRIPT_DIR}/../data/raw/ETT-small/" "ETTh2.csv" "ETTh2" # ~1h43m № 6 ------------------------ DONE

# ------------------------------------------------------------------
# | AUTOFORMER | AUTOFORMER | AUTOFORMER | AUTOFORMER | AUTOFORMER |
# ------------------------------------------------------------------

# ETTh1 (hourly)
# run_autoformer "${SCRIPT_DIR}/../data/raw/ETT-small/" "ETTh1.csv" "ETTh1"  # ~1h43m

# ETTh2 (hourly)
# run_autoformer "${SCRIPT_DIR}/../data/raw/ETT-small/" "ETTh2.csv" "ETTh2"  # ~1h43m № 7 ------------------------ DONE

# ------------------------------------------------------------------
# | CONVFORMER | CONVFORMER | CONVFORMER | CONVFORMER | CONVFORMER |
# ------------------------------------------------------------------

# ETTh1 (hourly)
run_convformer "${SCRIPT_DIR}/../data/raw/ETT-small/" "ETTh1.csv" "ETTh1"  # ~1h43m № 5 ------------------------ DONE

# ETTh2 (hourly)
run_convformer "${SCRIPT_DIR}/../data/raw/ETT-small/" "ETTh2.csv" "ETTh2"  # ~1h43m № 8 ------------------------ DONE

# -------------------------------------------------------------
# | PERFORMER | PERFORMER | PERFORMER | PERFORMER | PERFORMER |
# -------------------------------------------------------------

# ETTh1 (hourly)
# run_performer "${SCRIPT_DIR}/../data/raw/ETT-small/" "ETTh1.csv" "ETTh1"  # ~1h43m

# ETTh2 (hourly)
# run_performer "${SCRIPT_DIR}/../data/raw/ETT-small/" "ETTh2.csv" "ETTh2"  # ~1h43m № 9 ------------------------ DONE
