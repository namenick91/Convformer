# A Study of Modular Hybridization for Accuracy and Efficiency of Transformers in Long-Term Time Series Forecasting
A hybrid Transformer model combining a convolutional stem, linear attention (FAVOR+), and series decomposition for Long-Sequence Time-series Forecasting (LSTF).

## Overview
Long-term forecasting of multivariate time series is challenging due to complex patterns in real-world data, such as short-term motifs, long-range cross-series dependencies, and pronounced non-stationarity (level shifts, multi-seasonality, heteroscedasticity). We propose an architecture that enhances the Informer model by integrating three key modules. We introduce a compact two-layer convolutional block (COnvStem) into the embedding layer to improve local pattern extraction. To explicitly handle trend and seasonal components, we incorporate a series decomposition module from Autoformer. Finally, we replace the ProbSparse attention mechanism with FAVOR+ linear attention to efficiently capture global dependencies with low computational cost. Experiments on standard ETT benchmarks indicate reduced errors at long prediction horizons in several settings. Ablation studies show each module contributes, though gains are not universal across horizons. Overall, the method offers a favorable accuracy–time/memory trade-off on the evaluated configurations.

## Method Overview

The model combines three inductive biases: 
* (i) a ConvStem at the input for robust extraction of short-term local patterns; 
* (ii) FAVOR+ linear attention for scalable modeling of global dependencies; 
* (iii) Autoformer-style trend/seasonal decomposition after each self-attention block to stabilize non-stationarity. 

Evaluation uses the ETT benchmark (ETTh1, ETTh2) with input length 96 and horizons {24, 48, 168, 336, 720}; primary metrics are MSE/MAE (with RMSE/MAPE/MSPE also reported in the full benchmark), and train/inference time is recorded. Training employs MSE loss and Adam (lr=1e-4), batch size 32, early stopping within ≤10 epochs, averaging results over three runs; implementation is in PyTorch on a single NVIDIA GTX 1660 SUPER (6 GB), with 2 encoder layers and 1 decoder. Full hyperparameter/configuration details are provided in the repository.

## Get Started

1. Install Python 3.6, PyTorch 1.9.0.
2. Download data. You can obtain all the six benchmarks from [ETT](https://github.com/zhouhaoyi/ETDataset). **All the datasets are well pre-processed** and can be used easily.
3. Train the model. We provide the experiment scripts of all benchmarks under the folder `./convformer/scripts`. You can reproduce the experiment results by:

```bash
# inside ./convformer/scripts
bash ETT_script/Convformer.sh "./data/raw/ETT-small/" "ETTh1.csv" "ETTh1"
bash ETT_script/Convformer.sh "./data/raw/ETT-small/" "ETTh2.csv" "ETTh2"
```

or just

```bash
# inside ./convformer/scripts
bash main.sh
```

## Main Results

<p align="center">
    <img src=".convformer\results.png" height = "550" alt="" align=center />
</p>

## Limitations

* Benchmarks are limited to the ETT dataset; broader validation is pending.
* The cross-attention mechanism retains quadratic complexity.
* MAPE/MSPE exhibit sensitivity to feature scales.
* Uncertainty estimation (calibration/predictive intervals) is not addressed.

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/thuml/Autoformer

https://github.com/zhouhaoyi/Informer2020

https://github.com/zhouhaoyi/ETDataset
