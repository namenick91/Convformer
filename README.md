# Research on the Influence of Modular Hybridization on the Accuracy and Efficiency of Transformers for Long-Term Time Series Forecasting

A hybrid Transformer architecture (Convformer) combining a convolutional stem for local patterns, linear attention (FAVOR+) for global dependencies, and explicit series decomposition for Long-Sequence Time-series Forecasting (LSTF).

[📄 View the full paper (PDF)](./preprint.pdf)

## Overview

Long-term forecasting of multivariate time series is challenging due to complex patterns in real-world data, including short-term motifs (jumps, steps), long-range dependencies, and pronounced non-stationarity (trend shifts, multi-seasonality, heteroscedasticity). 

In this work, we propose **Convformer**, an architecture that enhances the Informer model by integrating three complementary inductive biases:
1.  **ConvStem:** A compact two-layer convolutional block integrated into the embedding layer to improve the extraction of local patterns and stabilize statistics of non-stationary inputs.
2.  **Series Decomposition:** An Autoformer-style module incorporated after each self-attention block to explicitly separate trend and seasonal components, passing a more stationarized data flow to the attention mechanism.
3.  **FAVOR+ Attention:** Replacing the ProbSparse mechanism with Performer's FAVOR+ linear attention to efficiently model global dependencies with linear time and memory complexity regarding sequence length.

Experiments on seven standard benchmarks (ETTh1, ETTh2, ECL, Exchange, Illness, Traffic, Weather) demonstrate reduced errors at long prediction horizons in multiple settings. Ablation studies confirm the contribution of each module, offering a favorable trade-off between accuracy and computational cost (time/memory).

## Method Overview

The architecture addresses the limitations of existing LSTF models (sensitivity to non-stationarity, local pattern ignorance, and quadratic complexity) via a modular approach:

<p align="center">
    <img src=".\convformer\architecture.png" height = "325" alt="Convformer Architecture" align=center />
</p>

*   **Input Representation:** Uses **ConvStem** (combining point-wise and depth-wise convolutions with Instance Normalization) instead of simple token embeddings to capture short-term motifs immediately.
*   **Encoder:** Focuses on modeling the **seasonal component**. It uses FAVOR+ self-attention and Series Decomposition to eliminate long-term trends from intermediate representations.
*   **Decoder:** Accumulates the **trend component** extracted from hidden states while refining the seasonal prediction using full cross-attention with the encoder's output.
*   **Efficiency:** The use of FAVOR+ (Fast Attention Via positive Orthogonal Random features) ensures predictable scalability ( $O(L)$ ) for very long sequences, unlike the $O(L^2)$  of canonical Transformers or $O(L \log L)$ of Informer.

## Get Started

1.  **Environment:** Install Python 3.6, PyTorch 1.9.0.
2.  **Data:** Download the datasets. You can obtain all the six benchmarks from [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) [[1](https://github.com/thuml/Autoformer)]. All the datasets are well pre-processed and can be used easily.
3.  **Training:** We provide experiment scripts for all benchmarks under `./convformer/scripts`.

To reproduce the experiment results:

```bash
# inside ./convformer/scripts
bash model_scripts/Convformer.sh "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)/../data/raw/ETT-small/" "ETTh1.csv" "ETTh1"
bash model_scripts/Convformer.sh "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)/../data/raw/ETT-small/" "ETTh2.csv" "ETTh2"
```

or just

```bash
# inside ./convformer/scripts
bash main.sh
```

## Main Results

Convformer demonstrates consistent improvements over Informer and Performer across most horizons and datasets. It remains competitive with Autoformer, particularly on shorter horizons and datasets with strong local variations, while offering a robust inference speed.

<p align="center">
    <img src=".\convformer\results.png" height = "550" alt="results" align=center />
</p>

*(Full benchmark results available in the paper Appendix).*

## Limitations

As discussed in the paper:
*   **Quadratic Cross-Attention:** While self-attention is linear (FAVOR+), the cross-attention mechanism in the decoder remains full-attention ( $O(L^2)$ ) to ensure precision, limiting maximum theoretical scalability compared to a fully linear model.
*   **Hyperparameter Scope:** The configuration (ranks, kernel sizes) was not exhaustively searched; results are based on a fixed set of parameters derived from baselines.
*   **Random Feature Sensitivity:** No systematic study was conducted on the impact of the number of random features ( $r$ ) or the specific map type beyond the chosen configuration.
*   **Uncertainty:** The model provides point forecasts (deterministic) and does not currently estimate predictive intervals.

## Acknowledgement

We appreciate the following open-source repositories for their code bases and datasets:

https://github.com/thuml/Autoformer

https://github.com/zhouhaoyi/Informer2020

https://github.com/idiap/fast-transformers

https://github.com/zhouhaoyi/ETDataset
