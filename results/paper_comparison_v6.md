# Reproduction v6 vs. paper (Table 3)

Records averaged: paper 195 comparable of 196 | reproduction 196 of 196 (0 without SQL from either model)

| Metric | Paper base | Repro base | Δ base | Paper FT | Repro FT | Δ FT |
|---|---:|---:|---:|---:|---:|---:|
| Exact Match (%) | 0.00 | 0.00 | 0.00 | 0.00 | 0.51 | 0.51 |
| String Similarity | 0.076 | 0.099 | 0.023 | 0.302 | 0.533 | 0.231 |
| Token Precision | 0.211 | 0.368 | 0.157 | 0.668 | 0.784 | 0.116 |
| Token Recall | 0.416 | 0.299 | -0.117 | 0.617 | 0.756 | 0.139 |
| Token F1 | 0.223 | 0.285 | 0.062 | 0.623 | 0.757 | 0.134 |
| Structural Precision | 0.298 | 0.271 | -0.027 | 0.351 | 0.769 | 0.418 |
| Structural Recall | 0.223 | 0.134 | -0.089 | 0.308 | 0.759 | 0.451 |
| Structural F1 | 0.244 | 0.169 | -0.075 | 0.320 | 0.753 | 0.433 |
| Component Jaccard | 0.212 | 0.265 | 0.053 | 0.233 | 0.292 | 0.059 |
| Geospatial Precision | 0.267 | 0.117 | -0.150 | 0.593 | 0.664 | 0.071 |
| Geospatial Recall | 0.120 | 0.135 | 0.015 | 0.470 | 0.658 | 0.188 |
| Geospatial F1 | 0.166 | 0.108 | -0.058 | 0.524 | 0.639 | 0.115 |
| Spatial Exact Match (%) | 0.00 | 0.51 | 0.51 | 7.69 | 28.06 | 20.37 |

## Spatial function call counts

| Model | Paper TP | Repro TP | Paper FP | Repro FP | Repro FN |
|---|---:|---:|---:|---:|---:|
| base | 77 | 86 | 211 | 340 | 533 |
| finetuned | 301 | 397 | 207 | 224 | 222 |

## New metrics (not in the paper): execution against the reference database

| Metric | Base | Fine-tuned | Δ |
|---|---:|---:|---:|
| Execution Accuracy | 0.000 | 0.158 | 0.158 |
| Executable Rate | 0.000 | 0.525 | 0.525 |

## By complexity tier (fine-tuned)

| Tier | n | Exec. Acc. | Token F1 | Structural F1 | Geospatial F1 | Spatial EM |
|---|---:|---:|---:|---:|---:|---:|
| Difícil | 49 | 0.143 | 0.753 | 0.748 | 0.703 | 26.53 |
| Fácil | 49 | 0.265 | 0.816 | 0.845 | 0.588 | 53.06 |
| Muito Difícil | 49 | 0.000 | 0.665 | 0.595 | 0.577 | 12.24 |
| Médio | 49 | 0.225 | 0.793 | 0.825 | 0.691 | 20.41 |
