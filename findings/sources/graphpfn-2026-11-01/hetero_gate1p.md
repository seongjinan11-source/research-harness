# Gate 1-P result (30 Oct 2026): path-aggregated feature columns into released GraphPFN, no training
Protocol: docs/gate1p_protocol.md (fixed before running). Same cells as Gate 1-R v2. Raw: results/gate1p/cells.jsonl.

## Means (test accuracy %)
              B5   B5g  oracle_val  PLLK  PLL2  UNIF   B1m    B1    B0
dataset  k                                                            
ACM      1  42.4  39.8        46.6   NaN   NaN  43.5  39.1  39.9  39.9
         3  48.0  70.4        67.7  63.9  63.3  63.6  66.4  58.7  43.0
         5  58.9  71.8        75.9  72.6  73.8  74.2  73.7  68.4  54.8
DBLP     1  49.2  45.4        28.9   NaN   NaN  26.1  24.6  26.0  27.4
         3  80.2  83.1        70.6  61.4  59.2  58.2  56.7  38.8  38.2
         5  88.7  88.7        91.1  91.1  90.9  87.6  86.2  52.7  49.6
FREEBASE 1  38.9  38.2        41.9   NaN   NaN  33.2  38.2  32.5  35.2
         3  46.4  54.5        54.2  44.1  44.7  44.5  50.4  50.6  35.7
         5  44.1  55.3        56.3  55.6  55.6  55.8  55.1  49.5  33.4
IMDB     1  33.8  31.4        37.6   NaN   NaN  34.3  34.5  34.5  37.7
         3  34.5  34.9        37.8  36.6  36.5  36.7  37.3  37.8  36.0
         5  38.0  40.2        39.5  39.8  40.0  40.1  39.9  39.9  38.5

## k=3 paired contrasts (5 context draws, bootstrap 95% CI)
 dataset       contrast   mean     lo     hi
     ACM  B5-oracle_val -19.71 -28.03 -11.39
     ACM B5g-oracle_val   2.73  -2.72   7.72
     ACM        B5-PLLK -15.92 -22.84  -8.73
     ACM       B5g-PLLK   6.51   0.04  11.66
     ACM        B5g-B1m   4.00  -2.04   9.29
    DBLP  B5-oracle_val   9.55  -2.47  21.59
    DBLP B5g-oracle_val  12.47  -0.93  25.95
    DBLP        B5-PLLK  18.81   6.96  30.67
    DBLP       B5g-PLLK  21.74   8.33  35.14
    DBLP        B5g-B1m  26.39  15.03  37.75
FREEBASE  B5-oracle_val  -7.80 -14.08  -2.86
FREEBASE B5g-oracle_val   0.30  -3.16   3.62
FREEBASE        B5-PLLK   2.32  -2.42   5.72
FREEBASE       B5g-PLLK  10.42   7.04  12.76
FREEBASE        B5g-B1m   4.08   0.62   7.80
    IMDB  B5-oracle_val  -3.27  -5.45  -1.21
    IMDB B5g-oracle_val  -2.92  -6.70   1.02
    IMDB        B5-PLLK  -2.08  -4.85  -0.25
    IMDB       B5g-PLLK  -1.73  -5.64   2.40
    IMDB        B5g-B1m  -2.48  -5.64   0.67

## Decision (pre-registered rule)
B5g >= val_oracle - 1 pp on 3 of 3 decision datasets (ACM +2.7, DBLP +12.5, Freebase +0.3).
-> The released PFN already uses typed paths in context when they are given as aggregated feature columns; a learned path adapter is NOT justified
   by the k<=3 gap. Any contribution must come from somewhere else (re-scope before any training).
Other observations: B5 (columns only, no graph) is much weaker on ACM/Freebase; DBLP k=1 B5 49.2 vs every graph method <= 28.9;
at k=5 B5g trails the val oracle on ACM (-4.1) and DBLP (-2.4), where explicit selection already matches the oracle.
Calibration check (v2, k=3): B0 is the top candidate by held-out NLL in 60% of ACM/Freebase/IMDB cells but by held-out accuracy in 0-40%,
so pseudo-likelihood selection favours under-confident feature-only predictions.
