# Gate 1-R result: real-data room for relation choice (released GraphPFN 1.3, no training)  [30 Oct 2026]
Protocol: docs/gate1_real_protocol.md. 100 cells = 4 datasets x k in {1,3,5,10,20} x 5 context draws. Raw: results/gate1_real/cells.jsonl.
Oracles are picked on TEST accuracy. To limit selection-on-test bias, the decision uses a FIXED oracle per (dataset, k): the candidate with the best
mean accuracy over the 5 draws (cell-wise maxima are also reported). Paired bootstrap over the 15 cells with k <= 5.

## Per (dataset, k) means (accuracy %)
 dataset  k   B0   LR   B1  B1m   B2   B3  B3all best_fixed_view  fixed_view best_fixed_all  fixed_all  room_cell  room_fixed  unrec_cell  unrec_fixed
     ACM  1 39.9 46.1 39.9 39.1 44.3 44.3   44.0             PAP        44.2            PAP       44.2        6.5         4.2         3.4          0.1
     ACM  3 43.0 50.2 58.7 66.4 61.7 61.6   61.8             PSP        63.9            B1m       66.4        6.2         5.2         6.2          4.6
     ACM  5 54.8 59.9 68.4 73.7 73.9 73.4   74.5             PSP        70.0            B1m       73.7        4.2         1.6         1.4         -0.8
     ACM 10 76.0 73.2 81.5 82.1 83.2 80.0   81.5             PAP        81.2            B1m       82.1        0.8        -0.3         1.8          0.6
     ACM 20 80.3 76.4 87.7 84.3 86.6 83.1   84.6             P-P        84.3             B1       87.7       -2.8        -3.4         3.2          3.1
    DBLP  1 27.4 37.0 26.0 24.6 26.0 26.0   26.2             APA        27.7            APA       27.7        2.3         1.7         5.3          1.5
    DBLP  3 38.2 48.7 38.8 56.7 63.0 64.1   59.4           APVPA        70.6          APVPA       70.6       31.8        31.8        11.2         11.2
    DBLP  5 49.6 55.2 52.7 86.2 88.8 91.1   91.0           APVPA        91.1          APVPA       91.1       38.3        38.3         0.1          0.1
    DBLP 10 65.7 66.3 66.3 89.9 89.6 93.0   93.0           APVPA        93.0          APVPA       93.0       26.6        26.6        -0.0         -0.0
    DBLP 20 70.1 72.8 70.7 92.5 89.4 93.2   93.2           APVPA        93.2          APVPA       93.2       22.5        22.5         0.0          0.0
FREEBASE  1 35.2 38.7 32.5 38.2 32.1 32.1   33.3             MAM        38.7            MAM       38.7        8.4         6.1         8.7          5.4
FREEBASE  3 35.7 35.4 50.6 50.4 38.4 39.3   42.3             MAM        50.4             B4       52.1       -0.2        -0.2        11.9          9.7
FREEBASE  5 33.4 31.3 49.5 55.1 55.2 55.1   55.2             MAM        55.5            MAM       55.5        6.0         6.0         1.4          0.3
FREEBASE 10 35.2 41.1 58.6 57.3 56.9 57.3   57.7             MAM        57.2             B1       58.6       -1.4        -1.4         2.6          1.0
FREEBASE 20 36.8 42.6 61.2 57.0 58.7 56.7   57.8             MAM        56.5             B1       61.2       -4.7        -4.7         3.5          3.4
    IMDB  1 37.7 37.1 34.5 34.5 33.7 33.7   34.4             MDM        33.9             B0       37.7        0.0        -0.6         3.5          3.2
    IMDB  3 36.0 32.8 37.8 37.3 36.6 36.6   36.5             MAM        37.1             B1       37.8       -0.4        -0.7         1.8          1.3
    IMDB  5 38.5 36.7 39.9 39.9 39.1 38.8   39.8             MAM        38.6             B1       39.9       -1.1        -1.3         0.9          0.1
    IMDB 10 41.7 38.5 41.5 41.6 41.9 42.0   41.5             MDM        41.7             B0       41.7        0.7         0.2         1.3          0.2
    IMDB 20 43.7 40.7 45.5 45.5 44.9 44.8   44.8             MAM        44.7             B4       45.7       -0.6        -0.8         1.2          0.9

## Decision quantities, k <= 5 (fixed oracle; 95% bootstrap CI over 15 cells)
 dataset  n_cells  room_fixed      room_ci  unrec_fixed    unrec_ci
     ACM       15        3.67  [-0.0, 7.3]         1.33 [-0.9, 3.5]
    DBLP       15       23.98 [14.6, 33.0]         4.29  [1.4, 7.6]
FREEBASE       15        3.98   [0.7, 7.3]         5.15  [2.9, 7.4]
    IMDB       15       -0.87 [-1.5, -0.2]         1.55  [0.4, 3.1]

## Decision (rule fixed in the protocol)
- Room >= 1 pp on 3 of 4 datasets (ACM 3.67 [-0.0, 7.3], DBLP 23.98 [14.6, 33.0], Freebase 3.98 [0.7, 7.3]); IMDB has none (-0.87). -> not "no room".
- Unrecovered >= 1 pp on all 4 datasets (ACM 1.33 with CI touching 0). -> explicit selection does not recover the room. -> continue to synthetic Gate 1.

## What the numbers say
1. Released GraphPFN on the collapsed heterograph cannot use long typed paths: DBLP B1 26-71 vs APVPA view 28-93 (room 22-38 pp for k >= 3).
2. TAG-style cross-fitted selection (B3/B3all) recovers the room fully once k >= 5 on DBLP (unrecovered 0.0-0.1) and nearly on ACM/Freebase at k = 5.
3. The unrecovered part is concentrated at k <= 3 (DBLP k=3 11.2, Freebase k=1/3 5.4/9.7, ACM k=3 4.6): a 2-fold split leaves 1-2 labels per class.
4. At k = 20 explicit selection again loses 3.1-3.4 pp on ACM/Freebase, where the collapsed heterograph B1 is best.
5. Context logistic regression on the same features beats feature-only GraphPFN (B0) at small k on ACM/DBLP/Freebase (e.g. DBLP k=1 37.0 vs 27.4).
6. IMDB: all methods within ~2 pp of each other at every k (34-46 %): no relation signal usable by GraphPFN here.
Caveats: the fixed oracle still uses test labels (5 draws share one test set); Freebase uses structural degree features; dense meta-paths are capped at 50 neighbours.
