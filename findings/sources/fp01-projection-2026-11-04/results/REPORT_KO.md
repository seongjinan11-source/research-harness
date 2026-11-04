# FP01 audited saved-probability results

This report is generated from saved probabilities; final scientific interpretation is written separately.

Synthetic final is a new-seed, same-family stratified bank. Real graphs are all reused development evidence. Seeds are averaged as metrics, never as probability ensembles. Full and edge are separate panels; no per-graph winner is selected.

Audit: PASS; twelve fits / 3072 new successful updates / 6144 task presentations / target updates0. Synthetic 16704, real 1800, reused baselines 375.

## Prespecified synthetic primary

Full/projected, missing50/75/90/95%, source>=2, common own-feature-missing queries. Average rates and three fixed training seeds within each base graph. Resample four base graphs within each of twelve N×K strata,5000times. Two accuracy contrasts have97.5%two-sided percentile intervals each (Bonferroni familywise95%).

| Arm | ACC% | BAcc% | macroF1 | NLL | Brier |
| --- | --- | --- | --- | --- | --- |
| A | 52.817 | 53.123 | 0.5140 | 1.00329 | 0.52538 |
| B | 52.800 | 53.076 | 0.5137 | 1.00335 | 0.52542 |
| C_global | 52.565 | 52.858 | 0.5116 | 0.99801 | 0.52409 |
| C | 53.752 | 54.079 | 0.5223 | 0.97400 | 0.51194 |

| Contrast | ACC difference%p [97.5%CI] | NLL difference |
| --- | --- | --- |
| C-minus-B | +0.952 [-0.585,+2.696] | -0.02935 |
| C-minus-C_global | +1.187 [+0.213,+2.446] | -0.02400 |

Prespecified adaptive-success gate: **False**. Both comparisons require positive interval lower bound and mean>=1%p, C−none positive, and C NLL at most .01nats worse than either comparator. No subgroup rescues this gate.

| Arm | Primary observed−none accuracy%p |
| --- | --- |
| A | -0.356 |
| B | -0.373 |
| C_global | -0.608 |
| C | +0.579 |

The shared none prediction makes C−B identical to a difference in feature utility. These are one comparison, not independent confirmations. Source0/1 are structural boundary checks. All raw/projected/orthogonal views, rates and class counts are preserved in summary.json; orthogonal first-cell75/95 diagnostics do not replace the primary.

## Real development: separate full and edge panels

The following rows average95/99/99.5% equally. Cora/CiteSeer are full graphs; PubMed/Coauthors are fixed2000-node induced subgraphs. Public pipelines retain historical input interfaces; LP and public predictions are reused once, then displayed beside both graph panels. There is no independent-real confidence interval.

### full

| Graph | Scope | Arm | ACC% | BAcc% | macroF1 | NLL | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Cora | full | A | 58.915 | 63.539 | 0.5826 | 1.6867 | 0.6130 |
| Cora | full | B | 58.919 | 63.540 | 0.5826 | 1.6868 | 0.6131 |
| Cora | full | C_global | 59.480 | 64.238 | 0.5890 | 1.6873 | 0.6095 |
| Cora | full | C | 59.735 | 64.261 | 0.5910 | 1.6967 | 0.6122 |
| Cora | full | lp_harmonic | 38.246 | 42.287 | 0.3798 | 1.5769 | 0.7183 |
| Cora | full | lp_restart | 59.056 | 63.286 | 0.5988 | 1.8061 | 0.8061 |
| Cora | full | public_graphpfn | 22.336 | 27.965 | 0.2117 | 2.0706 | 0.8707 |
| Cora | full | pcfi_public_graphpfn | 52.898 | 58.627 | 0.5269 | 1.4019 | 0.6203 |
| Cora | full | fisf_public_graphpfn | 55.481 | 60.624 | 0.5586 | 1.3563 | 0.6053 |
| CiteSeer | full | A | 44.385 | 43.216 | 0.4297 | 1.6426 | 0.7064 |
| CiteSeer | full | B | 44.608 | 43.186 | 0.4298 | 1.6422 | 0.7062 |
| CiteSeer | full | C_global | 45.607 | 43.721 | 0.4370 | 1.6249 | 0.6994 |
| CiteSeer | full | C | 45.827 | 44.035 | 0.4398 | 1.6193 | 0.6965 |
| CiteSeer | full | lp_harmonic | 34.228 | 37.663 | 0.2945 | 1.6267 | 0.7412 |
| CiteSeer | full | lp_restart | 38.410 | 41.964 | 0.3769 | 1.7504 | 0.8167 |
| CiteSeer | full | public_graphpfn | 19.130 | 21.555 | 0.1527 | 1.8435 | 0.8460 |
| CiteSeer | full | pcfi_public_graphpfn | 35.262 | 35.359 | 0.3291 | 1.7145 | 0.7917 |
| CiteSeer | full | fisf_public_graphpfn | 32.763 | 32.363 | 0.3055 | 1.7315 | 0.7984 |
| PubMed | induced2000 | A | 61.730 | 58.974 | 0.5690 | 1.4818 | 0.5873 |
| PubMed | induced2000 | B | 61.727 | 58.958 | 0.5689 | 1.4826 | 0.5874 |
| PubMed | induced2000 | C_global | 61.529 | 58.746 | 0.5673 | 1.4912 | 0.5891 |
| PubMed | induced2000 | C | 61.798 | 59.131 | 0.5701 | 1.4918 | 0.5885 |
| PubMed | induced2000 | lp_harmonic | 51.248 | 44.037 | 0.3841 | 0.8724 | 0.5071 |
| PubMed | induced2000 | lp_restart | 73.690 | 66.196 | 0.6534 | 1.0719 | 0.6484 |
| PubMed | induced2000 | public_graphpfn | 40.629 | 46.418 | 0.3937 | 1.0702 | 0.6471 |
| PubMed | induced2000 | pcfi_public_graphpfn | 60.831 | 58.708 | 0.5543 | 0.9826 | 0.5838 |
| PubMed | induced2000 | fisf_public_graphpfn | 57.205 | 57.338 | 0.5282 | 0.9679 | 0.5752 |
| CoauthorCS | induced2000 | A | 61.686 | 58.666 | 0.4999 | 3.3648 | 0.6737 |
| CoauthorCS | induced2000 | B | 61.691 | 58.682 | 0.5000 | 3.3650 | 0.6737 |
| CoauthorCS | induced2000 | C_global | 61.747 | 58.688 | 0.5009 | 3.3677 | 0.6736 |
| CoauthorCS | induced2000 | C | 61.750 | 58.770 | 0.5013 | 3.3665 | 0.6734 |
| CoauthorCS | induced2000 | lp_harmonic | 64.822 | 56.404 | 0.5157 | 1.5776 | 0.6438 |
| CoauthorCS | induced2000 | lp_restart | 68.891 | 61.689 | 0.5438 | 2.1624 | 0.8208 |
| CoauthorCS | induced2000 | public_graphpfn | 45.203 | 40.354 | 0.3545 | 2.9267 | 0.7856 |
| CoauthorCS | induced2000 | pcfi_public_graphpfn | 68.761 | 58.411 | 0.5350 | 2.8415 | 0.5476 |
| CoauthorCS | induced2000 | fisf_public_graphpfn | 69.601 | 61.006 | 0.5524 | 2.9359 | 0.5355 |
| CoauthorPhysics | induced2000 | A | 46.520 | 49.555 | 0.3722 | 5.8094 | 0.9962 |
| CoauthorPhysics | induced2000 | B | 46.500 | 49.476 | 0.3714 | 5.8099 | 0.9962 |
| CoauthorPhysics | induced2000 | C_global | 46.483 | 49.065 | 0.3700 | 5.8441 | 0.9988 |
| CoauthorPhysics | induced2000 | C | 46.387 | 48.724 | 0.3681 | 5.8459 | 1.0000 |
| CoauthorPhysics | induced2000 | lp_harmonic | 85.411 | 52.636 | 0.5611 | 0.5955 | 0.2641 |
| CoauthorPhysics | induced2000 | lp_restart | 84.111 | 63.911 | 0.6146 | 1.4935 | 0.7472 |
| CoauthorPhysics | induced2000 | public_graphpfn | 65.711 | 54.675 | 0.4687 | 1.2361 | 0.6130 |
| CoauthorPhysics | induced2000 | pcfi_public_graphpfn | 75.743 | 60.085 | 0.5501 | 1.0558 | 0.5250 |
| CoauthorPhysics | induced2000 | fisf_public_graphpfn | 76.900 | 63.472 | 0.5806 | 0.9843 | 0.4825 |

| Graph | Arm | Feature contrast | ACC difference%p [3reps] | NLL difference |
| --- | --- | --- | --- | --- |
| Cora | A | 95-minus-none | -1.675 [-1.78,-3.17,-0.07] | -0.01473 |
| Cora | A | 95-minus-shuffle95 | +0.108 [+1.82,-2.92,+1.43] | -0.04629 |
| Cora | B | 95-minus-none | -1.663 [-1.68,-3.21,-0.10] | -0.01470 |
| Cora | B | 95-minus-shuffle95 | +0.037 [+1.75,-3.14,+1.50] | -0.03820 |
| Cora | C_global | 95-minus-none | -0.189 [+0.52,-1.06,-0.03] | -0.01658 |
| Cora | C_global | 95-minus-shuffle95 | -0.156 [+1.50,-1.90,-0.07] | -0.00148 |
| Cora | C | 95-minus-none | -0.024 [+0.70,-0.53,-0.24] | -0.00515 |
| Cora | C | 95-minus-shuffle95 | +0.128 [+0.77,-0.07,-0.31] | -0.00151 |
| CiteSeer | A | 95-minus-none | -4.330 [-7.69,-2.68,-2.62] | +0.06494 |
| CiteSeer | A | 95-minus-shuffle95 | -0.932 [-1.83,+0.67,-1.64] | +0.01387 |
| CiteSeer | B | 95-minus-none | -3.613 [-7.76,-0.35,-2.72] | +0.06427 |
| CiteSeer | B | 95-minus-shuffle95 | -0.343 [-1.93,+2.79,-1.88] | +0.01358 |
| CiteSeer | C_global | 95-minus-none | -0.484 [-2.60,+2.89,-1.75] | +0.01421 |
| CiteSeer | C_global | 95-minus-shuffle95 | +1.964 [+3.55,+3.77,-1.43] | -0.02436 |
| CiteSeer | C | 95-minus-none | +0.003 [-0.67,+1.27,-0.59] | -0.00191 |
| CiteSeer | C | 95-minus-shuffle95 | +0.998 [+2.00,+1.48,-0.49] | -0.01378 |
| PubMed | A | 95-minus-none | -0.623 [-0.18,-0.48,-1.21] | -0.02279 |
| PubMed | A | 95-minus-shuffle95 | -0.191 [+0.53,-0.07,-1.03] | -0.00757 |
| PubMed | B | 95-minus-none | -0.689 [-0.20,-0.66,-1.21] | -0.02073 |
| PubMed | B | 95-minus-shuffle95 | -0.250 [+0.50,-0.22,-1.03] | -0.00539 |
| PubMed | C_global | 95-minus-none | -0.946 [+0.20,-0.59,-2.44] | -0.00113 |
| PubMed | C_global | 95-minus-shuffle95 | -0.903 [+1.42,-0.18,-3.96] | -0.00326 |
| PubMed | C | 95-minus-none | -0.440 [+0.22,-0.07,-1.47] | -0.00024 |
| PubMed | C | 95-minus-shuffle95 | -0.206 [+0.68,+0.31,-1.61] | -0.00001 |
| CoauthorCS | A | 95-minus-none | -0.219 [-0.07,-0.55,-0.04] | -0.00762 |
| CoauthorCS | A | 95-minus-shuffle95 | -0.051 [-0.11,-0.29,+0.24] | -0.00545 |
| CoauthorCS | B | 95-minus-none | -0.205 [-0.02,-0.55,-0.04] | -0.00702 |
| CoauthorCS | B | 95-minus-shuffle95 | -0.051 [-0.11,-0.29,+0.24] | -0.00353 |
| CoauthorCS | C_global | 95-minus-none | -0.022 [-0.09,+0.04,-0.02] | +0.00288 |
| CoauthorCS | C_global | 95-minus-shuffle95 | +0.315 [+0.09,+0.50,+0.35] | -0.00805 |
| CoauthorCS | C | 95-minus-none | +0.000 [+0.02,-0.07,+0.04] | -0.00091 |
| CoauthorCS | C | 95-minus-shuffle95 | +0.095 [+0.02,-0.00,+0.26] | -0.00241 |
| CoauthorPhysics | A | 95-minus-none | +1.268 [+0.26,+0.42,+3.12] | -0.14649 |
| CoauthorPhysics | A | 95-minus-shuffle95 | +0.763 [-0.04,+0.35,+1.98] | -0.10785 |
| CoauthorPhysics | B | 95-minus-none | +1.283 [+0.24,+0.44,+3.17] | -0.14655 |
| CoauthorPhysics | B | 95-minus-shuffle95 | +0.777 [-0.07,+0.37,+2.02] | -0.10900 |
| CoauthorPhysics | C_global | 95-minus-none | +0.477 [+0.13,-0.11,+1.41] | -0.01437 |
| CoauthorPhysics | C_global | 95-minus-shuffle95 | +0.403 [+0.09,-0.24,+1.36] | -0.00395 |
| CoauthorPhysics | C | 95-minus-none | +0.205 [-0.09,-0.09,+0.79] | -0.01064 |
| CoauthorPhysics | C | 95-minus-shuffle95 | +0.249 [-0.09,+0.04,+0.79] | -0.01699 |

### edge

| Graph | Scope | Arm | ACC% | BAcc% | macroF1 | NLL | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Cora | full | A | 57.865 | 58.964 | 0.5516 | 1.7921 | 0.6390 |
| Cora | full | B | 57.884 | 58.993 | 0.5519 | 1.7910 | 0.6388 |
| Cora | full | C_global | 60.178 | 61.206 | 0.5749 | 1.7084 | 0.6091 |
| Cora | full | C | 60.159 | 61.151 | 0.5742 | 1.7077 | 0.6087 |
| Cora | full | lp_harmonic | 38.246 | 42.287 | 0.3798 | 1.5769 | 0.7183 |
| Cora | full | lp_restart | 59.056 | 63.286 | 0.5988 | 1.8061 | 0.8061 |
| Cora | full | public_graphpfn | 22.336 | 27.965 | 0.2117 | 2.0706 | 0.8707 |
| Cora | full | pcfi_public_graphpfn | 52.898 | 58.627 | 0.5269 | 1.4019 | 0.6203 |
| Cora | full | fisf_public_graphpfn | 55.481 | 60.624 | 0.5586 | 1.3563 | 0.6053 |
| CiteSeer | full | A | 35.828 | 39.586 | 0.3112 | 1.8879 | 0.7812 |
| CiteSeer | full | B | 35.824 | 39.584 | 0.3113 | 1.8880 | 0.7811 |
| CiteSeer | full | C_global | 36.137 | 40.050 | 0.3127 | 1.8851 | 0.7774 |
| CiteSeer | full | C | 36.348 | 40.265 | 0.3144 | 1.8789 | 0.7749 |
| CiteSeer | full | lp_harmonic | 34.228 | 37.663 | 0.2945 | 1.6267 | 0.7412 |
| CiteSeer | full | lp_restart | 38.410 | 41.964 | 0.3769 | 1.7504 | 0.8167 |
| CiteSeer | full | public_graphpfn | 19.130 | 21.555 | 0.1527 | 1.8435 | 0.8460 |
| CiteSeer | full | pcfi_public_graphpfn | 35.262 | 35.359 | 0.3291 | 1.7145 | 0.7917 |
| CiteSeer | full | fisf_public_graphpfn | 32.763 | 32.363 | 0.3055 | 1.7315 | 0.7984 |
| PubMed | induced2000 | A | 65.876 | 61.655 | 0.6011 | 1.4006 | 0.5324 |
| PubMed | induced2000 | B | 65.839 | 61.669 | 0.6009 | 1.4006 | 0.5323 |
| PubMed | induced2000 | C_global | 65.820 | 61.830 | 0.6033 | 1.4042 | 0.5330 |
| PubMed | induced2000 | C | 66.034 | 62.212 | 0.6084 | 1.4053 | 0.5333 |
| PubMed | induced2000 | lp_harmonic | 51.248 | 44.037 | 0.3841 | 0.8724 | 0.5071 |
| PubMed | induced2000 | lp_restart | 73.690 | 66.196 | 0.6534 | 1.0719 | 0.6484 |
| PubMed | induced2000 | public_graphpfn | 40.629 | 46.418 | 0.3937 | 1.0702 | 0.6471 |
| PubMed | induced2000 | pcfi_public_graphpfn | 60.831 | 58.708 | 0.5543 | 0.9826 | 0.5838 |
| PubMed | induced2000 | fisf_public_graphpfn | 57.205 | 57.338 | 0.5282 | 0.9679 | 0.5752 |
| CoauthorCS | induced2000 | A | 72.573 | 58.577 | 0.5652 | 2.4656 | 0.4976 |
| CoauthorCS | induced2000 | B | 72.573 | 58.577 | 0.5652 | 2.4655 | 0.4976 |
| CoauthorCS | induced2000 | C_global | 72.656 | 58.630 | 0.5657 | 2.4606 | 0.4963 |
| CoauthorCS | induced2000 | C | 72.673 | 58.642 | 0.5658 | 2.4570 | 0.4956 |
| CoauthorCS | induced2000 | lp_harmonic | 64.822 | 56.404 | 0.5157 | 1.5776 | 0.6438 |
| CoauthorCS | induced2000 | lp_restart | 68.891 | 61.689 | 0.5438 | 2.1624 | 0.8208 |
| CoauthorCS | induced2000 | public_graphpfn | 45.203 | 40.354 | 0.3545 | 2.9267 | 0.7856 |
| CoauthorCS | induced2000 | pcfi_public_graphpfn | 68.761 | 58.411 | 0.5350 | 2.8415 | 0.5476 |
| CoauthorCS | induced2000 | fisf_public_graphpfn | 69.601 | 61.006 | 0.5524 | 2.9359 | 0.5355 |
| CoauthorPhysics | induced2000 | A | 86.255 | 58.809 | 0.6090 | 0.9282 | 0.2436 |
| CoauthorPhysics | induced2000 | B | 86.258 | 58.815 | 0.6090 | 0.9280 | 0.2436 |
| CoauthorPhysics | induced2000 | C_global | 86.155 | 58.115 | 0.6010 | 0.9365 | 0.2450 |
| CoauthorPhysics | induced2000 | C | 86.240 | 58.111 | 0.6015 | 0.9307 | 0.2438 |
| CoauthorPhysics | induced2000 | lp_harmonic | 85.411 | 52.636 | 0.5611 | 0.5955 | 0.2641 |
| CoauthorPhysics | induced2000 | lp_restart | 84.111 | 63.911 | 0.6146 | 1.4935 | 0.7472 |
| CoauthorPhysics | induced2000 | public_graphpfn | 65.711 | 54.675 | 0.4687 | 1.2361 | 0.6130 |
| CoauthorPhysics | induced2000 | pcfi_public_graphpfn | 75.743 | 60.085 | 0.5501 | 1.0558 | 0.5250 |
| CoauthorPhysics | induced2000 | fisf_public_graphpfn | 76.900 | 63.472 | 0.5806 | 0.9843 | 0.4825 |

| Graph | Arm | Feature contrast | ACC difference%p [3reps] | NLL difference |
| --- | --- | --- | --- | --- |
| Cora | A | 95-minus-none | -5.486 [-7.52,-4.40,-4.53] | +0.21172 |
| Cora | A | 95-minus-shuffle95 | -0.990 [-1.85,+0.07,-1.19] | +0.05353 |
| Cora | B | 95-minus-none | -5.416 [-7.35,-4.40,-4.50] | +0.20841 |
| Cora | B | 95-minus-shuffle95 | -0.944 [-1.68,-0.14,-1.01] | +0.05368 |
| Cora | C_global | 95-minus-none | +0.663 [-0.35,+0.39,+1.95] | -0.00396 |
| Cora | C_global | 95-minus-shuffle95 | +0.836 [+0.91,-0.49,+2.09] | -0.01988 |
| Cora | C | 95-minus-none | +0.245 [-0.17,+0.46,+0.45] | +0.00458 |
| Cora | C | 95-minus-shuffle95 | +0.467 [+0.66,+0.39,+0.35] | -0.02023 |
| CiteSeer | A | 95-minus-none | -1.627 [-2.18,-1.59,-1.12] | +0.03554 |
| CiteSeer | A | 95-minus-shuffle95 | -0.245 [+0.00,-0.07,-0.66] | -0.00364 |
| CiteSeer | B | 95-minus-none | -1.663 [-2.21,-1.69,-1.08] | +0.03588 |
| CiteSeer | B | 95-minus-shuffle95 | -0.315 [+0.04,-0.21,-0.77] | -0.00504 |
| CiteSeer | C_global | 95-minus-none | -0.842 [-1.33,-0.42,-0.77] | +0.02199 |
| CiteSeer | C_global | 95-minus-shuffle95 | -0.723 [-1.55,+0.56,-1.19] | +0.00468 |
| CiteSeer | C | 95-minus-none | -0.139 [-0.60,+0.56,-0.38] | +0.00759 |
| CiteSeer | C | 95-minus-shuffle95 | -0.266 [-1.09,+1.02,-0.73] | +0.00052 |
| PubMed | A | 95-minus-none | -1.448 [-2.45,-1.58,-0.31] | +0.00434 |
| PubMed | A | 95-minus-shuffle95 | +0.520 [+0.77,+0.24,+0.55] | -0.01497 |
| PubMed | B | 95-minus-none | -1.448 [-2.48,-1.58,-0.29] | +0.00428 |
| PubMed | B | 95-minus-shuffle95 | +0.520 [+0.77,+0.22,+0.57] | -0.01501 |
| PubMed | C_global | 95-minus-none | -1.867 [-2.15,-3.06,-0.40] | +0.01678 |
| PubMed | C_global | 95-minus-shuffle95 | -0.416 [-1.47,-1.52,+1.74] | +0.00257 |
| PubMed | C | 95-minus-none | -1.120 [-1.16,-1.82,-0.37] | +0.01378 |
| PubMed | C | 95-minus-shuffle95 | -0.395 [-0.96,-1.93,+1.72] | +0.01561 |
| CoauthorCS | A | 95-minus-none | -0.286 [-0.62,-0.09,-0.15] | +0.02603 |
| CoauthorCS | A | 95-minus-shuffle95 | -0.256 [-0.31,+0.04,-0.51] | +0.04254 |
| CoauthorCS | B | 95-minus-none | -0.286 [-0.62,-0.09,-0.15] | +0.02575 |
| CoauthorCS | B | 95-minus-shuffle95 | -0.256 [-0.31,+0.04,-0.51] | +0.04448 |
| CoauthorCS | C_global | 95-minus-none | -0.037 [+0.00,+0.00,-0.11] | +0.01066 |
| CoauthorCS | C_global | 95-minus-shuffle95 | -0.139 [-0.07,+0.00,-0.35] | +0.01720 |
| CoauthorCS | C | 95-minus-none | +0.015 [+0.02,+0.00,+0.02] | +0.00113 |
| CoauthorCS | C | 95-minus-shuffle95 | +0.007 [+0.00,+0.00,+0.02] | +0.00113 |
| CoauthorPhysics | A | 95-minus-none | -0.058 [-1.71,+0.31,+1.23] | +0.01931 |
| CoauthorPhysics | A | 95-minus-shuffle95 | -0.835 [-2.31,+0.11,-0.31] | +0.05476 |
| CoauthorPhysics | B | 95-minus-none | -0.050 [-1.71,+0.33,+1.23] | +0.01898 |
| CoauthorPhysics | B | 95-minus-shuffle95 | -0.761 [-2.11,+0.13,-0.31] | +0.04846 |
| CoauthorPhysics | C_global | 95-minus-none | -0.161 [-0.68,-0.04,+0.24] | +0.03780 |
| CoauthorPhysics | C_global | 95-minus-shuffle95 | -0.373 [-0.61,-0.22,-0.29] | +0.02825 |
| CoauthorPhysics | C | 95-minus-none | +0.095 [+0.00,-0.04,+0.33] | +0.01561 |
| CoauthorPhysics | C | 95-minus-shuffle95 | -0.110 [+0.18,-0.31,-0.20] | -0.00073 |

## Audit and cost scope

Common none maximum probability error: 0; source0/1 versus matched none maximum: 3.58e-07.

All checkpoint/source/bank/input/query/target hashes and finite probability arrays were checked and independently rescored. Fitted feature state digests and frozen graph digests were checked; no forward pass was needed. Balanced accuracy averages recalls of classes present in each truth vector; macroF1 uses the historical sklearn label-union convention. Query/source class counts are descriptive scoring metadata.

Inference timing is cold, mixed-shape worker timing including CPU probability return and excluding shared graph encoding; it is not a competitive latency benchmark. Fit seconds are a worker-time sum, not elapsed wall time. Inherited graph training, preflight and failed attempts are listed separately in budget.json.

Detailed per-rate/class/view, seed/repetition, paired feature effects and all predictions are indexed in summary.json, paired.json and prediction_index.jsonl. artifact_manifest.json points to immutable heavy artifacts by path/hash. No calibrated-Bayes, published-AMP-reproduction, unseen-family or independent-real-generalization claim follows from this report.

