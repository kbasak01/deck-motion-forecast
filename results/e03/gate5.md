# Gate 5 read-out

Simulated results only. Gate 5: **PICP@90% within [0.85, 0.95] on the `id` regime** (`docs/IMPLEMENTATION_PLAN.md` Phase 5, band unchanged). The cell it is read at was registered before the sweep ran (`docs/protocol.md` P5-D2) and is the cell Gates 3 and 4 are read at: the decision horizon, on the DOF that binds.

Computed from `probabilistic.csv` in `results/e03/`; the point accuracy of the same runs is in `baselines.csv` beside it, and the per-run source of truth is `probabilistic_by_seed.csv`.

## Outcome

- **Reading A -- PASS**: 6 of 6 rows inside the band.
- **Reading B -- NOT PASSED**: 102 of 216 rows inside the band. Outside: `dlinear_gaussian`/gaussian heave@1s 0.962 (FAIL), `dlinear_gaussian`/gaussian heave_rate@1s 0.970 (FAIL), `dlinear_gaussian`/gaussian heave_rate@2s 0.955 (FAIL), `dlinear_gaussian`/gaussian pitch_rate@2s 0.847 (FAIL), `dlinear_gaussian`/gaussian pitch_rate@3s 0.845 (FAIL), `dlinear_gaussian`/gaussian pitch_rate@5s 0.840 (FAIL), `dlinear_gaussian`/gaussian pitch_rate@10s 0.838 (FAIL), `dlinear_gaussian`/gaussian roll@1s 0.992 (FAIL), and 106 more.

## Reading A -- the gate -- pitch at 100 samples

Regime `id`. PASS iff `0.85 <= picp_mean <= 0.95`, inclusive at both ends.

**Reading A -- PASS**: 6 of 6 rows inside the band.

| model | head | dof | horizon_s | n_seeds | picp_mean | picp_std | picp_ci_lo | picp_ci_hi | mean_interval_width_mean | width_ratio_mean | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dlinear_gaussian | gaussian | pitch | 10.0000 | 3 | 0.8647 | 0.0001 | 0.8489 | 0.8804 | 3.0921 | 0.7573 | PASS |
| dlinear_quantile | quantile | pitch | 10.0000 | 3 | 0.8576 | 0.0002 | 0.8410 | 0.8741 | 3.0101 | 0.7372 | PASS |
| lstm_gaussian | gaussian | pitch | 10.0000 | 3 | 0.9172 | 0.0032 | 0.9100 | 0.9220 | 1.2751 | 0.3123 | PASS |
| lstm_quantile | quantile | pitch | 10.0000 | 3 | 0.9120 | 0.0071 | 0.9010 | 0.9192 | 1.1582 | 0.2837 | PASS |
| tcn_gaussian | gaussian | pitch | 10.0000 | 3 | 0.9176 | 0.0032 | 0.9106 | 0.9235 | 1.6214 | 0.3971 | PASS |
| tcn_quantile | quantile | pitch | 10.0000 | 3 | 0.9166 | 0.0026 | 0.9097 | 0.9224 | 1.4746 | 0.3611 | PASS |

## Reading B -- the surround -- every cell of `id`

Regime `id`. PASS iff `0.85 <= picp_mean <= 0.95`, inclusive at both ends.

**Reading B -- NOT PASSED**: 102 of 216 rows inside the band. Outside: `dlinear_gaussian`/gaussian heave@1s 0.962 (FAIL), `dlinear_gaussian`/gaussian heave_rate@1s 0.970 (FAIL), `dlinear_gaussian`/gaussian heave_rate@2s 0.955 (FAIL), `dlinear_gaussian`/gaussian pitch_rate@2s 0.847 (FAIL), `dlinear_gaussian`/gaussian pitch_rate@3s 0.845 (FAIL), `dlinear_gaussian`/gaussian pitch_rate@5s 0.840 (FAIL), `dlinear_gaussian`/gaussian pitch_rate@10s 0.838 (FAIL), `dlinear_gaussian`/gaussian roll@1s 0.992 (FAIL), and 106 more.

| model | head | dof | horizon_s | n_seeds | picp_mean | picp_std | picp_ci_lo | picp_ci_hi | mean_interval_width_mean | width_ratio_mean | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dlinear_gaussian | gaussian | heave | 1.0000 | 3 | 0.9624 | 0.0025 | 0.9512 | 0.9739 | 0.0726 | 0.0291 | FAIL |
| dlinear_gaussian | gaussian | heave | 2.0000 | 3 | 0.9444 | 0.0007 | 0.9324 | 0.9564 | 0.3962 | 0.1586 | PASS |
| dlinear_gaussian | gaussian | heave | 3.0000 | 3 | 0.9224 | 0.0007 | 0.9080 | 0.9375 | 0.8292 | 0.3320 | PASS |
| dlinear_gaussian | gaussian | heave | 5.0000 | 3 | 0.9196 | 0.0005 | 0.9051 | 0.9343 | 0.9751 | 0.3904 | PASS |
| dlinear_gaussian | gaussian | heave | 10.0000 | 3 | 0.9072 | 0.0003 | 0.8938 | 0.9210 | 1.8778 | 0.7519 | PASS |
| dlinear_gaussian | gaussian | heave | 15.0000 | 3 | 0.8856 | 0.0000 | 0.8703 | 0.9013 | 2.2599 | 0.9054 | PASS |
| dlinear_gaussian | gaussian | heave_rate | 1.0000 | 3 | 0.9701 | 0.0027 | 0.9605 | 0.9781 | 0.0431 | 0.0291 | FAIL |
| dlinear_gaussian | gaussian | heave_rate | 2.0000 | 3 | 0.9551 | 0.0008 | 0.9459 | 0.9644 | 0.2351 | 0.1589 | FAIL |
| dlinear_gaussian | gaussian | heave_rate | 3.0000 | 3 | 0.9344 | 0.0008 | 0.9235 | 0.9462 | 0.4919 | 0.3326 | PASS |
| dlinear_gaussian | gaussian | heave_rate | 5.0000 | 3 | 0.9274 | 0.0006 | 0.9159 | 0.9390 | 0.5785 | 0.3911 | PASS |
| dlinear_gaussian | gaussian | heave_rate | 10.0000 | 3 | 0.9007 | 0.0004 | 0.8885 | 0.9134 | 1.1141 | 0.7536 | PASS |
| dlinear_gaussian | gaussian | heave_rate | 15.0000 | 3 | 0.8826 | 0.0000 | 0.8692 | 0.8962 | 1.3408 | 0.9074 | PASS |
| dlinear_gaussian | gaussian | pitch | 1.0000 | 3 | 0.9229 | 0.0053 | 0.9046 | 0.9411 | 0.1196 | 0.0293 | PASS |
| dlinear_gaussian | gaussian | pitch | 2.0000 | 3 | 0.9047 | 0.0009 | 0.8888 | 0.9192 | 0.6524 | 0.1597 | PASS |
| dlinear_gaussian | gaussian | pitch | 3.0000 | 3 | 0.8947 | 0.0004 | 0.8797 | 0.9092 | 1.3653 | 0.3343 | PASS |
| dlinear_gaussian | gaussian | pitch | 5.0000 | 3 | 0.8867 | 0.0006 | 0.8714 | 0.9014 | 1.6056 | 0.3931 | PASS |
| dlinear_gaussian | gaussian | pitch | 10.0000 | 3 | 0.8647 | 0.0001 | 0.8489 | 0.8804 | 3.0921 | 0.7573 | PASS |
| dlinear_gaussian | gaussian | pitch | 15.0000 | 3 | 0.8753 | 0.0000 | 0.8606 | 0.8900 | 3.7212 | 0.9119 | PASS |
| dlinear_gaussian | gaussian | pitch_rate | 1.0000 | 3 | 0.8636 | 0.0057 | 0.8345 | 0.8904 | 0.0891 | 0.0293 | PASS |
| dlinear_gaussian | gaussian | pitch_rate | 2.0000 | 3 | 0.8470 | 0.0012 | 0.8230 | 0.8686 | 0.4864 | 0.1601 | FAIL |
| dlinear_gaussian | gaussian | pitch_rate | 3.0000 | 3 | 0.8454 | 0.0007 | 0.8239 | 0.8654 | 1.0179 | 0.3349 | FAIL |
| dlinear_gaussian | gaussian | pitch_rate | 5.0000 | 3 | 0.8404 | 0.0004 | 0.8199 | 0.8603 | 1.1971 | 0.3939 | FAIL |
| dlinear_gaussian | gaussian | pitch_rate | 10.0000 | 3 | 0.8385 | 0.0001 | 0.8187 | 0.8569 | 2.3053 | 0.7586 | FAIL |
| dlinear_gaussian | gaussian | pitch_rate | 15.0000 | 3 | 0.8729 | 0.0000 | 0.8572 | 0.8882 | 2.7744 | 0.9133 | PASS |
| dlinear_gaussian | gaussian | roll | 1.0000 | 3 | 0.9917 | 0.0047 | 0.9827 | 0.9975 | 0.3499 | 0.0289 | FAIL |
| dlinear_gaussian | gaussian | roll | 2.0000 | 3 | 0.9828 | 0.0006 | 0.9776 | 0.9881 | 1.9091 | 0.1575 | FAIL |
| dlinear_gaussian | gaussian | roll | 3.0000 | 3 | 0.9706 | 0.0006 | 0.9630 | 0.9780 | 3.9955 | 0.3296 | FAIL |
| dlinear_gaussian | gaussian | roll | 5.0000 | 3 | 0.9707 | 0.0006 | 0.9632 | 0.9781 | 4.6986 | 0.3875 | FAIL |
| dlinear_gaussian | gaussian | roll | 10.0000 | 3 | 0.9624 | 0.0003 | 0.9548 | 0.9712 | 9.0486 | 0.7454 | FAIL |
| dlinear_gaussian | gaussian | roll | 15.0000 | 3 | 0.9410 | 0.0000 | 0.9302 | 0.9531 | 10.8897 | 0.8962 | PASS |
| dlinear_gaussian | gaussian | roll_rate | 1.0000 | 3 | 0.9937 | 0.0044 | 0.9860 | 0.9986 | 0.1872 | 0.0289 | FAIL |
| dlinear_gaussian | gaussian | roll_rate | 2.0000 | 3 | 0.9866 | 0.0006 | 0.9823 | 0.9908 | 1.0214 | 0.1576 | FAIL |
| dlinear_gaussian | gaussian | roll_rate | 3.0000 | 3 | 0.9743 | 0.0008 | 0.9676 | 0.9812 | 2.1376 | 0.3298 | FAIL |
| dlinear_gaussian | gaussian | roll_rate | 5.0000 | 3 | 0.9738 | 0.0005 | 0.9673 | 0.9803 | 2.5138 | 0.3877 | FAIL |
| dlinear_gaussian | gaussian | roll_rate | 10.0000 | 3 | 0.9614 | 0.0001 | 0.9541 | 0.9695 | 4.8410 | 0.7460 | FAIL |
| dlinear_gaussian | gaussian | roll_rate | 15.0000 | 3 | 0.9387 | 0.0000 | 0.9283 | 0.9505 | 5.8260 | 0.8971 | PASS |
| dlinear_quantile | quantile | heave | 1.0000 | 3 | 0.9335 | 0.0055 | 0.9160 | 0.9509 | 0.0471 | 0.0189 | PASS |
| dlinear_quantile | quantile | heave | 2.0000 | 3 | 0.9149 | 0.0040 | 0.8974 | 0.9336 | 0.2976 | 0.1191 | PASS |
| dlinear_quantile | quantile | heave | 3.0000 | 3 | 0.9016 | 0.0008 | 0.8847 | 0.9184 | 0.7101 | 0.2843 | PASS |
| dlinear_quantile | quantile | heave | 5.0000 | 3 | 0.9009 | 0.0007 | 0.8846 | 0.9176 | 0.8686 | 0.3477 | PASS |
| dlinear_quantile | quantile | heave | 10.0000 | 3 | 0.9014 | 0.0003 | 0.8873 | 0.9156 | 1.8287 | 0.7322 | PASS |
| dlinear_quantile | quantile | heave | 15.0000 | 3 | 0.8861 | 0.0001 | 0.8708 | 0.9017 | 2.2701 | 0.9095 | PASS |
| dlinear_quantile | quantile | heave_rate | 1.0000 | 3 | 0.9343 | 0.0066 | 0.9187 | 0.9506 | 0.0280 | 0.0189 | PASS |
| dlinear_quantile | quantile | heave_rate | 2.0000 | 3 | 0.9167 | 0.0054 | 0.9007 | 0.9340 | 0.1766 | 0.1194 | PASS |
| dlinear_quantile | quantile | heave_rate | 3.0000 | 3 | 0.9076 | 0.0013 | 0.8937 | 0.9218 | 0.4214 | 0.2848 | PASS |
| dlinear_quantile | quantile | heave_rate | 5.0000 | 3 | 0.9043 | 0.0008 | 0.8913 | 0.9184 | 0.5141 | 0.3476 | PASS |
| dlinear_quantile | quantile | heave_rate | 10.0000 | 3 | 0.8934 | 0.0003 | 0.8804 | 0.9064 | 1.0846 | 0.7336 | PASS |
| dlinear_quantile | quantile | heave_rate | 15.0000 | 3 | 0.8828 | 0.0002 | 0.8691 | 0.8965 | 1.3468 | 0.9115 | PASS |
| dlinear_quantile | quantile | pitch | 1.0000 | 3 | 0.8645 | 0.0107 | 0.8355 | 0.8931 | 0.0795 | 0.0195 | PASS |
| dlinear_quantile | quantile | pitch | 2.0000 | 3 | 0.8508 | 0.0065 | 0.8270 | 0.8761 | 0.4949 | 0.1212 | PASS |
| dlinear_quantile | quantile | pitch | 3.0000 | 3 | 0.8603 | 0.0020 | 0.8411 | 0.8789 | 1.1724 | 0.2871 | PASS |
| dlinear_quantile | quantile | pitch | 5.0000 | 3 | 0.8603 | 0.0008 | 0.8424 | 0.8775 | 1.4267 | 0.3493 | PASS |
| dlinear_quantile | quantile | pitch | 10.0000 | 3 | 0.8576 | 0.0002 | 0.8410 | 0.8741 | 3.0101 | 0.7372 | PASS |
| dlinear_quantile | quantile | pitch | 15.0000 | 3 | 0.8757 | 0.0001 | 0.8610 | 0.8905 | 3.7380 | 0.9160 | PASS |
| dlinear_quantile | quantile | pitch_rate | 1.0000 | 3 | 0.8065 | 0.0099 | 0.7715 | 0.8412 | 0.0655 | 0.0215 | FAIL |
| dlinear_quantile | quantile | pitch_rate | 2.0000 | 3 | 0.7931 | 0.0059 | 0.7634 | 0.8241 | 0.3875 | 0.1275 | FAIL |
| dlinear_quantile | quantile | pitch_rate | 3.0000 | 3 | 0.8094 | 0.0020 | 0.7844 | 0.8333 | 0.8914 | 0.2933 | FAIL |
| dlinear_quantile | quantile | pitch_rate | 5.0000 | 3 | 0.8125 | 0.0008 | 0.7889 | 0.8348 | 1.0659 | 0.3507 | FAIL |
| dlinear_quantile | quantile | pitch_rate | 10.0000 | 3 | 0.8315 | 0.0002 | 0.8112 | 0.8507 | 2.2445 | 0.7386 | FAIL |
| dlinear_quantile | quantile | pitch_rate | 15.0000 | 3 | 0.8733 | 0.0001 | 0.8577 | 0.8887 | 2.7868 | 0.9174 | PASS |
| dlinear_quantile | quantile | roll | 1.0000 | 3 | 0.9727 | 0.0035 | 0.9619 | 0.9810 | 0.2272 | 0.0188 | FAIL |
| dlinear_quantile | quantile | roll | 2.0000 | 3 | 0.9668 | 0.0027 | 0.9571 | 0.9769 | 1.4342 | 0.1184 | FAIL |
| dlinear_quantile | quantile | roll | 3.0000 | 3 | 0.9609 | 0.0001 | 0.9524 | 0.9698 | 3.4218 | 0.2823 | FAIL |
| dlinear_quantile | quantile | roll | 5.0000 | 3 | 0.9624 | 0.0003 | 0.9542 | 0.9711 | 4.1734 | 0.3442 | FAIL |
| dlinear_quantile | quantile | roll | 10.0000 | 3 | 0.9596 | 0.0002 | 0.9516 | 0.9688 | 8.8086 | 0.7256 | FAIL |
| dlinear_quantile | quantile | roll | 15.0000 | 3 | 0.9422 | 0.0001 | 0.9316 | 0.9541 | 10.9387 | 0.9002 | PASS |
| dlinear_quantile | quantile | roll_rate | 1.0000 | 3 | 0.9739 | 0.0030 | 0.9644 | 0.9817 | 0.1215 | 0.0188 | FAIL |
| dlinear_quantile | quantile | roll_rate | 2.0000 | 3 | 0.9687 | 0.0029 | 0.9602 | 0.9786 | 0.7673 | 0.1184 | FAIL |
| dlinear_quantile | quantile | roll_rate | 3.0000 | 3 | 0.9635 | 0.0002 | 0.9562 | 0.9718 | 1.8307 | 0.2824 | FAIL |
| dlinear_quantile | quantile | roll_rate | 5.0000 | 3 | 0.9643 | 0.0003 | 0.9568 | 0.9723 | 2.2324 | 0.3443 | FAIL |
| dlinear_quantile | quantile | roll_rate | 10.0000 | 3 | 0.9577 | 0.0001 | 0.9499 | 0.9664 | 4.7126 | 0.7262 | FAIL |
| dlinear_quantile | quantile | roll_rate | 15.0000 | 3 | 0.9399 | 0.0001 | 0.9294 | 0.9515 | 5.8522 | 0.9011 | PASS |
| lstm_gaussian | gaussian | heave | 1.0000 | 3 | 0.9935 | 0.0008 | 0.9914 | 0.9949 | 0.0301 | 0.0121 | FAIL |
| lstm_gaussian | gaussian | heave | 2.0000 | 3 | 0.9909 | 0.0003 | 0.9894 | 0.9923 | 0.0342 | 0.0137 | FAIL |
| lstm_gaussian | gaussian | heave | 3.0000 | 3 | 0.9769 | 0.0009 | 0.9737 | 0.9796 | 0.0490 | 0.0196 | FAIL |
| lstm_gaussian | gaussian | heave | 5.0000 | 3 | 0.9398 | 0.0043 | 0.9322 | 0.9465 | 0.1511 | 0.0605 | PASS |
| lstm_gaussian | gaussian | heave | 10.0000 | 3 | 0.9210 | 0.0013 | 0.9160 | 0.9260 | 0.5194 | 0.2080 | PASS |
| lstm_gaussian | gaussian | heave | 15.0000 | 3 | 0.9098 | 0.0012 | 0.9054 | 0.9139 | 1.0591 | 0.4243 | PASS |
| lstm_gaussian | gaussian | heave_rate | 1.0000 | 3 | 0.9895 | 0.0006 | 0.9876 | 0.9914 | 0.0215 | 0.0145 | FAIL |
| lstm_gaussian | gaussian | heave_rate | 2.0000 | 3 | 0.9824 | 0.0018 | 0.9786 | 0.9856 | 0.0261 | 0.0177 | FAIL |
| lstm_gaussian | gaussian | heave_rate | 3.0000 | 3 | 0.9638 | 0.0004 | 0.9601 | 0.9668 | 0.0435 | 0.0294 | FAIL |
| lstm_gaussian | gaussian | heave_rate | 5.0000 | 3 | 0.9350 | 0.0030 | 0.9279 | 0.9405 | 0.0940 | 0.0636 | PASS |
| lstm_gaussian | gaussian | heave_rate | 10.0000 | 3 | 0.9177 | 0.0019 | 0.9120 | 0.9219 | 0.3794 | 0.2566 | PASS |
| lstm_gaussian | gaussian | heave_rate | 15.0000 | 3 | 0.9089 | 0.0003 | 0.9053 | 0.9123 | 0.7241 | 0.4901 | PASS |
| lstm_gaussian | gaussian | pitch | 1.0000 | 3 | 0.9896 | 0.0013 | 0.9870 | 0.9922 | 0.0595 | 0.0146 | FAIL |
| lstm_gaussian | gaussian | pitch | 2.0000 | 3 | 0.9781 | 0.0016 | 0.9735 | 0.9812 | 0.0907 | 0.0222 | FAIL |
| lstm_gaussian | gaussian | pitch | 3.0000 | 3 | 0.9558 | 0.0064 | 0.9444 | 0.9625 | 0.1472 | 0.0360 | FAIL |
| lstm_gaussian | gaussian | pitch | 5.0000 | 3 | 0.9375 | 0.0079 | 0.9254 | 0.9483 | 0.3420 | 0.0837 | PASS |
| lstm_gaussian | gaussian | pitch | 10.0000 | 3 | 0.9172 | 0.0032 | 0.9100 | 0.9220 | 1.2751 | 0.3123 | PASS |
| lstm_gaussian | gaussian | pitch | 15.0000 | 3 | 0.9076 | 0.0014 | 0.9035 | 0.9118 | 2.2352 | 0.5477 | PASS |
| lstm_gaussian | gaussian | pitch_rate | 1.0000 | 3 | 0.9824 | 0.0027 | 0.9772 | 0.9863 | 0.0658 | 0.0216 | FAIL |
| lstm_gaussian | gaussian | pitch_rate | 2.0000 | 3 | 0.9647 | 0.0058 | 0.9552 | 0.9723 | 0.0978 | 0.0322 | FAIL |
| lstm_gaussian | gaussian | pitch_rate | 3.0000 | 3 | 0.9430 | 0.0246 | 0.9058 | 0.9622 | 0.1360 | 0.0447 | PASS |
| lstm_gaussian | gaussian | pitch_rate | 5.0000 | 3 | 0.9383 | 0.0061 | 0.9276 | 0.9463 | 0.3728 | 0.1227 | PASS |
| lstm_gaussian | gaussian | pitch_rate | 10.0000 | 3 | 0.9152 | 0.0022 | 0.9098 | 0.9198 | 1.2078 | 0.3975 | PASS |
| lstm_gaussian | gaussian | pitch_rate | 15.0000 | 3 | 0.9066 | 0.0009 | 0.9030 | 0.9103 | 1.8484 | 0.6085 | PASS |
| lstm_gaussian | gaussian | roll | 1.0000 | 3 | 0.9918 | 0.0028 | 0.9869 | 0.9947 | 0.1339 | 0.0111 | FAIL |
| lstm_gaussian | gaussian | roll | 2.0000 | 3 | 0.9906 | 0.0015 | 0.9877 | 0.9934 | 0.1366 | 0.0113 | FAIL |
| lstm_gaussian | gaussian | roll | 3.0000 | 3 | 0.9822 | 0.0034 | 0.9776 | 0.9880 | 0.1551 | 0.0128 | FAIL |
| lstm_gaussian | gaussian | roll | 5.0000 | 3 | 0.9532 | 0.0073 | 0.9427 | 0.9644 | 0.2176 | 0.0179 | FAIL |
| lstm_gaussian | gaussian | roll | 10.0000 | 3 | 0.9242 | 0.0013 | 0.9196 | 0.9294 | 0.8294 | 0.0683 | PASS |
| lstm_gaussian | gaussian | roll | 15.0000 | 3 | 0.9135 | 0.0044 | 0.9072 | 0.9222 | 1.8962 | 0.1560 | PASS |
| lstm_gaussian | gaussian | roll_rate | 1.0000 | 3 | 0.9926 | 0.0012 | 0.9900 | 0.9945 | 0.0787 | 0.0121 | FAIL |
| lstm_gaussian | gaussian | roll_rate | 2.0000 | 3 | 0.9652 | 0.0267 | 0.9240 | 0.9863 | 0.0882 | 0.0136 | FAIL |
| lstm_gaussian | gaussian | roll_rate | 3.0000 | 3 | 0.9669 | 0.0013 | 0.9627 | 0.9708 | 0.1038 | 0.0160 | FAIL |
| lstm_gaussian | gaussian | roll_rate | 5.0000 | 3 | 0.9491 | 0.0087 | 0.9330 | 0.9585 | 0.1675 | 0.0258 | PASS |
| lstm_gaussian | gaussian | roll_rate | 10.0000 | 3 | 0.9172 | 0.0034 | 0.9100 | 0.9230 | 0.6534 | 0.1007 | PASS |
| lstm_gaussian | gaussian | roll_rate | 15.0000 | 3 | 0.9075 | 0.0022 | 0.9022 | 0.9124 | 1.2816 | 0.1973 | PASS |
| lstm_quantile | quantile | heave | 1.0000 | 3 | 0.9923 | 0.0018 | 0.9894 | 0.9948 | 0.0520 | 0.0208 | FAIL |
| lstm_quantile | quantile | heave | 2.0000 | 3 | 0.9855 | 0.0082 | 0.9742 | 0.9933 | 0.0503 | 0.0201 | FAIL |
| lstm_quantile | quantile | heave | 3.0000 | 3 | 0.9502 | 0.0593 | 0.8580 | 0.9882 | 0.0575 | 0.0230 | FAIL |
| lstm_quantile | quantile | heave | 5.0000 | 3 | 0.9303 | 0.0302 | 0.8880 | 0.9525 | 0.1314 | 0.0526 | PASS |
| lstm_quantile | quantile | heave | 10.0000 | 3 | 0.9172 | 0.0014 | 0.9119 | 0.9213 | 0.4602 | 0.1843 | PASS |
| lstm_quantile | quantile | heave | 15.0000 | 3 | 0.9083 | 0.0006 | 0.9048 | 0.9129 | 0.9621 | 0.3855 | PASS |
| lstm_quantile | quantile | heave_rate | 1.0000 | 3 | 0.9863 | 0.0047 | 0.9795 | 0.9923 | 0.0330 | 0.0223 | FAIL |
| lstm_quantile | quantile | heave_rate | 2.0000 | 3 | 0.9797 | 0.0132 | 0.9605 | 0.9909 | 0.0337 | 0.0228 | FAIL |
| lstm_quantile | quantile | heave_rate | 3.0000 | 3 | 0.9624 | 0.0201 | 0.9333 | 0.9790 | 0.0453 | 0.0306 | FAIL |
| lstm_quantile | quantile | heave_rate | 5.0000 | 3 | 0.9346 | 0.0145 | 0.9131 | 0.9483 | 0.0830 | 0.0561 | PASS |
| lstm_quantile | quantile | heave_rate | 10.0000 | 3 | 0.9203 | 0.0078 | 0.9110 | 0.9327 | 0.3434 | 0.2323 | PASS |
| lstm_quantile | quantile | heave_rate | 15.0000 | 3 | 0.9069 | 0.0001 | 0.9038 | 0.9105 | 0.6631 | 0.4488 | PASS |
| lstm_quantile | quantile | pitch | 1.0000 | 3 | 0.9884 | 0.0028 | 0.9853 | 0.9926 | 0.0855 | 0.0209 | FAIL |
| lstm_quantile | quantile | pitch | 2.0000 | 3 | 0.9814 | 0.0034 | 0.9766 | 0.9867 | 0.0989 | 0.0242 | FAIL |
| lstm_quantile | quantile | pitch | 3.0000 | 3 | 0.9645 | 0.0055 | 0.9568 | 0.9735 | 0.1414 | 0.0346 | FAIL |
| lstm_quantile | quantile | pitch | 5.0000 | 3 | 0.9419 | 0.0177 | 0.9173 | 0.9559 | 0.2967 | 0.0726 | PASS |
| lstm_quantile | quantile | pitch | 10.0000 | 3 | 0.9120 | 0.0071 | 0.9010 | 0.9192 | 1.1582 | 0.2837 | PASS |
| lstm_quantile | quantile | pitch | 15.0000 | 3 | 0.9105 | 0.0074 | 0.9032 | 0.9230 | 2.0459 | 0.5013 | PASS |
| lstm_quantile | quantile | pitch_rate | 1.0000 | 3 | 0.9859 | 0.0038 | 0.9804 | 0.9907 | 0.0808 | 0.0266 | FAIL |
| lstm_quantile | quantile | pitch_rate | 2.0000 | 3 | 0.9791 | 0.0039 | 0.9729 | 0.9846 | 0.1029 | 0.0339 | FAIL |
| lstm_quantile | quantile | pitch_rate | 3.0000 | 3 | 0.9659 | 0.0031 | 0.9600 | 0.9716 | 0.1260 | 0.0415 | FAIL |
| lstm_quantile | quantile | pitch_rate | 5.0000 | 3 | 0.9334 | 0.0274 | 0.8941 | 0.9527 | 0.3229 | 0.1063 | PASS |
| lstm_quantile | quantile | pitch_rate | 10.0000 | 3 | 0.9126 | 0.0030 | 0.9050 | 0.9176 | 1.0905 | 0.3589 | PASS |
| lstm_quantile | quantile | pitch_rate | 15.0000 | 3 | 0.9044 | 0.0042 | 0.8967 | 0.9098 | 1.7232 | 0.5673 | PASS |
| lstm_quantile | quantile | roll | 1.0000 | 3 | 0.9854 | 0.0016 | 0.9821 | 0.9887 | 0.2290 | 0.0189 | FAIL |
| lstm_quantile | quantile | roll | 2.0000 | 3 | 0.9791 | 0.0074 | 0.9678 | 0.9874 | 0.2252 | 0.0186 | FAIL |
| lstm_quantile | quantile | roll | 3.0000 | 3 | 0.9790 | 0.0042 | 0.9717 | 0.9853 | 0.2372 | 0.0196 | FAIL |
| lstm_quantile | quantile | roll | 5.0000 | 3 | 0.9543 | 0.0201 | 0.9219 | 0.9712 | 0.2596 | 0.0214 | FAIL |
| lstm_quantile | quantile | roll | 10.0000 | 3 | 0.9249 | 0.0098 | 0.9090 | 0.9345 | 0.7617 | 0.0627 | PASS |
| lstm_quantile | quantile | roll | 15.0000 | 3 | 0.9204 | 0.0056 | 0.9106 | 0.9293 | 1.7209 | 0.1416 | PASS |
| lstm_quantile | quantile | roll_rate | 1.0000 | 3 | 0.9763 | 0.0127 | 0.9586 | 0.9894 | 0.1330 | 0.0205 | FAIL |
| lstm_quantile | quantile | roll_rate | 2.0000 | 3 | 0.9787 | 0.0071 | 0.9668 | 0.9859 | 0.1380 | 0.0213 | FAIL |
| lstm_quantile | quantile | roll_rate | 3.0000 | 3 | 0.9813 | 0.0051 | 0.9761 | 0.9886 | 0.1439 | 0.0222 | FAIL |
| lstm_quantile | quantile | roll_rate | 5.0000 | 3 | 0.9650 | 0.0047 | 0.9580 | 0.9733 | 0.1797 | 0.0277 | FAIL |
| lstm_quantile | quantile | roll_rate | 10.0000 | 3 | 0.9219 | 0.0092 | 0.9039 | 0.9308 | 0.5958 | 0.0918 | PASS |
| lstm_quantile | quantile | roll_rate | 15.0000 | 3 | 0.9155 | 0.0046 | 0.9075 | 0.9238 | 1.1971 | 0.1843 | PASS |
| tcn_gaussian | gaussian | heave | 1.0000 | 3 | 0.9937 | 0.0025 | 0.9903 | 0.9970 | 0.0174 | 0.0070 | FAIL |
| tcn_gaussian | gaussian | heave | 2.0000 | 3 | 0.9926 | 0.0007 | 0.9912 | 0.9937 | 0.0393 | 0.0157 | FAIL |
| tcn_gaussian | gaussian | heave | 3.0000 | 3 | 0.9780 | 0.0020 | 0.9741 | 0.9810 | 0.0859 | 0.0344 | FAIL |
| tcn_gaussian | gaussian | heave | 5.0000 | 3 | 0.9517 | 0.0013 | 0.9470 | 0.9564 | 0.2430 | 0.0973 | FAIL |
| tcn_gaussian | gaussian | heave | 10.0000 | 3 | 0.9221 | 0.0016 | 0.9166 | 0.9284 | 0.6934 | 0.2777 | PASS |
| tcn_gaussian | gaussian | heave | 15.0000 | 3 | 0.9068 | 0.0013 | 0.9017 | 0.9116 | 1.2366 | 0.4955 | PASS |
| tcn_gaussian | gaussian | heave_rate | 1.0000 | 3 | 0.9922 | 0.0011 | 0.9901 | 0.9937 | 0.0175 | 0.0118 | FAIL |
| tcn_gaussian | gaussian | heave_rate | 2.0000 | 3 | 0.9812 | 0.0006 | 0.9792 | 0.9829 | 0.0413 | 0.0279 | FAIL |
| tcn_gaussian | gaussian | heave_rate | 3.0000 | 3 | 0.9598 | 0.0004 | 0.9566 | 0.9630 | 0.0753 | 0.0509 | FAIL |
| tcn_gaussian | gaussian | heave_rate | 5.0000 | 3 | 0.9426 | 0.0015 | 0.9367 | 0.9481 | 0.1341 | 0.0907 | PASS |
| tcn_gaussian | gaussian | heave_rate | 10.0000 | 3 | 0.9187 | 0.0027 | 0.9127 | 0.9253 | 0.4596 | 0.3109 | PASS |
| tcn_gaussian | gaussian | heave_rate | 15.0000 | 3 | 0.9034 | 0.0031 | 0.8973 | 0.9103 | 0.8088 | 0.5474 | PASS |
| tcn_gaussian | gaussian | pitch | 1.0000 | 3 | 0.9769 | 0.0113 | 0.9632 | 0.9893 | 0.0533 | 0.0131 | FAIL |
| tcn_gaussian | gaussian | pitch | 2.0000 | 3 | 0.9717 | 0.0010 | 0.9685 | 0.9749 | 0.1577 | 0.0386 | FAIL |
| tcn_gaussian | gaussian | pitch | 3.0000 | 3 | 0.9677 | 0.0019 | 0.9635 | 0.9718 | 0.2634 | 0.0645 | FAIL |
| tcn_gaussian | gaussian | pitch | 5.0000 | 3 | 0.9509 | 0.0023 | 0.9448 | 0.9566 | 0.5619 | 0.1376 | FAIL |
| tcn_gaussian | gaussian | pitch | 10.0000 | 3 | 0.9176 | 0.0032 | 0.9106 | 0.9235 | 1.6214 | 0.3971 | PASS |
| tcn_gaussian | gaussian | pitch | 15.0000 | 3 | 0.9056 | 0.0012 | 0.9015 | 0.9105 | 2.4822 | 0.6083 | PASS |
| tcn_gaussian | gaussian | pitch_rate | 1.0000 | 3 | 0.9753 | 0.0033 | 0.9692 | 0.9796 | 0.0864 | 0.0284 | FAIL |
| tcn_gaussian | gaussian | pitch_rate | 2.0000 | 3 | 0.9686 | 0.0034 | 0.9632 | 0.9742 | 0.1640 | 0.0540 | FAIL |
| tcn_gaussian | gaussian | pitch_rate | 3.0000 | 3 | 0.9562 | 0.0018 | 0.9501 | 0.9608 | 0.2484 | 0.0817 | FAIL |
| tcn_gaussian | gaussian | pitch_rate | 5.0000 | 3 | 0.9328 | 0.0019 | 0.9276 | 0.9389 | 0.5625 | 0.1851 | PASS |
| tcn_gaussian | gaussian | pitch_rate | 10.0000 | 3 | 0.9140 | 0.0017 | 0.9094 | 0.9191 | 1.4131 | 0.4650 | PASS |
| tcn_gaussian | gaussian | pitch_rate | 15.0000 | 3 | 0.9014 | 0.0017 | 0.8967 | 0.9069 | 2.0121 | 0.6624 | PASS |
| tcn_gaussian | gaussian | roll | 1.0000 | 3 | 0.9953 | 0.0030 | 0.9911 | 0.9979 | 0.0567 | 0.0047 | FAIL |
| tcn_gaussian | gaussian | roll | 2.0000 | 3 | 0.9849 | 0.0040 | 0.9797 | 0.9906 | 0.1011 | 0.0083 | FAIL |
| tcn_gaussian | gaussian | roll | 3.0000 | 3 | 0.9743 | 0.0041 | 0.9695 | 0.9807 | 0.1747 | 0.0144 | FAIL |
| tcn_gaussian | gaussian | roll | 5.0000 | 3 | 0.9648 | 0.0037 | 0.9594 | 0.9716 | 0.2981 | 0.0246 | FAIL |
| tcn_gaussian | gaussian | roll | 10.0000 | 3 | 0.9326 | 0.0040 | 0.9245 | 0.9391 | 1.0728 | 0.0884 | PASS |
| tcn_gaussian | gaussian | roll | 15.0000 | 3 | 0.9165 | 0.0054 | 0.9088 | 0.9263 | 2.2330 | 0.1838 | PASS |
| tcn_gaussian | gaussian | roll_rate | 1.0000 | 3 | 0.9819 | 0.0113 | 0.9679 | 0.9950 | 0.0506 | 0.0078 | FAIL |
| tcn_gaussian | gaussian | roll_rate | 2.0000 | 3 | 0.9738 | 0.0080 | 0.9624 | 0.9824 | 0.0951 | 0.0147 | FAIL |
| tcn_gaussian | gaussian | roll_rate | 3.0000 | 3 | 0.9656 | 0.0033 | 0.9598 | 0.9716 | 0.1208 | 0.0186 | FAIL |
| tcn_gaussian | gaussian | roll_rate | 5.0000 | 3 | 0.9495 | 0.0020 | 0.9441 | 0.9544 | 0.2672 | 0.0412 | PASS |
| tcn_gaussian | gaussian | roll_rate | 10.0000 | 3 | 0.9236 | 0.0037 | 0.9164 | 0.9308 | 0.8382 | 0.1292 | PASS |
| tcn_gaussian | gaussian | roll_rate | 15.0000 | 3 | 0.9122 | 0.0037 | 0.9059 | 0.9194 | 1.5021 | 0.2313 | PASS |
| tcn_quantile | quantile | heave | 1.0000 | 3 | 0.9929 | 0.0015 | 0.9906 | 0.9950 | 0.0298 | 0.0119 | FAIL |
| tcn_quantile | quantile | heave | 2.0000 | 3 | 0.9920 | 0.0011 | 0.9904 | 0.9938 | 0.0602 | 0.0241 | FAIL |
| tcn_quantile | quantile | heave | 3.0000 | 3 | 0.9860 | 0.0005 | 0.9844 | 0.9874 | 0.1038 | 0.0416 | FAIL |
| tcn_quantile | quantile | heave | 5.0000 | 3 | 0.9580 | 0.0028 | 0.9530 | 0.9639 | 0.2402 | 0.0961 | FAIL |
| tcn_quantile | quantile | heave | 10.0000 | 3 | 0.9215 | 0.0035 | 0.9131 | 0.9289 | 0.6379 | 0.2554 | PASS |
| tcn_quantile | quantile | heave | 15.0000 | 3 | 0.9039 | 0.0027 | 0.8966 | 0.9103 | 1.1279 | 0.4519 | PASS |
| tcn_quantile | quantile | heave_rate | 1.0000 | 3 | 0.9866 | 0.0054 | 0.9786 | 0.9916 | 0.0309 | 0.0209 | FAIL |
| tcn_quantile | quantile | heave_rate | 2.0000 | 3 | 0.9847 | 0.0015 | 0.9826 | 0.9877 | 0.0528 | 0.0357 | FAIL |
| tcn_quantile | quantile | heave_rate | 3.0000 | 3 | 0.9684 | 0.0019 | 0.9646 | 0.9729 | 0.0822 | 0.0556 | FAIL |
| tcn_quantile | quantile | heave_rate | 5.0000 | 3 | 0.9524 | 0.0025 | 0.9462 | 0.9584 | 0.1337 | 0.0904 | FAIL |
| tcn_quantile | quantile | heave_rate | 10.0000 | 3 | 0.9169 | 0.0028 | 0.9099 | 0.9230 | 0.4322 | 0.2923 | PASS |
| tcn_quantile | quantile | heave_rate | 15.0000 | 3 | 0.9003 | 0.0027 | 0.8937 | 0.9061 | 0.7554 | 0.5113 | PASS |
| tcn_quantile | quantile | pitch | 1.0000 | 3 | 0.9887 | 0.0018 | 0.9853 | 0.9910 | 0.0719 | 0.0176 | FAIL |
| tcn_quantile | quantile | pitch | 2.0000 | 3 | 0.9834 | 0.0004 | 0.9812 | 0.9855 | 0.1722 | 0.0422 | FAIL |
| tcn_quantile | quantile | pitch | 3.0000 | 3 | 0.9784 | 0.0014 | 0.9750 | 0.9813 | 0.2743 | 0.0672 | FAIL |
| tcn_quantile | quantile | pitch | 5.0000 | 3 | 0.9612 | 0.0005 | 0.9572 | 0.9656 | 0.5274 | 0.1291 | FAIL |
| tcn_quantile | quantile | pitch | 10.0000 | 3 | 0.9166 | 0.0026 | 0.9097 | 0.9224 | 1.4746 | 0.3611 | PASS |
| tcn_quantile | quantile | pitch | 15.0000 | 3 | 0.9047 | 0.0021 | 0.8991 | 0.9106 | 2.2714 | 0.5566 | PASS |
| tcn_quantile | quantile | pitch_rate | 1.0000 | 3 | 0.9850 | 0.0031 | 0.9798 | 0.9892 | 0.0932 | 0.0307 | FAIL |
| tcn_quantile | quantile | pitch_rate | 2.0000 | 3 | 0.9715 | 0.0057 | 0.9631 | 0.9777 | 0.1727 | 0.0568 | FAIL |
| tcn_quantile | quantile | pitch_rate | 3.0000 | 3 | 0.9706 | 0.0031 | 0.9642 | 0.9763 | 0.2444 | 0.0804 | FAIL |
| tcn_quantile | quantile | pitch_rate | 5.0000 | 3 | 0.9459 | 0.0028 | 0.9390 | 0.9521 | 0.5215 | 0.1716 | PASS |
| tcn_quantile | quantile | pitch_rate | 10.0000 | 3 | 0.9145 | 0.0017 | 0.9091 | 0.9197 | 1.3029 | 0.4288 | PASS |
| tcn_quantile | quantile | pitch_rate | 15.0000 | 3 | 0.9010 | 0.0014 | 0.8962 | 0.9067 | 1.8681 | 0.6150 | PASS |
| tcn_quantile | quantile | roll | 1.0000 | 3 | 0.9916 | 0.0020 | 0.9878 | 0.9934 | 0.1112 | 0.0092 | FAIL |
| tcn_quantile | quantile | roll | 2.0000 | 3 | 0.9893 | 0.0009 | 0.9874 | 0.9915 | 0.1963 | 0.0162 | FAIL |
| tcn_quantile | quantile | roll | 3.0000 | 3 | 0.9836 | 0.0013 | 0.9812 | 0.9869 | 0.2938 | 0.0242 | FAIL |
| tcn_quantile | quantile | roll | 5.0000 | 3 | 0.9751 | 0.0024 | 0.9699 | 0.9811 | 0.4093 | 0.0338 | FAIL |
| tcn_quantile | quantile | roll | 10.0000 | 3 | 0.9368 | 0.0019 | 0.9308 | 0.9425 | 1.0699 | 0.0881 | PASS |
| tcn_quantile | quantile | roll | 15.0000 | 3 | 0.9222 | 0.0043 | 0.9140 | 0.9317 | 2.1321 | 0.1755 | PASS |
| tcn_quantile | quantile | roll_rate | 1.0000 | 3 | 0.9805 | 0.0016 | 0.9768 | 0.9843 | 0.1087 | 0.0168 | FAIL |
| tcn_quantile | quantile | roll_rate | 2.0000 | 3 | 0.9744 | 0.0034 | 0.9693 | 0.9806 | 0.1638 | 0.0253 | FAIL |
| tcn_quantile | quantile | roll_rate | 3.0000 | 3 | 0.9734 | 0.0029 | 0.9669 | 0.9782 | 0.1920 | 0.0296 | FAIL |
| tcn_quantile | quantile | roll_rate | 5.0000 | 3 | 0.9594 | 0.0049 | 0.9483 | 0.9668 | 0.3244 | 0.0500 | FAIL |
| tcn_quantile | quantile | roll_rate | 10.0000 | 3 | 0.9250 | 0.0038 | 0.9157 | 0.9322 | 0.8296 | 0.1278 | PASS |
| tcn_quantile | quantile | roll_rate | 15.0000 | 3 | 0.9146 | 0.0042 | 0.9063 | 0.9226 | 1.4414 | 0.2220 | PASS |

## Coverage under distribution shift -- reported, not fixed

Each row is one cell's move from `id` to a held-out regime. Read `picp_delta` and `width_delta` **together**: an interval that keeps its coverage by growing has not kept its calibration.

| model | head | regime | dof | horizon_samples | horizon_s | picp_id | picp_ood | picp_delta | width_id | width_ood | width_delta | width_ratio_id | width_ratio_ood |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dlinear_gaussian | gaussian | unseen_heading | heave | 10 | 1.0000 | 0.9624 | 0.9790 | 0.0167 | 0.0726 | 0.0729 | 0.0003 | 0.0291 | 0.0284 |
| dlinear_gaussian | gaussian | unseen_heading | heave | 20 | 2.0000 | 0.9444 | 0.9588 | 0.0144 | 0.3962 | 0.3984 | 0.0022 | 0.1586 | 0.1554 |
| dlinear_gaussian | gaussian | unseen_heading | heave | 30 | 3.0000 | 0.9224 | 0.9368 | 0.0143 | 0.8292 | 0.8418 | 0.0126 | 0.3320 | 0.3282 |
| dlinear_gaussian | gaussian | unseen_heading | heave | 50 | 5.0000 | 0.9196 | 0.9311 | 0.0115 | 0.9751 | 0.9920 | 0.0170 | 0.3904 | 0.3867 |
| dlinear_gaussian | gaussian | unseen_heading | heave | 100 | 10.0000 | 0.9072 | 0.9212 | 0.0141 | 1.8778 | 1.8955 | 0.0177 | 0.7519 | 0.7388 |
| dlinear_gaussian | gaussian | unseen_heading | heave | 150 | 15.0000 | 0.8856 | 0.8920 | 0.0064 | 2.2599 | 2.2706 | 0.0107 | 0.9054 | 0.8850 |
| dlinear_gaussian | gaussian | unseen_heading | heave_rate | 10 | 1.0000 | 0.9701 | 0.9927 | 0.0226 | 0.0431 | 0.0437 | 0.0006 | 0.0291 | 0.0296 |
| dlinear_gaussian | gaussian | unseen_heading | heave_rate | 20 | 2.0000 | 0.9551 | 0.9786 | 0.0236 | 0.2351 | 0.2388 | 0.0037 | 0.1589 | 0.1616 |
| dlinear_gaussian | gaussian | unseen_heading | heave_rate | 30 | 3.0000 | 0.9344 | 0.9607 | 0.0262 | 0.4919 | 0.5045 | 0.0125 | 0.3326 | 0.3415 |
| dlinear_gaussian | gaussian | unseen_heading | heave_rate | 50 | 5.0000 | 0.9274 | 0.9521 | 0.0247 | 0.5785 | 0.5945 | 0.0160 | 0.3911 | 0.4024 |
| dlinear_gaussian | gaussian | unseen_heading | heave_rate | 100 | 10.0000 | 0.9007 | 0.9258 | 0.0251 | 1.1141 | 1.1360 | 0.0219 | 0.7536 | 0.7689 |
| dlinear_gaussian | gaussian | unseen_heading | heave_rate | 150 | 15.0000 | 0.8826 | 0.8941 | 0.0115 | 1.3408 | 1.3608 | 0.0201 | 0.9074 | 0.9211 |
| dlinear_gaussian | gaussian | unseen_heading | pitch | 10 | 1.0000 | 0.9229 | 1.0000 | 0.0771 | 0.1196 | 0.1398 | 0.0202 | 0.0293 | 0.4553 |
| dlinear_gaussian | gaussian | unseen_heading | pitch | 20 | 2.0000 | 0.9047 | 1.0000 | 0.0953 | 0.6524 | 0.7640 | 0.1116 | 0.1597 | 2.4887 |
| dlinear_gaussian | gaussian | unseen_heading | pitch | 30 | 3.0000 | 0.8947 | 1.0000 | 0.1053 | 1.3653 | 1.6141 | 0.2488 | 0.3343 | 5.2583 |
| dlinear_gaussian | gaussian | unseen_heading | pitch | 50 | 5.0000 | 0.8867 | 1.0000 | 0.1133 | 1.6056 | 1.9023 | 0.2966 | 0.3931 | 6.1978 |
| dlinear_gaussian | gaussian | unseen_heading | pitch | 100 | 10.0000 | 0.8647 | 1.0000 | 0.1353 | 3.0921 | 3.6346 | 0.5425 | 0.7573 | 11.8453 |
| dlinear_gaussian | gaussian | unseen_heading | pitch | 150 | 15.0000 | 0.8753 | 1.0000 | 0.1247 | 3.7212 | 4.3540 | 0.6328 | 0.9119 | 14.1875 |
| dlinear_gaussian | gaussian | unseen_heading | pitch_rate | 10 | 1.0000 | 0.8636 | 1.0000 | 0.1364 | 0.0891 | 0.1042 | 0.0150 | 0.0293 | 0.4816 |
| dlinear_gaussian | gaussian | unseen_heading | pitch_rate | 20 | 2.0000 | 0.8470 | 1.0000 | 0.1530 | 0.4864 | 0.5695 | 0.0831 | 0.1601 | 2.6324 |
| dlinear_gaussian | gaussian | unseen_heading | pitch_rate | 30 | 3.0000 | 0.8454 | 1.0000 | 0.1546 | 1.0179 | 1.2032 | 0.1853 | 0.3349 | 5.5623 |
| dlinear_gaussian | gaussian | unseen_heading | pitch_rate | 50 | 5.0000 | 0.8404 | 1.0000 | 0.1596 | 1.1971 | 1.4180 | 0.2210 | 0.3939 | 6.5567 |
| dlinear_gaussian | gaussian | unseen_heading | pitch_rate | 100 | 10.0000 | 0.8385 | 1.0000 | 0.1615 | 2.3053 | 2.7094 | 0.4041 | 0.7586 | 12.5308 |
| dlinear_gaussian | gaussian | unseen_heading | pitch_rate | 150 | 15.0000 | 0.8729 | 1.0000 | 0.1271 | 2.7744 | 3.2456 | 0.4713 | 0.9133 | 15.0099 |
| dlinear_gaussian | gaussian | unseen_heading | roll | 10 | 1.0000 | 0.9917 | 0.9634 | -0.0283 | 0.3499 | 0.2947 | -0.0552 | 0.0289 | 0.0177 |
| dlinear_gaussian | gaussian | unseen_heading | roll | 20 | 2.0000 | 0.9828 | 0.9417 | -0.0411 | 1.9091 | 1.6107 | -0.2983 | 0.1575 | 0.0968 |
| dlinear_gaussian | gaussian | unseen_heading | roll | 30 | 3.0000 | 0.9706 | 0.9203 | -0.0502 | 3.9955 | 3.4031 | -0.5923 | 0.3296 | 0.2045 |
| dlinear_gaussian | gaussian | unseen_heading | roll | 50 | 5.0000 | 0.9707 | 0.9148 | -0.0559 | 4.6986 | 4.0107 | -0.6879 | 0.3875 | 0.2410 |
| dlinear_gaussian | gaussian | unseen_heading | roll | 100 | 10.0000 | 0.9624 | 0.8922 | -0.0703 | 9.0486 | 7.6631 | -1.3855 | 0.7454 | 0.4602 |
| dlinear_gaussian | gaussian | unseen_heading | roll | 150 | 15.0000 | 0.9410 | 0.8537 | -0.0873 | 10.8897 | 9.1799 | -1.7098 | 0.8962 | 0.5509 |
| dlinear_gaussian | gaussian | unseen_heading | roll_rate | 10 | 1.0000 | 0.9937 | 0.9633 | -0.0304 | 0.1872 | 0.1562 | -0.0310 | 0.0289 | 0.0174 |
| dlinear_gaussian | gaussian | unseen_heading | roll_rate | 20 | 2.0000 | 0.9866 | 0.9396 | -0.0470 | 1.0214 | 0.8537 | -0.1677 | 0.1576 | 0.0949 |
| dlinear_gaussian | gaussian | unseen_heading | roll_rate | 30 | 3.0000 | 0.9743 | 0.9153 | -0.0590 | 2.1376 | 1.8037 | -0.3339 | 0.3298 | 0.2006 |
| dlinear_gaussian | gaussian | unseen_heading | roll_rate | 50 | 5.0000 | 0.9738 | 0.9069 | -0.0669 | 2.5138 | 2.1257 | -0.3881 | 0.3877 | 0.2363 |
| dlinear_gaussian | gaussian | unseen_heading | roll_rate | 100 | 10.0000 | 0.9614 | 0.8737 | -0.0877 | 4.8410 | 4.0614 | -0.7796 | 0.7460 | 0.4512 |
| dlinear_gaussian | gaussian | unseen_heading | roll_rate | 150 | 15.0000 | 0.9387 | 0.8335 | -0.1052 | 5.8260 | 4.8653 | -0.9607 | 0.8971 | 0.5403 |
| dlinear_quantile | quantile | unseen_heading | heave | 10 | 1.0000 | 0.9335 | 0.9587 | 0.0252 | 0.0471 | 0.0516 | 0.0044 | 0.0189 | 0.0201 |
| dlinear_quantile | quantile | unseen_heading | heave | 20 | 2.0000 | 0.9149 | 0.9348 | 0.0199 | 0.2976 | 0.3238 | 0.0262 | 0.1191 | 0.1263 |
| dlinear_quantile | quantile | unseen_heading | heave | 30 | 3.0000 | 0.9016 | 0.9183 | 0.0167 | 0.7101 | 0.7534 | 0.0432 | 0.2843 | 0.2937 |
| dlinear_quantile | quantile | unseen_heading | heave | 50 | 5.0000 | 0.9009 | 0.9177 | 0.0169 | 0.8686 | 0.9093 | 0.0406 | 0.3477 | 0.3545 |
| dlinear_quantile | quantile | unseen_heading | heave | 100 | 10.0000 | 0.9014 | 0.9192 | 0.0178 | 1.8287 | 1.8842 | 0.0556 | 0.7322 | 0.7345 |
| dlinear_quantile | quantile | unseen_heading | heave | 150 | 15.0000 | 0.8861 | 0.8944 | 0.0083 | 2.2701 | 2.2954 | 0.0253 | 0.9095 | 0.8947 |
| dlinear_quantile | quantile | unseen_heading | heave_rate | 10 | 1.0000 | 0.9343 | 0.9743 | 0.0400 | 0.0280 | 0.0309 | 0.0029 | 0.0189 | 0.0209 |
| dlinear_quantile | quantile | unseen_heading | heave_rate | 20 | 2.0000 | 0.9167 | 0.9548 | 0.0381 | 0.1766 | 0.1941 | 0.0174 | 0.1194 | 0.1314 |
| dlinear_quantile | quantile | unseen_heading | heave_rate | 30 | 3.0000 | 0.9076 | 0.9405 | 0.0329 | 0.4214 | 0.4515 | 0.0301 | 0.2848 | 0.3056 |
| dlinear_quantile | quantile | unseen_heading | heave_rate | 50 | 5.0000 | 0.9043 | 0.9375 | 0.0332 | 0.5141 | 0.5447 | 0.0306 | 0.3476 | 0.3686 |
| dlinear_quantile | quantile | unseen_heading | heave_rate | 100 | 10.0000 | 0.8934 | 0.9226 | 0.0292 | 1.0846 | 1.1292 | 0.0447 | 0.7336 | 0.7643 |
| dlinear_quantile | quantile | unseen_heading | heave_rate | 150 | 15.0000 | 0.8828 | 0.8963 | 0.0135 | 1.3468 | 1.3757 | 0.0289 | 0.9115 | 0.9311 |
| dlinear_quantile | quantile | unseen_heading | pitch | 10 | 1.0000 | 0.8645 | 1.0000 | 0.1355 | 0.0795 | 0.0988 | 0.0193 | 0.0195 | 0.3220 |
| dlinear_quantile | quantile | unseen_heading | pitch | 20 | 2.0000 | 0.8508 | 1.0000 | 0.1492 | 0.4949 | 0.6209 | 0.1260 | 0.1212 | 2.0228 |
| dlinear_quantile | quantile | unseen_heading | pitch | 30 | 3.0000 | 0.8603 | 1.0000 | 0.1397 | 1.1724 | 1.4445 | 0.2721 | 0.2871 | 4.7059 |
| dlinear_quantile | quantile | unseen_heading | pitch | 50 | 5.0000 | 0.8603 | 1.0000 | 0.1397 | 1.4267 | 1.7426 | 0.3159 | 0.3493 | 5.6777 |
| dlinear_quantile | quantile | unseen_heading | pitch | 100 | 10.0000 | 0.8576 | 1.0000 | 0.1424 | 3.0101 | 3.6130 | 0.6029 | 0.7372 | 11.7750 |
| dlinear_quantile | quantile | unseen_heading | pitch | 150 | 15.0000 | 0.8757 | 1.0000 | 0.1243 | 3.7380 | 4.4015 | 0.6635 | 0.9160 | 14.3422 |
| dlinear_quantile | quantile | unseen_heading | pitch_rate | 10 | 1.0000 | 0.8065 | 1.0000 | 0.1935 | 0.0655 | 0.0737 | 0.0082 | 0.0215 | 0.3405 |
| dlinear_quantile | quantile | unseen_heading | pitch_rate | 20 | 2.0000 | 0.7931 | 1.0000 | 0.2069 | 0.3875 | 0.4629 | 0.0754 | 0.1275 | 2.1396 |
| dlinear_quantile | quantile | unseen_heading | pitch_rate | 30 | 3.0000 | 0.8094 | 1.0000 | 0.1906 | 0.8914 | 1.0768 | 0.1854 | 0.2933 | 4.9780 |
| dlinear_quantile | quantile | unseen_heading | pitch_rate | 50 | 5.0000 | 0.8125 | 1.0000 | 0.1875 | 1.0659 | 1.2990 | 0.2331 | 0.3507 | 6.0065 |
| dlinear_quantile | quantile | unseen_heading | pitch_rate | 100 | 10.0000 | 0.8315 | 1.0000 | 0.1685 | 2.2445 | 2.6933 | 0.4488 | 0.7386 | 12.4564 |
| dlinear_quantile | quantile | unseen_heading | pitch_rate | 150 | 15.0000 | 0.8733 | 1.0000 | 0.1267 | 2.7868 | 3.2810 | 0.4942 | 0.9174 | 15.1735 |
| dlinear_quantile | quantile | unseen_heading | roll | 10 | 1.0000 | 0.9727 | 0.9502 | -0.0224 | 0.2272 | 0.2090 | -0.0182 | 0.0188 | 0.0126 |
| dlinear_quantile | quantile | unseen_heading | roll | 20 | 2.0000 | 0.9668 | 0.9111 | -0.0557 | 1.4342 | 1.3113 | -0.1229 | 0.1184 | 0.0788 |
| dlinear_quantile | quantile | unseen_heading | roll | 30 | 3.0000 | 0.9609 | 0.9037 | -0.0572 | 3.4218 | 3.0468 | -0.3750 | 0.2823 | 0.1831 |
| dlinear_quantile | quantile | unseen_heading | roll | 50 | 5.0000 | 0.9624 | 0.9027 | -0.0598 | 4.1734 | 3.6755 | -0.4979 | 0.3442 | 0.2209 |
| dlinear_quantile | quantile | unseen_heading | roll | 100 | 10.0000 | 0.9596 | 0.8896 | -0.0700 | 8.8086 | 7.6176 | -1.1909 | 0.7256 | 0.4574 |
| dlinear_quantile | quantile | unseen_heading | roll | 150 | 15.0000 | 0.9422 | 0.8565 | -0.0857 | 10.9387 | 9.2800 | -1.6588 | 0.9002 | 0.5569 |
| dlinear_quantile | quantile | unseen_heading | roll_rate | 10 | 1.0000 | 0.9739 | 0.9448 | -0.0292 | 0.1215 | 0.1108 | -0.0108 | 0.0188 | 0.0123 |
| dlinear_quantile | quantile | unseen_heading | roll_rate | 20 | 2.0000 | 0.9687 | 0.9030 | -0.0657 | 0.7673 | 0.6949 | -0.0724 | 0.1184 | 0.0773 |
| dlinear_quantile | quantile | unseen_heading | roll_rate | 30 | 3.0000 | 0.9635 | 0.8938 | -0.0697 | 1.8307 | 1.6147 | -0.2160 | 0.2824 | 0.1796 |
| dlinear_quantile | quantile | unseen_heading | roll_rate | 50 | 5.0000 | 0.9643 | 0.8916 | -0.0727 | 2.2324 | 1.9478 | -0.2847 | 0.3443 | 0.2165 |
| dlinear_quantile | quantile | unseen_heading | roll_rate | 100 | 10.0000 | 0.9577 | 0.8703 | -0.0874 | 4.7126 | 4.0373 | -0.6753 | 0.7262 | 0.4485 |
| dlinear_quantile | quantile | unseen_heading | roll_rate | 150 | 15.0000 | 0.9399 | 0.8364 | -0.1034 | 5.8522 | 4.9184 | -0.9339 | 0.9011 | 0.5462 |
| lstm_gaussian | gaussian | unseen_heading | heave | 10 | 1.0000 | 0.9935 | 0.1063 | -0.8872 | 0.0301 | 0.0818 | 0.0516 | 0.0121 | 0.0319 |
| lstm_gaussian | gaussian | unseen_heading | heave | 20 | 2.0000 | 0.9909 | 0.0733 | -0.9176 | 0.0342 | 0.0950 | 0.0608 | 0.0137 | 0.0371 |
| lstm_gaussian | gaussian | unseen_heading | heave | 30 | 3.0000 | 0.9769 | 0.0949 | -0.8820 | 0.0490 | 0.1351 | 0.0861 | 0.0196 | 0.0527 |
| lstm_gaussian | gaussian | unseen_heading | heave | 50 | 5.0000 | 0.9398 | 0.4243 | -0.5155 | 0.1511 | 0.4208 | 0.2697 | 0.0605 | 0.1641 |
| lstm_gaussian | gaussian | unseen_heading | heave | 100 | 10.0000 | 0.9210 | 0.6335 | -0.2875 | 0.5194 | 1.3276 | 0.8082 | 0.2080 | 0.5175 |
| lstm_gaussian | gaussian | unseen_heading | heave | 150 | 15.0000 | 0.9098 | 0.6575 | -0.2523 | 1.0591 | 2.1193 | 1.0602 | 0.4243 | 0.8260 |
| lstm_gaussian | gaussian | unseen_heading | heave_rate | 10 | 1.0000 | 0.9895 | 0.0686 | -0.9209 | 0.0215 | 0.0598 | 0.0383 | 0.0145 | 0.0405 |
| lstm_gaussian | gaussian | unseen_heading | heave_rate | 20 | 2.0000 | 0.9824 | 0.0799 | -0.9025 | 0.0261 | 0.0706 | 0.0445 | 0.0177 | 0.0478 |
| lstm_gaussian | gaussian | unseen_heading | heave_rate | 30 | 3.0000 | 0.9638 | 0.2021 | -0.7617 | 0.0435 | 0.1190 | 0.0755 | 0.0294 | 0.0805 |
| lstm_gaussian | gaussian | unseen_heading | heave_rate | 50 | 5.0000 | 0.9350 | 0.2565 | -0.6785 | 0.0940 | 0.2530 | 0.1590 | 0.0636 | 0.1713 |
| lstm_gaussian | gaussian | unseen_heading | heave_rate | 100 | 10.0000 | 0.9177 | 0.5321 | -0.3856 | 0.3794 | 0.8732 | 0.4938 | 0.2566 | 0.5910 |
| lstm_gaussian | gaussian | unseen_heading | heave_rate | 150 | 15.0000 | 0.9089 | 0.6827 | -0.2262 | 0.7241 | 1.5073 | 0.7832 | 0.4901 | 1.0202 |
| lstm_gaussian | gaussian | unseen_heading | pitch | 10 | 1.0000 | 0.9896 | 0.0728 | -0.9168 | 0.0595 | 0.1579 | 0.0984 | 0.0146 | 0.5144 |
| lstm_gaussian | gaussian | unseen_heading | pitch | 20 | 2.0000 | 0.9781 | 0.0929 | -0.8851 | 0.0907 | 0.2554 | 0.1647 | 0.0222 | 0.8320 |
| lstm_gaussian | gaussian | unseen_heading | pitch | 30 | 3.0000 | 0.9558 | 0.1294 | -0.8263 | 0.1472 | 0.4115 | 0.2643 | 0.0360 | 1.3405 |
| lstm_gaussian | gaussian | unseen_heading | pitch | 50 | 5.0000 | 0.9375 | 0.3549 | -0.5826 | 0.3420 | 0.8624 | 0.5204 | 0.0837 | 2.8099 |
| lstm_gaussian | gaussian | unseen_heading | pitch | 100 | 10.0000 | 0.9172 | 0.5696 | -0.3476 | 1.2751 | 2.5467 | 1.2716 | 0.3123 | 8.2997 |
| lstm_gaussian | gaussian | unseen_heading | pitch | 150 | 15.0000 | 0.9076 | 0.6604 | -0.2471 | 2.2352 | 3.3927 | 1.1575 | 0.5477 | 11.0552 |
| lstm_gaussian | gaussian | unseen_heading | pitch_rate | 10 | 1.0000 | 0.9824 | 0.0849 | -0.8975 | 0.0658 | 0.1696 | 0.1038 | 0.0216 | 0.7839 |
| lstm_gaussian | gaussian | unseen_heading | pitch_rate | 20 | 2.0000 | 0.9647 | 0.1247 | -0.8400 | 0.0978 | 0.2570 | 0.1592 | 0.0322 | 1.1879 |
| lstm_gaussian | gaussian | unseen_heading | pitch_rate | 30 | 3.0000 | 0.9430 | 0.1758 | -0.7672 | 0.1360 | 0.3083 | 0.1723 | 0.0447 | 1.4252 |
| lstm_gaussian | gaussian | unseen_heading | pitch_rate | 50 | 5.0000 | 0.9383 | 0.3398 | -0.5985 | 0.3728 | 1.0073 | 0.6345 | 0.1227 | 4.6576 |
| lstm_gaussian | gaussian | unseen_heading | pitch_rate | 100 | 10.0000 | 0.9152 | 0.5945 | -0.3207 | 1.2078 | 2.3703 | 1.1625 | 0.3975 | 10.9626 |
| lstm_gaussian | gaussian | unseen_heading | pitch_rate | 150 | 15.0000 | 0.9066 | 0.7152 | -0.1914 | 1.8484 | 3.1721 | 1.3237 | 0.6085 | 14.6697 |
| lstm_gaussian | gaussian | unseen_heading | roll | 10 | 1.0000 | 0.9918 | 0.0841 | -0.9077 | 0.1339 | 0.4135 | 0.2796 | 0.0111 | 0.0249 |
| lstm_gaussian | gaussian | unseen_heading | roll | 20 | 2.0000 | 0.9906 | 0.0746 | -0.9161 | 0.1366 | 0.4300 | 0.2934 | 0.0113 | 0.0258 |
| lstm_gaussian | gaussian | unseen_heading | roll | 30 | 3.0000 | 0.9822 | 0.0883 | -0.8939 | 0.1551 | 0.5427 | 0.3876 | 0.0128 | 0.0326 |
| lstm_gaussian | gaussian | unseen_heading | roll | 50 | 5.0000 | 0.9532 | 0.1303 | -0.8229 | 0.2176 | 0.6801 | 0.4624 | 0.0179 | 0.0409 |
| lstm_gaussian | gaussian | unseen_heading | roll | 100 | 10.0000 | 0.9242 | 0.3854 | -0.5387 | 0.8294 | 1.9983 | 1.1689 | 0.0683 | 0.1200 |
| lstm_gaussian | gaussian | unseen_heading | roll | 150 | 15.0000 | 0.9135 | 0.4654 | -0.4481 | 1.8962 | 3.4087 | 1.5125 | 0.1560 | 0.2046 |
| lstm_gaussian | gaussian | unseen_heading | roll_rate | 10 | 1.0000 | 0.9926 | 0.0778 | -0.9148 | 0.0787 | 0.2738 | 0.1951 | 0.0121 | 0.0304 |
| lstm_gaussian | gaussian | unseen_heading | roll_rate | 20 | 2.0000 | 0.9652 | 0.0844 | -0.8808 | 0.0882 | 0.3113 | 0.2230 | 0.0136 | 0.0346 |
| lstm_gaussian | gaussian | unseen_heading | roll_rate | 30 | 3.0000 | 0.9669 | 0.0946 | -0.8724 | 0.1038 | 0.3507 | 0.2468 | 0.0160 | 0.0390 |
| lstm_gaussian | gaussian | unseen_heading | roll_rate | 50 | 5.0000 | 0.9491 | 0.2788 | -0.6704 | 0.1675 | 0.7278 | 0.5603 | 0.0258 | 0.0809 |
| lstm_gaussian | gaussian | unseen_heading | roll_rate | 100 | 10.0000 | 0.9172 | 0.4877 | -0.4295 | 0.6534 | 1.4646 | 0.8112 | 0.1007 | 0.1627 |
| lstm_gaussian | gaussian | unseen_heading | roll_rate | 150 | 15.0000 | 0.9075 | 0.5087 | -0.3988 | 1.2816 | 2.1930 | 0.9113 | 0.1973 | 0.2436 |
| lstm_quantile | quantile | unseen_heading | heave | 10 | 1.0000 | 0.9923 | 0.1855 | -0.8067 | 0.0520 | 0.0814 | 0.0293 | 0.0208 | 0.0317 |
| lstm_quantile | quantile | unseen_heading | heave | 20 | 2.0000 | 0.9855 | 0.1049 | -0.8806 | 0.0503 | 0.0867 | 0.0364 | 0.0201 | 0.0338 |
| lstm_quantile | quantile | unseen_heading | heave | 30 | 3.0000 | 0.9502 | 0.0963 | -0.8538 | 0.0575 | 0.1070 | 0.0495 | 0.0230 | 0.0417 |
| lstm_quantile | quantile | unseen_heading | heave | 50 | 5.0000 | 0.9303 | 0.2974 | -0.6329 | 0.1314 | 0.3181 | 0.1867 | 0.0526 | 0.1240 |
| lstm_quantile | quantile | unseen_heading | heave | 100 | 10.0000 | 0.9172 | 0.5463 | -0.3709 | 0.4602 | 1.1129 | 0.6527 | 0.1843 | 0.4338 |
| lstm_quantile | quantile | unseen_heading | heave | 150 | 15.0000 | 0.9083 | 0.7204 | -0.1879 | 0.9621 | 2.0201 | 1.0580 | 0.3855 | 0.7874 |
| lstm_quantile | quantile | unseen_heading | heave_rate | 10 | 1.0000 | 0.9863 | 0.1294 | -0.8568 | 0.0330 | 0.0557 | 0.0228 | 0.0223 | 0.0377 |
| lstm_quantile | quantile | unseen_heading | heave_rate | 20 | 2.0000 | 0.9797 | 0.1018 | -0.8779 | 0.0337 | 0.0591 | 0.0254 | 0.0228 | 0.0400 |
| lstm_quantile | quantile | unseen_heading | heave_rate | 30 | 3.0000 | 0.9624 | 0.1670 | -0.7954 | 0.0453 | 0.0909 | 0.0456 | 0.0306 | 0.0615 |
| lstm_quantile | quantile | unseen_heading | heave_rate | 50 | 5.0000 | 0.9346 | 0.2023 | -0.7323 | 0.0830 | 0.1908 | 0.1078 | 0.0561 | 0.1291 |
| lstm_quantile | quantile | unseen_heading | heave_rate | 100 | 10.0000 | 0.9203 | 0.5325 | -0.3878 | 0.3434 | 0.7795 | 0.4361 | 0.2323 | 0.5276 |
| lstm_quantile | quantile | unseen_heading | heave_rate | 150 | 15.0000 | 0.9069 | 0.7206 | -0.1863 | 0.6631 | 1.3518 | 0.6886 | 0.4488 | 0.9149 |
| lstm_quantile | quantile | unseen_heading | pitch | 10 | 1.0000 | 0.9884 | 0.1353 | -0.8531 | 0.0855 | 0.1468 | 0.0614 | 0.0209 | 0.4783 |
| lstm_quantile | quantile | unseen_heading | pitch | 20 | 2.0000 | 0.9814 | 0.1200 | -0.8614 | 0.0989 | 0.2019 | 0.1030 | 0.0242 | 0.6577 |
| lstm_quantile | quantile | unseen_heading | pitch | 30 | 3.0000 | 0.9645 | 0.1463 | -0.8182 | 0.1414 | 0.3274 | 0.1860 | 0.0346 | 1.0665 |
| lstm_quantile | quantile | unseen_heading | pitch | 50 | 5.0000 | 0.9419 | 0.3627 | -0.5792 | 0.2967 | 0.7440 | 0.4473 | 0.0726 | 2.4239 |
| lstm_quantile | quantile | unseen_heading | pitch | 100 | 10.0000 | 0.9120 | 0.6115 | -0.3005 | 1.1582 | 2.6074 | 1.4492 | 0.2837 | 8.4977 |
| lstm_quantile | quantile | unseen_heading | pitch | 150 | 15.0000 | 0.9105 | 0.8010 | -0.1095 | 2.0459 | 3.8893 | 1.8434 | 0.5013 | 12.6731 |
| lstm_quantile | quantile | unseen_heading | pitch_rate | 10 | 1.0000 | 0.9859 | 0.2092 | -0.7766 | 0.0808 | 0.1652 | 0.0844 | 0.0266 | 0.7634 |
| lstm_quantile | quantile | unseen_heading | pitch_rate | 20 | 2.0000 | 0.9791 | 0.1903 | -0.7887 | 0.1029 | 0.2186 | 0.1157 | 0.0339 | 1.0105 |
| lstm_quantile | quantile | unseen_heading | pitch_rate | 30 | 3.0000 | 0.9659 | 0.2206 | -0.7452 | 0.1260 | 0.2660 | 0.1400 | 0.0415 | 1.2297 |
| lstm_quantile | quantile | unseen_heading | pitch_rate | 50 | 5.0000 | 0.9334 | 0.3518 | -0.5817 | 0.3229 | 0.8459 | 0.5230 | 0.1063 | 3.9114 |
| lstm_quantile | quantile | unseen_heading | pitch_rate | 100 | 10.0000 | 0.9126 | 0.6831 | -0.2296 | 1.0905 | 2.3701 | 1.2796 | 0.3589 | 10.9618 |
| lstm_quantile | quantile | unseen_heading | pitch_rate | 150 | 15.0000 | 0.9044 | 0.8434 | -0.0610 | 1.7232 | 3.2341 | 1.5109 | 0.5673 | 14.9563 |
| lstm_quantile | quantile | unseen_heading | roll | 10 | 1.0000 | 0.9854 | 0.1507 | -0.8347 | 0.2290 | 0.3680 | 0.1390 | 0.0189 | 0.0221 |
| lstm_quantile | quantile | unseen_heading | roll | 20 | 2.0000 | 0.9791 | 0.1039 | -0.8753 | 0.2252 | 0.3701 | 0.1449 | 0.0186 | 0.0222 |
| lstm_quantile | quantile | unseen_heading | roll | 30 | 3.0000 | 0.9790 | 0.0924 | -0.8865 | 0.2372 | 0.3980 | 0.1609 | 0.0196 | 0.0239 |
| lstm_quantile | quantile | unseen_heading | roll | 50 | 5.0000 | 0.9543 | 0.1155 | -0.8388 | 0.2596 | 0.4821 | 0.2225 | 0.0214 | 0.0290 |
| lstm_quantile | quantile | unseen_heading | roll | 100 | 10.0000 | 0.9249 | 0.3644 | -0.5605 | 0.7617 | 1.6292 | 0.8675 | 0.0627 | 0.0978 |
| lstm_quantile | quantile | unseen_heading | roll | 150 | 15.0000 | 0.9204 | 0.5085 | -0.4119 | 1.7209 | 2.9880 | 1.2670 | 0.1416 | 0.1793 |
| lstm_quantile | quantile | unseen_heading | roll_rate | 10 | 1.0000 | 0.9763 | 0.1240 | -0.8523 | 0.1330 | 0.2044 | 0.0714 | 0.0205 | 0.0227 |
| lstm_quantile | quantile | unseen_heading | roll_rate | 20 | 2.0000 | 0.9787 | 0.1045 | -0.8742 | 0.1380 | 0.2222 | 0.0842 | 0.0213 | 0.0247 |
| lstm_quantile | quantile | unseen_heading | roll_rate | 30 | 3.0000 | 0.9813 | 0.1015 | -0.8798 | 0.1439 | 0.2522 | 0.1083 | 0.0222 | 0.0280 |
| lstm_quantile | quantile | unseen_heading | roll_rate | 50 | 5.0000 | 0.9650 | 0.2195 | -0.7456 | 0.1797 | 0.4896 | 0.3099 | 0.0277 | 0.0544 |
| lstm_quantile | quantile | unseen_heading | roll_rate | 100 | 10.0000 | 0.9219 | 0.5023 | -0.4196 | 0.5958 | 1.3655 | 0.7697 | 0.0918 | 0.1517 |
| lstm_quantile | quantile | unseen_heading | roll_rate | 150 | 15.0000 | 0.9155 | 0.5987 | -0.3168 | 1.1971 | 2.1946 | 0.9974 | 0.1843 | 0.2437 |
| tcn_gaussian | gaussian | unseen_heading | heave | 10 | 1.0000 | 0.9937 | 0.3844 | -0.6093 | 0.0174 | 0.0295 | 0.0120 | 0.0070 | 0.0115 |
| tcn_gaussian | gaussian | unseen_heading | heave | 20 | 2.0000 | 0.9926 | 0.2370 | -0.7556 | 0.0393 | 0.0678 | 0.0285 | 0.0157 | 0.0264 |
| tcn_gaussian | gaussian | unseen_heading | heave | 30 | 3.0000 | 0.9780 | 0.2399 | -0.7382 | 0.0859 | 0.1502 | 0.0644 | 0.0344 | 0.0586 |
| tcn_gaussian | gaussian | unseen_heading | heave | 50 | 5.0000 | 0.9517 | 0.4613 | -0.4903 | 0.2430 | 0.4823 | 0.2393 | 0.0973 | 0.1880 |
| tcn_gaussian | gaussian | unseen_heading | heave | 100 | 10.0000 | 0.9221 | 0.6330 | -0.2891 | 0.6934 | 1.2913 | 0.5979 | 0.2777 | 0.5034 |
| tcn_gaussian | gaussian | unseen_heading | heave | 150 | 15.0000 | 0.9068 | 0.8159 | -0.0909 | 1.2366 | 2.3453 | 1.1087 | 0.4955 | 0.9141 |
| tcn_gaussian | gaussian | unseen_heading | heave_rate | 10 | 1.0000 | 0.9922 | 0.2258 | -0.7664 | 0.0175 | 0.0272 | 0.0097 | 0.0118 | 0.0184 |
| tcn_gaussian | gaussian | unseen_heading | heave_rate | 20 | 2.0000 | 0.9812 | 0.2278 | -0.7534 | 0.0413 | 0.0675 | 0.0262 | 0.0279 | 0.0457 |
| tcn_gaussian | gaussian | unseen_heading | heave_rate | 30 | 3.0000 | 0.9598 | 0.3489 | -0.6110 | 0.0753 | 0.1323 | 0.0569 | 0.0509 | 0.0895 |
| tcn_gaussian | gaussian | unseen_heading | heave_rate | 50 | 5.0000 | 0.9426 | 0.5137 | -0.4290 | 0.1341 | 0.2574 | 0.1232 | 0.0907 | 0.1742 |
| tcn_gaussian | gaussian | unseen_heading | heave_rate | 100 | 10.0000 | 0.9187 | 0.7078 | -0.2109 | 0.4596 | 0.7624 | 0.3028 | 0.3109 | 0.5160 |
| tcn_gaussian | gaussian | unseen_heading | heave_rate | 150 | 15.0000 | 0.9034 | 0.8096 | -0.0938 | 0.8088 | 1.3412 | 0.5324 | 0.5474 | 0.9078 |
| tcn_gaussian | gaussian | unseen_heading | pitch | 10 | 1.0000 | 0.9769 | 0.4247 | -0.5522 | 0.0533 | 0.0779 | 0.0246 | 0.0131 | 0.2538 |
| tcn_gaussian | gaussian | unseen_heading | pitch | 20 | 2.0000 | 0.9717 | 0.3680 | -0.6037 | 0.1577 | 0.2447 | 0.0870 | 0.0386 | 0.7972 |
| tcn_gaussian | gaussian | unseen_heading | pitch | 30 | 3.0000 | 0.9677 | 0.3417 | -0.6260 | 0.2634 | 0.4622 | 0.1988 | 0.0645 | 1.5057 |
| tcn_gaussian | gaussian | unseen_heading | pitch | 50 | 5.0000 | 0.9509 | 0.4832 | -0.4677 | 0.5619 | 0.9095 | 0.3476 | 0.1376 | 2.9631 |
| tcn_gaussian | gaussian | unseen_heading | pitch | 100 | 10.0000 | 0.9176 | 0.8322 | -0.0854 | 1.6214 | 2.5837 | 0.9623 | 0.3971 | 8.4204 |
| tcn_gaussian | gaussian | unseen_heading | pitch | 150 | 15.0000 | 0.9056 | 0.9055 | -0.0001 | 2.4822 | 3.9177 | 1.4355 | 0.6083 | 12.7660 |
| tcn_gaussian | gaussian | unseen_heading | pitch_rate | 10 | 1.0000 | 0.9753 | 0.3814 | -0.5939 | 0.0864 | 0.1160 | 0.0296 | 0.0284 | 0.5363 |
| tcn_gaussian | gaussian | unseen_heading | pitch_rate | 20 | 2.0000 | 0.9686 | 0.3782 | -0.5904 | 0.1640 | 0.2608 | 0.0968 | 0.0540 | 1.2055 |
| tcn_gaussian | gaussian | unseen_heading | pitch_rate | 30 | 3.0000 | 0.9562 | 0.4600 | -0.4962 | 0.2484 | 0.3688 | 0.1204 | 0.0817 | 1.7048 |
| tcn_gaussian | gaussian | unseen_heading | pitch_rate | 50 | 5.0000 | 0.9328 | 0.6154 | -0.3173 | 0.5625 | 0.7784 | 0.2158 | 0.1851 | 3.5991 |
| tcn_gaussian | gaussian | unseen_heading | pitch_rate | 100 | 10.0000 | 0.9140 | 0.8998 | -0.0142 | 1.4131 | 2.0226 | 0.6095 | 0.4650 | 9.3545 |
| tcn_gaussian | gaussian | unseen_heading | pitch_rate | 150 | 15.0000 | 0.9014 | 0.9259 | 0.0245 | 2.0121 | 2.8964 | 0.8842 | 0.6624 | 13.3947 |
| tcn_gaussian | gaussian | unseen_heading | roll | 10 | 1.0000 | 0.9953 | 0.1292 | -0.8661 | 0.0567 | 0.1072 | 0.0505 | 0.0047 | 0.0064 |
| tcn_gaussian | gaussian | unseen_heading | roll | 20 | 2.0000 | 0.9849 | 0.1062 | -0.8787 | 0.1011 | 0.2062 | 0.1051 | 0.0083 | 0.0124 |
| tcn_gaussian | gaussian | unseen_heading | roll | 30 | 3.0000 | 0.9743 | 0.1138 | -0.8605 | 0.1747 | 0.3640 | 0.1894 | 0.0144 | 0.0219 |
| tcn_gaussian | gaussian | unseen_heading | roll | 50 | 5.0000 | 0.9648 | 0.1755 | -0.7892 | 0.2981 | 0.6963 | 0.3982 | 0.0246 | 0.0418 |
| tcn_gaussian | gaussian | unseen_heading | roll | 100 | 10.0000 | 0.9326 | 0.4346 | -0.4981 | 1.0728 | 2.1475 | 1.0747 | 0.0884 | 0.1290 |
| tcn_gaussian | gaussian | unseen_heading | roll | 150 | 15.0000 | 0.9165 | 0.5516 | -0.3649 | 2.2330 | 4.2639 | 2.0309 | 0.1838 | 0.2559 |
| tcn_gaussian | gaussian | unseen_heading | roll_rate | 10 | 1.0000 | 0.9819 | 0.1043 | -0.8776 | 0.0506 | 0.1033 | 0.0527 | 0.0078 | 0.0115 |
| tcn_gaussian | gaussian | unseen_heading | roll_rate | 20 | 2.0000 | 0.9738 | 0.1164 | -0.8575 | 0.0951 | 0.1945 | 0.0994 | 0.0147 | 0.0216 |
| tcn_gaussian | gaussian | unseen_heading | roll_rate | 30 | 3.0000 | 0.9656 | 0.1403 | -0.8252 | 0.1208 | 0.2722 | 0.1514 | 0.0186 | 0.0303 |
| tcn_gaussian | gaussian | unseen_heading | roll_rate | 50 | 5.0000 | 0.9495 | 0.2374 | -0.7121 | 0.2672 | 0.5080 | 0.2408 | 0.0412 | 0.0565 |
| tcn_gaussian | gaussian | unseen_heading | roll_rate | 100 | 10.0000 | 0.9236 | 0.4643 | -0.4593 | 0.8382 | 1.4260 | 0.5878 | 0.1292 | 0.1584 |
| tcn_gaussian | gaussian | unseen_heading | roll_rate | 150 | 15.0000 | 0.9122 | 0.5744 | -0.3378 | 1.5021 | 2.5699 | 1.0677 | 0.2313 | 0.2854 |
| tcn_quantile | quantile | unseen_heading | heave | 10 | 1.0000 | 0.9929 | 0.4379 | -0.5550 | 0.0298 | 0.0443 | 0.0145 | 0.0119 | 0.0173 |
| tcn_quantile | quantile | unseen_heading | heave | 20 | 2.0000 | 0.9920 | 0.2477 | -0.7443 | 0.0602 | 0.0807 | 0.0206 | 0.0241 | 0.0315 |
| tcn_quantile | quantile | unseen_heading | heave | 30 | 3.0000 | 0.9860 | 0.1817 | -0.8044 | 0.1038 | 0.1324 | 0.0286 | 0.0416 | 0.0516 |
| tcn_quantile | quantile | unseen_heading | heave | 50 | 5.0000 | 0.9580 | 0.2060 | -0.7520 | 0.2402 | 0.2974 | 0.0573 | 0.0961 | 0.1159 |
| tcn_quantile | quantile | unseen_heading | heave | 100 | 10.0000 | 0.9215 | 0.2661 | -0.6554 | 0.6379 | 0.6580 | 0.0201 | 0.2554 | 0.2565 |
| tcn_quantile | quantile | unseen_heading | heave | 150 | 15.0000 | 0.9039 | 0.4250 | -0.4789 | 1.1279 | 1.2034 | 0.0755 | 0.4519 | 0.4691 |
| tcn_quantile | quantile | unseen_heading | heave_rate | 10 | 1.0000 | 0.9866 | 0.2409 | -0.7457 | 0.0309 | 0.0380 | 0.0071 | 0.0209 | 0.0257 |
| tcn_quantile | quantile | unseen_heading | heave_rate | 20 | 2.0000 | 0.9847 | 0.1792 | -0.8055 | 0.0528 | 0.0653 | 0.0124 | 0.0357 | 0.0442 |
| tcn_quantile | quantile | unseen_heading | heave_rate | 30 | 3.0000 | 0.9684 | 0.1860 | -0.7824 | 0.0822 | 0.0967 | 0.0145 | 0.0556 | 0.0655 |
| tcn_quantile | quantile | unseen_heading | heave_rate | 50 | 5.0000 | 0.9524 | 0.2556 | -0.6968 | 0.1337 | 0.1461 | 0.0124 | 0.0904 | 0.0989 |
| tcn_quantile | quantile | unseen_heading | heave_rate | 100 | 10.0000 | 0.9169 | 0.3329 | -0.5840 | 0.4322 | 0.3939 | -0.0383 | 0.2923 | 0.2666 |
| tcn_quantile | quantile | unseen_heading | heave_rate | 150 | 15.0000 | 0.9003 | 0.4464 | -0.4540 | 0.7554 | 0.6788 | -0.0766 | 0.5113 | 0.4594 |
| tcn_quantile | quantile | unseen_heading | pitch | 10 | 1.0000 | 0.9887 | 0.4816 | -0.5071 | 0.0719 | 0.0920 | 0.0201 | 0.0176 | 0.2998 |
| tcn_quantile | quantile | unseen_heading | pitch | 20 | 2.0000 | 0.9834 | 0.2902 | -0.6931 | 0.1722 | 0.1937 | 0.0216 | 0.0422 | 0.6311 |
| tcn_quantile | quantile | unseen_heading | pitch | 30 | 3.0000 | 0.9784 | 0.2386 | -0.7399 | 0.2743 | 0.3198 | 0.0455 | 0.0672 | 1.0418 |
| tcn_quantile | quantile | unseen_heading | pitch | 50 | 5.0000 | 0.9612 | 0.2926 | -0.6687 | 0.5274 | 0.5579 | 0.0304 | 0.1291 | 1.8176 |
| tcn_quantile | quantile | unseen_heading | pitch | 100 | 10.0000 | 0.9166 | 0.5003 | -0.4163 | 1.4746 | 1.3315 | -0.1431 | 0.3611 | 4.3394 |
| tcn_quantile | quantile | unseen_heading | pitch | 150 | 15.0000 | 0.9047 | 0.6467 | -0.2581 | 2.2714 | 2.1121 | -0.1593 | 0.5566 | 6.8823 |
| tcn_quantile | quantile | unseen_heading | pitch_rate | 10 | 1.0000 | 0.9850 | 0.2897 | -0.6954 | 0.0932 | 0.0994 | 0.0061 | 0.0307 | 0.4592 |
| tcn_quantile | quantile | unseen_heading | pitch_rate | 20 | 2.0000 | 0.9715 | 0.2924 | -0.6790 | 0.1727 | 0.1881 | 0.0154 | 0.0568 | 0.8696 |
| tcn_quantile | quantile | unseen_heading | pitch_rate | 30 | 3.0000 | 0.9706 | 0.3271 | -0.6435 | 0.2444 | 0.2463 | 0.0019 | 0.0804 | 1.1385 |
| tcn_quantile | quantile | unseen_heading | pitch_rate | 50 | 5.0000 | 0.9459 | 0.4639 | -0.4820 | 0.5215 | 0.4373 | -0.0841 | 0.1716 | 2.0221 |
| tcn_quantile | quantile | unseen_heading | pitch_rate | 100 | 10.0000 | 0.9145 | 0.6658 | -0.2487 | 1.3029 | 1.0308 | -0.2721 | 0.4288 | 4.7675 |
| tcn_quantile | quantile | unseen_heading | pitch_rate | 150 | 15.0000 | 0.9010 | 0.7382 | -0.1628 | 1.8681 | 1.5075 | -0.3606 | 0.6150 | 6.9717 |
| tcn_quantile | quantile | unseen_heading | roll | 10 | 1.0000 | 0.9916 | 0.1869 | -0.8046 | 0.1112 | 0.1533 | 0.0421 | 0.0092 | 0.0092 |
| tcn_quantile | quantile | unseen_heading | roll | 20 | 2.0000 | 0.9893 | 0.0946 | -0.8947 | 0.1963 | 0.2175 | 0.0212 | 0.0162 | 0.0131 |
| tcn_quantile | quantile | unseen_heading | roll | 30 | 3.0000 | 0.9836 | 0.0812 | -0.9024 | 0.2938 | 0.3233 | 0.0295 | 0.0242 | 0.0194 |
| tcn_quantile | quantile | unseen_heading | roll | 50 | 5.0000 | 0.9751 | 0.0831 | -0.8921 | 0.4093 | 0.4585 | 0.0492 | 0.0338 | 0.0276 |
| tcn_quantile | quantile | unseen_heading | roll | 100 | 10.0000 | 0.9368 | 0.1498 | -0.7870 | 1.0699 | 1.0303 | -0.0396 | 0.0881 | 0.0619 |
| tcn_quantile | quantile | unseen_heading | roll | 150 | 15.0000 | 0.9222 | 0.1846 | -0.7376 | 2.1321 | 2.1074 | -0.0247 | 0.1755 | 0.1265 |
| tcn_quantile | quantile | unseen_heading | roll_rate | 10 | 1.0000 | 0.9805 | 0.0961 | -0.8844 | 0.1087 | 0.1084 | -0.0003 | 0.0168 | 0.0121 |
| tcn_quantile | quantile | unseen_heading | roll_rate | 20 | 2.0000 | 0.9744 | 0.0908 | -0.8837 | 0.1638 | 0.1761 | 0.0122 | 0.0253 | 0.0196 |
| tcn_quantile | quantile | unseen_heading | roll_rate | 30 | 3.0000 | 0.9734 | 0.0911 | -0.8823 | 0.1920 | 0.2092 | 0.0172 | 0.0296 | 0.0233 |
| tcn_quantile | quantile | unseen_heading | roll_rate | 50 | 5.0000 | 0.9594 | 0.1057 | -0.8537 | 0.3244 | 0.3100 | -0.0144 | 0.0500 | 0.0345 |
| tcn_quantile | quantile | unseen_heading | roll_rate | 100 | 10.0000 | 0.9250 | 0.1529 | -0.7721 | 0.8296 | 0.6494 | -0.1803 | 0.1278 | 0.0721 |
| tcn_quantile | quantile | unseen_heading | roll_rate | 150 | 15.0000 | 0.9146 | 0.2253 | -0.6894 | 1.4414 | 1.2814 | -0.1601 | 0.2220 | 0.1423 |
| dlinear_gaussian | gaussian | unseen_seastate | heave | 10 | 1.0000 | 0.9624 | 0.4133 | -0.5491 | 0.0726 | 0.0420 | -0.0306 | 0.0291 | 0.0098 |
| dlinear_gaussian | gaussian | unseen_seastate | heave | 20 | 2.0000 | 0.9444 | 0.3384 | -0.6060 | 0.3962 | 0.2191 | -0.1771 | 0.1586 | 0.0510 |
| dlinear_gaussian | gaussian | unseen_seastate | heave | 30 | 3.0000 | 0.9224 | 0.2842 | -0.6382 | 0.8292 | 0.4173 | -0.4119 | 0.3320 | 0.0972 |
| dlinear_gaussian | gaussian | unseen_seastate | heave | 50 | 5.0000 | 0.9196 | 0.3831 | -0.5365 | 0.9751 | 0.5198 | -0.4553 | 0.3904 | 0.1211 |
| dlinear_gaussian | gaussian | unseen_seastate | heave | 100 | 10.0000 | 0.9072 | 0.4315 | -0.4757 | 1.8778 | 1.1248 | -0.7530 | 0.7519 | 0.2620 |
| dlinear_gaussian | gaussian | unseen_seastate | heave | 150 | 15.0000 | 0.8856 | 0.3825 | -0.5031 | 2.2599 | 1.3241 | -0.9358 | 0.9054 | 0.3085 |
| dlinear_gaussian | gaussian | unseen_seastate | heave_rate | 10 | 1.0000 | 0.9701 | 0.5542 | -0.4159 | 0.0431 | 0.0283 | -0.0148 | 0.0291 | 0.0117 |
| dlinear_gaussian | gaussian | unseen_seastate | heave_rate | 20 | 2.0000 | 0.9551 | 0.4633 | -0.4918 | 0.2351 | 0.1475 | -0.0875 | 0.1589 | 0.0611 |
| dlinear_gaussian | gaussian | unseen_seastate | heave_rate | 30 | 3.0000 | 0.9344 | 0.3966 | -0.5378 | 0.4919 | 0.2810 | -0.2110 | 0.3326 | 0.1163 |
| dlinear_gaussian | gaussian | unseen_seastate | heave_rate | 50 | 5.0000 | 0.9274 | 0.5093 | -0.4180 | 0.5785 | 0.3500 | -0.2285 | 0.3911 | 0.1449 |
| dlinear_gaussian | gaussian | unseen_seastate | heave_rate | 100 | 10.0000 | 0.9007 | 0.5335 | -0.3672 | 1.1141 | 0.7574 | -0.3567 | 0.7536 | 0.3136 |
| dlinear_gaussian | gaussian | unseen_seastate | heave_rate | 150 | 15.0000 | 0.8826 | 0.4763 | -0.4064 | 1.3408 | 0.8915 | -0.4492 | 0.9074 | 0.3692 |
| dlinear_gaussian | gaussian | unseen_seastate | pitch | 10 | 1.0000 | 0.9229 | 0.7212 | -0.2017 | 0.1196 | 0.0941 | -0.0255 | 0.0293 | 0.0159 |
| dlinear_gaussian | gaussian | unseen_seastate | pitch | 20 | 2.0000 | 0.9047 | 0.6661 | -0.2386 | 0.6524 | 0.4912 | -0.1612 | 0.1597 | 0.0830 |
| dlinear_gaussian | gaussian | unseen_seastate | pitch | 30 | 3.0000 | 0.8947 | 0.6278 | -0.2669 | 1.3653 | 0.9355 | -0.4298 | 0.3343 | 0.1580 |
| dlinear_gaussian | gaussian | unseen_seastate | pitch | 50 | 5.0000 | 0.8867 | 0.6686 | -0.2181 | 1.6056 | 1.1653 | -0.4403 | 0.3931 | 0.1969 |
| dlinear_gaussian | gaussian | unseen_seastate | pitch | 100 | 10.0000 | 0.8647 | 0.6630 | -0.2017 | 3.0921 | 2.5216 | -0.5705 | 0.7573 | 0.4261 |
| dlinear_gaussian | gaussian | unseen_seastate | pitch | 150 | 15.0000 | 0.8753 | 0.6493 | -0.2260 | 3.7212 | 2.9683 | -0.7529 | 0.9119 | 0.5015 |
| dlinear_gaussian | gaussian | unseen_seastate | pitch_rate | 10 | 1.0000 | 0.8636 | 0.7421 | -0.1215 | 0.0891 | 0.0747 | -0.0145 | 0.0293 | 0.0181 |
| dlinear_gaussian | gaussian | unseen_seastate | pitch_rate | 20 | 2.0000 | 0.8470 | 0.6973 | -0.1497 | 0.4864 | 0.3897 | -0.0966 | 0.1601 | 0.0943 |
| dlinear_gaussian | gaussian | unseen_seastate | pitch_rate | 30 | 3.0000 | 0.8454 | 0.6677 | -0.1777 | 1.0179 | 0.7423 | -0.2757 | 0.3349 | 0.1796 |
| dlinear_gaussian | gaussian | unseen_seastate | pitch_rate | 50 | 5.0000 | 0.8404 | 0.6870 | -0.1535 | 1.1971 | 0.9246 | -0.2724 | 0.3939 | 0.2238 |
| dlinear_gaussian | gaussian | unseen_seastate | pitch_rate | 100 | 10.0000 | 0.8385 | 0.6886 | -0.1499 | 2.3053 | 2.0007 | -0.3046 | 0.7586 | 0.4843 |
| dlinear_gaussian | gaussian | unseen_seastate | pitch_rate | 150 | 15.0000 | 0.8729 | 0.6998 | -0.1730 | 2.7744 | 2.3552 | -0.4192 | 0.9133 | 0.5698 |
| dlinear_gaussian | gaussian | unseen_seastate | roll | 10 | 1.0000 | 0.9917 | 0.7325 | -0.2592 | 0.3499 | 0.2053 | -0.1447 | 0.0289 | 0.0100 |
| dlinear_gaussian | gaussian | unseen_seastate | roll | 20 | 2.0000 | 0.9828 | 0.6579 | -0.3249 | 1.9091 | 1.0714 | -0.8376 | 0.1575 | 0.0520 |
| dlinear_gaussian | gaussian | unseen_seastate | roll | 30 | 3.0000 | 0.9706 | 0.6092 | -0.3614 | 3.9955 | 2.0406 | -1.9549 | 0.3296 | 0.0991 |
| dlinear_gaussian | gaussian | unseen_seastate | roll | 50 | 5.0000 | 0.9707 | 0.7141 | -0.2566 | 4.6986 | 2.5419 | -2.1567 | 0.3875 | 0.1234 |
| dlinear_gaussian | gaussian | unseen_seastate | roll | 100 | 10.0000 | 0.9624 | 0.7088 | -0.2537 | 9.0486 | 5.5003 | -3.5483 | 0.7454 | 0.2669 |
| dlinear_gaussian | gaussian | unseen_seastate | roll | 150 | 15.0000 | 0.9410 | 0.6533 | -0.2877 | 10.8897 | 6.4747 | -4.4150 | 0.8962 | 0.3140 |
| dlinear_gaussian | gaussian | unseen_seastate | roll_rate | 10 | 1.0000 | 0.9937 | 0.7723 | -0.2214 | 0.1872 | 0.1141 | -0.0731 | 0.0289 | 0.0105 |
| dlinear_gaussian | gaussian | unseen_seastate | roll_rate | 20 | 2.0000 | 0.9866 | 0.6918 | -0.2948 | 1.0214 | 0.5955 | -0.4258 | 0.1576 | 0.0549 |
| dlinear_gaussian | gaussian | unseen_seastate | roll_rate | 30 | 3.0000 | 0.9743 | 0.6418 | -0.3325 | 2.1376 | 1.1342 | -1.0034 | 0.3298 | 0.1045 |
| dlinear_gaussian | gaussian | unseen_seastate | roll_rate | 50 | 5.0000 | 0.9738 | 0.7464 | -0.2275 | 2.5138 | 1.4129 | -1.1009 | 0.3877 | 0.1301 |
| dlinear_gaussian | gaussian | unseen_seastate | roll_rate | 100 | 10.0000 | 0.9614 | 0.7257 | -0.2357 | 4.8410 | 3.0572 | -1.7838 | 0.7460 | 0.2814 |
| dlinear_gaussian | gaussian | unseen_seastate | roll_rate | 150 | 15.0000 | 0.9387 | 0.6683 | -0.2704 | 5.8260 | 3.5988 | -2.2272 | 0.8971 | 0.3311 |
| dlinear_quantile | quantile | unseen_seastate | heave | 10 | 1.0000 | 0.9335 | 0.3369 | -0.5966 | 0.0471 | 0.0290 | -0.0181 | 0.0189 | 0.0068 |
| dlinear_quantile | quantile | unseen_seastate | heave | 20 | 2.0000 | 0.9149 | 0.2886 | -0.6263 | 0.2976 | 0.1721 | -0.1254 | 0.1191 | 0.0401 |
| dlinear_quantile | quantile | unseen_seastate | heave | 30 | 3.0000 | 0.9016 | 0.2507 | -0.6509 | 0.7101 | 0.3683 | -0.3418 | 0.2843 | 0.0858 |
| dlinear_quantile | quantile | unseen_seastate | heave | 50 | 5.0000 | 0.9009 | 0.3307 | -0.5702 | 0.8686 | 0.4846 | -0.3840 | 0.3477 | 0.1129 |
| dlinear_quantile | quantile | unseen_seastate | heave | 100 | 10.0000 | 0.9014 | 0.4196 | -0.4817 | 1.8287 | 1.1065 | -0.7221 | 0.7322 | 0.2578 |
| dlinear_quantile | quantile | unseen_seastate | heave | 150 | 15.0000 | 0.8861 | 0.3829 | -0.5032 | 2.2701 | 1.3226 | -0.9475 | 0.9095 | 0.3081 |
| dlinear_quantile | quantile | unseen_seastate | heave_rate | 10 | 1.0000 | 0.9343 | 0.4537 | -0.4806 | 0.0280 | 0.0194 | -0.0086 | 0.0189 | 0.0080 |
| dlinear_quantile | quantile | unseen_seastate | heave_rate | 20 | 2.0000 | 0.9167 | 0.3956 | -0.5210 | 0.1766 | 0.1152 | -0.0614 | 0.1194 | 0.0477 |
| dlinear_quantile | quantile | unseen_seastate | heave_rate | 30 | 3.0000 | 0.9076 | 0.3497 | -0.5579 | 0.4214 | 0.2437 | -0.1776 | 0.2848 | 0.1009 |
| dlinear_quantile | quantile | unseen_seastate | heave_rate | 50 | 5.0000 | 0.9043 | 0.4347 | -0.4696 | 0.5141 | 0.2976 | -0.2164 | 0.3476 | 0.1232 |
| dlinear_quantile | quantile | unseen_seastate | heave_rate | 100 | 10.0000 | 0.8934 | 0.5155 | -0.3778 | 1.0846 | 0.7279 | -0.3566 | 0.7336 | 0.3014 |
| dlinear_quantile | quantile | unseen_seastate | heave_rate | 150 | 15.0000 | 0.8828 | 0.4765 | -0.4063 | 1.3468 | 0.8905 | -0.4563 | 0.9115 | 0.3687 |
| dlinear_quantile | quantile | unseen_seastate | pitch | 10 | 1.0000 | 0.8645 | 0.6346 | -0.2299 | 0.0795 | 0.0668 | -0.0128 | 0.0195 | 0.0113 |
| dlinear_quantile | quantile | unseen_seastate | pitch | 20 | 2.0000 | 0.8508 | 0.6035 | -0.2473 | 0.4949 | 0.3893 | -0.1056 | 0.1212 | 0.0658 |
| dlinear_quantile | quantile | unseen_seastate | pitch | 30 | 3.0000 | 0.8603 | 0.5859 | -0.2744 | 1.1724 | 0.8130 | -0.3595 | 0.2871 | 0.1373 |
| dlinear_quantile | quantile | unseen_seastate | pitch | 50 | 5.0000 | 0.8603 | 0.6084 | -0.2519 | 1.4267 | 0.9681 | -0.4586 | 0.3493 | 0.1636 |
| dlinear_quantile | quantile | unseen_seastate | pitch | 100 | 10.0000 | 0.8576 | 0.6483 | -0.2092 | 3.0101 | 2.4166 | -0.5935 | 0.7372 | 0.4084 |
| dlinear_quantile | quantile | unseen_seastate | pitch | 150 | 15.0000 | 0.8757 | 0.6490 | -0.2268 | 3.7380 | 2.9650 | -0.7730 | 0.9160 | 0.5010 |
| dlinear_quantile | quantile | unseen_seastate | pitch_rate | 10 | 1.0000 | 0.8065 | 0.6636 | -0.1428 | 0.0655 | 0.0600 | -0.0054 | 0.0215 | 0.0145 |
| dlinear_quantile | quantile | unseen_seastate | pitch_rate | 20 | 2.0000 | 0.7931 | 0.6354 | -0.1577 | 0.3875 | 0.3299 | -0.0576 | 0.1275 | 0.0798 |
| dlinear_quantile | quantile | unseen_seastate | pitch_rate | 30 | 3.0000 | 0.8094 | 0.6248 | -0.1846 | 0.8914 | 0.6628 | -0.2286 | 0.2933 | 0.1604 |
| dlinear_quantile | quantile | unseen_seastate | pitch_rate | 50 | 5.0000 | 0.8125 | 0.6280 | -0.1845 | 1.0659 | 0.7743 | -0.2916 | 0.3507 | 0.1874 |
| dlinear_quantile | quantile | unseen_seastate | pitch_rate | 100 | 10.0000 | 0.8315 | 0.6745 | -0.1570 | 2.2445 | 1.9224 | -0.3220 | 0.7386 | 0.4653 |
| dlinear_quantile | quantile | unseen_seastate | pitch_rate | 150 | 15.0000 | 0.8733 | 0.6991 | -0.1742 | 2.7868 | 2.3525 | -0.4343 | 0.9174 | 0.5691 |
| dlinear_quantile | quantile | unseen_seastate | roll | 10 | 1.0000 | 0.9727 | 0.6759 | -0.2967 | 0.2272 | 0.1409 | -0.0863 | 0.0188 | 0.0068 |
| dlinear_quantile | quantile | unseen_seastate | roll | 20 | 2.0000 | 0.9668 | 0.6160 | -0.3508 | 1.4342 | 0.8432 | -0.5910 | 0.1184 | 0.0410 |
| dlinear_quantile | quantile | unseen_seastate | roll | 30 | 3.0000 | 0.9609 | 0.5817 | -0.3792 | 3.4218 | 1.8018 | -1.6200 | 0.2823 | 0.0875 |
| dlinear_quantile | quantile | unseen_seastate | roll | 50 | 5.0000 | 0.9624 | 0.6648 | -0.2976 | 4.1734 | 2.1458 | -2.0276 | 0.3442 | 0.1042 |
| dlinear_quantile | quantile | unseen_seastate | roll | 100 | 10.0000 | 0.9596 | 0.6932 | -0.2665 | 8.8086 | 5.2747 | -3.5339 | 0.7256 | 0.2560 |
| dlinear_quantile | quantile | unseen_seastate | roll | 150 | 15.0000 | 0.9422 | 0.6516 | -0.2906 | 10.9387 | 6.4674 | -4.4713 | 0.9002 | 0.3137 |
| dlinear_quantile | quantile | unseen_seastate | roll_rate | 10 | 1.0000 | 0.9739 | 0.7046 | -0.2693 | 0.1215 | 0.0781 | -0.0435 | 0.0188 | 0.0072 |
| dlinear_quantile | quantile | unseen_seastate | roll_rate | 20 | 2.0000 | 0.9687 | 0.6428 | -0.3259 | 0.7673 | 0.4673 | -0.3000 | 0.1184 | 0.0431 |
| dlinear_quantile | quantile | unseen_seastate | roll_rate | 30 | 3.0000 | 0.9635 | 0.6106 | -0.3530 | 1.8307 | 0.9947 | -0.8360 | 0.2824 | 0.0916 |
| dlinear_quantile | quantile | unseen_seastate | roll_rate | 50 | 5.0000 | 0.9643 | 0.6878 | -0.2765 | 2.2324 | 1.1742 | -1.0583 | 0.3443 | 0.1081 |
| dlinear_quantile | quantile | unseen_seastate | roll_rate | 100 | 10.0000 | 0.9577 | 0.7082 | -0.2494 | 4.7126 | 2.9263 | -1.7863 | 0.7262 | 0.2694 |
| dlinear_quantile | quantile | unseen_seastate | roll_rate | 150 | 15.0000 | 0.9399 | 0.6663 | -0.2735 | 5.8522 | 3.5947 | -2.2575 | 0.9011 | 0.3308 |
| lstm_gaussian | gaussian | unseen_seastate | heave | 10 | 1.0000 | 0.9935 | 0.6199 | -0.3736 | 0.0301 | 0.2870 | 0.2568 | 0.0121 | 0.0668 |
| lstm_gaussian | gaussian | unseen_seastate | heave | 20 | 2.0000 | 0.9909 | 0.5841 | -0.4068 | 0.0342 | 0.2872 | 0.2530 | 0.0137 | 0.0669 |
| lstm_gaussian | gaussian | unseen_seastate | heave | 30 | 3.0000 | 0.9769 | 0.4941 | -0.4828 | 0.0490 | 0.2589 | 0.2100 | 0.0196 | 0.0603 |
| lstm_gaussian | gaussian | unseen_seastate | heave | 50 | 5.0000 | 0.9398 | 0.3641 | -0.5757 | 0.1511 | 0.3264 | 0.1752 | 0.0605 | 0.0760 |
| lstm_gaussian | gaussian | unseen_seastate | heave | 100 | 10.0000 | 0.9210 | 0.3728 | -0.5482 | 0.5194 | 0.8792 | 0.3598 | 0.2080 | 0.2048 |
| lstm_gaussian | gaussian | unseen_seastate | heave | 150 | 15.0000 | 0.9098 | 0.3956 | -0.5142 | 1.0591 | 1.5202 | 0.4611 | 0.4243 | 0.3542 |
| lstm_gaussian | gaussian | unseen_seastate | heave_rate | 10 | 1.0000 | 0.9895 | 0.6275 | -0.3620 | 0.0215 | 0.2183 | 0.1968 | 0.0145 | 0.0904 |
| lstm_gaussian | gaussian | unseen_seastate | heave_rate | 20 | 2.0000 | 0.9824 | 0.5759 | -0.4065 | 0.0261 | 0.1818 | 0.1557 | 0.0177 | 0.0753 |
| lstm_gaussian | gaussian | unseen_seastate | heave_rate | 30 | 3.0000 | 0.9638 | 0.5060 | -0.4578 | 0.0435 | 0.1621 | 0.1186 | 0.0294 | 0.0671 |
| lstm_gaussian | gaussian | unseen_seastate | heave_rate | 50 | 5.0000 | 0.9350 | 0.4394 | -0.4956 | 0.0940 | 0.2324 | 0.1383 | 0.0636 | 0.0962 |
| lstm_gaussian | gaussian | unseen_seastate | heave_rate | 100 | 10.0000 | 0.9177 | 0.4353 | -0.4824 | 0.3794 | 0.6377 | 0.2584 | 0.2566 | 0.2641 |
| lstm_gaussian | gaussian | unseen_seastate | heave_rate | 150 | 15.0000 | 0.9089 | 0.4662 | -0.4427 | 0.7241 | 1.0983 | 0.3741 | 0.4901 | 0.4548 |
| lstm_gaussian | gaussian | unseen_seastate | pitch | 10 | 1.0000 | 0.9896 | 0.6352 | -0.3544 | 0.0595 | 0.5796 | 0.5201 | 0.0146 | 0.0979 |
| lstm_gaussian | gaussian | unseen_seastate | pitch | 20 | 2.0000 | 0.9781 | 0.5721 | -0.4060 | 0.0907 | 0.5824 | 0.4916 | 0.0222 | 0.0984 |
| lstm_gaussian | gaussian | unseen_seastate | pitch | 30 | 3.0000 | 0.9558 | 0.4789 | -0.4768 | 0.1472 | 0.6155 | 0.4683 | 0.0360 | 0.1040 |
| lstm_gaussian | gaussian | unseen_seastate | pitch | 50 | 5.0000 | 0.9375 | 0.5370 | -0.4005 | 0.3420 | 1.0555 | 0.7135 | 0.0837 | 0.1783 |
| lstm_gaussian | gaussian | unseen_seastate | pitch | 100 | 10.0000 | 0.9172 | 0.5014 | -0.4158 | 1.2751 | 2.3248 | 1.0497 | 0.3123 | 0.3929 |
| lstm_gaussian | gaussian | unseen_seastate | pitch | 150 | 15.0000 | 0.9076 | 0.5570 | -0.3506 | 2.2352 | 3.7201 | 1.4849 | 0.5477 | 0.6285 |
| lstm_gaussian | gaussian | unseen_seastate | pitch_rate | 10 | 1.0000 | 0.9824 | 0.6512 | -0.3312 | 0.0658 | 0.5943 | 0.5286 | 0.0216 | 0.1438 |
| lstm_gaussian | gaussian | unseen_seastate | pitch_rate | 20 | 2.0000 | 0.9647 | 0.5441 | -0.4206 | 0.0978 | 0.5116 | 0.4138 | 0.0322 | 0.1238 |
| lstm_gaussian | gaussian | unseen_seastate | pitch_rate | 30 | 3.0000 | 0.9430 | 0.5728 | -0.3702 | 0.1360 | 0.6330 | 0.4970 | 0.0447 | 0.1531 |
| lstm_gaussian | gaussian | unseen_seastate | pitch_rate | 50 | 5.0000 | 0.9383 | 0.5468 | -0.3915 | 0.3728 | 1.0655 | 0.6927 | 0.1227 | 0.2579 |
| lstm_gaussian | gaussian | unseen_seastate | pitch_rate | 100 | 10.0000 | 0.9152 | 0.5357 | -0.3795 | 1.2078 | 2.2741 | 1.0663 | 0.3975 | 0.5504 |
| lstm_gaussian | gaussian | unseen_seastate | pitch_rate | 150 | 15.0000 | 0.9066 | 0.6060 | -0.3006 | 1.8484 | 3.0890 | 1.2406 | 0.6085 | 0.7473 |
| lstm_gaussian | gaussian | unseen_seastate | roll | 10 | 1.0000 | 0.9918 | 0.5704 | -0.4214 | 0.1339 | 1.8450 | 1.7111 | 0.0111 | 0.0896 |
| lstm_gaussian | gaussian | unseen_seastate | roll | 20 | 2.0000 | 0.9906 | 0.5537 | -0.4369 | 0.1366 | 1.7839 | 1.6473 | 0.0113 | 0.0867 |
| lstm_gaussian | gaussian | unseen_seastate | roll | 30 | 3.0000 | 0.9822 | 0.5072 | -0.4749 | 0.1551 | 1.5219 | 1.3668 | 0.0128 | 0.0739 |
| lstm_gaussian | gaussian | unseen_seastate | roll | 50 | 5.0000 | 0.9532 | 0.4585 | -0.4947 | 0.2176 | 1.0855 | 0.8679 | 0.0179 | 0.0527 |
| lstm_gaussian | gaussian | unseen_seastate | roll | 100 | 10.0000 | 0.9242 | 0.4413 | -0.4829 | 0.8294 | 1.7414 | 0.9120 | 0.0683 | 0.0845 |
| lstm_gaussian | gaussian | unseen_seastate | roll | 150 | 15.0000 | 0.9135 | 0.4331 | -0.4804 | 1.8962 | 3.0657 | 1.1695 | 0.1560 | 0.1487 |
| lstm_gaussian | gaussian | unseen_seastate | roll_rate | 10 | 1.0000 | 0.9926 | 0.5910 | -0.4015 | 0.0787 | 1.1382 | 1.0596 | 0.0121 | 0.1049 |
| lstm_gaussian | gaussian | unseen_seastate | roll_rate | 20 | 2.0000 | 0.9652 | 0.5330 | -0.4322 | 0.0882 | 0.9595 | 0.8713 | 0.0136 | 0.0884 |
| lstm_gaussian | gaussian | unseen_seastate | roll_rate | 30 | 3.0000 | 0.9669 | 0.5000 | -0.4669 | 0.1038 | 0.7596 | 0.6558 | 0.0160 | 0.0700 |
| lstm_gaussian | gaussian | unseen_seastate | roll_rate | 50 | 5.0000 | 0.9491 | 0.5388 | -0.4103 | 0.1675 | 0.8351 | 0.6676 | 0.0258 | 0.0769 |
| lstm_gaussian | gaussian | unseen_seastate | roll_rate | 100 | 10.0000 | 0.9172 | 0.4702 | -0.4470 | 0.6534 | 1.2936 | 0.6402 | 0.1007 | 0.1191 |
| lstm_gaussian | gaussian | unseen_seastate | roll_rate | 150 | 15.0000 | 0.9075 | 0.4867 | -0.4208 | 1.2816 | 2.1788 | 0.8972 | 0.1973 | 0.2005 |
| lstm_quantile | quantile | unseen_seastate | heave | 10 | 1.0000 | 0.9923 | 0.4232 | -0.5691 | 0.0520 | 0.1054 | 0.0534 | 0.0208 | 0.0245 |
| lstm_quantile | quantile | unseen_seastate | heave | 20 | 2.0000 | 0.9855 | 0.4011 | -0.5844 | 0.0503 | 0.1021 | 0.0518 | 0.0201 | 0.0238 |
| lstm_quantile | quantile | unseen_seastate | heave | 30 | 3.0000 | 0.9502 | 0.3669 | -0.5833 | 0.0575 | 0.1112 | 0.0537 | 0.0230 | 0.0259 |
| lstm_quantile | quantile | unseen_seastate | heave | 50 | 5.0000 | 0.9303 | 0.2532 | -0.6771 | 0.1314 | 0.1821 | 0.0507 | 0.0526 | 0.0424 |
| lstm_quantile | quantile | unseen_seastate | heave | 100 | 10.0000 | 0.9172 | 0.2731 | -0.6441 | 0.4602 | 0.5506 | 0.0904 | 0.1843 | 0.1283 |
| lstm_quantile | quantile | unseen_seastate | heave | 150 | 15.0000 | 0.9083 | 0.3452 | -0.5631 | 0.9621 | 1.2152 | 0.2530 | 0.3855 | 0.2831 |
| lstm_quantile | quantile | unseen_seastate | heave_rate | 10 | 1.0000 | 0.9863 | 0.4325 | -0.5537 | 0.0330 | 0.0701 | 0.0372 | 0.0223 | 0.0290 |
| lstm_quantile | quantile | unseen_seastate | heave_rate | 20 | 2.0000 | 0.9797 | 0.4122 | -0.5675 | 0.0337 | 0.0718 | 0.0381 | 0.0228 | 0.0297 |
| lstm_quantile | quantile | unseen_seastate | heave_rate | 30 | 3.0000 | 0.9624 | 0.3514 | -0.6110 | 0.0453 | 0.0796 | 0.0343 | 0.0306 | 0.0329 |
| lstm_quantile | quantile | unseen_seastate | heave_rate | 50 | 5.0000 | 0.9346 | 0.2564 | -0.6782 | 0.0830 | 0.1071 | 0.0241 | 0.0561 | 0.0443 |
| lstm_quantile | quantile | unseen_seastate | heave_rate | 100 | 10.0000 | 0.9203 | 0.3154 | -0.6049 | 0.3434 | 0.4154 | 0.0720 | 0.2323 | 0.1720 |
| lstm_quantile | quantile | unseen_seastate | heave_rate | 150 | 15.0000 | 0.9069 | 0.3543 | -0.5526 | 0.6631 | 0.8075 | 0.1444 | 0.4488 | 0.3344 |
| lstm_quantile | quantile | unseen_seastate | pitch | 10 | 1.0000 | 0.9884 | 0.4828 | -0.5056 | 0.0855 | 0.2258 | 0.1403 | 0.0209 | 0.0381 |
| lstm_quantile | quantile | unseen_seastate | pitch | 20 | 2.0000 | 0.9814 | 0.4122 | -0.5692 | 0.0989 | 0.2305 | 0.1316 | 0.0242 | 0.0389 |
| lstm_quantile | quantile | unseen_seastate | pitch | 30 | 3.0000 | 0.9645 | 0.3462 | -0.6183 | 0.1414 | 0.2642 | 0.1228 | 0.0346 | 0.0446 |
| lstm_quantile | quantile | unseen_seastate | pitch | 50 | 5.0000 | 0.9419 | 0.3514 | -0.5905 | 0.2967 | 0.4658 | 0.1691 | 0.0726 | 0.0787 |
| lstm_quantile | quantile | unseen_seastate | pitch | 100 | 10.0000 | 0.9120 | 0.3985 | -0.5135 | 1.1582 | 1.5654 | 0.4072 | 0.2837 | 0.2646 |
| lstm_quantile | quantile | unseen_seastate | pitch | 150 | 15.0000 | 0.9105 | 0.4863 | -0.4242 | 2.0459 | 2.9224 | 0.8765 | 0.5013 | 0.4938 |
| lstm_quantile | quantile | unseen_seastate | pitch_rate | 10 | 1.0000 | 0.9859 | 0.4829 | -0.5030 | 0.0808 | 0.1998 | 0.1190 | 0.0266 | 0.0483 |
| lstm_quantile | quantile | unseen_seastate | pitch_rate | 20 | 2.0000 | 0.9791 | 0.4134 | -0.5656 | 0.1029 | 0.2083 | 0.1054 | 0.0339 | 0.0504 |
| lstm_quantile | quantile | unseen_seastate | pitch_rate | 30 | 3.0000 | 0.9659 | 0.3932 | -0.5727 | 0.1260 | 0.2308 | 0.1048 | 0.0415 | 0.0558 |
| lstm_quantile | quantile | unseen_seastate | pitch_rate | 50 | 5.0000 | 0.9334 | 0.3533 | -0.5802 | 0.3229 | 0.4642 | 0.1413 | 0.1063 | 0.1123 |
| lstm_quantile | quantile | unseen_seastate | pitch_rate | 100 | 10.0000 | 0.9126 | 0.4050 | -0.5076 | 1.0905 | 1.5036 | 0.4131 | 0.3589 | 0.3639 |
| lstm_quantile | quantile | unseen_seastate | pitch_rate | 150 | 15.0000 | 0.9044 | 0.4901 | -0.4143 | 1.7232 | 2.3694 | 0.6463 | 0.5673 | 0.5732 |
| lstm_quantile | quantile | unseen_seastate | roll | 10 | 1.0000 | 0.9854 | 0.4177 | -0.5678 | 0.2290 | 0.4758 | 0.2469 | 0.0189 | 0.0231 |
| lstm_quantile | quantile | unseen_seastate | roll | 20 | 2.0000 | 0.9791 | 0.4081 | -0.5710 | 0.2252 | 0.4658 | 0.2407 | 0.0186 | 0.0226 |
| lstm_quantile | quantile | unseen_seastate | roll | 30 | 3.0000 | 0.9790 | 0.3859 | -0.5930 | 0.2372 | 0.4602 | 0.2230 | 0.0196 | 0.0223 |
| lstm_quantile | quantile | unseen_seastate | roll | 50 | 5.0000 | 0.9543 | 0.3472 | -0.6072 | 0.2596 | 0.4544 | 0.1948 | 0.0214 | 0.0221 |
| lstm_quantile | quantile | unseen_seastate | roll | 100 | 10.0000 | 0.9249 | 0.3401 | -0.5848 | 0.7617 | 1.0363 | 0.2746 | 0.0627 | 0.0503 |
| lstm_quantile | quantile | unseen_seastate | roll | 150 | 15.0000 | 0.9204 | 0.3829 | -0.5375 | 1.7209 | 2.2718 | 0.5509 | 0.1416 | 0.1102 |
| lstm_quantile | quantile | unseen_seastate | roll_rate | 10 | 1.0000 | 0.9763 | 0.4229 | -0.5534 | 0.1330 | 0.2777 | 0.1446 | 0.0205 | 0.0256 |
| lstm_quantile | quantile | unseen_seastate | roll_rate | 20 | 2.0000 | 0.9787 | 0.3951 | -0.5836 | 0.1380 | 0.2714 | 0.1334 | 0.0213 | 0.0250 |
| lstm_quantile | quantile | unseen_seastate | roll_rate | 30 | 3.0000 | 0.9813 | 0.3712 | -0.6102 | 0.1439 | 0.2704 | 0.1265 | 0.0222 | 0.0249 |
| lstm_quantile | quantile | unseen_seastate | roll_rate | 50 | 5.0000 | 0.9650 | 0.3841 | -0.5810 | 0.1797 | 0.3241 | 0.1445 | 0.0277 | 0.0299 |
| lstm_quantile | quantile | unseen_seastate | roll_rate | 100 | 10.0000 | 0.9219 | 0.3673 | -0.5545 | 0.5958 | 0.8684 | 0.2726 | 0.0918 | 0.0799 |
| lstm_quantile | quantile | unseen_seastate | roll_rate | 150 | 15.0000 | 0.9155 | 0.3956 | -0.5198 | 1.1971 | 1.7443 | 0.5472 | 0.1843 | 0.1605 |
| tcn_gaussian | gaussian | unseen_seastate | heave | 10 | 1.0000 | 0.9937 | 0.8990 | -0.0947 | 0.0174 | 0.0566 | 0.0392 | 0.0070 | 0.0132 |
| tcn_gaussian | gaussian | unseen_seastate | heave | 20 | 2.0000 | 0.9926 | 0.8279 | -0.1648 | 0.0393 | 0.1315 | 0.0922 | 0.0157 | 0.0306 |
| tcn_gaussian | gaussian | unseen_seastate | heave | 30 | 3.0000 | 0.9780 | 0.6911 | -0.2869 | 0.0859 | 0.2422 | 0.1564 | 0.0344 | 0.0564 |
| tcn_gaussian | gaussian | unseen_seastate | heave | 50 | 5.0000 | 0.9517 | 0.5658 | -0.3858 | 0.2430 | 0.6066 | 0.3636 | 0.0973 | 0.1413 |
| tcn_gaussian | gaussian | unseen_seastate | heave | 100 | 10.0000 | 0.9221 | 0.5135 | -0.4086 | 0.6934 | 1.3932 | 0.6997 | 0.2777 | 0.3246 |
| tcn_gaussian | gaussian | unseen_seastate | heave | 150 | 15.0000 | 0.9068 | 0.6562 | -0.2506 | 1.2366 | 3.2426 | 2.0059 | 0.4955 | 0.7555 |
| tcn_gaussian | gaussian | unseen_seastate | heave_rate | 10 | 1.0000 | 0.9922 | 0.8325 | -0.1597 | 0.0175 | 0.0612 | 0.0436 | 0.0118 | 0.0253 |
| tcn_gaussian | gaussian | unseen_seastate | heave_rate | 20 | 2.0000 | 0.9812 | 0.7093 | -0.2719 | 0.0413 | 0.1190 | 0.0777 | 0.0279 | 0.0493 |
| tcn_gaussian | gaussian | unseen_seastate | heave_rate | 30 | 3.0000 | 0.9598 | 0.5868 | -0.3731 | 0.0753 | 0.1800 | 0.1046 | 0.0509 | 0.0745 |
| tcn_gaussian | gaussian | unseen_seastate | heave_rate | 50 | 5.0000 | 0.9426 | 0.5194 | -0.4232 | 0.1341 | 0.2523 | 0.1182 | 0.0907 | 0.1045 |
| tcn_gaussian | gaussian | unseen_seastate | heave_rate | 100 | 10.0000 | 0.9187 | 0.5374 | -0.3814 | 0.4596 | 0.9557 | 0.4962 | 0.3109 | 0.3958 |
| tcn_gaussian | gaussian | unseen_seastate | heave_rate | 150 | 15.0000 | 0.9034 | 0.6407 | -0.2627 | 0.8088 | 1.7062 | 0.8974 | 0.5474 | 0.7065 |
| tcn_gaussian | gaussian | unseen_seastate | pitch | 10 | 1.0000 | 0.9769 | 0.8029 | -0.1740 | 0.0533 | 0.1362 | 0.0829 | 0.0131 | 0.0230 |
| tcn_gaussian | gaussian | unseen_seastate | pitch | 20 | 2.0000 | 0.9717 | 0.6948 | -0.2769 | 0.1577 | 0.4249 | 0.2672 | 0.0386 | 0.0718 |
| tcn_gaussian | gaussian | unseen_seastate | pitch | 30 | 3.0000 | 0.9677 | 0.6626 | -0.3052 | 0.2634 | 0.7786 | 0.5152 | 0.0645 | 0.1315 |
| tcn_gaussian | gaussian | unseen_seastate | pitch | 50 | 5.0000 | 0.9509 | 0.6950 | -0.2559 | 0.5619 | 1.4573 | 0.8954 | 0.1376 | 0.2462 |
| tcn_gaussian | gaussian | unseen_seastate | pitch | 100 | 10.0000 | 0.9176 | 0.6439 | -0.2737 | 1.6214 | 3.3936 | 1.7722 | 0.3971 | 0.5735 |
| tcn_gaussian | gaussian | unseen_seastate | pitch | 150 | 15.0000 | 0.9056 | 0.7754 | -0.1302 | 2.4822 | 6.7681 | 4.2859 | 0.6083 | 1.1436 |
| tcn_gaussian | gaussian | unseen_seastate | pitch_rate | 10 | 1.0000 | 0.9753 | 0.7405 | -0.2348 | 0.0864 | 0.2120 | 0.1256 | 0.0284 | 0.0513 |
| tcn_gaussian | gaussian | unseen_seastate | pitch_rate | 20 | 2.0000 | 0.9686 | 0.6794 | -0.2891 | 0.1640 | 0.4482 | 0.2843 | 0.0540 | 0.1084 |
| tcn_gaussian | gaussian | unseen_seastate | pitch_rate | 30 | 3.0000 | 0.9562 | 0.6563 | -0.2999 | 0.2484 | 0.5606 | 0.3122 | 0.0817 | 0.1356 |
| tcn_gaussian | gaussian | unseen_seastate | pitch_rate | 50 | 5.0000 | 0.9328 | 0.5699 | -0.3629 | 0.5625 | 1.0407 | 0.4782 | 0.1851 | 0.2519 |
| tcn_gaussian | gaussian | unseen_seastate | pitch_rate | 100 | 10.0000 | 0.9140 | 0.6502 | -0.2638 | 1.4131 | 3.0648 | 1.6517 | 0.4650 | 0.7418 |
| tcn_gaussian | gaussian | unseen_seastate | pitch_rate | 150 | 15.0000 | 0.9014 | 0.8033 | -0.0981 | 2.0121 | 4.1850 | 2.1729 | 0.6624 | 1.0124 |
| tcn_gaussian | gaussian | unseen_seastate | roll | 10 | 1.0000 | 0.9953 | 0.8835 | -0.1119 | 0.0567 | 0.2240 | 0.1673 | 0.0047 | 0.0109 |
| tcn_gaussian | gaussian | unseen_seastate | roll | 20 | 2.0000 | 0.9849 | 0.8102 | -0.1746 | 0.1011 | 0.3590 | 0.2579 | 0.0083 | 0.0174 |
| tcn_gaussian | gaussian | unseen_seastate | roll | 30 | 3.0000 | 0.9743 | 0.7242 | -0.2501 | 0.1747 | 0.6111 | 0.4365 | 0.0144 | 0.0297 |
| tcn_gaussian | gaussian | unseen_seastate | roll | 50 | 5.0000 | 0.9648 | 0.6827 | -0.2820 | 0.2981 | 0.8346 | 0.5365 | 0.0246 | 0.0405 |
| tcn_gaussian | gaussian | unseen_seastate | roll | 100 | 10.0000 | 0.9326 | 0.6874 | -0.2453 | 1.0728 | 2.6379 | 1.5651 | 0.0884 | 0.1280 |
| tcn_gaussian | gaussian | unseen_seastate | roll | 150 | 15.0000 | 0.9165 | 0.7902 | -0.1263 | 2.2330 | 7.6719 | 5.4388 | 0.1838 | 0.3721 |
| tcn_gaussian | gaussian | unseen_seastate | roll_rate | 10 | 1.0000 | 0.9819 | 0.8243 | -0.1576 | 0.0506 | 0.1695 | 0.1189 | 0.0078 | 0.0156 |
| tcn_gaussian | gaussian | unseen_seastate | roll_rate | 20 | 2.0000 | 0.9738 | 0.7317 | -0.2422 | 0.0951 | 0.2955 | 0.2004 | 0.0147 | 0.0272 |
| tcn_gaussian | gaussian | unseen_seastate | roll_rate | 30 | 3.0000 | 0.9656 | 0.6766 | -0.2889 | 0.1208 | 0.3504 | 0.2296 | 0.0186 | 0.0323 |
| tcn_gaussian | gaussian | unseen_seastate | roll_rate | 50 | 5.0000 | 0.9495 | 0.6764 | -0.2731 | 0.2672 | 0.6356 | 0.3685 | 0.0412 | 0.0585 |
| tcn_gaussian | gaussian | unseen_seastate | roll_rate | 100 | 10.0000 | 0.9236 | 0.6074 | -0.3162 | 0.8382 | 1.8925 | 1.0543 | 0.1292 | 0.1742 |
| tcn_gaussian | gaussian | unseen_seastate | roll_rate | 150 | 15.0000 | 0.9122 | 0.7455 | -0.1667 | 1.5021 | 4.5379 | 3.0358 | 0.2313 | 0.4175 |
| tcn_quantile | quantile | unseen_seastate | heave | 10 | 1.0000 | 0.9929 | 0.8677 | -0.1252 | 0.0298 | 0.0745 | 0.0446 | 0.0119 | 0.0173 |
| tcn_quantile | quantile | unseen_seastate | heave | 20 | 2.0000 | 0.9920 | 0.8147 | -0.1773 | 0.0602 | 0.1445 | 0.0843 | 0.0241 | 0.0336 |
| tcn_quantile | quantile | unseen_seastate | heave | 30 | 3.0000 | 0.9860 | 0.7199 | -0.2661 | 0.1038 | 0.2321 | 0.1283 | 0.0416 | 0.0541 |
| tcn_quantile | quantile | unseen_seastate | heave | 50 | 5.0000 | 0.9580 | 0.5408 | -0.4171 | 0.2402 | 0.4699 | 0.2297 | 0.0961 | 0.1094 |
| tcn_quantile | quantile | unseen_seastate | heave | 100 | 10.0000 | 0.9215 | 0.5657 | -0.3558 | 0.6379 | 1.3364 | 0.6985 | 0.2554 | 0.3113 |
| tcn_quantile | quantile | unseen_seastate | heave | 150 | 15.0000 | 0.9039 | 0.6228 | -0.2810 | 1.1279 | 2.5626 | 1.4347 | 0.4519 | 0.5970 |
| tcn_quantile | quantile | unseen_seastate | heave_rate | 10 | 1.0000 | 0.9866 | 0.8339 | -0.1527 | 0.0309 | 0.0742 | 0.0432 | 0.0209 | 0.0307 |
| tcn_quantile | quantile | unseen_seastate | heave_rate | 20 | 2.0000 | 0.9847 | 0.7431 | -0.2416 | 0.0528 | 0.1249 | 0.0721 | 0.0357 | 0.0517 |
| tcn_quantile | quantile | unseen_seastate | heave_rate | 30 | 3.0000 | 0.9684 | 0.6366 | -0.3318 | 0.0822 | 0.1816 | 0.0994 | 0.0556 | 0.0752 |
| tcn_quantile | quantile | unseen_seastate | heave_rate | 50 | 5.0000 | 0.9524 | 0.5738 | -0.3786 | 0.1337 | 0.2557 | 0.1220 | 0.0904 | 0.1059 |
| tcn_quantile | quantile | unseen_seastate | heave_rate | 100 | 10.0000 | 0.9169 | 0.5539 | -0.3630 | 0.4322 | 0.8197 | 0.3875 | 0.2923 | 0.3394 |
| tcn_quantile | quantile | unseen_seastate | heave_rate | 150 | 15.0000 | 0.9003 | 0.6461 | -0.2542 | 0.7554 | 1.5646 | 0.8092 | 0.5113 | 0.6479 |
| tcn_quantile | quantile | unseen_seastate | pitch | 10 | 1.0000 | 0.9887 | 0.8686 | -0.1201 | 0.0719 | 0.1981 | 0.1261 | 0.0176 | 0.0335 |
| tcn_quantile | quantile | unseen_seastate | pitch | 20 | 2.0000 | 0.9834 | 0.7280 | -0.2554 | 0.1722 | 0.4093 | 0.2372 | 0.0422 | 0.0691 |
| tcn_quantile | quantile | unseen_seastate | pitch | 30 | 3.0000 | 0.9784 | 0.6474 | -0.3310 | 0.2743 | 0.6046 | 0.3303 | 0.0672 | 0.1021 |
| tcn_quantile | quantile | unseen_seastate | pitch | 50 | 5.0000 | 0.9612 | 0.6567 | -0.3046 | 0.5274 | 1.0581 | 0.5307 | 0.1291 | 0.1788 |
| tcn_quantile | quantile | unseen_seastate | pitch | 100 | 10.0000 | 0.9166 | 0.6583 | -0.2584 | 1.4746 | 3.0853 | 1.6107 | 0.3611 | 0.5214 |
| tcn_quantile | quantile | unseen_seastate | pitch | 150 | 15.0000 | 0.9047 | 0.7397 | -0.1651 | 2.2714 | 5.1293 | 2.8579 | 0.5566 | 0.8667 |
| tcn_quantile | quantile | unseen_seastate | pitch_rate | 10 | 1.0000 | 0.9850 | 0.7915 | -0.1935 | 0.0932 | 0.2417 | 0.1484 | 0.0307 | 0.0585 |
| tcn_quantile | quantile | unseen_seastate | pitch_rate | 20 | 2.0000 | 0.9715 | 0.6912 | -0.2803 | 0.1727 | 0.3853 | 0.2126 | 0.0568 | 0.0932 |
| tcn_quantile | quantile | unseen_seastate | pitch_rate | 30 | 3.0000 | 0.9706 | 0.7276 | -0.2430 | 0.2444 | 0.5245 | 0.2802 | 0.0804 | 0.1269 |
| tcn_quantile | quantile | unseen_seastate | pitch_rate | 50 | 5.0000 | 0.9459 | 0.6521 | -0.2938 | 0.5215 | 1.0112 | 0.4898 | 0.1716 | 0.2447 |
| tcn_quantile | quantile | unseen_seastate | pitch_rate | 100 | 10.0000 | 0.9145 | 0.6645 | -0.2500 | 1.3029 | 2.6471 | 1.3443 | 0.4288 | 0.6407 |
| tcn_quantile | quantile | unseen_seastate | pitch_rate | 150 | 15.0000 | 0.9010 | 0.8054 | -0.0956 | 1.8681 | 4.1119 | 2.2438 | 0.6150 | 0.9947 |
| tcn_quantile | quantile | unseen_seastate | roll | 10 | 1.0000 | 0.9916 | 0.8807 | -0.1109 | 0.1112 | 0.2796 | 0.1683 | 0.0092 | 0.0136 |
| tcn_quantile | quantile | unseen_seastate | roll | 20 | 2.0000 | 0.9893 | 0.7945 | -0.1948 | 0.1963 | 0.4695 | 0.2732 | 0.0162 | 0.0228 |
| tcn_quantile | quantile | unseen_seastate | roll | 30 | 3.0000 | 0.9836 | 0.7371 | -0.2465 | 0.2938 | 0.6915 | 0.3977 | 0.0242 | 0.0336 |
| tcn_quantile | quantile | unseen_seastate | roll | 50 | 5.0000 | 0.9751 | 0.6834 | -0.2918 | 0.4093 | 0.8853 | 0.4761 | 0.0338 | 0.0430 |
| tcn_quantile | quantile | unseen_seastate | roll | 100 | 10.0000 | 0.9368 | 0.6519 | -0.2848 | 1.0699 | 2.3321 | 1.2622 | 0.0881 | 0.1132 |
| tcn_quantile | quantile | unseen_seastate | roll | 150 | 15.0000 | 0.9222 | 0.6888 | -0.2334 | 2.1321 | 4.9216 | 2.7894 | 0.1755 | 0.2387 |
| tcn_quantile | quantile | unseen_seastate | roll_rate | 10 | 1.0000 | 0.9805 | 0.8212 | -0.1593 | 0.1087 | 0.2635 | 0.1548 | 0.0168 | 0.0243 |
| tcn_quantile | quantile | unseen_seastate | roll_rate | 20 | 2.0000 | 0.9744 | 0.7627 | -0.2117 | 0.1638 | 0.3801 | 0.2163 | 0.0253 | 0.0350 |
| tcn_quantile | quantile | unseen_seastate | roll_rate | 30 | 3.0000 | 0.9734 | 0.7283 | -0.2451 | 0.1920 | 0.4267 | 0.2347 | 0.0296 | 0.0393 |
| tcn_quantile | quantile | unseen_seastate | roll_rate | 50 | 5.0000 | 0.9594 | 0.7501 | -0.2093 | 0.3244 | 0.7417 | 0.4173 | 0.0500 | 0.0683 |
| tcn_quantile | quantile | unseen_seastate | roll_rate | 100 | 10.0000 | 0.9250 | 0.6489 | -0.2761 | 0.8296 | 1.9117 | 1.0821 | 0.1278 | 0.1760 |
| tcn_quantile | quantile | unseen_seastate | roll_rate | 150 | 15.0000 | 0.9146 | 0.6689 | -0.2458 | 1.4414 | 3.4112 | 1.9698 | 0.2220 | 0.3139 |
| dlinear_gaussian | gaussian | unseen_vessel | heave | 10 | 1.0000 | 0.9624 | 0.9718 | 0.0094 | 0.0726 | 0.0694 | -0.0032 | 0.0291 | 0.0323 |
| dlinear_gaussian | gaussian | unseen_vessel | heave | 20 | 2.0000 | 0.9444 | 0.9544 | 0.0100 | 0.3962 | 0.3858 | -0.0104 | 0.1586 | 0.1793 |
| dlinear_gaussian | gaussian | unseen_vessel | heave | 30 | 3.0000 | 0.9224 | 0.9361 | 0.0137 | 0.8292 | 0.8204 | -0.0088 | 0.3320 | 0.3812 |
| dlinear_gaussian | gaussian | unseen_vessel | heave | 50 | 5.0000 | 0.9196 | 0.9326 | 0.0130 | 0.9751 | 0.9740 | -0.0011 | 0.3904 | 0.4525 |
| dlinear_gaussian | gaussian | unseen_vessel | heave | 100 | 10.0000 | 0.9072 | 0.9335 | 0.0264 | 1.8778 | 1.8793 | 0.0015 | 0.7519 | 0.8730 |
| dlinear_gaussian | gaussian | unseen_vessel | heave | 150 | 15.0000 | 0.8856 | 0.9162 | 0.0306 | 2.2599 | 2.2643 | 0.0044 | 0.9054 | 1.0518 |
| dlinear_gaussian | gaussian | unseen_vessel | heave_rate | 10 | 1.0000 | 0.9701 | 0.9862 | 0.0160 | 0.0431 | 0.0412 | -0.0019 | 0.0291 | 0.0341 |
| dlinear_gaussian | gaussian | unseen_vessel | heave_rate | 20 | 2.0000 | 0.9551 | 0.9752 | 0.0201 | 0.2351 | 0.2289 | -0.0062 | 0.1589 | 0.1896 |
| dlinear_gaussian | gaussian | unseen_vessel | heave_rate | 30 | 3.0000 | 0.9344 | 0.9602 | 0.0258 | 0.4919 | 0.4867 | -0.0053 | 0.3326 | 0.4032 |
| dlinear_gaussian | gaussian | unseen_vessel | heave_rate | 50 | 5.0000 | 0.9274 | 0.9558 | 0.0284 | 0.5785 | 0.5778 | -0.0007 | 0.3911 | 0.4786 |
| dlinear_gaussian | gaussian | unseen_vessel | heave_rate | 100 | 10.0000 | 0.9007 | 0.9472 | 0.0465 | 1.1141 | 1.1148 | 0.0008 | 0.7536 | 0.9234 |
| dlinear_gaussian | gaussian | unseen_vessel | heave_rate | 150 | 15.0000 | 0.8826 | 0.9288 | 0.0462 | 1.3408 | 1.3432 | 0.0025 | 0.9074 | 1.1125 |
| dlinear_gaussian | gaussian | unseen_vessel | pitch | 10 | 1.0000 | 0.9229 | 0.9801 | 0.0572 | 0.1196 | 0.1142 | -0.0054 | 0.0293 | 0.0329 |
| dlinear_gaussian | gaussian | unseen_vessel | pitch | 20 | 2.0000 | 0.9047 | 0.9693 | 0.0646 | 0.6524 | 0.6348 | -0.0176 | 0.1597 | 0.1829 |
| dlinear_gaussian | gaussian | unseen_vessel | pitch | 30 | 3.0000 | 0.8947 | 0.9557 | 0.0610 | 1.3653 | 1.3497 | -0.0156 | 0.3343 | 0.3887 |
| dlinear_gaussian | gaussian | unseen_vessel | pitch | 50 | 5.0000 | 0.8867 | 0.9495 | 0.0628 | 1.6056 | 1.6024 | -0.0032 | 0.3931 | 0.4614 |
| dlinear_gaussian | gaussian | unseen_vessel | pitch | 100 | 10.0000 | 0.8647 | 0.9247 | 0.0600 | 3.0921 | 3.0919 | -0.0001 | 0.7573 | 0.8898 |
| dlinear_gaussian | gaussian | unseen_vessel | pitch | 150 | 15.0000 | 0.8753 | 0.9127 | 0.0375 | 3.7212 | 3.7254 | 0.0041 | 0.9119 | 1.0717 |
| dlinear_gaussian | gaussian | unseen_vessel | pitch_rate | 10 | 1.0000 | 0.8636 | 0.9795 | 0.1159 | 0.0891 | 0.0851 | -0.0040 | 0.0293 | 0.0368 |
| dlinear_gaussian | gaussian | unseen_vessel | pitch_rate | 20 | 2.0000 | 0.8470 | 0.9695 | 0.1225 | 0.4864 | 0.4731 | -0.0132 | 0.1601 | 0.2046 |
| dlinear_gaussian | gaussian | unseen_vessel | pitch_rate | 30 | 3.0000 | 0.8454 | 0.9591 | 0.1137 | 1.0179 | 1.0061 | -0.0119 | 0.3349 | 0.4350 |
| dlinear_gaussian | gaussian | unseen_vessel | pitch_rate | 50 | 5.0000 | 0.8404 | 0.9533 | 0.1129 | 1.1971 | 1.1944 | -0.0026 | 0.3939 | 0.5163 |
| dlinear_gaussian | gaussian | unseen_vessel | pitch_rate | 100 | 10.0000 | 0.8385 | 0.9293 | 0.0908 | 2.3053 | 2.3047 | -0.0006 | 0.7586 | 0.9960 |
| dlinear_gaussian | gaussian | unseen_vessel | pitch_rate | 150 | 15.0000 | 0.8729 | 0.9287 | 0.0558 | 2.7744 | 2.7768 | 0.0025 | 0.9133 | 1.1995 |
| dlinear_gaussian | gaussian | unseen_vessel | roll | 10 | 1.0000 | 0.9917 | 0.9871 | -0.0046 | 0.3499 | 0.3342 | -0.0157 | 0.0289 | 0.0470 |
| dlinear_gaussian | gaussian | unseen_vessel | roll | 20 | 2.0000 | 0.9828 | 0.9767 | -0.0062 | 1.9091 | 1.8576 | -0.0515 | 0.1575 | 0.2610 |
| dlinear_gaussian | gaussian | unseen_vessel | roll | 30 | 3.0000 | 0.9706 | 0.9664 | -0.0042 | 3.9955 | 3.9499 | -0.0456 | 0.3296 | 0.5552 |
| dlinear_gaussian | gaussian | unseen_vessel | roll | 50 | 5.0000 | 0.9707 | 0.9719 | 0.0012 | 4.6986 | 4.6893 | -0.0093 | 0.3875 | 0.6592 |
| dlinear_gaussian | gaussian | unseen_vessel | roll | 100 | 10.0000 | 0.9624 | 0.9725 | 0.0100 | 9.0486 | 9.0483 | -0.0003 | 0.7454 | 1.2715 |
| dlinear_gaussian | gaussian | unseen_vessel | roll | 150 | 15.0000 | 0.9410 | 0.9600 | 0.0190 | 10.8897 | 10.9020 | 0.0123 | 0.8962 | 1.5310 |
| dlinear_gaussian | gaussian | unseen_vessel | roll_rate | 10 | 1.0000 | 0.9937 | 0.9931 | -0.0006 | 0.1872 | 0.1789 | -0.0083 | 0.0289 | 0.0542 |
| dlinear_gaussian | gaussian | unseen_vessel | roll_rate | 20 | 2.0000 | 0.9866 | 0.9866 | 0.0001 | 1.0214 | 0.9941 | -0.0273 | 0.1576 | 0.3013 |
| dlinear_gaussian | gaussian | unseen_vessel | roll_rate | 30 | 3.0000 | 0.9743 | 0.9796 | 0.0053 | 2.1376 | 2.1137 | -0.0239 | 0.3298 | 0.6404 |
| dlinear_gaussian | gaussian | unseen_vessel | roll_rate | 50 | 5.0000 | 0.9738 | 0.9834 | 0.0096 | 2.5138 | 2.5094 | -0.0044 | 0.3877 | 0.7603 |
| dlinear_gaussian | gaussian | unseen_vessel | roll_rate | 100 | 10.0000 | 0.9614 | 0.9831 | 0.0217 | 4.8410 | 4.8420 | 0.0010 | 0.7460 | 1.4669 |
| dlinear_gaussian | gaussian | unseen_vessel | roll_rate | 150 | 15.0000 | 0.9387 | 0.9730 | 0.0343 | 5.8260 | 5.8340 | 0.0080 | 0.8971 | 1.7676 |
| dlinear_quantile | quantile | unseen_vessel | heave | 10 | 1.0000 | 0.9335 | 0.9524 | 0.0189 | 0.0471 | 0.0456 | -0.0016 | 0.0189 | 0.0212 |
| dlinear_quantile | quantile | unseen_vessel | heave | 20 | 2.0000 | 0.9149 | 0.9339 | 0.0190 | 0.2976 | 0.2913 | -0.0062 | 0.1191 | 0.1354 |
| dlinear_quantile | quantile | unseen_vessel | heave | 30 | 3.0000 | 0.9016 | 0.9191 | 0.0175 | 0.7101 | 0.6987 | -0.0114 | 0.2843 | 0.3247 |
| dlinear_quantile | quantile | unseen_vessel | heave | 50 | 5.0000 | 0.9009 | 0.9187 | 0.0178 | 0.8686 | 0.8660 | -0.0026 | 0.3477 | 0.4024 |
| dlinear_quantile | quantile | unseen_vessel | heave | 100 | 10.0000 | 0.9014 | 0.9299 | 0.0286 | 1.8287 | 1.8284 | -0.0002 | 0.7322 | 0.8494 |
| dlinear_quantile | quantile | unseen_vessel | heave | 150 | 15.0000 | 0.8861 | 0.9169 | 0.0308 | 2.2701 | 2.2743 | 0.0042 | 0.9095 | 1.0565 |
| dlinear_quantile | quantile | unseen_vessel | heave_rate | 10 | 1.0000 | 0.9343 | 0.9713 | 0.0370 | 0.0280 | 0.0270 | -0.0010 | 0.0189 | 0.0224 |
| dlinear_quantile | quantile | unseen_vessel | heave_rate | 20 | 2.0000 | 0.9167 | 0.9560 | 0.0393 | 0.1766 | 0.1728 | -0.0038 | 0.1194 | 0.1432 |
| dlinear_quantile | quantile | unseen_vessel | heave_rate | 30 | 3.0000 | 0.9076 | 0.9434 | 0.0358 | 0.4214 | 0.4145 | -0.0069 | 0.2848 | 0.3434 |
| dlinear_quantile | quantile | unseen_vessel | heave_rate | 50 | 5.0000 | 0.9043 | 0.9417 | 0.0374 | 0.5141 | 0.5126 | -0.0015 | 0.3476 | 0.4246 |
| dlinear_quantile | quantile | unseen_vessel | heave_rate | 100 | 10.0000 | 0.8934 | 0.9427 | 0.0493 | 1.0846 | 1.0843 | -0.0003 | 0.7336 | 0.8981 |
| dlinear_quantile | quantile | unseen_vessel | heave_rate | 150 | 15.0000 | 0.8828 | 0.9293 | 0.0465 | 1.3468 | 1.3492 | 0.0024 | 0.9115 | 1.1174 |
| dlinear_quantile | quantile | unseen_vessel | pitch | 10 | 1.0000 | 0.8645 | 0.9581 | 0.0936 | 0.0795 | 0.0750 | -0.0045 | 0.0195 | 0.0216 |
| dlinear_quantile | quantile | unseen_vessel | pitch | 20 | 2.0000 | 0.8508 | 0.9431 | 0.0923 | 0.4949 | 0.4795 | -0.0154 | 0.1212 | 0.1381 |
| dlinear_quantile | quantile | unseen_vessel | pitch | 30 | 3.0000 | 0.8603 | 0.9357 | 0.0754 | 1.1724 | 1.1496 | -0.0229 | 0.2871 | 0.3311 |
| dlinear_quantile | quantile | unseen_vessel | pitch | 50 | 5.0000 | 0.8603 | 0.9325 | 0.0722 | 1.4267 | 1.4219 | -0.0048 | 0.3493 | 0.4094 |
| dlinear_quantile | quantile | unseen_vessel | pitch | 100 | 10.0000 | 0.8576 | 0.9184 | 0.0608 | 3.0101 | 3.0072 | -0.0029 | 0.7372 | 0.8655 |
| dlinear_quantile | quantile | unseen_vessel | pitch | 150 | 15.0000 | 0.8757 | 0.9128 | 0.0370 | 3.7380 | 3.7418 | 0.0039 | 0.9160 | 1.0764 |
| dlinear_quantile | quantile | unseen_vessel | pitch_rate | 10 | 1.0000 | 0.8065 | 0.9515 | 0.1450 | 0.0655 | 0.0559 | -0.0095 | 0.0215 | 0.0242 |
| dlinear_quantile | quantile | unseen_vessel | pitch_rate | 20 | 2.0000 | 0.7931 | 0.9402 | 0.1471 | 0.3875 | 0.3577 | -0.0298 | 0.1275 | 0.1547 |
| dlinear_quantile | quantile | unseen_vessel | pitch_rate | 30 | 3.0000 | 0.8094 | 0.9390 | 0.1296 | 0.8914 | 0.8570 | -0.0344 | 0.2933 | 0.3706 |
| dlinear_quantile | quantile | unseen_vessel | pitch_rate | 50 | 5.0000 | 0.8125 | 0.9364 | 0.1240 | 1.0659 | 1.0591 | -0.0068 | 0.3507 | 0.4579 |
| dlinear_quantile | quantile | unseen_vessel | pitch_rate | 100 | 10.0000 | 0.8315 | 0.9235 | 0.0920 | 2.2445 | 2.2414 | -0.0030 | 0.7386 | 0.9687 |
| dlinear_quantile | quantile | unseen_vessel | pitch_rate | 150 | 15.0000 | 0.8733 | 0.9288 | 0.0555 | 2.7868 | 2.7891 | 0.0022 | 0.9174 | 1.2048 |
| dlinear_quantile | quantile | unseen_vessel | roll | 10 | 1.0000 | 0.9727 | 0.9780 | 0.0053 | 0.2272 | 0.2193 | -0.0078 | 0.0188 | 0.0308 |
| dlinear_quantile | quantile | unseen_vessel | roll | 20 | 2.0000 | 0.9668 | 0.9675 | 0.0007 | 1.4342 | 1.4027 | -0.0315 | 0.1184 | 0.1971 |
| dlinear_quantile | quantile | unseen_vessel | roll | 30 | 3.0000 | 0.9609 | 0.9577 | -0.0032 | 3.4218 | 3.3645 | -0.0573 | 0.2823 | 0.4729 |
| dlinear_quantile | quantile | unseen_vessel | roll | 50 | 5.0000 | 0.9624 | 0.9653 | 0.0029 | 4.1734 | 4.1716 | -0.0018 | 0.3442 | 0.5864 |
| dlinear_quantile | quantile | unseen_vessel | roll | 100 | 10.0000 | 0.9596 | 0.9719 | 0.0123 | 8.8086 | 8.8033 | -0.0053 | 0.7256 | 1.2370 |
| dlinear_quantile | quantile | unseen_vessel | roll | 150 | 15.0000 | 0.9422 | 0.9613 | 0.0190 | 10.9387 | 10.9502 | 0.0115 | 0.9002 | 1.5378 |
| dlinear_quantile | quantile | unseen_vessel | roll_rate | 10 | 1.0000 | 0.9739 | 0.9874 | 0.0134 | 0.1215 | 0.1174 | -0.0042 | 0.0188 | 0.0356 |
| dlinear_quantile | quantile | unseen_vessel | roll_rate | 20 | 2.0000 | 0.9687 | 0.9802 | 0.0115 | 0.7673 | 0.7506 | -0.0166 | 0.1184 | 0.2275 |
| dlinear_quantile | quantile | unseen_vessel | roll_rate | 30 | 3.0000 | 0.9635 | 0.9730 | 0.0094 | 1.8307 | 1.8002 | -0.0305 | 0.2824 | 0.5455 |
| dlinear_quantile | quantile | unseen_vessel | roll_rate | 50 | 5.0000 | 0.9643 | 0.9787 | 0.0144 | 2.2324 | 2.2282 | -0.0042 | 0.3443 | 0.6751 |
| dlinear_quantile | quantile | unseen_vessel | roll_rate | 100 | 10.0000 | 0.9577 | 0.9826 | 0.0249 | 4.7126 | 4.7095 | -0.0030 | 0.7262 | 1.4268 |
| dlinear_quantile | quantile | unseen_vessel | roll_rate | 150 | 15.0000 | 0.9399 | 0.9741 | 0.0342 | 5.8522 | 5.8597 | 0.0075 | 0.9011 | 1.7754 |
| lstm_gaussian | gaussian | unseen_vessel | heave | 10 | 1.0000 | 0.9935 | 0.5152 | -0.4782 | 0.0301 | 0.0327 | 0.0025 | 0.0121 | 0.0152 |
| lstm_gaussian | gaussian | unseen_vessel | heave | 20 | 2.0000 | 0.9909 | 0.3297 | -0.6612 | 0.0342 | 0.0388 | 0.0045 | 0.0137 | 0.0180 |
| lstm_gaussian | gaussian | unseen_vessel | heave | 30 | 3.0000 | 0.9769 | 0.2920 | -0.6849 | 0.0490 | 0.0543 | 0.0053 | 0.0196 | 0.0252 |
| lstm_gaussian | gaussian | unseen_vessel | heave | 50 | 5.0000 | 0.9398 | 0.4742 | -0.4656 | 0.1511 | 0.1459 | -0.0052 | 0.0605 | 0.0678 |
| lstm_gaussian | gaussian | unseen_vessel | heave | 100 | 10.0000 | 0.9210 | 0.6934 | -0.2276 | 0.5194 | 0.4788 | -0.0405 | 0.2080 | 0.2224 |
| lstm_gaussian | gaussian | unseen_vessel | heave | 150 | 15.0000 | 0.9098 | 0.8409 | -0.0689 | 1.0591 | 0.9991 | -0.0600 | 0.4243 | 0.4641 |
| lstm_gaussian | gaussian | unseen_vessel | heave_rate | 10 | 1.0000 | 0.9895 | 0.3609 | -0.6286 | 0.0215 | 0.0232 | 0.0017 | 0.0145 | 0.0192 |
| lstm_gaussian | gaussian | unseen_vessel | heave_rate | 20 | 2.0000 | 0.9824 | 0.3238 | -0.6586 | 0.0261 | 0.0283 | 0.0022 | 0.0177 | 0.0235 |
| lstm_gaussian | gaussian | unseen_vessel | heave_rate | 30 | 3.0000 | 0.9638 | 0.4118 | -0.5520 | 0.0435 | 0.0420 | -0.0014 | 0.0294 | 0.0348 |
| lstm_gaussian | gaussian | unseen_vessel | heave_rate | 50 | 5.0000 | 0.9350 | 0.4743 | -0.4607 | 0.0940 | 0.0881 | -0.0059 | 0.0636 | 0.0730 |
| lstm_gaussian | gaussian | unseen_vessel | heave_rate | 100 | 10.0000 | 0.9177 | 0.6886 | -0.2291 | 0.3794 | 0.3220 | -0.0574 | 0.2566 | 0.2667 |
| lstm_gaussian | gaussian | unseen_vessel | heave_rate | 150 | 15.0000 | 0.9089 | 0.8495 | -0.0594 | 0.7241 | 0.6501 | -0.0740 | 0.4901 | 0.5384 |
| lstm_gaussian | gaussian | unseen_vessel | pitch | 10 | 1.0000 | 0.9896 | 0.4617 | -0.5280 | 0.0595 | 0.0653 | 0.0058 | 0.0146 | 0.0188 |
| lstm_gaussian | gaussian | unseen_vessel | pitch | 20 | 2.0000 | 0.9781 | 0.3776 | -0.6005 | 0.0907 | 0.0887 | -0.0020 | 0.0222 | 0.0256 |
| lstm_gaussian | gaussian | unseen_vessel | pitch | 30 | 3.0000 | 0.9558 | 0.4145 | -0.5413 | 0.1472 | 0.1416 | -0.0056 | 0.0360 | 0.0408 |
| lstm_gaussian | gaussian | unseen_vessel | pitch | 50 | 5.0000 | 0.9375 | 0.5304 | -0.4071 | 0.3420 | 0.2670 | -0.0750 | 0.0837 | 0.0769 |
| lstm_gaussian | gaussian | unseen_vessel | pitch | 100 | 10.0000 | 0.9172 | 0.7925 | -0.1247 | 1.2751 | 1.1093 | -0.1658 | 0.3123 | 0.3193 |
| lstm_gaussian | gaussian | unseen_vessel | pitch | 150 | 15.0000 | 0.9076 | 0.8568 | -0.0508 | 2.2352 | 2.0404 | -0.1948 | 0.5477 | 0.5870 |
| lstm_gaussian | gaussian | unseen_vessel | pitch_rate | 10 | 1.0000 | 0.9824 | 0.4108 | -0.5716 | 0.0658 | 0.0650 | -0.0007 | 0.0216 | 0.0281 |
| lstm_gaussian | gaussian | unseen_vessel | pitch_rate | 20 | 2.0000 | 0.9647 | 0.4210 | -0.5437 | 0.0978 | 0.0914 | -0.0064 | 0.0322 | 0.0395 |
| lstm_gaussian | gaussian | unseen_vessel | pitch_rate | 30 | 3.0000 | 0.9430 | 0.4456 | -0.4975 | 0.1360 | 0.1149 | -0.0210 | 0.0447 | 0.0497 |
| lstm_gaussian | gaussian | unseen_vessel | pitch_rate | 50 | 5.0000 | 0.9383 | 0.5889 | -0.3494 | 0.3728 | 0.2855 | -0.0873 | 0.1227 | 0.1234 |
| lstm_gaussian | gaussian | unseen_vessel | pitch_rate | 100 | 10.0000 | 0.9152 | 0.7835 | -0.1317 | 1.2078 | 0.9589 | -0.2489 | 0.3975 | 0.4144 |
| lstm_gaussian | gaussian | unseen_vessel | pitch_rate | 150 | 15.0000 | 0.9066 | 0.9016 | -0.0049 | 1.8484 | 1.6253 | -0.2231 | 0.6085 | 0.7021 |
| lstm_gaussian | gaussian | unseen_vessel | roll | 10 | 1.0000 | 0.9918 | 0.2880 | -0.7038 | 0.1339 | 0.1287 | -0.0052 | 0.0111 | 0.0181 |
| lstm_gaussian | gaussian | unseen_vessel | roll | 20 | 2.0000 | 0.9906 | 0.2292 | -0.7614 | 0.1366 | 0.1373 | 0.0007 | 0.0113 | 0.0193 |
| lstm_gaussian | gaussian | unseen_vessel | roll | 30 | 3.0000 | 0.9822 | 0.1947 | -0.7875 | 0.1551 | 0.1646 | 0.0095 | 0.0128 | 0.0231 |
| lstm_gaussian | gaussian | unseen_vessel | roll | 50 | 5.0000 | 0.9532 | 0.1876 | -0.7656 | 0.2176 | 0.2325 | 0.0149 | 0.0179 | 0.0327 |
| lstm_gaussian | gaussian | unseen_vessel | roll | 100 | 10.0000 | 0.9242 | 0.3619 | -0.5622 | 0.8294 | 0.8215 | -0.0079 | 0.0683 | 0.1154 |
| lstm_gaussian | gaussian | unseen_vessel | roll | 150 | 15.0000 | 0.9135 | 0.5466 | -0.3669 | 1.8962 | 1.8871 | -0.0091 | 0.1560 | 0.2650 |
| lstm_gaussian | gaussian | unseen_vessel | roll_rate | 10 | 1.0000 | 0.9926 | 0.2553 | -0.7372 | 0.0787 | 0.0784 | -0.0003 | 0.0121 | 0.0238 |
| lstm_gaussian | gaussian | unseen_vessel | roll_rate | 20 | 2.0000 | 0.9652 | 0.2024 | -0.7628 | 0.0882 | 0.0898 | 0.0015 | 0.0136 | 0.0272 |
| lstm_gaussian | gaussian | unseen_vessel | roll_rate | 30 | 3.0000 | 0.9669 | 0.2075 | -0.7594 | 0.1038 | 0.1064 | 0.0026 | 0.0160 | 0.0322 |
| lstm_gaussian | gaussian | unseen_vessel | roll_rate | 50 | 5.0000 | 0.9491 | 0.2423 | -0.7068 | 0.1675 | 0.1657 | -0.0019 | 0.0258 | 0.0502 |
| lstm_gaussian | gaussian | unseen_vessel | roll_rate | 100 | 10.0000 | 0.9172 | 0.4671 | -0.4501 | 0.6534 | 0.6335 | -0.0199 | 0.1007 | 0.1919 |
| lstm_gaussian | gaussian | unseen_vessel | roll_rate | 150 | 15.0000 | 0.9075 | 0.6343 | -0.2733 | 1.2816 | 1.2896 | 0.0079 | 0.1973 | 0.3907 |
| lstm_quantile | quantile | unseen_vessel | heave | 10 | 1.0000 | 0.9923 | 0.4311 | -0.5612 | 0.0520 | 0.0322 | -0.0198 | 0.0208 | 0.0150 |
| lstm_quantile | quantile | unseen_vessel | heave | 20 | 2.0000 | 0.9855 | 0.2634 | -0.7221 | 0.0503 | 0.0332 | -0.0170 | 0.0201 | 0.0154 |
| lstm_quantile | quantile | unseen_vessel | heave | 30 | 3.0000 | 0.9502 | 0.1956 | -0.7546 | 0.0575 | 0.0379 | -0.0197 | 0.0230 | 0.0176 |
| lstm_quantile | quantile | unseen_vessel | heave | 50 | 5.0000 | 0.9303 | 0.2988 | -0.6314 | 0.1314 | 0.0946 | -0.0367 | 0.0526 | 0.0440 |
| lstm_quantile | quantile | unseen_vessel | heave | 100 | 10.0000 | 0.9172 | 0.5565 | -0.3607 | 0.4602 | 0.3596 | -0.1006 | 0.1843 | 0.1671 |
| lstm_quantile | quantile | unseen_vessel | heave | 150 | 15.0000 | 0.9083 | 0.7498 | -0.1585 | 0.9621 | 0.8282 | -0.1339 | 0.3855 | 0.3847 |
| lstm_quantile | quantile | unseen_vessel | heave_rate | 10 | 1.0000 | 0.9863 | 0.3630 | -0.6232 | 0.0330 | 0.0202 | -0.0128 | 0.0223 | 0.0167 |
| lstm_quantile | quantile | unseen_vessel | heave_rate | 20 | 2.0000 | 0.9797 | 0.2624 | -0.7173 | 0.0337 | 0.0211 | -0.0126 | 0.0228 | 0.0175 |
| lstm_quantile | quantile | unseen_vessel | heave_rate | 30 | 3.0000 | 0.9624 | 0.2777 | -0.6848 | 0.0453 | 0.0274 | -0.0179 | 0.0306 | 0.0227 |
| lstm_quantile | quantile | unseen_vessel | heave_rate | 50 | 5.0000 | 0.9346 | 0.2858 | -0.6489 | 0.0830 | 0.0597 | -0.0234 | 0.0561 | 0.0494 |
| lstm_quantile | quantile | unseen_vessel | heave_rate | 100 | 10.0000 | 0.9203 | 0.5660 | -0.3543 | 0.3434 | 0.2624 | -0.0810 | 0.2323 | 0.2174 |
| lstm_quantile | quantile | unseen_vessel | heave_rate | 150 | 15.0000 | 0.9069 | 0.7810 | -0.1259 | 0.6631 | 0.5627 | -0.1005 | 0.4488 | 0.4660 |
| lstm_quantile | quantile | unseen_vessel | pitch | 10 | 1.0000 | 0.9884 | 0.3496 | -0.6388 | 0.0855 | 0.0545 | -0.0309 | 0.0209 | 0.0157 |
| lstm_quantile | quantile | unseen_vessel | pitch | 20 | 2.0000 | 0.9814 | 0.2668 | -0.7146 | 0.0989 | 0.0614 | -0.0375 | 0.0242 | 0.0177 |
| lstm_quantile | quantile | unseen_vessel | pitch | 30 | 3.0000 | 0.9645 | 0.2657 | -0.6988 | 0.1414 | 0.0926 | -0.0488 | 0.0346 | 0.0267 |
| lstm_quantile | quantile | unseen_vessel | pitch | 50 | 5.0000 | 0.9419 | 0.3550 | -0.5869 | 0.2967 | 0.1781 | -0.1186 | 0.0726 | 0.0513 |
| lstm_quantile | quantile | unseen_vessel | pitch | 100 | 10.0000 | 0.9120 | 0.6614 | -0.2506 | 1.1582 | 0.9034 | -0.2548 | 0.2837 | 0.2600 |
| lstm_quantile | quantile | unseen_vessel | pitch | 150 | 15.0000 | 0.9105 | 0.8125 | -0.0980 | 2.0459 | 1.8060 | -0.2399 | 0.5013 | 0.5195 |
| lstm_quantile | quantile | unseen_vessel | pitch_rate | 10 | 1.0000 | 0.9859 | 0.3679 | -0.6179 | 0.0808 | 0.0469 | -0.0338 | 0.0266 | 0.0203 |
| lstm_quantile | quantile | unseen_vessel | pitch_rate | 20 | 2.0000 | 0.9791 | 0.3144 | -0.6646 | 0.1029 | 0.0609 | -0.0420 | 0.0339 | 0.0263 |
| lstm_quantile | quantile | unseen_vessel | pitch_rate | 30 | 3.0000 | 0.9659 | 0.3035 | -0.6624 | 0.1260 | 0.0765 | -0.0494 | 0.0415 | 0.0331 |
| lstm_quantile | quantile | unseen_vessel | pitch_rate | 50 | 5.0000 | 0.9334 | 0.4033 | -0.5301 | 0.3229 | 0.2005 | -0.1225 | 0.1063 | 0.0867 |
| lstm_quantile | quantile | unseen_vessel | pitch_rate | 100 | 10.0000 | 0.9126 | 0.6859 | -0.2267 | 1.0905 | 0.8294 | -0.2611 | 0.3589 | 0.3584 |
| lstm_quantile | quantile | unseen_vessel | pitch_rate | 150 | 15.0000 | 0.9044 | 0.8388 | -0.0656 | 1.7232 | 1.4680 | -0.2551 | 0.5673 | 0.6342 |
| lstm_quantile | quantile | unseen_vessel | roll | 10 | 1.0000 | 0.9854 | 0.3072 | -0.6782 | 0.2290 | 0.1509 | -0.0780 | 0.0189 | 0.0212 |
| lstm_quantile | quantile | unseen_vessel | roll | 20 | 2.0000 | 0.9791 | 0.2300 | -0.7491 | 0.2252 | 0.1498 | -0.0754 | 0.0186 | 0.0210 |
| lstm_quantile | quantile | unseen_vessel | roll | 30 | 3.0000 | 0.9790 | 0.1971 | -0.7819 | 0.2372 | 0.1531 | -0.0841 | 0.0196 | 0.0215 |
| lstm_quantile | quantile | unseen_vessel | roll | 50 | 5.0000 | 0.9543 | 0.1684 | -0.7859 | 0.2596 | 0.1807 | -0.0789 | 0.0214 | 0.0254 |
| lstm_quantile | quantile | unseen_vessel | roll | 100 | 10.0000 | 0.9249 | 0.3359 | -0.5890 | 0.7617 | 0.6457 | -0.1160 | 0.0627 | 0.0907 |
| lstm_quantile | quantile | unseen_vessel | roll | 150 | 15.0000 | 0.9204 | 0.4865 | -0.4339 | 1.7209 | 1.5629 | -0.1580 | 0.1416 | 0.2195 |
| lstm_quantile | quantile | unseen_vessel | roll_rate | 10 | 1.0000 | 0.9763 | 0.2941 | -0.6821 | 0.1330 | 0.0843 | -0.0487 | 0.0205 | 0.0256 |
| lstm_quantile | quantile | unseen_vessel | roll_rate | 20 | 2.0000 | 0.9787 | 0.2463 | -0.7324 | 0.1380 | 0.0867 | -0.0513 | 0.0213 | 0.0263 |
| lstm_quantile | quantile | unseen_vessel | roll_rate | 30 | 3.0000 | 0.9813 | 0.2109 | -0.7704 | 0.1439 | 0.0932 | -0.0507 | 0.0222 | 0.0282 |
| lstm_quantile | quantile | unseen_vessel | roll_rate | 50 | 5.0000 | 0.9650 | 0.2228 | -0.7423 | 0.1797 | 0.1201 | -0.0595 | 0.0277 | 0.0364 |
| lstm_quantile | quantile | unseen_vessel | roll_rate | 100 | 10.0000 | 0.9219 | 0.4243 | -0.4976 | 0.5958 | 0.4859 | -0.1099 | 0.0918 | 0.1472 |
| lstm_quantile | quantile | unseen_vessel | roll_rate | 150 | 15.0000 | 0.9155 | 0.5738 | -0.3417 | 1.1971 | 1.0644 | -0.1328 | 0.1843 | 0.3225 |
| tcn_gaussian | gaussian | unseen_vessel | heave | 10 | 1.0000 | 0.9937 | 0.5498 | -0.4439 | 0.0174 | 0.0139 | -0.0036 | 0.0070 | 0.0064 |
| tcn_gaussian | gaussian | unseen_vessel | heave | 20 | 2.0000 | 0.9926 | 0.3979 | -0.5948 | 0.0393 | 0.0327 | -0.0065 | 0.0157 | 0.0152 |
| tcn_gaussian | gaussian | unseen_vessel | heave | 30 | 3.0000 | 0.9780 | 0.4266 | -0.5514 | 0.0859 | 0.0718 | -0.0140 | 0.0344 | 0.0334 |
| tcn_gaussian | gaussian | unseen_vessel | heave | 50 | 5.0000 | 0.9517 | 0.5965 | -0.3552 | 0.2430 | 0.2124 | -0.0306 | 0.0973 | 0.0987 |
| tcn_gaussian | gaussian | unseen_vessel | heave | 100 | 10.0000 | 0.9221 | 0.7953 | -0.1268 | 0.6934 | 0.6100 | -0.0835 | 0.2777 | 0.2833 |
| tcn_gaussian | gaussian | unseen_vessel | heave | 150 | 15.0000 | 0.9068 | 0.8960 | -0.0108 | 1.2366 | 1.1449 | -0.0917 | 0.4955 | 0.5319 |
| tcn_gaussian | gaussian | unseen_vessel | heave_rate | 10 | 1.0000 | 0.9922 | 0.4050 | -0.5872 | 0.0175 | 0.0141 | -0.0034 | 0.0118 | 0.0117 |
| tcn_gaussian | gaussian | unseen_vessel | heave_rate | 20 | 2.0000 | 0.9812 | 0.4435 | -0.5377 | 0.0413 | 0.0329 | -0.0084 | 0.0279 | 0.0273 |
| tcn_gaussian | gaussian | unseen_vessel | heave_rate | 30 | 3.0000 | 0.9598 | 0.5685 | -0.3913 | 0.0753 | 0.0607 | -0.0147 | 0.0509 | 0.0503 |
| tcn_gaussian | gaussian | unseen_vessel | heave_rate | 50 | 5.0000 | 0.9426 | 0.5251 | -0.4175 | 0.1341 | 0.1097 | -0.0244 | 0.0907 | 0.0909 |
| tcn_gaussian | gaussian | unseen_vessel | heave_rate | 100 | 10.0000 | 0.9187 | 0.7470 | -0.1718 | 0.4596 | 0.3804 | -0.0791 | 0.3109 | 0.3151 |
| tcn_gaussian | gaussian | unseen_vessel | heave_rate | 150 | 15.0000 | 0.9034 | 0.8920 | -0.0114 | 0.8088 | 0.7164 | -0.0924 | 0.5474 | 0.5934 |
| tcn_gaussian | gaussian | unseen_vessel | pitch | 10 | 1.0000 | 0.9769 | 0.7457 | -0.2312 | 0.0533 | 0.0398 | -0.0135 | 0.0131 | 0.0115 |
| tcn_gaussian | gaussian | unseen_vessel | pitch | 20 | 2.0000 | 0.9717 | 0.5469 | -0.4247 | 0.1577 | 0.1158 | -0.0418 | 0.0386 | 0.0334 |
| tcn_gaussian | gaussian | unseen_vessel | pitch | 30 | 3.0000 | 0.9677 | 0.4850 | -0.4827 | 0.2634 | 0.2135 | -0.0499 | 0.0645 | 0.0615 |
| tcn_gaussian | gaussian | unseen_vessel | pitch | 50 | 5.0000 | 0.9509 | 0.6714 | -0.2795 | 0.5619 | 0.4120 | -0.1499 | 0.1376 | 0.1186 |
| tcn_gaussian | gaussian | unseen_vessel | pitch | 100 | 10.0000 | 0.9176 | 0.8394 | -0.0782 | 1.6214 | 1.3777 | -0.2437 | 0.3971 | 0.3965 |
| tcn_gaussian | gaussian | unseen_vessel | pitch | 150 | 15.0000 | 0.9056 | 0.9019 | -0.0037 | 2.4822 | 2.2525 | -0.2297 | 0.6083 | 0.6480 |
| tcn_gaussian | gaussian | unseen_vessel | pitch_rate | 10 | 1.0000 | 0.9753 | 0.6062 | -0.3691 | 0.0864 | 0.0594 | -0.0271 | 0.0284 | 0.0257 |
| tcn_gaussian | gaussian | unseen_vessel | pitch_rate | 20 | 2.0000 | 0.9686 | 0.5278 | -0.4408 | 0.1640 | 0.1242 | -0.0398 | 0.0540 | 0.0537 |
| tcn_gaussian | gaussian | unseen_vessel | pitch_rate | 30 | 3.0000 | 0.9562 | 0.6165 | -0.3397 | 0.2484 | 0.1677 | -0.0806 | 0.0817 | 0.0725 |
| tcn_gaussian | gaussian | unseen_vessel | pitch_rate | 50 | 5.0000 | 0.9328 | 0.6720 | -0.2607 | 0.5625 | 0.4122 | -0.1504 | 0.1851 | 0.1782 |
| tcn_gaussian | gaussian | unseen_vessel | pitch_rate | 100 | 10.0000 | 0.9140 | 0.8344 | -0.0796 | 1.4131 | 1.1161 | -0.2970 | 0.4650 | 0.4823 |
| tcn_gaussian | gaussian | unseen_vessel | pitch_rate | 150 | 15.0000 | 0.9014 | 0.9459 | 0.0445 | 2.0121 | 1.7625 | -0.2496 | 0.6624 | 0.7614 |
| tcn_gaussian | gaussian | unseen_vessel | roll | 10 | 1.0000 | 0.9953 | 0.3270 | -0.6683 | 0.0567 | 0.0433 | -0.0134 | 0.0047 | 0.0061 |
| tcn_gaussian | gaussian | unseen_vessel | roll | 20 | 2.0000 | 0.9849 | 0.2183 | -0.7666 | 0.1011 | 0.0865 | -0.0145 | 0.0083 | 0.0122 |
| tcn_gaussian | gaussian | unseen_vessel | roll | 30 | 3.0000 | 0.9743 | 0.1805 | -0.7938 | 0.1747 | 0.1589 | -0.0158 | 0.0144 | 0.0223 |
| tcn_gaussian | gaussian | unseen_vessel | roll | 50 | 5.0000 | 0.9648 | 0.2108 | -0.7539 | 0.2981 | 0.2757 | -0.0224 | 0.0246 | 0.0388 |
| tcn_gaussian | gaussian | unseen_vessel | roll | 100 | 10.0000 | 0.9326 | 0.4400 | -0.4926 | 1.0728 | 1.0258 | -0.0470 | 0.0884 | 0.1441 |
| tcn_gaussian | gaussian | unseen_vessel | roll | 150 | 15.0000 | 0.9165 | 0.6272 | -0.2893 | 2.2330 | 2.2262 | -0.0069 | 0.1838 | 0.3126 |
| tcn_gaussian | gaussian | unseen_vessel | roll_rate | 10 | 1.0000 | 0.9819 | 0.2772 | -0.7047 | 0.0506 | 0.0440 | -0.0066 | 0.0078 | 0.0133 |
| tcn_gaussian | gaussian | unseen_vessel | roll_rate | 20 | 2.0000 | 0.9738 | 0.2222 | -0.7516 | 0.0951 | 0.0847 | -0.0104 | 0.0147 | 0.0257 |
| tcn_gaussian | gaussian | unseen_vessel | roll_rate | 30 | 3.0000 | 0.9656 | 0.2301 | -0.7354 | 0.1208 | 0.1120 | -0.0088 | 0.0186 | 0.0339 |
| tcn_gaussian | gaussian | unseen_vessel | roll_rate | 50 | 5.0000 | 0.9495 | 0.3192 | -0.6303 | 0.2672 | 0.2356 | -0.0316 | 0.0412 | 0.0714 |
| tcn_gaussian | gaussian | unseen_vessel | roll_rate | 100 | 10.0000 | 0.9236 | 0.5424 | -0.3813 | 0.8382 | 0.8033 | -0.0349 | 0.1292 | 0.2434 |
| tcn_gaussian | gaussian | unseen_vessel | roll_rate | 150 | 15.0000 | 0.9122 | 0.7138 | -0.1983 | 1.5021 | 1.5136 | 0.0115 | 0.2313 | 0.4586 |
| tcn_quantile | quantile | unseen_vessel | heave | 10 | 1.0000 | 0.9929 | 0.6864 | -0.3065 | 0.0298 | 0.0249 | -0.0049 | 0.0119 | 0.0116 |
| tcn_quantile | quantile | unseen_vessel | heave | 20 | 2.0000 | 0.9920 | 0.4885 | -0.5035 | 0.0602 | 0.0464 | -0.0138 | 0.0241 | 0.0216 |
| tcn_quantile | quantile | unseen_vessel | heave | 30 | 3.0000 | 0.9860 | 0.4529 | -0.5332 | 0.1038 | 0.0800 | -0.0238 | 0.0416 | 0.0372 |
| tcn_quantile | quantile | unseen_vessel | heave | 50 | 5.0000 | 0.9580 | 0.6448 | -0.3132 | 0.2402 | 0.1958 | -0.0444 | 0.0961 | 0.0910 |
| tcn_quantile | quantile | unseen_vessel | heave | 100 | 10.0000 | 0.9215 | 0.7666 | -0.1549 | 0.6379 | 0.5122 | -0.1257 | 0.2554 | 0.2379 |
| tcn_quantile | quantile | unseen_vessel | heave | 150 | 15.0000 | 0.9039 | 0.8531 | -0.0507 | 1.1279 | 0.9471 | -0.1808 | 0.4519 | 0.4400 |
| tcn_quantile | quantile | unseen_vessel | heave_rate | 10 | 1.0000 | 0.9866 | 0.5213 | -0.4653 | 0.0309 | 0.0230 | -0.0079 | 0.0209 | 0.0191 |
| tcn_quantile | quantile | unseen_vessel | heave_rate | 20 | 2.0000 | 0.9847 | 0.5356 | -0.4492 | 0.0528 | 0.0423 | -0.0105 | 0.0357 | 0.0350 |
| tcn_quantile | quantile | unseen_vessel | heave_rate | 30 | 3.0000 | 0.9684 | 0.6702 | -0.2982 | 0.0822 | 0.0647 | -0.0175 | 0.0556 | 0.0536 |
| tcn_quantile | quantile | unseen_vessel | heave_rate | 50 | 5.0000 | 0.9524 | 0.5713 | -0.3811 | 0.1337 | 0.1016 | -0.0321 | 0.0904 | 0.0842 |
| tcn_quantile | quantile | unseen_vessel | heave_rate | 100 | 10.0000 | 0.9169 | 0.7332 | -0.1837 | 0.4322 | 0.3331 | -0.0990 | 0.2923 | 0.2759 |
| tcn_quantile | quantile | unseen_vessel | heave_rate | 150 | 15.0000 | 0.9003 | 0.8731 | -0.0272 | 0.7554 | 0.6181 | -0.1373 | 0.5113 | 0.5120 |
| tcn_quantile | quantile | unseen_vessel | pitch | 10 | 1.0000 | 0.9887 | 0.8548 | -0.1339 | 0.0719 | 0.0533 | -0.0186 | 0.0176 | 0.0154 |
| tcn_quantile | quantile | unseen_vessel | pitch | 20 | 2.0000 | 0.9834 | 0.7316 | -0.2517 | 0.1722 | 0.1241 | -0.0481 | 0.0422 | 0.0357 |
| tcn_quantile | quantile | unseen_vessel | pitch | 30 | 3.0000 | 0.9784 | 0.6555 | -0.3230 | 0.2743 | 0.2067 | -0.0676 | 0.0672 | 0.0595 |
| tcn_quantile | quantile | unseen_vessel | pitch | 50 | 5.0000 | 0.9612 | 0.7659 | -0.1953 | 0.5274 | 0.3617 | -0.1657 | 0.1291 | 0.1041 |
| tcn_quantile | quantile | unseen_vessel | pitch | 100 | 10.0000 | 0.9166 | 0.8334 | -0.0832 | 1.4746 | 1.1286 | -0.3460 | 0.3611 | 0.3248 |
| tcn_quantile | quantile | unseen_vessel | pitch | 150 | 15.0000 | 0.9047 | 0.8725 | -0.0322 | 2.2714 | 1.8869 | -0.3845 | 0.5566 | 0.5428 |
| tcn_quantile | quantile | unseen_vessel | pitch_rate | 10 | 1.0000 | 0.9850 | 0.7736 | -0.2115 | 0.0932 | 0.0640 | -0.0293 | 0.0307 | 0.0277 |
| tcn_quantile | quantile | unseen_vessel | pitch_rate | 20 | 2.0000 | 0.9715 | 0.7398 | -0.2317 | 0.1727 | 0.1249 | -0.0478 | 0.0568 | 0.0540 |
| tcn_quantile | quantile | unseen_vessel | pitch_rate | 30 | 3.0000 | 0.9706 | 0.7948 | -0.1759 | 0.2444 | 0.1623 | -0.0820 | 0.0804 | 0.0702 |
| tcn_quantile | quantile | unseen_vessel | pitch_rate | 50 | 5.0000 | 0.9459 | 0.7188 | -0.2271 | 0.5215 | 0.3523 | -0.1692 | 0.1716 | 0.1523 |
| tcn_quantile | quantile | unseen_vessel | pitch_rate | 100 | 10.0000 | 0.9145 | 0.8342 | -0.0803 | 1.3029 | 0.9469 | -0.3560 | 0.4288 | 0.4092 |
| tcn_quantile | quantile | unseen_vessel | pitch_rate | 150 | 15.0000 | 0.9010 | 0.9097 | 0.0087 | 1.8681 | 1.4724 | -0.3958 | 0.6150 | 0.6360 |
| tcn_quantile | quantile | unseen_vessel | roll | 10 | 1.0000 | 0.9916 | 0.4810 | -0.5106 | 0.1112 | 0.0853 | -0.0259 | 0.0092 | 0.0120 |
| tcn_quantile | quantile | unseen_vessel | roll | 20 | 2.0000 | 0.9893 | 0.3780 | -0.6113 | 0.1963 | 0.1556 | -0.0406 | 0.0162 | 0.0219 |
| tcn_quantile | quantile | unseen_vessel | roll | 30 | 3.0000 | 0.9836 | 0.3282 | -0.6554 | 0.2938 | 0.2467 | -0.0471 | 0.0242 | 0.0347 |
| tcn_quantile | quantile | unseen_vessel | roll | 50 | 5.0000 | 0.9751 | 0.3278 | -0.6473 | 0.4093 | 0.3514 | -0.0578 | 0.0338 | 0.0494 |
| tcn_quantile | quantile | unseen_vessel | roll | 100 | 10.0000 | 0.9368 | 0.4576 | -0.4792 | 1.0699 | 0.9235 | -0.1465 | 0.0881 | 0.1298 |
| tcn_quantile | quantile | unseen_vessel | roll | 150 | 15.0000 | 0.9222 | 0.6001 | -0.3222 | 2.1321 | 1.8773 | -0.2548 | 0.1755 | 0.2636 |
| tcn_quantile | quantile | unseen_vessel | roll_rate | 10 | 1.0000 | 0.9805 | 0.4429 | -0.5375 | 0.1087 | 0.0865 | -0.0222 | 0.0168 | 0.0262 |
| tcn_quantile | quantile | unseen_vessel | roll_rate | 20 | 2.0000 | 0.9744 | 0.3929 | -0.5815 | 0.1638 | 0.1348 | -0.0290 | 0.0253 | 0.0409 |
| tcn_quantile | quantile | unseen_vessel | roll_rate | 30 | 3.0000 | 0.9734 | 0.4212 | -0.5522 | 0.1920 | 0.1670 | -0.0250 | 0.0296 | 0.0506 |
| tcn_quantile | quantile | unseen_vessel | roll_rate | 50 | 5.0000 | 0.9594 | 0.4296 | -0.5298 | 0.3244 | 0.2677 | -0.0566 | 0.0500 | 0.0811 |
| tcn_quantile | quantile | unseen_vessel | roll_rate | 100 | 10.0000 | 0.9250 | 0.5476 | -0.3774 | 0.8296 | 0.7110 | -0.1187 | 0.1278 | 0.2154 |
| tcn_quantile | quantile | unseen_vessel | roll_rate | 150 | 15.0000 | 0.9146 | 0.6717 | -0.2430 | 1.4414 | 1.2692 | -0.1722 | 0.2220 | 0.3846 |

## Notes

- Simulated results only. The generator has no process noise, so the achievable sharpness is unrealistically high and every interval here is narrower than one fitted to real deck motion would be (`docs/protocol.md` P4-D16).
- **Coverage is never a result on its own.** A wide enough interval covers everything. `width_ratio` is the reading that makes the width interpretable: it is the interval width over that of an unconditional interval matched to the scored partition's own spread, so **1.0 means no sharper than knowing only the variance**, the same device `nrmse` provides for RMSE (P4-D3).
- **The heads are not parameter-matched** to each other or to their Phase 4 point rows: the final projection widens with the head, so a quantile head carries ~9x the head parameters of a point one (P5-D5). A head-vs-head difference is not an architecture result.
- **`best_val_loss` is not comparable across heads** -- each model is early-stopped on its own objective, named in `val_loss_name` (P5-D4).
- The degradation table covers `unseen_heading`, `unseen_seastate`, `unseen_vessel` and is **reported, not fixed** -- it is the finding this phase exists to produce, and the same coverage-under-shift story that motivates conformal prediction. The deltas are **unpaired**: the two regimes score different realizations, so no common resample exists and no interval on a delta would be honest (P3-D13).
- On `unseen_heading`, coverage on pitch and pitch_rate is **predicted in advance** to be near 1.0 at meaningless width: that regime's test set is beam seas, where the pitch heading factor sits on the P1-D2 residual floor ~26 dB down, so a head fitted where pitch has amplitude emits intervals scaled to a signal the test set does not contain (P5-D6). Near-perfect coverage there is not calibration.
