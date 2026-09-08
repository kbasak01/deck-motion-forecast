# Phase 3 baselines (Gate 3)

Simulated results only. Gate cell: **pitch at 10 s (100 samples), `unseen_heading` regime**, threshold 0.8 skill vs persistence.

Every number here traces to the CSVs it was written from: `results/e04/e04b_channels_ood/baselines.csv` (aggregated), `results/e04/e04b_channels_ood/baselines_by_seed.csv` (one row per run, the source of truth), `results/e04/e04b_channels_ood/baselines_by_cell.csv` (per grid cell) and `results/e04/e04b_channels_ood/baselines_controls.csv` (negative controls).

## The gate cell

| model | n_seeds | deterministic | rmse_mean | rmse_persistence | skill_mean | skill_std | skill_ci_lo | skill_ci_hi | n_params |
|---|---|---|---|---|---|---|---|---|---|
| ar20 | 1 | yes | 0.5069 | 0.1063 | -21.7403 | n/a | -23.1870 | -20.1501 | 27450 |
| damped_persistence | 1 | yes | 0.0936 | 0.1063 | 0.2247 | n/a | 0.2129 | 0.2373 | 3 |
| dlinear_ols | 1 | yes | 0.0706 | 0.1063 | 0.5587 | n/a | 0.5499 | 0.5673 | 60300 |
| persistence | 1 | yes | 0.1063 | 0.1063 | 0.0000 | n/a | 0.0000 | 0.0000 | 0 |
| window_mean | 1 | yes | 0.0936 | 0.1063 | 0.2246 | n/a | 0.2129 | 0.2373 | 0 |

`skill_std` is seed-to-seed spread; `skill_ci_lo`/`skill_ci_hi` are a bootstrap over held-out **realizations** and are what decide whether a value near the threshold is distinguishable from it -- on a single-seed row. On a multi-seed row they are the envelope of the per-run intervals, not a calibrated interval for the seed mean; see the caveats.

## Negative controls

Kept out of the headline table so they cannot be mistaken for models.

| control | subject_model | null_model | worst_excess | tol | passed |
|---|---|---|---|---|---|
| shuffle | shuffled | window_mean | 0.0028 | 0.0200 | yes |

The shuffle control's null is the **window-mean forecast**, not zero skill: a model fitted to time-shuffled targets degenerates to the conditional mean, which inverts to the window mean, and the window mean beats persistence at long horizons on a narrowband signal. Testing against zero would report leakage on a clean pipeline.

## Full table

### `unseen_heading`

| model | dof | horizon_s | n_seeds | rmse_mean | rmse_std | mae_mean | rmse_persistence | skill_mean | skill_std | skill_ci_lo | skill_ci_hi | nrmse_mean | n_params | fit_time_s_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ar20 | heave | 1.0000 | 1 | 0.0121 | n/a | 0.0085 | 0.4420 | 0.9993 | n/a | 0.9992 | 0.9993 | 0.0155 | 27450 | 0.0103 |
| ar20 | heave | 2.0000 | 1 | 0.0668 | n/a | 0.0464 | 0.8431 | 0.9937 | n/a | 0.9937 | 0.9938 | 0.0857 | 27450 | 0.0103 |
| ar20 | heave | 3.0000 | 1 | 0.1803 | n/a | 0.1247 | 1.1676 | 0.9761 | n/a | 0.9759 | 0.9764 | 0.2313 | 27450 | 0.0103 |
| ar20 | heave | 5.0000 | 1 | 0.3959 | n/a | 0.2751 | 1.4948 | 0.9299 | n/a | 0.9289 | 0.9307 | 0.5077 | 27450 | 0.0103 |
| ar20 | heave | 10.0000 | 1 | 0.6189 | n/a | 0.4365 | 0.8089 | 0.4145 | n/a | 0.3777 | 0.4436 | 0.7937 | 27450 | 0.0103 |
| ar20 | heave | 15.0000 | 1 | 0.7055 | n/a | 0.5032 | 1.1495 | 0.6234 | n/a | 0.6161 | 0.6295 | 0.9046 | 27450 | 0.0103 |
| ar20 | pitch | 1.0000 | 1 | 0.0559 | n/a | 0.0341 | 0.0642 | 0.2410 | n/a | 0.1763 | 0.3096 | 0.5993 | 27450 | 0.0103 |
| ar20 | pitch | 2.0000 | 1 | 0.3383 | n/a | 0.2055 | 0.1193 | -7.0369 | n/a | -7.7175 | -6.3028 | 3.6251 | 27450 | 0.0103 |
| ar20 | pitch | 3.0000 | 1 | 0.8394 | n/a | 0.5070 | 0.1581 | -27.1762 | n/a | -29.5268 | -24.5736 | 8.9957 | 27450 | 0.0103 |
| ar20 | pitch | 5.0000 | 1 | 0.9659 | n/a | 0.5894 | 0.1747 | -29.5702 | n/a | -31.8875 | -26.9862 | 10.3528 | 27450 | 0.0103 |
| ar20 | pitch | 10.0000 | 1 | 0.5069 | n/a | 0.3277 | 0.1063 | -21.7403 | n/a | -23.1870 | -20.1501 | 5.4343 | 27450 | 0.0103 |
| ar20 | pitch | 15.0000 | 1 | 0.4341 | n/a | 0.2973 | 0.1393 | -8.7108 | n/a | -9.2927 | -8.0927 | 4.6537 | 27450 | 0.0103 |
| ar20 | roll | 1.0000 | 1 | 0.0382 | n/a | 0.0278 | 2.6984 | 0.9998 | n/a | 0.9998 | 0.9998 | 0.0076 | 27450 | 0.0103 |
| ar20 | roll | 2.0000 | 1 | 0.1879 | n/a | 0.1363 | 5.1909 | 0.9987 | n/a | 0.9986 | 0.9988 | 0.0371 | 27450 | 0.0103 |
| ar20 | roll | 3.0000 | 1 | 0.4978 | n/a | 0.3584 | 7.2910 | 0.9953 | n/a | 0.9949 | 0.9957 | 0.0984 | 27450 | 0.0103 |
| ar20 | roll | 5.0000 | 1 | 1.3539 | n/a | 0.9490 | 9.7606 | 0.9808 | n/a | 0.9792 | 0.9820 | 0.2676 | 27450 | 0.0103 |
| ar20 | roll | 10.0000 | 1 | 1.6900 | n/a | 1.2422 | 5.0272 | 0.8870 | n/a | 0.8742 | 0.8965 | 0.3338 | 27450 | 0.0103 |
| ar20 | roll | 15.0000 | 1 | 2.4866 | n/a | 1.8028 | 7.3741 | 0.8863 | n/a | 0.8779 | 0.8933 | 0.4909 | 27450 | 0.0103 |
| damped_persistence | heave | 1.0000 | 1 | 0.5505 | n/a | 0.3602 | 0.4420 | -0.5513 | n/a | -0.5768 | -0.5212 | 0.7063 | 3 | 0.1222 |
| damped_persistence | heave | 2.0000 | 1 | 0.7902 | n/a | 0.5145 | 0.8431 | 0.1216 | n/a | 0.1071 | 0.1387 | 1.0137 | 3 | 0.1222 |
| damped_persistence | heave | 3.0000 | 1 | 0.8677 | n/a | 0.5615 | 1.1676 | 0.4477 | n/a | 0.4393 | 0.4578 | 1.1130 | 3 | 0.1222 |
| damped_persistence | heave | 5.0000 | 1 | 0.8284 | n/a | 0.5316 | 1.4948 | 0.6929 | n/a | 0.6908 | 0.6953 | 1.0624 | 3 | 0.1222 |
| damped_persistence | heave | 10.0000 | 1 | 0.7410 | n/a | 0.4891 | 0.8089 | 0.1608 | n/a | 0.1234 | 0.1916 | 0.9502 | 3 | 0.1222 |
| damped_persistence | heave | 15.0000 | 1 | 0.8280 | n/a | 0.5276 | 1.1495 | 0.4812 | n/a | 0.4651 | 0.4999 | 1.0616 | 3 | 0.1222 |
| damped_persistence | pitch | 1.0000 | 1 | 0.0732 | n/a | 0.0537 | 0.0642 | -0.3005 | n/a | -0.3159 | -0.2837 | 0.7845 | 3 | 0.1222 |
| damped_persistence | pitch | 2.0000 | 1 | 0.0979 | n/a | 0.0716 | 0.1193 | 0.3269 | n/a | 0.3189 | 0.3355 | 1.0491 | 3 | 0.1222 |
| damped_persistence | pitch | 3.0000 | 1 | 0.1021 | n/a | 0.0743 | 0.1581 | 0.5834 | n/a | 0.5788 | 0.5883 | 1.0938 | 3 | 0.1222 |
| damped_persistence | pitch | 5.0000 | 1 | 0.0940 | n/a | 0.0681 | 0.1747 | 0.7104 | n/a | 0.7097 | 0.7111 | 1.0077 | 3 | 0.1222 |
| damped_persistence | pitch | 10.0000 | 1 | 0.0936 | n/a | 0.0687 | 0.1063 | 0.2247 | n/a | 0.2129 | 0.2373 | 1.0034 | 3 | 0.1222 |
| damped_persistence | pitch | 15.0000 | 1 | 0.0949 | n/a | 0.0687 | 0.1393 | 0.5365 | n/a | 0.5279 | 0.5448 | 1.0167 | 3 | 0.1222 |
| damped_persistence | roll | 1.0000 | 1 | 3.4171 | n/a | 1.9996 | 2.6984 | -0.6036 | n/a | -0.6152 | -0.5883 | 0.6758 | 3 | 0.1222 |
| damped_persistence | roll | 2.0000 | 1 | 5.0673 | n/a | 2.9548 | 5.1909 | 0.0470 | n/a | 0.0403 | 0.0559 | 1.0018 | 3 | 0.1222 |
| damped_persistence | roll | 3.0000 | 1 | 5.6650 | n/a | 3.2895 | 7.2910 | 0.3963 | n/a | 0.3923 | 0.4014 | 1.1199 | 3 | 0.1222 |
| damped_persistence | roll | 5.0000 | 1 | 5.3594 | n/a | 3.1006 | 9.7606 | 0.6985 | n/a | 0.6974 | 0.6998 | 1.0594 | 3 | 0.1222 |
| damped_persistence | roll | 10.0000 | 1 | 4.7195 | n/a | 2.7949 | 5.0272 | 0.1187 | n/a | 0.0971 | 0.1353 | 0.9323 | 3 | 0.1222 |
| damped_persistence | roll | 15.0000 | 1 | 5.6361 | n/a | 3.2231 | 7.3741 | 0.4158 | n/a | 0.4055 | 0.4285 | 1.1127 | 3 | 0.1222 |
| dlinear_ols | heave | 1.0000 | 1 | 0.0024 | n/a | 0.0016 | 0.4420 | 1.0000 | n/a | 1.0000 | 1.0000 | 0.0031 | 60300 | 0.1990 |
| dlinear_ols | heave | 2.0000 | 1 | 0.0233 | n/a | 0.0155 | 0.8431 | 0.9992 | n/a | 0.9992 | 0.9993 | 0.0299 | 60300 | 0.1990 |
| dlinear_ols | heave | 3.0000 | 1 | 0.0866 | n/a | 0.0572 | 1.1676 | 0.9945 | n/a | 0.9944 | 0.9946 | 0.1111 | 60300 | 0.1990 |
| dlinear_ols | heave | 5.0000 | 1 | 0.2287 | n/a | 0.1482 | 1.4948 | 0.9766 | n/a | 0.9761 | 0.9770 | 0.2934 | 60300 | 0.1990 |
| dlinear_ols | heave | 10.0000 | 1 | 0.4968 | n/a | 0.3336 | 0.8089 | 0.6228 | n/a | 0.6048 | 0.6379 | 0.6370 | 60300 | 0.1990 |
| dlinear_ols | heave | 15.0000 | 1 | 0.6813 | n/a | 0.4633 | 1.1495 | 0.6488 | n/a | 0.6440 | 0.6529 | 0.8735 | 60300 | 0.1990 |
| dlinear_ols | pitch | 1.0000 | 1 | 0.0004 | n/a | 0.0003 | 0.0642 | 1.0000 | n/a | 1.0000 | 1.0000 | 0.0038 | 60300 | 0.1990 |
| dlinear_ols | pitch | 2.0000 | 1 | 0.0034 | n/a | 0.0026 | 0.1193 | 0.9992 | n/a | 0.9992 | 0.9992 | 0.0366 | 60300 | 0.1990 |
| dlinear_ols | pitch | 3.0000 | 1 | 0.0123 | n/a | 0.0093 | 0.1581 | 0.9939 | n/a | 0.9938 | 0.9940 | 0.1323 | 60300 | 0.1990 |
| dlinear_ols | pitch | 5.0000 | 1 | 0.0295 | n/a | 0.0220 | 0.1747 | 0.9715 | n/a | 0.9710 | 0.9719 | 0.3161 | 60300 | 0.1990 |
| dlinear_ols | pitch | 10.0000 | 1 | 0.0706 | n/a | 0.0529 | 0.1063 | 0.5587 | n/a | 0.5499 | 0.5673 | 0.7570 | 60300 | 0.1990 |
| dlinear_ols | pitch | 15.0000 | 1 | 0.0936 | n/a | 0.0695 | 0.1393 | 0.5485 | n/a | 0.5413 | 0.5549 | 1.0034 | 60300 | 0.1990 |
| dlinear_ols | roll | 1.0000 | 1 | 0.0111 | n/a | 0.0070 | 2.6984 | 1.0000 | n/a | 1.0000 | 1.0000 | 0.0022 | 60300 | 0.1990 |
| dlinear_ols | roll | 2.0000 | 1 | 0.1078 | n/a | 0.0674 | 5.1909 | 0.9996 | n/a | 0.9996 | 0.9996 | 0.0213 | 60300 | 0.1990 |
| dlinear_ols | roll | 3.0000 | 1 | 0.3996 | n/a | 0.2492 | 7.2910 | 0.9970 | n/a | 0.9969 | 0.9970 | 0.0790 | 60300 | 0.1990 |
| dlinear_ols | roll | 5.0000 | 1 | 1.0434 | n/a | 0.6445 | 9.7606 | 0.9886 | n/a | 0.9883 | 0.9888 | 0.2063 | 60300 | 0.1990 |
| dlinear_ols | roll | 10.0000 | 1 | 2.3904 | n/a | 1.4985 | 5.0272 | 0.7739 | n/a | 0.7647 | 0.7811 | 0.4722 | 60300 | 0.1990 |
| dlinear_ols | roll | 15.0000 | 1 | 3.4577 | n/a | 2.1661 | 7.3741 | 0.7801 | n/a | 0.7767 | 0.7832 | 0.6826 | 60300 | 0.1990 |
| persistence | heave | 1.0000 | 1 | 0.4420 | n/a | 0.3006 | 0.4420 | 0.0000 | n/a | 0.0000 | 0.0000 | 0.5671 | 0 | 0.0000 |
| persistence | heave | 2.0000 | 1 | 0.8431 | n/a | 0.5708 | 0.8431 | 0.0000 | n/a | 0.0000 | 0.0000 | 1.0815 | 0 | 0.0000 |
| persistence | heave | 3.0000 | 1 | 1.1676 | n/a | 0.7839 | 1.1676 | 0.0000 | n/a | 0.0000 | 0.0000 | 1.4977 | 0 | 0.0000 |
| persistence | heave | 5.0000 | 1 | 1.4948 | n/a | 0.9719 | 1.4948 | 0.0000 | n/a | 0.0000 | 0.0000 | 1.9171 | 0 | 0.0000 |
| persistence | heave | 10.0000 | 1 | 0.8089 | n/a | 0.4965 | 0.8089 | 0.0000 | n/a | 0.0000 | 0.0000 | 1.0372 | 0 | 0.0000 |
| persistence | heave | 15.0000 | 1 | 1.1495 | n/a | 0.7647 | 1.1495 | 0.0000 | n/a | 0.0000 | 0.0000 | 1.4739 | 0 | 0.0000 |
| persistence | pitch | 1.0000 | 1 | 0.0642 | n/a | 0.0479 | 0.0642 | 0.0000 | n/a | 0.0000 | 0.0000 | 0.6879 | 0 | 0.0000 |
| persistence | pitch | 2.0000 | 1 | 0.1193 | n/a | 0.0888 | 0.1193 | 0.0000 | n/a | 0.0000 | 0.0000 | 1.2787 | 0 | 0.0000 |
| persistence | pitch | 3.0000 | 1 | 0.1581 | n/a | 0.1171 | 0.1581 | 0.0000 | n/a | 0.0000 | 0.0000 | 1.6947 | 0 | 0.0000 |
| persistence | pitch | 5.0000 | 1 | 0.1747 | n/a | 0.1264 | 0.1747 | 0.0000 | n/a | 0.0000 | 0.0000 | 1.8724 | 0 | 0.0000 |
| persistence | pitch | 10.0000 | 1 | 0.1063 | n/a | 0.0799 | 0.1063 | 0.0000 | n/a | 0.0000 | 0.0000 | 1.1396 | 0 | 0.0000 |
| persistence | pitch | 15.0000 | 1 | 0.1393 | n/a | 0.0991 | 0.1393 | 0.0000 | n/a | 0.0000 | 0.0000 | 1.4934 | 0 | 0.0000 |
| persistence | roll | 1.0000 | 1 | 2.6984 | n/a | 1.6330 | 2.6984 | 0.0000 | n/a | 0.0000 | 0.0000 | 0.5336 | 0 | 0.0000 |
| persistence | roll | 2.0000 | 1 | 5.1909 | n/a | 3.1275 | 5.1909 | 0.0000 | n/a | 0.0000 | 0.0000 | 1.0263 | 0 | 0.0000 |
| persistence | roll | 3.0000 | 1 | 7.2910 | n/a | 4.3604 | 7.2910 | 0.0000 | n/a | 0.0000 | 0.0000 | 1.4413 | 0 | 0.0000 |
| persistence | roll | 5.0000 | 1 | 9.7606 | n/a | 5.6951 | 9.7606 | 0.0000 | n/a | 0.0000 | 0.0000 | 1.9295 | 0 | 0.0000 |
| persistence | roll | 10.0000 | 1 | 5.0272 | n/a | 2.8028 | 5.0272 | 0.0000 | n/a | 0.0000 | 0.0000 | 0.9931 | 0 | 0.0000 |
| persistence | roll | 15.0000 | 1 | 7.3741 | n/a | 4.4408 | 7.3741 | 0.0000 | n/a | 0.0000 | 0.0000 | 1.4558 | 0 | 0.0000 |
| window_mean | heave | 1.0000 | 1 | 0.8377 | n/a | 0.5374 | 0.4420 | -2.5923 | n/a | -2.6730 | -2.4972 | 1.0748 | 0 | 0.0000 |
| window_mean | heave | 2.0000 | 1 | 0.8616 | n/a | 0.5522 | 0.8431 | -0.0444 | n/a | -0.0667 | -0.0178 | 1.1053 | 0 | 0.0000 |
| window_mean | heave | 3.0000 | 1 | 0.8649 | n/a | 0.5541 | 1.1676 | 0.4512 | n/a | 0.4409 | 0.4634 | 1.1095 | 0 | 0.0000 |
| window_mean | heave | 5.0000 | 1 | 0.8147 | n/a | 0.5222 | 1.4948 | 0.7030 | n/a | 0.7008 | 0.7055 | 1.0448 | 0 | 0.0000 |
| window_mean | heave | 10.0000 | 1 | 0.7412 | n/a | 0.4893 | 0.8089 | 0.1604 | n/a | 0.1229 | 0.1912 | 0.9504 | 0 | 0.0000 |
| window_mean | heave | 15.0000 | 1 | 0.8280 | n/a | 0.5276 | 1.1495 | 0.4812 | n/a | 0.4651 | 0.4999 | 1.0616 | 0 | 0.0000 |
| window_mean | pitch | 1.0000 | 1 | 0.0982 | n/a | 0.0714 | 0.0642 | -1.3382 | n/a | -1.3777 | -1.2956 | 1.0519 | 0 | 0.0000 |
| window_mean | pitch | 2.0000 | 1 | 0.1005 | n/a | 0.0731 | 0.1193 | 0.2909 | n/a | 0.2799 | 0.3026 | 1.0768 | 0 | 0.0000 |
| window_mean | pitch | 3.0000 | 1 | 0.1000 | n/a | 0.0726 | 0.1581 | 0.5998 | n/a | 0.5947 | 0.6053 | 1.0721 | 0 | 0.0000 |
| window_mean | pitch | 5.0000 | 1 | 0.0934 | n/a | 0.0677 | 0.1747 | 0.7144 | n/a | 0.7137 | 0.7152 | 1.0006 | 0 | 0.0000 |
| window_mean | pitch | 10.0000 | 1 | 0.0936 | n/a | 0.0687 | 0.1063 | 0.2246 | n/a | 0.2129 | 0.2373 | 1.0035 | 0 | 0.0000 |
| window_mean | pitch | 15.0000 | 1 | 0.0949 | n/a | 0.0687 | 0.1393 | 0.5365 | n/a | 0.5279 | 0.5448 | 1.0167 | 0 | 0.0000 |
| window_mean | roll | 1.0000 | 1 | 5.6547 | n/a | 3.2550 | 2.6984 | -3.3914 | n/a | -3.4342 | -3.3350 | 1.1183 | 0 | 0.0000 |
| window_mean | roll | 2.0000 | 1 | 5.7716 | n/a | 3.3177 | 5.1909 | -0.2362 | n/a | -0.2480 | -0.2207 | 1.1411 | 0 | 0.0000 |
| window_mean | roll | 3.0000 | 1 | 5.7216 | n/a | 3.2888 | 7.2910 | 0.3842 | n/a | 0.3790 | 0.3909 | 1.1310 | 0 | 0.0000 |
| window_mean | roll | 5.0000 | 1 | 5.2018 | n/a | 3.0053 | 9.7606 | 0.7160 | n/a | 0.7148 | 0.7174 | 1.0283 | 0 | 0.0000 |
| window_mean | roll | 10.0000 | 1 | 4.7242 | n/a | 2.7978 | 5.0272 | 0.1169 | n/a | 0.0953 | 0.1336 | 0.9332 | 0 | 0.0000 |
| window_mean | roll | 15.0000 | 1 | 5.6361 | n/a | 3.2231 | 7.3741 | 0.4158 | n/a | 0.4055 | 0.4285 | 1.1127 | 0 | 0.0000 |

### `unseen_vessel`

| model | dof | horizon_s | n_seeds | rmse_mean | rmse_std | mae_mean | rmse_persistence | skill_mean | skill_std | skill_ci_lo | skill_ci_hi | nrmse_mean | n_params | fit_time_s_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ar20 | heave | 1.0000 | 1 | 0.0052 | n/a | 0.0034 | 0.3614 | 0.9998 | n/a | 0.9998 | 0.9998 | 0.0079 | 27450 | 0.0122 |
| ar20 | heave | 2.0000 | 1 | 0.0286 | n/a | 0.0187 | 0.6908 | 0.9983 | n/a | 0.9980 | 0.9985 | 0.0438 | 27450 | 0.0122 |
| ar20 | heave | 3.0000 | 1 | 0.0812 | n/a | 0.0526 | 0.9601 | 0.9928 | n/a | 0.9919 | 0.9937 | 0.1241 | 27450 | 0.0122 |
| ar20 | heave | 5.0000 | 1 | 0.2408 | n/a | 0.1512 | 1.2448 | 0.9626 | n/a | 0.9594 | 0.9658 | 0.3680 | 27450 | 0.0122 |
| ar20 | heave | 10.0000 | 1 | 0.4259 | n/a | 0.2619 | 0.7165 | 0.6466 | n/a | 0.6178 | 0.6743 | 0.6509 | 27450 | 0.0122 |
| ar20 | heave | 15.0000 | 1 | 0.5366 | n/a | 0.3387 | 0.9502 | 0.6811 | n/a | 0.6523 | 0.7066 | 0.8200 | 27450 | 0.0122 |
| ar20 | pitch | 1.0000 | 1 | 0.0199 | n/a | 0.0131 | 0.6874 | 0.9992 | n/a | 0.9990 | 0.9993 | 0.0189 | 27450 | 0.0122 |
| ar20 | pitch | 2.0000 | 1 | 0.1243 | n/a | 0.0814 | 1.2853 | 0.9906 | n/a | 0.9893 | 0.9919 | 0.1178 | 27450 | 0.0122 |
| ar20 | pitch | 3.0000 | 1 | 0.3226 | n/a | 0.2078 | 1.7223 | 0.9649 | n/a | 0.9600 | 0.9694 | 0.3057 | 27450 | 0.0122 |
| ar20 | pitch | 5.0000 | 1 | 0.5546 | n/a | 0.3407 | 1.9910 | 0.9224 | n/a | 0.9133 | 0.9308 | 0.5254 | 27450 | 0.0122 |
| ar20 | pitch | 10.0000 | 1 | 0.8592 | n/a | 0.5267 | 1.1519 | 0.4437 | n/a | 0.4228 | 0.4664 | 0.8134 | 27450 | 0.0122 |
| ar20 | pitch | 15.0000 | 1 | 1.0387 | n/a | 0.6310 | 1.5950 | 0.5759 | n/a | 0.5522 | 0.5961 | 0.9830 | 27450 | 0.0122 |
| ar20 | roll | 1.0000 | 1 | 0.0256 | n/a | 0.0145 | 0.9927 | 0.9993 | n/a | 0.9993 | 0.9994 | 0.0118 | 27450 | 0.0122 |
| ar20 | roll | 2.0000 | 1 | 0.1322 | n/a | 0.0737 | 1.9286 | 0.9953 | n/a | 0.9949 | 0.9956 | 0.0611 | 27450 | 0.0122 |
| ar20 | roll | 3.0000 | 1 | 0.3644 | n/a | 0.1998 | 2.7551 | 0.9825 | n/a | 0.9811 | 0.9836 | 0.1685 | 27450 | 0.0122 |
| ar20 | roll | 5.0000 | 1 | 1.0666 | n/a | 0.5617 | 3.9136 | 0.9257 | n/a | 0.9209 | 0.9294 | 0.4932 | 27450 | 0.0122 |
| ar20 | roll | 10.0000 | 1 | 2.0565 | n/a | 1.0113 | 3.2712 | 0.6048 | n/a | 0.5863 | 0.6182 | 0.9507 | 27450 | 0.0122 |
| ar20 | roll | 15.0000 | 1 | 2.2008 | n/a | 1.1061 | 1.6783 | -0.7195 | n/a | -0.9564 | -0.4814 | 1.0167 | 27450 | 0.0122 |
| damped_persistence | heave | 1.0000 | 1 | 0.4536 | n/a | 0.2746 | 0.3614 | -0.5758 | n/a | -0.6192 | -0.5327 | 0.6938 | 3 | 0.1184 |
| damped_persistence | heave | 2.0000 | 1 | 0.6544 | n/a | 0.3946 | 0.6908 | 0.1026 | n/a | 0.0770 | 0.1288 | 1.0007 | 3 | 0.1184 |
| damped_persistence | heave | 3.0000 | 1 | 0.7227 | n/a | 0.4335 | 0.9601 | 0.4333 | n/a | 0.4168 | 0.4501 | 1.1048 | 3 | 0.1184 |
| damped_persistence | heave | 5.0000 | 1 | 0.6998 | n/a | 0.4165 | 1.2448 | 0.6839 | n/a | 0.6768 | 0.6909 | 1.0696 | 3 | 0.1184 |
| damped_persistence | heave | 10.0000 | 1 | 0.6241 | n/a | 0.3787 | 0.7165 | 0.2413 | n/a | 0.1591 | 0.3062 | 0.9538 | 3 | 0.1184 |
| damped_persistence | heave | 15.0000 | 1 | 0.6859 | n/a | 0.4076 | 0.9502 | 0.4790 | n/a | 0.4442 | 0.5124 | 1.0481 | 3 | 0.1184 |
| damped_persistence | pitch | 1.0000 | 1 | 0.8143 | n/a | 0.4917 | 0.6874 | -0.4034 | n/a | -0.4473 | -0.3607 | 0.7719 | 3 | 0.1184 |
| damped_persistence | pitch | 2.0000 | 1 | 1.0961 | n/a | 0.6597 | 1.2853 | 0.2728 | n/a | 0.2496 | 0.2952 | 1.0388 | 3 | 0.1184 |
| damped_persistence | pitch | 3.0000 | 1 | 1.1536 | n/a | 0.6911 | 1.7223 | 0.5514 | n/a | 0.5379 | 0.5647 | 1.0929 | 3 | 0.1184 |
| damped_persistence | pitch | 5.0000 | 1 | 1.0797 | n/a | 0.6434 | 1.9910 | 0.7059 | n/a | 0.7004 | 0.7109 | 1.0228 | 3 | 0.1184 |
| damped_persistence | pitch | 10.0000 | 1 | 1.0479 | n/a | 0.6289 | 1.1519 | 0.1724 | n/a | 0.1095 | 0.2336 | 0.9921 | 3 | 0.1184 |
| damped_persistence | pitch | 15.0000 | 1 | 1.0770 | n/a | 0.6436 | 1.5950 | 0.5440 | n/a | 0.5207 | 0.5639 | 1.0193 | 3 | 0.1184 |
| damped_persistence | roll | 1.0000 | 1 | 1.3744 | n/a | 0.6311 | 0.9927 | -0.9168 | n/a | -0.9376 | -0.8904 | 0.6352 | 3 | 0.1184 |
| damped_persistence | roll | 2.0000 | 1 | 2.1001 | n/a | 0.9562 | 1.9286 | -0.1858 | n/a | -0.2011 | -0.1665 | 0.9708 | 3 | 0.1184 |
| damped_persistence | roll | 3.0000 | 1 | 2.4447 | n/a | 1.1012 | 2.7551 | 0.2127 | n/a | 0.2009 | 0.2277 | 1.1305 | 3 | 0.1184 |
| damped_persistence | roll | 5.0000 | 1 | 2.5484 | n/a | 1.1245 | 3.9136 | 0.5760 | n/a | 0.5684 | 0.5859 | 1.1784 | 3 | 0.1184 |
| damped_persistence | roll | 10.0000 | 1 | 1.8944 | n/a | 0.8664 | 3.2712 | 0.6646 | n/a | 0.6504 | 0.6750 | 0.8757 | 3 | 0.1184 |
| damped_persistence | roll | 15.0000 | 1 | 2.2556 | n/a | 1.0221 | 1.6783 | -0.8062 | n/a | -1.0259 | -0.5900 | 1.0420 | 3 | 0.1184 |
| dlinear_ols | heave | 1.0000 | 1 | 0.0027 | n/a | 0.0014 | 0.3614 | 0.9999 | n/a | 0.9999 | 1.0000 | 0.0041 | 60300 | 0.1714 |
| dlinear_ols | heave | 2.0000 | 1 | 0.0259 | n/a | 0.0137 | 0.6908 | 0.9986 | n/a | 0.9982 | 0.9989 | 0.0396 | 60300 | 0.1714 |
| dlinear_ols | heave | 3.0000 | 1 | 0.0966 | n/a | 0.0508 | 0.9601 | 0.9899 | n/a | 0.9872 | 0.9924 | 0.1476 | 60300 | 0.1714 |
| dlinear_ols | heave | 5.0000 | 1 | 0.2591 | n/a | 0.1325 | 1.2448 | 0.9567 | n/a | 0.9446 | 0.9679 | 0.3960 | 60300 | 0.1714 |
| dlinear_ols | heave | 10.0000 | 1 | 0.4958 | n/a | 0.2801 | 0.7165 | 0.5212 | n/a | 0.4608 | 0.5797 | 0.7577 | 60300 | 0.1714 |
| dlinear_ols | heave | 15.0000 | 1 | 0.6174 | n/a | 0.3707 | 0.9502 | 0.5778 | n/a | 0.5252 | 0.6265 | 0.9434 | 60300 | 0.1714 |
| dlinear_ols | pitch | 1.0000 | 1 | 0.0043 | n/a | 0.0026 | 0.6874 | 1.0000 | n/a | 1.0000 | 1.0000 | 0.0041 | 60300 | 0.1714 |
| dlinear_ols | pitch | 2.0000 | 1 | 0.0410 | n/a | 0.0246 | 1.2853 | 0.9990 | n/a | 0.9989 | 0.9991 | 0.0389 | 60300 | 0.1714 |
| dlinear_ols | pitch | 3.0000 | 1 | 0.1474 | n/a | 0.0883 | 1.7223 | 0.9927 | n/a | 0.9918 | 0.9934 | 0.1396 | 60300 | 0.1714 |
| dlinear_ols | pitch | 5.0000 | 1 | 0.3522 | n/a | 0.2091 | 1.9910 | 0.9687 | n/a | 0.9646 | 0.9723 | 0.3336 | 60300 | 0.1714 |
| dlinear_ols | pitch | 10.0000 | 1 | 0.7858 | n/a | 0.4767 | 1.1519 | 0.5346 | n/a | 0.5141 | 0.5587 | 0.7440 | 60300 | 0.1714 |
| dlinear_ols | pitch | 15.0000 | 1 | 1.0426 | n/a | 0.6274 | 1.5950 | 0.5727 | n/a | 0.5428 | 0.5989 | 0.9867 | 60300 | 0.1714 |
| dlinear_ols | roll | 1.0000 | 1 | 0.0076 | n/a | 0.0031 | 0.9927 | 0.9999 | n/a | 0.9999 | 1.0000 | 0.0035 | 60300 | 0.1714 |
| dlinear_ols | roll | 2.0000 | 1 | 0.0732 | n/a | 0.0295 | 1.9286 | 0.9986 | n/a | 0.9983 | 0.9989 | 0.0339 | 60300 | 0.1714 |
| dlinear_ols | roll | 3.0000 | 1 | 0.2708 | n/a | 0.1090 | 2.7551 | 0.9903 | n/a | 0.9884 | 0.9924 | 0.1252 | 60300 | 0.1714 |
| dlinear_ols | roll | 5.0000 | 1 | 0.7087 | n/a | 0.2848 | 3.9136 | 0.9672 | n/a | 0.9611 | 0.9738 | 0.3277 | 60300 | 0.1714 |
| dlinear_ols | roll | 10.0000 | 1 | 1.4823 | n/a | 0.6303 | 3.2712 | 0.7947 | n/a | 0.7776 | 0.8136 | 0.6852 | 60300 | 0.1714 |
| dlinear_ols | roll | 15.0000 | 1 | 1.9508 | n/a | 0.8570 | 1.6783 | -0.3511 | n/a | -0.6166 | -0.0781 | 0.9012 | 60300 | 0.1714 |
| persistence | heave | 1.0000 | 1 | 0.3614 | n/a | 0.2245 | 0.3614 | 0.0000 | n/a | 0.0000 | 0.0000 | 0.5527 | 0 | 0.0000 |
| persistence | heave | 2.0000 | 1 | 0.6908 | n/a | 0.4276 | 0.6908 | 0.0000 | n/a | 0.0000 | 0.0000 | 1.0564 | 0 | 0.0000 |
| persistence | heave | 3.0000 | 1 | 0.9601 | n/a | 0.5905 | 0.9601 | 0.0000 | n/a | 0.0000 | 0.0000 | 1.4677 | 0 | 0.0000 |
| persistence | heave | 5.0000 | 1 | 1.2448 | n/a | 0.7470 | 1.2448 | 0.0000 | n/a | 0.0000 | 0.0000 | 1.9026 | 0 | 0.0000 |
| persistence | heave | 10.0000 | 1 | 0.7165 | n/a | 0.4044 | 0.7165 | 0.0000 | n/a | 0.0000 | 0.0000 | 1.0950 | 0 | 0.0000 |
| persistence | heave | 15.0000 | 1 | 0.9502 | n/a | 0.5694 | 0.9502 | 0.0000 | n/a | 0.0000 | 0.0000 | 1.4520 | 0 | 0.0000 |
| persistence | pitch | 1.0000 | 1 | 0.6874 | n/a | 0.4189 | 0.6874 | 0.0000 | n/a | 0.0000 | 0.0000 | 0.6516 | 0 | 0.0000 |
| persistence | pitch | 2.0000 | 1 | 1.2853 | n/a | 0.7815 | 1.2853 | 0.0000 | n/a | 0.0000 | 0.0000 | 1.2182 | 0 | 0.0000 |
| persistence | pitch | 3.0000 | 1 | 1.7223 | n/a | 1.0423 | 1.7223 | 0.0000 | n/a | 0.0000 | 0.0000 | 1.6318 | 0 | 0.0000 |
| persistence | pitch | 5.0000 | 1 | 1.9910 | n/a | 1.1746 | 1.9910 | 0.0000 | n/a | 0.0000 | 0.0000 | 1.8860 | 0 | 0.0000 |
| persistence | pitch | 10.0000 | 1 | 1.1519 | n/a | 0.7000 | 1.1519 | 0.0000 | n/a | 0.0000 | 0.0000 | 1.0906 | 0 | 0.0000 |
| persistence | pitch | 15.0000 | 1 | 1.5950 | n/a | 0.9220 | 1.5950 | 0.0000 | n/a | 0.0000 | 0.0000 | 1.5094 | 0 | 0.0000 |
| persistence | roll | 1.0000 | 1 | 0.9927 | n/a | 0.4759 | 0.9927 | 0.0000 | n/a | 0.0000 | 0.0000 | 0.4588 | 0 | 0.0000 |
| persistence | roll | 2.0000 | 1 | 1.9286 | n/a | 0.9188 | 1.9286 | 0.0000 | n/a | 0.0000 | 0.0000 | 0.8915 | 0 | 0.0000 |
| persistence | roll | 3.0000 | 1 | 2.7551 | n/a | 1.2991 | 2.7551 | 0.0000 | n/a | 0.0000 | 0.0000 | 1.2740 | 0 | 0.0000 |
| persistence | roll | 5.0000 | 1 | 3.9136 | n/a | 1.7857 | 3.9136 | 0.0000 | n/a | 0.0000 | 0.0000 | 1.8097 | 0 | 0.0000 |
| persistence | roll | 10.0000 | 1 | 3.2712 | n/a | 1.3657 | 3.2712 | 0.0000 | n/a | 0.0000 | 0.0000 | 1.5122 | 0 | 0.0000 |
| persistence | roll | 15.0000 | 1 | 1.6783 | n/a | 0.8838 | 1.6783 | 0.0000 | n/a | 0.0000 | 0.0000 | 0.7754 | 0 | 0.0000 |
| window_mean | heave | 1.0000 | 1 | 0.6974 | n/a | 0.4157 | 0.3614 | -2.7246 | n/a | -2.8593 | -2.5888 | 1.0666 | 0 | 0.0000 |
| window_mean | heave | 2.0000 | 1 | 0.7176 | n/a | 0.4275 | 0.6908 | -0.0791 | n/a | -0.1182 | -0.0396 | 1.0973 | 0 | 0.0000 |
| window_mean | heave | 3.0000 | 1 | 0.7224 | n/a | 0.4299 | 0.9601 | 0.4339 | n/a | 0.4142 | 0.4538 | 1.1043 | 0 | 0.0000 |
| window_mean | heave | 5.0000 | 1 | 0.6883 | n/a | 0.4093 | 1.2448 | 0.6943 | n/a | 0.6869 | 0.7014 | 1.0520 | 0 | 0.0000 |
| window_mean | heave | 10.0000 | 1 | 0.6243 | n/a | 0.3788 | 0.7165 | 0.2409 | n/a | 0.1586 | 0.3059 | 0.9540 | 0 | 0.0000 |
| window_mean | heave | 15.0000 | 1 | 0.6859 | n/a | 0.4076 | 0.9502 | 0.4790 | n/a | 0.4442 | 0.5124 | 1.0481 | 0 | 0.0000 |
| window_mean | pitch | 1.0000 | 1 | 1.1096 | n/a | 0.6640 | 0.6874 | -1.6058 | n/a | -1.7158 | -1.4973 | 1.0519 | 0 | 0.0000 |
| window_mean | pitch | 2.0000 | 1 | 1.1374 | n/a | 0.6805 | 1.2853 | 0.2169 | n/a | 0.1855 | 0.2469 | 1.0780 | 0 | 0.0000 |
| window_mean | pitch | 3.0000 | 1 | 1.1372 | n/a | 0.6793 | 1.7223 | 0.5640 | n/a | 0.5490 | 0.5787 | 1.0775 | 0 | 0.0000 |
| window_mean | pitch | 5.0000 | 1 | 1.0723 | n/a | 0.6392 | 1.9910 | 0.7100 | n/a | 0.7044 | 0.7150 | 1.0157 | 0 | 0.0000 |
| window_mean | pitch | 10.0000 | 1 | 1.0480 | n/a | 0.6289 | 1.1519 | 0.1724 | n/a | 0.1095 | 0.2335 | 0.9922 | 0 | 0.0000 |
| window_mean | pitch | 15.0000 | 1 | 1.0770 | n/a | 0.6436 | 1.5950 | 0.5440 | n/a | 0.5207 | 0.5639 | 1.0193 | 0 | 0.0000 |
| window_mean | roll | 1.0000 | 1 | 2.2879 | n/a | 1.0349 | 0.9927 | -4.3113 | n/a | -4.3792 | -4.2261 | 1.0574 | 0 | 0.0000 |
| window_mean | roll | 2.0000 | 1 | 2.4302 | n/a | 1.0917 | 1.9286 | -0.5878 | n/a | -0.6119 | -0.5579 | 1.1234 | 0 | 0.0000 |
| window_mean | roll | 3.0000 | 1 | 2.5199 | n/a | 1.1237 | 2.7551 | 0.1634 | n/a | 0.1492 | 0.1817 | 1.1653 | 0 | 0.0000 |
| window_mean | roll | 5.0000 | 1 | 2.5111 | n/a | 1.1050 | 3.9136 | 0.5883 | n/a | 0.5805 | 0.5985 | 1.1612 | 0 | 0.0000 |
| window_mean | roll | 10.0000 | 1 | 1.8933 | n/a | 0.8662 | 3.2712 | 0.6650 | n/a | 0.6507 | 0.6754 | 0.8752 | 0 | 0.0000 |
| window_mean | roll | 15.0000 | 1 | 2.2557 | n/a | 1.0221 | 1.6783 | -0.8063 | n/a | -1.0260 | -0.5901 | 1.0421 | 0 | 0.0000 |

## Caveats

- **These are simulated results.** No real deck data is used anywhere in this project.
- **`unseen_heading` pitch sits on the P1-D2 residual floor.** That regime's test set *is* beam seas, where the pitch heading factor is floored at `eps = 0.05` -- about 26 dB below its maximum -- and is therefore driven by an engineering stand-in for hull asymmetry rather than by the pitch physics. Its persistence RMSE is correspondingly tiny and its skill is dominated by that stand-in. Never quote that cell without this sentence.
- **The `id` regime pools 4 headings x 4 sea states x 3 speeds**, and per P1-D2 roll in head seas is on the same residual floor. A pooled roll skill of 0.85 could be 0.93 at beam and 0.6 at head, or the reverse; `baselines_by_cell.csv` is what distinguishes them, and the follow-on decision depends on which it is.
- **`n_windows` is a window count, not an independent-sample count.** The 542 880 windows of `unseen_heading/test` reported here are cut at a stride far shorter than the lookback, so consecutive windows share almost all of their input samples and their forecast targets overlap. They come from a much smaller number of independently simulated realizations -- `baselines_by_seed.csv` carries that count in `n_realizations` -- which is why `skill_ci_lo`/`skill_ci_hi` bootstrap whole realizations and never windows.
- **`nrmse_mean` is there because skill is not comparable across horizons on this signal.** Persistence error tracks the target's autocorrelation, so the skill denominator oscillates with the signal's own period instead of growing with lead time: on `id`/test in the `ideal` mode, persistence RMSE for roll *falls* from 7.09 deg at 50 samples (5 s) to 3.79 deg at 100 (10 s), because 10 s is about one roll period (P3-D5/P3-D6). A skill-vs-horizon curve therefore has dips that belong to the reference and not to the model. `nrmse_mean` is `rmse_mean / signal_std`, where `signal_std` is the standard deviation of the held-out target itself at that lead time in corpus units: 1.0 means no better than predicting the partition mean, lower is better, and it is the column to read when the horizon or the vessel varies.
- **`fit_time_s_mean` is not apples-to-apples.** It is CPU wall-clock for the closed-form models and GPU wall-clock *including data loading* for the SGD-fitted ones. No throughput claim is made from it.
- **`imu` observation mode is not a drop-in comparison.** Per P1-D6/P2-D8, `imu` inputs must be scored against `imu` targets, which changes the persistence denominator; skill scores across observation modes are therefore not comparable.
