# Statistical appendix

## Scenario-level numerical quality control

| scenario | method_trials | passing_method_trials | nonfinite_values | maximum_absolute_value |
| --- | --- | --- | --- | --- |
| latent_common_driver | 20 | 20 | 0 | 9.34 |
| order1_unconfounded | 20 | 20 | 0 | 9.25 |
| order3_unconfounded | 20 | 14 | 43690 | 1.484e+308 |
| variable_lag_unconfounded | 20 | 14 | 16426 | 1.526e+308 |

`passes_qc` requires all values to be finite and `max(abs(X)) < 1e+06`. Counts include both methods; method inputs were identical within each paired repeat.

## Descriptive statistics for retained scenarios

| method | scenario | endpoint | n | mean | sd | median | ci95_low | ci95_high | minimum | maximum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fast_gcstar_cgc | latent_common_driver | T_obs | 10 | 0.0333333 | 0.0157135 | 0.0333333 | 0.0244444 | 0.0433333 | 0.0111111 | 0.0666667 |
| fast_gcstar_cgc | latent_common_driver | cumulative_instability | 10 | 0.0866667 | 0.0428094 | 0.0777778 | 0.0622222 | 0.112222 | 0.0111111 | 0.155556 |
| fast_gcstar_cgc | order1_unconfounded | T_obs | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| fast_gcstar_cgc | order1_unconfounded | cumulative_instability | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| fast_gcstar_fcgc | latent_common_driver | T_obs | 10 | 0.0188889 | 0.0139074 | 0.0222222 | 0.0111111 | 0.0266667 | 0 | 0.0444444 |
| fast_gcstar_fcgc | latent_common_driver | cumulative_instability | 10 | 0.0444444 | 0.0335385 | 0.0444444 | 0.0255556 | 0.0644444 | 0 | 0.111111 |
| fast_gcstar_fcgc | order1_unconfounded | T_obs | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| fast_gcstar_fcgc | order1_unconfounded | cumulative_instability | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Paired comparisons

| comparison | method | endpoint | n_pairs | mean_difference | median_difference | ci95_low | ci95_high | positive_pairs | negative_pairs | zero_pairs | p_exact | p_holm_primary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| latent_common_driver - order1_unconfounded | fast_gcstar_cgc | T_obs | 10 | 0.0333333 | 0.0333333 | 0.0244444 | 0.0433333 | 10 | 0 | 0 | 0.00195312 | 0.00390625 |
| latent_common_driver - order1_unconfounded | fast_gcstar_cgc | cumulative_instability | 10 | 0.0866667 | 0.0777778 | 0.0622222 | 0.112222 | 10 | 0 | 0 | 0.00195312 |  |
| latent_common_driver - order1_unconfounded | fast_gcstar_fcgc | T_obs | 10 | 0.0188889 | 0.0222222 | 0.0111111 | 0.0266667 | 8 | 0 | 2 | 0.0078125 | 0.0078125 |
| latent_common_driver - order1_unconfounded | fast_gcstar_fcgc | cumulative_instability | 10 | 0.0444444 | 0.0444444 | 0.0255556 | 0.0644444 | 8 | 0 | 2 | 0.0078125 |  |
| c-GC - c-GC* within latent_common_driver | paired_methods | T_obs | 10 | 0.0144444 | 0.0111111 | 0.00333333 | 0.0255556 | 7 | 1 | 2 | 0.0703125 |  |
| c-GC - c-GC* within latent_common_driver | paired_methods | cumulative_instability | 10 | 0.0422222 | 0.0444444 | 0.0255556 | 0.0577778 | 9 | 0 | 1 | 0.00390625 |  |

The `p_holm_primary` column is populated only for the two prespecified `T_obs` comparisons. Method contrasts and cumulative-instability tests are secondary and are reported without multiplicity-adjusted claims.
