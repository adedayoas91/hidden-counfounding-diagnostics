

def scenario_omitted_lag_order(
    *,
    T: int = 2000,
    d: int = 10,
    true_order: int = 3,
    noise_scale: float = 1.0,
    seed: int = 0,
) -> ScenarioResult:
    """Higher-order VAR system observed with misspecified lag order.

    Mechanism: If true data is VAR(3) but analyst assumes VAR(1),
    the omitted lags appear as spurious confounding structure.
    """
    rng = np.random.default_rng(seed)
    
    if true_order == 1:
        a1 = scale_to_stability(make_sparse_matrix(d, edge_prob=0.10, seed=seed + 1) * 0.60)
        lag_matrices = [a1]
    elif true_order == 3:
        a1 = make_sparse_matrix(d, edge_prob=0.10, seed=seed + 1) * 0.60
        a2 = make_sparse_matrix(d, edge_prob=0.08, seed=seed + 2) * 0.35
        a3 = make_sparse_matrix(d, edge_prob=0.06, seed=seed + 3) * 0.20
        total_radius = spectral_radius(a1) + spectral_radius(a2) + spectral_radius(a3)
        if total_radius > 1e-12:
            scale = 0.80 / total_radius
            a1, a2, a3 = scale * a1, scale * a2, scale * a3
        lag_matrices = [a1, a2, a3]
    else:
        raise ValueError(f"Unsupported true_order: {true_order}")
    
    X = simulate_var(T=T, lag_matrices=lag_matrices, noise_scale=noise_scale, seed=seed + 10)
    
    return ScenarioResult(
        X=X,
        ground_truth_compact=compact_ground_truth(lag_matrices),
        metadata={
            "scenario": "omitted_lag_order",
            "latent": False,
            "true_order": true_order,
            "observed_order": 1,
            "mechanism": "omitted_lag_order",
            "expected_signature": {
                "description": "Higher-order dynamics misidentified as latent confounding",
                "higher_order": True,
                "lag_structure": "complex",
            },
            "noise_scale": noise_scale,
        },
    )


def scenario_undersampled_markov(
    *,
    T: int = 2500,
    d: int = 10,
    undersample: int = 2,
    noise_scale: float = 1.0,
    seed: int = 0,
) -> ScenarioResult:
    """Markov(1) system observed at coarser time resolution.

    Mechanism: Observing every k-th timepoint of a true Markov(1) system
    induces apparent higher-order dependence.
    """
    rng = np.random.default_rng(seed)
    
    # Generate at high temporal resolution
    T_fine = T * undersample + 500
    a1 = scale_to_stability(make_sparse_matrix(d, edge_prob=0.10, seed=seed + 1) * 0.70)
    X_fine = simulate_var(T=T_fine, lag_matrices=[a1], noise_scale=noise_scale, seed=seed + 10)
    
    # Subsample every k-th observation
    X = X_fine[::undersample][:T]
    
    return ScenarioResult(
        X=X,
        ground_truth_compact=compact_ground_truth([a1]),
        metadata={
            "scenario": "undersampled_markov",
            "latent": False,
            "true_order": 1,
            "observed_order": undersample,
            "mechanism": "undersampled_markov",
            "expected_signature": {
                "description": "True Markov(1) appears as higher-order due to undersampling",
                "undersampling_factor": undersample,
                "induced_order": undersample,
            },
            "undersample": undersample,
            "noise_scale": noise_scale,
        },
    )


def scenario_measurement_noise(
    *,
    T: int = 2000,
    d: int = 10,
    obs_noise_scale: float = 0.5,
    noise_scale: float = 1.0,
    seed: int = 0,
) -> ScenarioResult:
    """Markov(1) system with added measurement noise.

    Mechanism: Observation noise introduces spurious correlations
    and can distort causal structure inference.
    """
    rng = np.random.default_rng(seed)
    
    a1 = scale_to_stability(make_sparse_matrix(d, edge_prob=0.10, seed=seed + 1) * 0.65)
    X = simulate_var(T=T, lag_matrices=[a1], noise_scale=noise_scale, seed=seed + 10)
    
    # Add measurement noise
    obs_noise = rng.normal(scale=obs_noise_scale, size=(T, d))
    X_noisy = X + obs_noise
    
    return ScenarioResult(
        X=X_noisy,
        ground_truth_compact=compact_ground_truth([a1]),
        metadata={
            "scenario": "measurement_noise",
            "latent": False,
            "true_order": 1,
            "observed_order": 1,
            "mechanism": "measurement_noise",
            "expected_signature": {
                "description": "Observation noise distorts inferred causal structure",
                "noise_distortion": True,
                "spurious_correlations": True,
            },
            "obs_noise_scale": obs_noise_scale,
            "process_noise_scale": noise_scale,
        },
    )


def scenario_time_varying_coefficients(
    *,
    T: int = 2000,
    d: int = 10,
    n_change_points: int = 2,
    noise_scale: float = 1.0,
    seed: int = 0,
) -> ScenarioResult:
    """VAR system with time-varying coefficients.

    Mechanism: Coefficients evolve over time, creating non-stationary dynamics
    that can mimic hidden confounding structure.
    """
    rng = np.random.default_rng(seed)
    
    # Generate change points
    change_points = sorted(rng.choice(range(200, T - 200), size=n_change_points, replace=False))
    change_points = [0] + list(change_points) + [T]
    
    # Generate different coefficient matrices for each regime
    a1_base = make_sparse_matrix(d, edge_prob=0.10, seed=seed + 1) * 0.50
    
    X = np.zeros((T, d), dtype=float)
    noise = rng.normal(scale=noise_scale, size=(T, d))
    
    for regime_idx in range(len(change_points) - 1):
        t_start = change_points[regime_idx]
        t_end = change_points[regime_idx + 1]
        
        # Vary coefficients by regime
        scale_factor = 0.5 + 0.5 * (regime_idx / (n_change_points + 1))
        a1_regime = scale_to_stability(a1_base * scale_factor)
        
        for t in range(max(1, t_start), t_end):
            X[t] = a1_regime @ X[t - 1] + noise[t]
    
    return ScenarioResult(
        X=X,
        ground_truth_compact=compact_ground_truth([a1_base]),
        metadata={
            "scenario": "time_varying_coefficients",
            "latent": False,
            "true_order": 1,
            "observed_order": 1,
            "mechanism": "time_varying_coefficients",
            "expected_signature": {
                "description": "Time-varying coefficients create apparent non-stationarity",
                "nonstationarity": True,
                "time_varying": True,
                "n_change_points": n_change_points,
            },
            "n_change_points": n_change_points,
            "noise_scale": noise_scale,
        },
    )


def scenario_regime_shift(
    *,
    T: int = 2000,
    d: int = 10,
    n_regimes: int = 2,
    noise_scale: float = 1.0,
    seed: int = 0,
) -> ScenarioResult:
    """System with discrete regime changes.

    Mechanism: System switches between multiple stable regimes,
    creating heterogeneous dynamics.
    """
    rng = np.random.default_rng(seed)
    
    # Generate regime-specific coefficient matrices
    a1_regimes = []
    for regime_idx in range(n_regimes):
        a1 = make_sparse_matrix(d, edge_prob=0.10, seed=seed + 1 + regime_idx) * 0.55
        a1_regimes.append(scale_to_stability(a1))
    
    # Generate regime sequence
    regime_sequence = rng.integers(0, n_regimes, size=T)
    regime_persistence = np.random.geometric(0.05, size=T)
    for t in range(1, T):
        if regime_persistence[t] > 10:
            regime_sequence[t] = regime_sequence[t - 1]
    regime_sequence = np.clip(regime_sequence, 0, n_regimes - 1)
    
    # Simulate with regime switching
    X = np.zeros((T, d), dtype=float)
    noise = rng.normal(scale=noise_scale, size=(T, d))
    
    for t in range(1, T):
        regime = regime_sequence[t]
        X[t] = a1_regimes[regime] @ X[t - 1] + noise[t]
    
    return ScenarioResult(
        X=X,
        ground_truth_compact=compact_ground_truth([a1_regimes[0]]),  # Use first regime as base truth
        metadata={
            "scenario": "regime_shift",
            "latent": False,
            "true_order": 1,
            "observed_order": 1,
            "mechanism": "regime_shift",
            "expected_signature": {
                "description": "Regime switching creates heterogeneous dynamics",
                "nonstationarity": True,
                "regime_switching": True,
                "n_regimes": n_regimes,
            },
            "n_regimes": n_regimes,
            "noise_scale": noise_scale,
        },
    )


def scenario_nonlinear_markov(
    *,
    T: int = 2000,
    d: int = 6,
    nonlinearity_strength: float = 0.3,
    noise_scale: float = 1.0,
    seed: int = 0,
) -> ScenarioResult:
    """Nonlinear Markov(1) system.

    Mechanism: True data-generating process is nonlinear,
    but linear VAR analysis misspecifies causal structure.
    """
    rng = np.random.default_rng(seed)
    
    # Generate base linear structure
    a1 = scale_to_stability(make_sparse_matrix(d, edge_prob=0.15, seed=seed + 1) * 0.50)
    
    X = np.zeros((T + 300, d), dtype=float)
    noise = rng.normal(scale=noise_scale, size=(T + 300, d))
    
    for t in range(1, T + 300):
        # Linear part
        linear = a1 @ X[t - 1]
        
        # Nonlinear part: polynomial and interaction terms
        x_prev = X[t - 1]
        nonlinear = nonlinearity_strength * (
            0.1 * np.tanh(x_prev) +
            0.05 * (x_prev ** 2 - 1) +
            0.05 * (x_prev[0] * x_prev)
        )
        
        X[t] = linear + nonlinear + noise[t]
    
    return ScenarioResult(
        X=X[300:],
        ground_truth_compact=compact_ground_truth([a1]),
        metadata={
            "scenario": "nonlinear_markov",
            "latent": False,
            "true_order": 1,
            "observed_order": 1,
            "mechanism": "nonlinear_markov",
            "expected_signature": {
                "description": "Nonlinear Markov(1) misidentified by linear VAR analysis",
                "nonlinearity": True,
                "nonlinear_distortion": True,
            },
            "nonlinearity_strength": nonlinearity_strength,
            "noise_scale": noise_scale,
        },
    )


def scenario_nonlinear_hidden_driver(
    *,
    T: int = 2000,
    d: int = 10,
    conf_strength: float = 0.35,
    nonlinearity_strength: float = 0.3,
    noise_scale: float = 1.0,
    seed: int = 0,
) -> ScenarioResult:
    """Observed system driven by nonlinear hidden confounder.

    Mechanism: A hidden variable follows nonlinear dynamics
    and influences all observed variables, creating spurious correlations.
    """
    rng = np.random.default_rng(seed)
    
    # Observed system linear dynamics
    a1 = scale_to_stability(make_sparse_matrix(d, edge_prob=0.10, seed=seed + 1) * 0.50)
    
    total = T + 400
    
    # Hidden variable with nonlinear dynamics
    latent = np.zeros(total, dtype=float)
    latent_noise = rng.normal(scale=1.0, size=total)
    
    for t in range(1, total):
        # Nonlinear AR(1) for hidden variable
        latent[t] = 0.7 * latent[t - 1] + nonlinearity_strength * np.tanh(latent[t - 1]) + latent_noise[t]
    
    conf_weights = rng.normal(scale=conf_strength, size=d)
    
    # Observed variables
    X = np.zeros((total, d), dtype=float)
    noise = rng.normal(scale=noise_scale, size=(total, d))
    
    for t in range(1, total):
        X[t] = a1 @ X[t - 1] + conf_weights * latent[t - 1] + noise[t]
    
    return ScenarioResult(
        X=X[400:],
        ground_truth_compact=compact_ground_truth([a1]),
        metadata={
            "scenario": "nonlinear_hidden_driver",
            "latent": True,
            "true_order": 1,
            "observed_order": 1,
            "mechanism": "nonlinear_hidden_driver",
            "expected_signature": {
                "description": "Nonlinear hidden driver creates confounding",
                "confounding": True,
                "hidden_nonlinearity": True,
            },
            "conf_strength": conf_strength,
            "nonlinearity_strength": nonlinearity_strength,
            "noise_scale": noise_scale,
        },
    )


def scenario_observation_filtering(
    *,
    T: int = 2000,
    d: int = 10,
    filter_type: str = "low_pass",
    filter_order: int = 2,
    noise_scale: float = 1.0,
    seed: int = 0,
) -> ScenarioResult:
    """Markov(1) system observed through temporal filtering.

    Mechanism: Filtering true Markov(1) observations through a moving average
    or other temporal filter induces apparent higher-order dependence.
    """
    rng = np.random.default_rng(seed)
    
    # Generate true Markov(1) system at higher resolution
    T_fine = T + 2 * filter_order + 100
    a1 = scale_to_stability(make_sparse_matrix(d, edge_prob=0.10, seed=seed + 1) * 0.65)
    X_fine = simulate_var(T=T_fine, lag_matrices=[a1], noise_scale=noise_scale, seed=seed + 10)
    
    # Apply temporal filtering
    if filter_type == "low_pass":
        kernel = np.ones(filter_order + 1) / (filter_order + 1)
        X_filtered = np.zeros_like(X_fine)
        for col in range(d):
            X_filtered[:, col] = np.convolve(X_fine[:, col], kernel, mode='same')
        X = X_filtered[filter_order + 50 : filter_order + 50 + T]
    elif filter_type == "diff":
        X_filtered = np.diff(X_fine, axis=0)
        X = X_filtered[50:50 + T]
    else:
        raise ValueError(f"Unknown filter_type: {filter_type}")
    
    return ScenarioResult(
        X=X,
        ground_truth_compact=compact_ground_truth([a1]),
        metadata={
            "scenario": "observation_filtering",
            "latent": False,
            "true_order": 1,
            "observed_order": filter_order + 1,
            "mechanism": "observation_filtering",
            "expected_signature": {
                "description": "Temporal filtering induces higher-order apparent structure",
                "filtering": True,
                "induced_order": filter_order + 1,
            },
            "filter_type": filter_type,
            "filter_order": filter_order,
            "noise_scale": noise_scale,
        },
    )


def scenario_partial_observation_hidden_nodes_strong(
    *,
    T: int = 2000,
    d_observed: int = 8,
    d_hidden: int = 4,
    hidden_strength: float = 0.6,
    noise_scale: float = 1.0,
    seed: int = 0,
) -> ScenarioResult:
    """Partial observation with strong hidden node coupling.

    Mechanism: Hidden nodes are strongly coupled to observed nodes,
    creating spurious correlations among observed variables.
    """
    rng = np.random.default_rng(seed)
    
    d_total = d_observed + d_hidden
    
    # Full system coefficient matrix
    a1 = scale_to_stability(make_sparse_matrix(d_total, edge_prob=0.12, seed=seed + 1))
    
    # Strengthen hidden->observed couplings
    hidden_indices = np.arange(d_observed, d_total)
    observed_indices = np.arange(d_observed)
    a1[np.ix_(observed_indices, hidden_indices)] *= hidden_strength
    
    # Normalize to stability
    a1 = scale_to_stability(a1, target_radius=0.80)
    
    X_full = simulate_var(T=T, lag_matrices=[a1], noise_scale=noise_scale, seed=seed + 10)
    
    # Extract observed part
    X = X_full[:, observed_indices]
    
    # Ground truth restricted to observed nodes
    observed_truth = (np.abs(a1[np.ix_(observed_indices, observed_indices)]) > 1e-12).astype(int)
    np.fill_diagonal(observed_truth, 0)
    
    return ScenarioResult(
        X=X,
        ground_truth_compact=observed_truth,
        metadata={
            "scenario": "partial_observation_hidden_nodes_strong",
            "latent": True,
            "true_order": 1,
            "observed_order": 1,
            "mechanism": "partial_observation_hidden_nodes_strong",
            "expected_signature": {
                "description": "Strong hidden node coupling creates spurious observed correlations",
                "spurious_correlations": True,
                "hidden_confounding": True,
            },
            "d_observed": d_observed,
            "d_hidden": d_hidden,
            "hidden_strength": hidden_strength,
            "noise_scale": noise_scale,
        },
    )


SCENARIOS = {
    "order1_unconfounded": scenario_order1_unconfounded,
    "order3_unconfounded": scenario_order3_unconfounded,
    "latent_common_driver": scenario_latent_common_driver,
    "variable_lag_unconfounded": scenario_variable_lag_unconfounded,
    "hidden_nodes": scenario_hidden_nodes,
    "omitted_lag_order": scenario_omitted_lag_order,
    "undersampled_markov": scenario_undersampled_markov,
    "measurement_noise": scenario_measurement_noise,
    "time_varying_coefficients": scenario_time_varying_coefficients,
    "regime_shift": scenario_regime_shift,
    "nonlinear_markov": scenario_nonlinear_markov,
    "nonlinear_hidden_driver": scenario_nonlinear_hidden_driver,
    "observation_filtering": scenario_observation_filtering,
    "partial_observation_hidden_nodes_strong": scenario_partial_observation_hidden_nodes_strong,
}
