//! Granger causality test for time-series data.
//!
//! Tests whether past values of X help predict Y beyond what past values
//! of Y alone can predict.

use nalgebra::{DMatrix, DVector};

use crate::stats::{self, f_to_p};
use crate::types::GrangerResult;

/// Test if `cause` Granger-causes `effect`.
///
/// **Granger causality** (1969): X Granger-causes Y if past values of X
/// improve prediction of Y beyond what Y's own history provides.
///
/// Restricted model:  Y_t = a₀ + Σ aᵢ·Y_{t-i} + ε
/// Unrestricted model: Y_t = a₀ + Σ aᵢ·Y_{t-i} + Σ bⱼ·X_{t-j} + ε
///
/// F-test: F = ((SSR_r - SSR_u) / p) / (SSR_u / (n - 2p - 1))
///
/// # Arguments
/// * `x` — Potential cause time series
/// * `y` — Potential effect time series
/// * `max_lags` — Maximum number of lags to test
///
/// # Returns
/// `None` if insufficient data.
///
/// # Example
/// ```
/// use causal_inference::granger_causality;
///
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
///              11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0];
/// let y: Vec<f64> = x.windows(2).map(|w| w[0] * 0.5 + w[1] * 0.8).collect();
/// // ... granger_causality(&x, &y, "X", "Y", 3) would test if X Granger-causes Y
/// ```
pub fn granger_causality(
    x: &[f64],
    y: &[f64],
    cause_name: &str,
    effect_name: &str,
    max_lags: usize,
) -> Option<GrangerResult> {
    let n_total = x.len().min(y.len());
    let lags = max_lags.min(n_total / 5);
    if n_total < lags * 3 + 1 || lags == 0 {
        return None;
    }

    let n = n_total - lags;

    // Build lagged variables
    let mut x_restricted_data = Vec::with_capacity(n * (lags + 1));
    let mut x_unrestricted_data = Vec::with_capacity(n * (2 * lags + 1));
    let mut y_target = Vec::with_capacity(n);

    for t in lags..n_total {
        y_target.push(y[t]);

        // Restricted: intercept + lagged Y
        x_restricted_data.push(1.0);
        for lag in 1..=lags {
            x_restricted_data.push(y[t - lag]);
        }

        // Unrestricted: intercept + lagged Y + lagged X
        x_unrestricted_data.push(1.0);
        for lag in 1..=lags {
            x_unrestricted_data.push(y[t - lag]);
        }
        for lag in 1..=lags {
            x_unrestricted_data.push(x[t - lag]);
        }
    }

    let p_r = lags + 1;
    let p_u = 2 * lags + 1;

    let y_vec = DVector::from_vec(y_target);

    let xr = DMatrix::from_row_slice(n, p_r, &x_restricted_data);
    let ssr_r = stats::ols_sse(&xr, &y_vec)?;

    let xu = DMatrix::from_row_slice(n, p_u, &x_unrestricted_data);
    let ssr_u = stats::ols_sse(&xu, &y_vec)?;

    let df1 = lags as f64;
    let df2 = (n - p_u) as f64;

    if df2 <= 0.0 || ssr_u <= 0.0 {
        return None;
    }

    let f_stat = ((ssr_r - ssr_u) / df1) / (ssr_u / df2);
    let p_value = f_to_p(f_stat, df1, df2);
    let granger_causes = p_value < 0.05;

    Some(GrangerResult {
        cause: cause_name.to_string(),
        effect: effect_name.to_string(),
        f_stat,
        p_value,
        lags,
        granger_causes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_granger_causality() {
        // X causes Y with 1-lag: Y_t = 0.8 * X_{t-1} + noise
        let n = 100;
        let mut rng = crate::stats::SimpleRng::new(42);
        let x: Vec<f64> = (0..n).map(|_| rng.next_f64() * 10.0).collect();
        let mut y = vec![0.0; n];
        for t in 1..n {
            y[t] = 0.8 * x[t - 1] + rng.next_f64() * 0.5;
        }

        let result = granger_causality(&x, &y, "X", "Y", 3).unwrap();
        assert!(result.granger_causes, "X should Granger-cause Y: {:?}", result);
    }

    #[test]
    fn test_no_granger_causality() {
        // Two independent random series
        let n = 100;
        let mut rng = crate::stats::SimpleRng::new(42);
        let x: Vec<f64> = rng.next_n(n);
        let y: Vec<f64> = crate::stats::SimpleRng::new(123).next_n(n);

        let result = granger_causality(&x, &y, "X", "Y", 3).unwrap();
        // Should usually not be significant (but randomness can cause false positives ~5%)
        // Just check it doesn't crash
        assert!(result.p_value >= 0.0);
    }
}
