//! Statistical helper functions: OLS, p-values, F-tests.

use nalgebra::{DMatrix, DVector};

/// Compute OLS regression: Y = Xβ + ε
/// Returns (coefficients, residuals, R², standard errors)
pub fn ols(x: &DMatrix<f64>, y: &DVector<f64>) -> Option<OlsResult> {
    let n = x.nrows();
    let p = x.ncols();
    if n <= p {
        return None;
    }

    let xtx = x.transpose() * x;
    let xtx_inv = xtx.try_inverse()?;
    let xty = x.transpose() * y;
    let beta = &xtx_inv * &xty;

    let y_hat = x * &beta;
    let residuals = y - &y_hat;
    let sse = residuals.dot(&residuals);

    let y_mean = y.mean();
    let sst: f64 = y.iter().map(|v| (v - y_mean).powi(2)).sum();
    let r_squared = if sst > 1e-10 { (1.0 - sse / sst).max(0.0) } else { 0.0 };

    let mse = sse / (n - p) as f64;
    let var_beta = mse * xtx_inv.diagonal();
    let std_errors: Vec<f64> = (0..p).map(|i| var_beta[i].max(0.0).sqrt()).collect();

    Some(OlsResult {
        coefficients: beta.as_slice().to_vec(),
        residuals: residuals.as_slice().to_vec(),
        r_squared,
        std_errors,
        sse,
        mse,
        n,
        p,
    })
}

/// Result of an OLS regression.
#[derive(Debug, Clone)]
pub struct OlsResult {
    /// Estimated coefficients β
    pub coefficients: Vec<f64>,
    /// Residuals (Y - Ŷ)
    pub residuals: Vec<f64>,
    /// Coefficient of determination
    pub r_squared: f64,
    /// Standard errors of coefficients
    pub std_errors: Vec<f64>,
    /// Sum of squared errors
    pub sse: f64,
    /// Mean squared error
    pub mse: f64,
    /// Number of observations
    pub n: usize,
    /// Number of parameters
    pub p: usize,
}

impl OlsResult {
    /// t-statistic for coefficient at index `i`
    pub fn t_stat(&self, i: usize) -> f64 {
        if i >= self.coefficients.len() || self.std_errors[i] < 1e-15 {
            return 0.0;
        }
        self.coefficients[i] / self.std_errors[i]
    }

    /// p-value for coefficient at index `i`
    pub fn p_value(&self, i: usize) -> f64 {
        t_to_p(self.t_stat(i), (self.n - self.p) as f64)
    }
}

/// Sum of squared errors from OLS regression.
pub fn ols_sse(x: &DMatrix<f64>, y: &DVector<f64>) -> Option<f64> {
    let xtx_inv = (x.transpose() * x).try_inverse()?;
    let beta = &xtx_inv * (x.transpose() * y);
    let residuals = y - x * &beta;
    Some(residuals.dot(&residuals))
}

/// Build a design matrix with intercept: [1, x₁, x₂, ...]
pub fn design_matrix(columns: &[&[f64]]) -> DMatrix<f64> {
    let n = columns.first().map_or(0, |c| c.len());
    let p = columns.len() + 1; // +1 for intercept
    let mut data = Vec::with_capacity(n * p);
    for i in 0..n {
        data.push(1.0); // intercept
        for col in columns {
            data.push(col[i]);
        }
    }
    DMatrix::from_row_slice(n, p, &data)
}

/// Convert t-statistic to two-tailed p-value.
pub fn t_to_p(t: f64, df: f64) -> f64 {
    if !t.is_finite() || df <= 0.0 {
        return 1.0;
    }
    use statrs::distribution::{StudentsT, ContinuousCDF};
    match StudentsT::new(0.0, 1.0, df) {
        Ok(dist) => {
            let cdf = dist.cdf(t.abs());
            if cdf.is_finite() { 2.0 * (1.0 - cdf) } else { 1.0 }
        }
        Err(_) => {
            use statrs::distribution::Normal;
            let normal = Normal::new(0.0, 1.0).unwrap();
            2.0 * (1.0 - normal.cdf(t.abs()))
        }
    }
}

/// Convert F-statistic to p-value.
pub fn f_to_p(f: f64, df1: f64, df2: f64) -> f64 {
    if df1 <= 0.0 || df2 <= 0.0 || f < 0.0 || !f.is_finite() {
        return 1.0;
    }
    use statrs::distribution::{FisherSnedecor, ContinuousCDF};
    match FisherSnedecor::new(df1, df2) {
        Ok(dist) => {
            let cdf = dist.cdf(f);
            if cdf.is_finite() { 1.0 - cdf } else { 1.0 }
        }
        Err(_) => 1.0,
    }
}

/// Simple LCG pseudo-random number generator (deterministic, for tests/refutation).
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Returns a value in [-1, 1)
    pub fn next_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.state >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0
    }

    /// Returns n values in [-1, 1)
    pub fn next_n(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.next_f64()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ols_simple() {
        // Y = 2 + 3X + noise
        let x_vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y_vals: Vec<f64> = x_vals.iter().map(|x| 2.0 + 3.0 * x).collect();

        let x = design_matrix(&[&x_vals]);
        let y = DVector::from_vec(y_vals);
        let result = ols(&x, &y).unwrap();

        assert!((result.coefficients[0] - 2.0).abs() < 0.01, "intercept");
        assert!((result.coefficients[1] - 3.0).abs() < 0.01, "slope");
        assert!(result.r_squared > 0.99);
    }

    #[test]
    fn test_t_to_p() {
        // Large t-stat should give small p-value
        assert!(t_to_p(5.0, 100.0) < 0.001);
        // Zero t-stat should give p ≈ 1.0
        assert!((t_to_p(0.0, 100.0) - 1.0).abs() < 0.01);
    }
}
