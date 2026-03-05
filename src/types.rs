//! Core types for causal inference results.

use std::fmt;

/// Result of estimating a causal effect.
#[derive(Debug, Clone)]
pub struct CausalEffect {
    /// The treatment (cause) variable
    pub treatment: String,
    /// The outcome (effect) variable
    pub outcome: String,
    /// Estimated causal effect (regression coefficient)
    pub effect: f64,
    /// Standard error of the estimate
    pub std_error: f64,
    /// t-statistic
    pub t_stat: f64,
    /// p-value (two-tailed)
    pub p_value: f64,
    /// R² of the regression
    pub r_squared: f64,
    /// Number of observations used
    pub n_obs: usize,
    /// Adjustment set used (confounders controlled for)
    pub adjustment_set: Vec<String>,
    /// Method used for estimation
    pub method: EstimationMethod,
}

impl CausalEffect {
    /// Is this effect statistically significant at the given alpha level?
    pub fn is_significant(&self, alpha: f64) -> bool {
        self.p_value < alpha
    }

    /// 95% confidence interval: (lower, upper)
    pub fn ci_95(&self) -> (f64, f64) {
        (self.effect - 1.96 * self.std_error, self.effect + 1.96 * self.std_error)
    }

    /// Confidence interval at arbitrary level (e.g., 0.99 for 99%)
    pub fn ci(&self, level: f64) -> (f64, f64) {
        use statrs::distribution::{Normal, ContinuousCDF};
        let z = Normal::new(0.0, 1.0)
            .unwrap()
            .inverse_cdf((1.0 + level) / 2.0);
        (self.effect - z * self.std_error, self.effect + z * self.std_error)
    }
}

impl fmt::Display for CausalEffect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (lo, hi) = self.ci_95();
        let sig = if self.is_significant(0.05) { "✅" } else { "❌" };
        write!(
            f,
            "{} → {}: β={:.4} [{:.4}, {:.4}] p={:.4} {} R²={:.3} n={} method={:?}",
            self.treatment, self.outcome,
            self.effect, lo, hi,
            self.p_value, sig,
            self.r_squared, self.n_obs,
            self.method
        )
    }
}

/// Estimation method used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EstimationMethod {
    /// OLS with backdoor adjustment (controlling for confounders)
    BackdoorAdjustment,
    /// Instrumental variable estimation (2SLS)
    InstrumentalVariable,
    /// Front-door adjustment (via mediator)
    FrontDoor,
    /// Simple bivariate regression (no adjustment)
    Bivariate,
}

/// Granger causality test result.
#[derive(Debug, Clone)]
pub struct GrangerResult {
    /// Potential cause variable
    pub cause: String,
    /// Potential effect variable
    pub effect: String,
    /// F-statistic
    pub f_stat: f64,
    /// p-value
    pub p_value: f64,
    /// Number of lags tested
    pub lags: usize,
    /// Does X Granger-cause Y at α=0.05?
    pub granger_causes: bool,
}

impl fmt::Display for GrangerResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let arrow = if self.granger_causes { "⟹" } else { "⊬" };
        write!(
            f,
            "{} {} {} (F={:.3}, p={:.4}, lags={})",
            self.cause, arrow, self.effect,
            self.f_stat, self.p_value, self.lags
        )
    }
}

/// Result of a refutation test.
#[derive(Debug, Clone)]
pub struct RefutationResult {
    /// Name of the test
    pub test_name: String,
    /// Original estimated effect
    pub original_effect: f64,
    /// Effect after refutation manipulation
    pub refuted_effect: f64,
    /// Did the estimate survive refutation?
    pub passed: bool,
    /// p-value (interpretation depends on test)
    pub p_value: f64,
}

impl fmt::Display for RefutationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = if self.passed { "✅ PASSED" } else { "❌ FAILED" };
        write!(
            f,
            "{}: {} (original={:.4}, refuted={:.4}, p={:.4})",
            self.test_name, status,
            self.original_effect, self.refuted_effect, self.p_value
        )
    }
}

/// Counterfactual query result.
#[derive(Debug, Clone)]
pub struct Counterfactual {
    /// Human-readable question
    pub question: String,
    /// Treatment variable
    pub treatment: String,
    /// Outcome variable
    pub outcome: String,
    /// Observed outcome value
    pub observed: f64,
    /// Counterfactual outcome value
    pub counterfactual: f64,
    /// Individual causal effect (counterfactual - observed)
    pub individual_effect: f64,
}

impl fmt::Display for Counterfactual {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Q: {}\n  Observed {}: {:.4}\n  Counterfactual {}: {:.4}\n  Effect: {:+.4}",
            self.question, self.outcome, self.observed,
            self.outcome, self.counterfactual, self.individual_effect
        )
    }
}

/// Overall verdict for a causal relationship after all tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeVerdict {
    /// Significant causal effect, passed refutation tests → strong evidence
    StrongCausal,
    /// Significant effect but some refutations failed → moderate evidence
    ModerateCausal,
    /// Granger-causal but not backdoor-significant → temporal precedence only
    GrangerOnly,
    /// Not enough data to determine
    InsufficientData,
    /// Effect not significant or refuted
    NoEvidence,
}
