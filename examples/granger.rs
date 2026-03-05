//! Granger causality test for time series data.
//!
//! Does oil price Granger-cause stock market volatility?
//!
//! Run: `cargo run --example granger`

use causal_inference::granger_causality;

fn main() {
    // Simulate: oil shocks → VIX with 1-2 period lag
    let n = 200;
    let mut rng = SimpleRng(42);

    let oil: Vec<f64> = (0..n).map(|_| 70.0 + rng.f64() * 15.0).collect();
    let mut vix = vec![20.0; n];
    for t in 1..n {
        // VIX responds directly to oil level with 1-period lag
        vix[t] = 0.3 * vix[t-1] + 0.5 * oil[t-1] + rng.f64() * 2.0;
    }

    // Test: does oil Granger-cause VIX?
    let result = granger_causality(&oil, &vix, "oil_price", "vix", 5)
        .expect("test failed");
    println!("Oil → VIX: {}", result);

    // Reverse test: does VIX Granger-cause oil?
    let reverse = granger_causality(&vix, &oil, "vix", "oil_price", 5)
        .expect("reverse test failed");
    println!("VIX → Oil: {}", reverse);

    println!("\nInterpretation:");
    if result.granger_causes && !reverse.granger_causes {
        println!("  Oil Granger-causes VIX (unidirectional)");
    } else if result.granger_causes && reverse.granger_causes {
        println!("  Bidirectional Granger causality (feedback loop)");
    } else {
        println!("  No clear Granger-causal direction detected");
    }
}

struct SimpleRng(u64);
impl SimpleRng {
    fn f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0
    }
}
