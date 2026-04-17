use pyo3::prelude::*;
use pyo3::pyclass;

fn standard_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

#[link(name = "m")]
unsafe extern "C" {
    #[link_name = "erf"]
    fn c_erf(x: f64) -> f64;
}

fn erf(x: f64) -> f64 {
    unsafe { c_erf(x) }
}

fn poisson_weights(mean: f64, tail_mass: f64) -> Vec<f64> {
    if mean <= 0.0 {
        return vec![1.0];
    }

    let mut weights = Vec::new();
    let mut weight = (-mean).exp();
    let mut cumulative = 0.0;
    let mut n = 0.0;

    loop {
        weights.push(weight);
        cumulative += weight;
        if 1.0 - cumulative <= tail_mass {
            break;
        }
        n += 1.0;
        weight *= mean / n;
    }

    let total: f64 = weights.iter().sum();
    weights.into_iter().map(|w| w / total).collect()
}

#[derive(Clone)]
struct ProbabilityTerm {
    weight: f64,
    score_shift: f64,
    inv_sqrt_variance: Option<f64>,
}

fn build_terms(
    steps_remaining: usize,
    diffusion_sigma: f64,
    jump_intensity: f64,
    jump_mean: f64,
    jump_sigma: f64,
    terminal_threshold: f64,
    poisson_tail_mass: f64,
) -> Vec<ProbabilityTerm> {
    if steps_remaining == 0 {
        return Vec::new();
    }

    let horizon = steps_remaining as f64;
    let jump_rate = jump_intensity * horizon;
    let weights = poisson_weights(jump_rate, poisson_tail_mass);
    let diffusion_variance = horizon * diffusion_sigma * diffusion_sigma;

    weights
        .into_iter()
        .enumerate()
        .map(|(n_jumps, weight)| {
            let jumps = n_jumps as f64;
            let future_variance = diffusion_variance + jumps * jump_sigma * jump_sigma;
            ProbabilityTerm {
                weight,
                score_shift: -terminal_threshold + jumps * jump_mean,
                inv_sqrt_variance: if future_variance <= 0.0 {
                    None
                } else {
                    Some(1.0 / future_variance.sqrt())
                },
            }
        })
        .collect()
}

fn eval_terms(score: f64, terms: &[ProbabilityTerm], terminal_threshold: f64) -> f64 {
    if terms.is_empty() {
        return if score > terminal_threshold { 1.0 } else { 0.0 };
    }

    let mut probability = 0.0;
    for term in terms {
        let shifted_score = score + term.score_shift;
        let conditional = if let Some(inv_sqrt_variance) = term.inv_sqrt_variance {
            standard_normal_cdf(shifted_score * inv_sqrt_variance)
        } else if shifted_score > 0.0 {
            1.0
        } else {
            0.0
        };
        probability += term.weight * conditional;
    }
    probability.clamp(0.0, 1.0)
}

#[pyclass]
struct ProbabilityKernel {
    terminal_threshold: f64,
    terms_by_steps: Vec<Vec<ProbabilityTerm>>,
}

#[pymethods]
impl ProbabilityKernel {
    #[new]
    fn new(
        max_steps: usize,
        diffusion_sigma: f64,
        jump_intensity: f64,
        jump_mean: f64,
        jump_sigma: f64,
        terminal_threshold: f64,
        poisson_tail_mass: f64,
    ) -> Self {
        let mut terms_by_steps = Vec::with_capacity(max_steps + 1);
        terms_by_steps.push(Vec::new());
        for steps_remaining in 1..=max_steps {
            terms_by_steps.push(build_terms(
                steps_remaining,
                diffusion_sigma,
                jump_intensity,
                jump_mean,
                jump_sigma,
                terminal_threshold,
                poisson_tail_mass,
            ));
        }
        Self {
            terminal_threshold,
            terms_by_steps,
        }
    }

    fn true_probability(&self, score: f64, steps_remaining: usize) -> f64 {
        let index = steps_remaining.min(self.terms_by_steps.len().saturating_sub(1));
        eval_terms(score, &self.terms_by_steps[index], self.terminal_threshold)
    }
}

#[pyfunction]
fn true_probability(
    score: f64,
    steps_remaining: usize,
    diffusion_sigma: f64,
    jump_intensity: f64,
    jump_mean: f64,
    jump_sigma: f64,
    terminal_threshold: f64,
    poisson_tail_mass: f64,
) -> f64 {
    let terms = build_terms(
        steps_remaining,
        diffusion_sigma,
        jump_intensity,
        jump_mean,
        jump_sigma,
        terminal_threshold,
        poisson_tail_mass,
    );
    eval_terms(score, &terms, terminal_threshold)
}

#[pymodule]
fn _rust_sim(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ProbabilityKernel>()?;
    module.add_function(wrap_pyfunction!(true_probability, module)?)?;
    Ok(())
}
