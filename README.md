# ARIMA in Pure C++

To teach myself more about time-series models, I coded ARIMA in pure C++. No external libraries[^1], just math and C++.

[^1]: Python libraries are used to graph and compare against industry standard packages.

Please read `doc/how_to_get_started.md` if you want to run it yourself.

## What is ARIMA?

ARIMA stands for **Auto-Regressive Integrated Moving Average** - a generalization of autoregressive (**AR**) and moving average (**MA**) models. The autoregressive model uses past values of the series to predict the future; the moving average focuses on the relationship between an observation and a residual error. The **I** in ARIMA stands for integrated, which preprocesses our data to ensure stationarity (the assumption that statistical properties of data remain constant over time).

## Implementation Overview
### Step 1: Data Preprocessing and Model Selection
The algorithm first ensures stationarity using the **Augmented Dickey-Fuller test**, which performs the regression `Δy_t = α + βy_{t-1} + ε_t` and tests H₀: β = 0 (unit root). If non-stationary, the data is differenced until stationary.

**Automatic parameter selection** uses statistical significance testing:
- **AR order (p)**: Determined by the **Partial Autocorrelation Function (PACF)** computed via the Durbin-Levinson recursion. The algorithm finds the last lag where |PACF(k)| > 1.96/√T (95% confidence band).
- **MA order (q)**: Determined by the **Autocorrelation Function (ACF)** using the same significance testing approach.

### Step 2: Parameter Estimation

**AR Parameters via Yule-Walker Equations**: 
The autoregressive coefficients are estimated by solving the system Γφ = γ, where Γ is the Toeplitz matrix of autocovariances:
```
[γ(0)  γ(1)  ... γ(p-1)] [φ₁]   [γ(1)]
[γ(1)  γ(0)  ... γ(p-2)] [φ₂] = [γ(2)]
[  ⋮     ⋮    ⋱    ⋮]  [⋮]   [ ⋮ ]
[γ(p-1)... γ(1)  γ(0)  ]  [φₚ]   [γ(p)]
```

**MA Parameters via Conditional Least Squares**:
With AR coefficients fixed, the algorithm minimizes the sum of squared residuals:
`SSE = Σε²_t + λΣθ²_j` where `ε_t = y_t - μ - Σφᵢy_{t-i} - Σθⱼε_{t-j}`

This is solved using gradient descent with finite differences for gradient computation and L2 regularization (λ = 0.01) to prevent overfitting.

## Implementation Challenges and Solutions

Building ARIMA from scratch presented several numerical and algorithmic challenges:

### Numerical Stability Issues
**Problem**: Gradient descent with finite differences is sensitive to step size - too large causes oscillation, too small causes slow convergence.
**Solution**: Implemented adaptive learning rate that halves when the objective function increases, combined with gradient clipping (±100.0) to prevent explosive updates.

### Optimization Convergence
**Problem**: MA parameter estimation often got stuck in local minima or diverged due to the non-linear nature of the residual equations.
**Solution**: Added multiple safeguards:
- L2 regularization (λ = 0.01) to prevent overfitting
- Parameter bounds (MA coefficients clipped to ±0.9 for stability)  
- NaN/infinity detection with automatic parameter reset
- Convergence criteria based on both iteration count (>50) and objective function change (<1e-6)

### Differencing Inversion
**Problem**: Converting differenced predictions back to original scale accumulated errors, especially for multi-step forecasts.
**Solution**: Implemented careful cumulative sum inversion using the last observed value as the starting point, with bounds checking to prevent unrealistic predictions.

### Data Scaling Artifacts
**Problem**: Working with raw stock prices (often in hundreds) caused numerical precision issues in gradient computation.
**Solution**: Standardized all data to zero mean and unit variance before fitting, then inverted the scaling during prediction to maintain interpretability.

## Implementation Differences

In industry-standard implementations such as Python's *statsmodels*, parameter estimation is typically done via **Maximum Likelihood Estimation (MLE)**. MLE jointly estimates all parameters (AR coefficients, MA coefficients, and error variance) by maximizing the likelihood function `L(θ) = P(data | θ)`, assuming residuals follow a normal distribution. This joint optimization uses advanced numerical solvers and careful handling of likelihood surfaces.

I attempted solving it from a different angle:  
- **AR parameters** estimated via **Yule–Walker equations** (a closed-form system of linear equations based on autocovariances).  
- **MA parameters** estimated via **conditional least squares** using gradient descent with finite-difference gradients and regularization.  

This two-step approach optimizes computational simplicity. The results naturally differ slightly from MLE-based methods (typically within 1–3%), but remain mathematically sound and give a practical perspective on how ARIMA can be built up “from first principles.”  


## Future Enhancements

Several improvements could further enhance this implementation:

### Advanced Optimization
- **Quasi-Newton methods** (L-BFGS) for faster MA parameter convergence

### Performance Improvements
- **Parallel processing** for parameter grid search and cross-validation
- **Memory optimization** for handling larger datasets
- **GPU acceleration** for matrix operations in high-frequency trading scenarios
