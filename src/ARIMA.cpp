#include "ARIMA.h"
#include "PACF.h"
#include "utils.h"

#include <iostream>
#include <numeric>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <algorithm>

namespace {
    constexpr double SINGULAR_MATRIX_THRESHOLD = 1e-10;
    constexpr double DEFAULT_AR_COEF = 0.1;
    constexpr double INITIAL_MA_COEF = 0.1;
    constexpr double L2_REGULARIZATION_LAMBDA = 0.01;
    constexpr double GRADIENT_STEP_SIZE = 1e-5;
    constexpr double GRADIENT_CLIP_THRESHOLD = 100.0;
    constexpr double INITIAL_LEARNING_RATE = 0.001;
    constexpr double LEARNING_RATE_DECAY = 0.5;
    constexpr int MAX_ITERATIONS = 10000;
    constexpr double CONVERGENCE_TOLERANCE = 1e-6;
    constexpr int MIN_CONVERGENCE_ITERATIONS = 50;
    constexpr int DEBUG_PRINT_INTERVAL = 100;
    constexpr double MA_COEFFICIENT_BOUND = 0.9;
    constexpr double COEFFICIENT_BOUND = 1.0;
}

ARIMA::ARIMA(int p, int d, int q, const std::vector<double>& data, const std::vector<double>& original_data, const std::vector<double>& ar_coef, const std::vector<double>& ma_coef)
    : p(p), d(d), q(q), data(data), original_data_(original_data), ar_coef(ar_coef), ma_coef(ma_coef), differenced(true), mu_(0.0) {
}

// Helper function to solve Yule-Walker equations
std::vector<double> ARIMA::solve_yule_walker(const std::vector<double>& gamma, int p) const {
    std::vector<std::vector<double>> Gamma(p, std::vector<double>(p, 0.0));
    std::vector<double> phi(p, 0.0);
    std::vector<double> rhs(gamma.begin() + 1, gamma.begin() + 1 + p);

    // Build Toeplitz matrix
    for (int i = 0; i < p; ++i) {
        for (int j = 0; j < p; ++j) {
            Gamma[i][j] = gamma[std::abs(i - j)];
        }
    }

    // Gaussian elimination
    for (int i = 0; i < p; ++i) {
        double pivot = Gamma[i][i];
        if (std::abs(pivot) < SINGULAR_MATRIX_THRESHOLD) {
#ifdef DEBUG
            std::cout << "Warning: Singular matrix in Yule-Walker, using default AR coefficients" << std::endl;
#endif
            return std::vector<double>(p, DEFAULT_AR_COEF);
        }
        for (int j = i + 1; j < p; ++j) {
            double factor = Gamma[j][i] / pivot;
            for (int k = i; k < p; ++k) {
                Gamma[j][k] -= factor * Gamma[i][k];
            }
            rhs[j] -= factor * rhs[i];
        }
    }

    // Back substitution
    for (int i = p - 1; i >= 0; --i) {
        double sum = 0.0;
        for (int j = i + 1; j < p; ++j) {
            sum += Gamma[i][j] * phi[j];
        }
        phi[i] = (rhs[i] - sum) / Gamma[i][i];
        phi[i] = std::max(std::min(phi[i], COEFFICIENT_BOUND), -COEFFICIENT_BOUND);
    }
    return phi;
}

// Helper function to compute residuals
std::vector<double> ARIMA::compute_residuals(const std::vector<double>& params, const std::vector<double>& scaled_data) const {
    const size_t n = scaled_data.size();
    std::vector<double> residuals(n, 0.0);
    const double mu = params[0];
    const std::vector<double> phi(params.begin() + 1, params.begin() + 1 + p);
    const std::vector<double> theta(params.begin() + 1 + p, params.end());
    for (size_t t = 0; t < n; ++t) {
        double ar_part = 0.0;
        for (int i = 0; i < p; ++i) {
            if (t >= static_cast<size_t>(i + 1)) {
                ar_part += phi[i] * scaled_data[t - i - 1];
            }
        }
        double ma_part = 0.0;
        for (int j = 0; j < q; ++j) {
            if (t >= static_cast<size_t>(j + 1)) {
                ma_part += theta[j] * residuals[t - j - 1];
            }
        }
        residuals[t] = scaled_data[t] - mu - ar_part - ma_part;
        if (std::isnan(residuals[t]) || std::isinf(residuals[t])) {
            residuals[t] = 0.0;
        }
    }
    return residuals;
}

// Helper function to compute SSE with L2 regularization
double ARIMA::compute_sse(const std::vector<double>& residuals, const std::vector<double>& params) const {
    const size_t m = std::max(static_cast<size_t>(p), static_cast<size_t>(q));
    double sse = 0.0;
    for (size_t t = m; t < residuals.size(); ++t) {
        sse += residuals[t] * residuals[t];
    }
    // L2 regularization on theta
    constexpr double lambda = L2_REGULARIZATION_LAMBDA;
    for (size_t k = p + 1; k < params.size(); ++k) {
        sse += lambda * params[k] * params[k];
    }
    return sse;
}

// Helper function to scale data
std::vector<double> ARIMA::scale_data(const std::vector<double>& input) {
    double mean = std::accumulate(input.begin(), input.end(), 0.0) / input.size();
    double sq_sum = 0.0;
    for (double x : input) {
        sq_sum += (x - mean) * (x - mean);
    }
    double std_dev = std::sqrt(sq_sum / (input.size() - 1));
    std_dev = std_dev > 0.0 ? std_dev : 1.0;
    data_mean_ = mean;
    data_std_ = std_dev;
    std::vector<double> scaled(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        scaled[i] = (input[i] - mean) / std_dev;
    }
    return scaled;
}

void ARIMA::fit() {
    if (!differenced) {
#ifdef DEBUG
        std::cout << "Model not differenced!" << std::endl;
#endif
        return;
    }
#ifdef DEBUG
    std::cout << "Starting to fit" << std::endl;
    std::cout << "Sneak peek: " << data[0] << " " << data[data.size()-1] << " size is " << data.size() << std::endl;
#endif

    // Scale data
    scaled_data_ = scale_data(data);
    mu_ = 0.0; // Mean of scaled data
#ifdef DEBUG
    std::cout << "avg of scaled data is 0 (after scaling)" << std::endl;
#endif

    size_t m = std::max(static_cast<size_t>(p), static_cast<size_t>(q));
    if (data.size() <= m + 1) {
#ifdef DEBUG
        std::cout << "Data size too small for p=" << p << ", q=" << q << std::endl;
#endif
        return;
    }

    // Step 1: Yule-Walker for AR coefficients
    std::vector<double> gamma = autocovariances(scaled_data_, p, true);
    try {
        ar_coef = solve_yule_walker(gamma, p);
    } catch (const std::runtime_error& e) {
#ifdef DEBUG
        std::cout << "Yule-Walker failed: " << e.what() << std::endl;
#endif
        ar_coef.assign(p, DEFAULT_AR_COEF);
    }

    // Step 2: Compute initial residuals with AR coefficients
    std::vector<double> initial_params;
    initial_params.push_back(mu_);
    initial_params.insert(initial_params.end(), ar_coef.begin(), ar_coef.end());
    initial_params.insert(initial_params.end(), q, INITIAL_MA_COEF);
    residuals_ = compute_residuals(initial_params, scaled_data_);

    // Step 3: CLS for MA coefficients and mu
    std::vector<double> params;
    params.push_back(mu_);
    params.insert(params.end(), ar_coef.begin(), ar_coef.end());
    for (int i = 0; i < q; ++i) {
        params.push_back(INITIAL_MA_COEF);
    }

    // Gradient descent parameters
    double learning_rate = INITIAL_LEARNING_RATE;
    constexpr int max_iter = MAX_ITERATIONS;
    constexpr double tol = CONVERGENCE_TOLERANCE;
    constexpr double grad_clip = GRADIENT_CLIP_THRESHOLD;
    double prev_sse = std::numeric_limits<double>::max();

    // Gradient descent for mu and theta
    for (int iter = 0; iter < max_iter; ++iter) {
        residuals_ = compute_residuals(params, scaled_data_);
        double current_sse = compute_sse(residuals_, params);
        if (std::isnan(current_sse) || std::isinf(current_sse)) {
#ifdef DEBUG
            std::cout << "SSE is NaN or Inf at iteration " << iter << std::endl;
#endif
            params.assign(params.size(), INITIAL_MA_COEF);
            params[0] = mu_;
            residuals_ = compute_residuals(params, scaled_data_);
            break;
        }
        if (current_sse > prev_sse && iter > 0) {
            learning_rate *= LEARNING_RATE_DECAY;
        }
        prev_sse = current_sse;
        std::vector<double> grad(params.size(), 0.0);
        constexpr double h = GRADIENT_STEP_SIZE;

        // Central differences for gradients
        for (size_t k = 0; k <= static_cast<size_t>(q); ++k) {
            double old = params[k];
            params[k] = old + h;
            std::vector<double> temp_residuals = compute_residuals(params, scaled_data_);
            double sse_plus = compute_sse(temp_residuals, params);
            params[k] = old - h;
            temp_residuals = compute_residuals(params, scaled_data_);
            double sse_minus = compute_sse(temp_residuals, params);
            params[k] = old;
            if (std::isnan(sse_plus) || std::isinf(sse_plus) || std::isnan(sse_minus) || std::isinf(sse_minus)) {
                grad[k] = 0.0;
            } else {
                grad[k] = (sse_plus - sse_minus) / (2.0 * h);
                grad[k] = std::max(std::min(grad[k], grad_clip), -grad_clip);
            }
        }

        // Update mu and theta
        bool converged = false;
        const double sse_diff = std::abs(current_sse - prev_sse);
        if (sse_diff < tol && iter > MIN_CONVERGENCE_ITERATIONS) {
            converged = true;
        }
        for (size_t k = 0; k <= static_cast<size_t>(q); ++k) {
            params[k] -= learning_rate * grad[k];
            if (k > 0) {
                params[k] = std::max(std::min(params[k], MA_COEFFICIENT_BOUND), -MA_COEFFICIENT_BOUND);
            }
        }
        if (converged) {
#ifdef DEBUG
            std::cout << "Converged after " << iter + 1 << " iterations" << std::endl;
#endif
            break;
        }
#ifdef DEBUG
        if (iter % DEBUG_PRINT_INTERVAL == 0) {
            std::cout << "Iteration " << iter << ": SSE = " << current_sse << std::endl;
        }
#endif
    }

    // Store fitted parameters
    mu_ = params[0];
    ma_coef.assign(params.begin() + 1 + p, params.end());
    residuals_ = compute_residuals(params, scaled_data_);
#ifdef DEBUG
    std::cout << "Fitted mu: " << mu_ << std::endl;
    std::cout << "Fitted AR coef: ";
    for (double c : ar_coef) std::cout << c << " ";
    std::cout << "\nFitted MA coef: ";
    for (double c : ma_coef) std::cout << c << " ";
    std::cout << "\nSSE: " << compute_sse(residuals_, params) << std::endl;
#endif
}

bool ARIMA::is_differenced() const {
    return differenced;
}

std::vector<double> ARIMA::get_parameters() const {
    if (!differenced) {
#ifdef DEBUG
        std::cout << "Model not differenced yet!" << std::endl;
#endif
        return {};
    }
    if (ar_coef.empty() || ma_coef.empty()) {
#ifdef DEBUG
        std::cout << "Model not fitted yet!" << std::endl;
#endif
        return {};
    }
    std::vector<double> params;
    params.push_back(mu_);
    params.insert(params.end(), ar_coef.begin(), ar_coef.end());
    params.insert(params.end(), ma_coef.begin(), ma_coef.end());
    return params;
}

std::vector<double> ARIMA::predict(int steps) const {
    if (!differenced) {
#ifdef DEBUG
        std::cout << "Model not differenced yet!" << std::endl;
#endif
        return {};
    }
    if (ar_coef.empty() || ma_coef.empty()) {
#ifdef DEBUG
        std::cout << "Model not fitted yet!" << std::endl;
#endif
        return {};
    }
#ifdef DEBUG
    std::cout << "Predicting " << steps << " steps ahead..." << std::endl;
#endif
    std::vector<double> z_forecast(steps, 0.0);
    std::vector<double> z_extended = scaled_data_;
    std::vector<double> eps_extended = residuals_;
    for (int h = 0; h < steps; ++h) {
        double ar_part = 0.0;
        for (int i = 0; i < p; ++i) {
            size_t idx = z_extended.size() > static_cast<size_t>(i) ? z_extended.size() - 1 - i : 0;
            ar_part += ar_coef[i] * (idx < z_extended.size() ? z_extended[idx] : 0.0);
        }
        double ma_part = 0.0;
        for (int j = 0; j < q; ++j) {
            size_t idx = eps_extended.size() > static_cast<size_t>(j) ? eps_extended.size() - 1 - j : 0;
            ma_part += ma_coef[j] * (idx < eps_extended.size() ? eps_extended[idx] : 0.0);
        }
        double pred = mu_ + ar_part + ma_part;
        if (std::isnan(pred) || std::isinf(pred)) {
            pred = 0.0;
        }
        z_forecast[h] = pred;
        z_extended.push_back(pred);
        eps_extended.push_back(0.0);
    }
    // Invert differencing and scaling
    std::vector<double> y_forecast(steps, 0.0);
    if (original_data_.empty()) {
#ifdef DEBUG
        std::cout << "Original data not provided!" << std::endl;
#endif
        return z_forecast;
    }
    double last_y = original_data_.back();
    double cum = last_y;
    for (int h = 0; h < steps; ++h) {
        cum += z_forecast[h] * data_std_;
        y_forecast[h] = cum;
        if (std::isnan(y_forecast[h]) || std::isinf(y_forecast[h])) {
            y_forecast[h] = last_y;
        }
    }
    return y_forecast;
}
