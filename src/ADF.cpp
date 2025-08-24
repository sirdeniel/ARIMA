#include "ADF.h"

#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>

namespace adf {
    constexpr double SQRT_2 = 1.4142135623730951;
    constexpr double MACKINNON_INTERCEPT = 1.84038;
    constexpr double MACKINNON_SLOPE = 1.214;
    constexpr double MIN_P_VALUE = 1e-12;
    constexpr double MAX_P_VALUE = 1.0 - 1e-12;
    constexpr double CRITICAL_VALUE_5_PERCENT = -2.86;
    constexpr int NUM_REGRESSORS = 2;  // intercept + y_{t-1}
    
    static inline double stdnorm_cdf(double x) {
        return 0.5 * std::erfc(-x / SQRT_2);
    }
    
    static inline double approx_adf_pvalue_const(double t_stat) {
        const double z = MACKINNON_INTERCEPT + MACKINNON_SLOPE * t_stat;
        double p = stdnorm_cdf(z);
        if (p < MIN_P_VALUE) p = MIN_P_VALUE;
        if (p > MAX_P_VALUE) p = MAX_P_VALUE;
        return p;
    }

Result adfuller(const std::vector<double>& series, int max_lags)
{
    const std::size_t n = series.size();
    if (n < 2) {
        return Result{0.0, 1.0, 0, false};
    }

    const std::size_t m = n - 1; // length of constructed vectors

    // z = [Δy2, …, Δy_n]
    std::vector<double> z;
    z.reserve(m);

    // x2 = [y1, …, y_{n-1}]
    std::vector<double> x2;
    x2.reserve(m);

    // intercept column of ones (length m)
    std::vector<double> intercept(m, 1.0);

    for (std::size_t t = 1; t < n; ++t) {
        z.push_back(series[t] - series[t - 1]);
        x2.push_back(series[t - 1]);
    }


    // statistics
    const std::size_t nn = z.size();
    const double sum_y = std::accumulate(x2.begin(), x2.end(), 0.0);
    const double sum_y2 = std::inner_product(x2.begin(), x2.end(), x2.begin(), 0.0);
    const double sum_dy = std::accumulate(z.begin(), z.end(), 0.0);
    const double sum_y_dy = std::inner_product(x2.begin(), x2.end(), z.begin(), 0.0);
    const double det = static_cast<double>(nn) * sum_y2 - sum_y * sum_y;

    double beta = 0.0, alpha = 0.0;
    if (det != 0.0) {
        beta = (static_cast<double>(nn) * sum_y_dy - sum_y * sum_dy) / det;
        alpha = (sum_y2 * sum_dy - sum_y * sum_y_dy) / det;
    }

    #ifdef DEBUG
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "n = " << nn << "\n";
    std::cout << "sum_y_{t-1} = " << sum_y << "\n";
    std::cout << "sum_y_{t-1}^2 = " << sum_y2 << "\n";
    std::cout << "sum_Δy_t = " << sum_dy << "\n";
    std::cout << "sum_y_{t-1}*Δy_t = " << sum_y_dy << "\n";
    std::cout << "det = " << det << "\n";
    std::cout << "alpha = " << alpha << ", beta = " << beta << "\n";
    #endif

    // Residuals and SSE
    double sse = 0.0;
    for (std::size_t i = 0; i < nn; ++i) {
        const double e = z[i] - alpha - beta * x2[i];
        sse += e * e;
    }

    // Error variance and SE(beta)
    const int dof = static_cast<int>(nn) - NUM_REGRESSORS;
    const double sigma2 = (dof > 0) ? (sse / dof) : 0.0;
    const double se_beta = (det != 0.0) ? std::sqrt(sigma2 * (static_cast<double>(nn) / det)) : 0.0;

    // ADF t-stat is the t-stat on beta
    const double adf_t = (se_beta > 0.0) ? (beta / se_beta) : 0.0;
    const double p_val = approx_adf_pvalue_const(adf_t);

    // Decision using asymptotic 5% critical value (constant-only case)
    const bool is_stationary = (adf_t < CRITICAL_VALUE_5_PERCENT);

    #ifdef DEBUG
    std::cout << "SSE = " << sse << ", sigma^2 = " << sigma2 << ", SE(beta) = " << se_beta << "\n";
    std::cout << "ADF t-stat = " << adf_t << ", p-value ~= " << p_val << "\n";
    std::cout << "Decision@5% (crit=" << CRITICAL_VALUE_5_PERCENT << ") stationary? " << (is_stationary ? "true" : "false") << "\n";
    #endif


    int used_lags = (max_lags > 0) ? std::min<int>(max_lags, static_cast<int>(m) - 1) : 0;

    return Result{adf_t, p_val, used_lags, is_stationary};
}


}
