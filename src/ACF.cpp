#include "ACF.h"

#include <iostream>

#include <vector> 
#include <cmath>
#include <limits>


// Compute mean of a vector (returns 0.0 for empty input)
static inline double mean(const std::vector<double>& x) {
    if (x.empty()) return 0.0;
    double s = 0.0;
    for (double v : x) s += v;
    return s / static_cast<double>(x.size());
}

// Autocovariances gamma(0..m). Biased: divide by N, Unbiased: divide by N-k.
static std::vector<double> autocovariances(const std::vector<double>& x, int m, bool unbiased) {
    const int N = static_cast<int>(x.size());
    std::vector<double> g(m + 1, 0.0);
    if (N == 0) return g;
    const double mu = mean(x);
    for (int k = 0; k <= m; ++k) {
        double s = 0.0;
        const int limit = N - k;
        for (int t = 0; t < limit; ++t) {
            s += (x[t] - mu) * (x[t + k] - mu);
        }
        const double denom = unbiased ? static_cast<double>(N - k) : static_cast<double>(N);
        g[k] = (denom > 0.0 ? (s / denom) : 0.0);
    }
    return g;
}


double acf_at_m(const std::vector<double>& x, int m, bool unbiased) {
    if (m < 0) return std::numeric_limits<double>::quiet_NaN();
    const int N = static_cast<int>(x.size());
    if (N == 0) return std::numeric_limits<double>::quiet_NaN();

    const std::vector<double> g = autocovariances(x, std::max(0, m), unbiased);
    if (g.empty() || g[0] == 0.0) return std::numeric_limits<double>::quiet_NaN();

    if (m == 0) return 1.0;
    if (static_cast<int>(g.size()) <= m) return std::numeric_limits<double>::quiet_NaN();
    return g[m] / g[0];
}

