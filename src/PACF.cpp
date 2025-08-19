#include "PACF.h"

#include <vector>
#include <cmath>
#include <limits>

namespace {

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

} // namespace

std::vector<double> pacf_durbin_levinson(const std::vector<double>& x, int m, bool unbiased) {
    std::vector<double> phi; // phi[k] will store PACF at lag k
    if (m < 0) return phi;

    const int N = static_cast<int>(x.size());
    if (N < 2 || m == 0) {
        phi.assign(std::max(1, m + 1), 0.0);
        if (!phi.empty()) phi[0] = 1.0;
        return phi;
    }

    const int m_eff = std::min(m, N - 1);
    std::vector<double> g = autocovariances(x, m_eff, unbiased);

    phi.assign(m_eff + 1, 0.0);
    phi[0] = 1.0; // by convention

    if (g[0] == 0.0) {
        // zero-variance series → undefined PACF; return zeros except phi[0]=1
        return phi;
    }

    // Durbin–Levinson recursion
    std::vector<std::vector<double>> PHI(m_eff + 1); // PHI[k][j] = phi_{k,j}
    PHI[0] = std::vector<double>(1, 1.0);

    double E = g[0]; // prediction error variance at order k

    for (int k = 1; k <= m_eff; ++k) {
        // numerator: g[k] - sum_{j=1}^{k-1} PHI[k-1][j] * g[k-j]
        double num = g[k];
        for (int j = 1; j <= k - 1; ++j) {
            num -= PHI[k - 1][j] * g[k - j];
        }

        const double denom = E;
        double a_k = 0.0;
        if (std::fabs(denom) > 0.0) {
            a_k = num / denom; // reflection coefficient
        }

        // Update PHI_k from PHI_{k-1}
        PHI[k] = std::vector<double>(k + 1, 0.0);
        PHI[k][k] = a_k;
        for (int j = 1; j <= k - 1; ++j) {
            PHI[k][j] = PHI[k - 1][j] - a_k * PHI[k - 1][k - j];
        }

        // Update prediction error variance
        E = E * (1.0 - a_k * a_k);

        // Store PACF at lag k
        phi[k] = PHI[k][k];
    }

    return phi;
}

double pacf_at_m(const std::vector<double>& x, int m, bool unbiased) {
    if (m < 1) return std::numeric_limits<double>::quiet_NaN();
    const std::vector<double> phi = pacf_durbin_levinson(x, m, unbiased);
    if (static_cast<int>(phi.size()) <= m) return std::numeric_limits<double>::quiet_NaN();
    return phi[m];
}

