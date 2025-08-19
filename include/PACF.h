#ifndef PACF_H
#define PACF_H

#include <vector>

// Compute PACF up to lag m using the Durbin–Levinson recursion.
// Returns a vector phi where phi[k] is the partial autocorrelation at lag k, for k=0..m.
// phi[0] is defined as 1.0.
// If unbiased is true, use an unbiased autocovariance estimator (divide by N-k); otherwise biased (divide by N).
std::vector<double> pacf_durbin_levinson(const std::vector<double>& x, int m, bool unbiased = false);

// Convenience: PACF at a single lag m (m >= 1). Returns NaN if not computable.
double pacf_at_m(const std::vector<double>& x, int m, bool unbiased = false);

#endif // PACF_H

