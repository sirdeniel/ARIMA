#ifndef ACF_H
#define ACF_H

#include <vector>


static std::vector<double> autocovariances(const std::vector<double>& x, int m, bool unbiased);
double acf_at_m(const std::vector<double>& x, int m, bool unbiased);


#endif
