#ifndef ADF_H
#define ADF_H

#include <vector>

namespace adf {

struct Result {
    double statistic;
    double p_value;
    int used_lags;
    bool stationary;
};

Result adfuller(const std::vector<double>& series, int max_lags = 1);

}


#endif // ADF_H
