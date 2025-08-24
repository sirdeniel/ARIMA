#ifndef ARIMA_H
#define ARIMA_H

#include <vector>

class ARIMA {
private:
    const int p;
    const int d;
    const int q;
    std::vector<double> data;
    std::vector<double> original_data_;
    std::vector<double> ar_coef;
    std::vector<double> ma_coef;
    std::vector<double> residuals_;
    std::vector<double> scaled_data_;
    bool differenced;
    double mu_;
    double data_std_;
    double data_mean_;
    
    // Helper methods
    std::vector<double> solve_yule_walker(const std::vector<double>& gamma, int p) const;
    std::vector<double> compute_residuals(const std::vector<double>& params, const std::vector<double>& scaled_data) const;
    double compute_sse(const std::vector<double>& residuals, const std::vector<double>& params) const;
    std::vector<double> scale_data(const std::vector<double>& input);
    
public:
    ARIMA(int p, int d, int q, const std::vector<double>& data, const std::vector<double>& original_data, const std::vector<double>& ar_coef, const std::vector<double>& ma_coef);
    
    void fit();
    bool is_differenced() const;
    std::vector<double> get_parameters() const;
    std::vector<double> predict(int steps) const;
};





#endif
