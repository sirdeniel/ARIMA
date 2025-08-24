#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <utility>

// String manipulation
void split(std::string str, std::string splitBy, std::vector<std::string>& tokens);

// Time series utilities
std::vector<double> difference(const std::vector<double>& series);
double mean(const std::vector<double>& x);
std::vector<double> autocovariances(const std::vector<double>& x, int max_lag, bool unbiased = false);

// Configuration management
struct Config {
    int holdout_size;
    std::string stock;
};

bool load_configuration(Config& config);

// File I/O utilities
bool load_stock_data(const std::string& stock_symbol, 
                    std::vector<double>& close_prices, 
                    std::vector<std::string>& dates);

void write_arima_output(const std::vector<double>& train_data,
                       const std::vector<double>& actual_holdout,
                       const std::vector<double>& predictions,
                       int holdout_size);

#endif
