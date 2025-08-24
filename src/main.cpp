#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <iomanip>
#include <stdexcept>

#include "utils.h"
#include "PACF.h"
#include "ADF.h"
#include "ARIMA.h"

namespace {
    constexpr int MAX_DIFFERENCING = 2;
}

int main() {
    // Load configuration
    Config config;
    if (!load_configuration(config)) {
        return 1;
    }
    
    // Load stock data
    std::vector<double> close_prices;
    std::vector<std::string> dates;
    if (!load_stock_data(config.stock, close_prices, dates)) {
        return 1;
    }
    
    // Split data: exclude last holdout_size points for validation
    const std::vector<double> train_data(close_prices.begin(), close_prices.end() - config.holdout_size);
    const std::vector<double> actual_holdout(close_prices.end() - config.holdout_size, close_prices.end());
    
#ifdef VERBOSE
    std::cout << "Total data points: " << close_prices.size() << std::endl;
    std::cout << "Training data points: " << train_data.size() << std::endl;
    std::cout << "Holdout (actual) data points: " << actual_holdout.size() << std::endl;
#endif
    
    // Apply differencing until stationary
    std::vector<double> differenced_data = train_data;
    int d = 0;
    
    auto result = adf::adfuller(differenced_data, 1);
    while (!result.stationary && differenced_data.size() > 2 && d < MAX_DIFFERENCING) {
        differenced_data = difference(differenced_data);
        ++d;
        result = adf::adfuller(differenced_data, 1);
    }
    
#ifdef VERBOSE
    std::cout << "For company " << config.stock << std::endl;
    if (d > MAX_DIFFERENCING) {
        std::cout << "WARNING: Required " << d << " differencing steps. Data might be problematic" << std::endl;
    }
#endif
    
    // Determine ARIMA parameters
    const int K = static_cast<int>(std::sqrt(static_cast<double>(differenced_data.size())));
    const int p = pick_p_from_pacf(differenced_data, K, false);
    const int q = pick_q_from_acf(differenced_data, K, false);
    
#ifdef VERBOSE
    std::cout << "ARIMA parameters: p=" << p << " d=" << d << " q=" << q << std::endl;
#endif

    // fit ARIMA model
    ARIMA model(p, d, q, differenced_data, train_data, {}, {});
    model.fit();
 
    const auto predictions = model.predict(config.holdout_size);
    
#ifdef VERBOSE
    std::cout << "Predictions for holdout period:" << std::endl;
    for (int i = 0; i < config.holdout_size; ++i) {
        std::cout << "Step " << (i + 1) << " - Predicted: " 
                  << std::fixed << std::setprecision(4) << predictions[i] 
                  << ", Actual: " << actual_holdout[i] 
                  << ", Error: " << (predictions[i] - actual_holdout[i]) << std::endl;
    }
#endif

    write_arima_output(train_data, actual_holdout, predictions, config.holdout_size);
    
    return 0;
}
