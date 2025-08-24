#include "utils.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>

// function found here https://stackoverflow.com/questions/10058606/splitting-a-string-by-a-character
void split(std::string str, std::string splitBy, std::vector<std::string>& tokens)
{
    /* Store the original string in the array, so we can loop the rest
     * of the algorithm. */
    tokens.push_back(str);

    // Store the split index in a 'size_t' (unsigned integer) type.
    size_t splitAt;
    // Store the size of what we're splicing out.
    size_t splitLen = splitBy.size();
    // Create a string for temporarily storing the fragment we're processing.
    std::string frag;
    // Loop infinitely - break is internal.
    while(true)
    {
        /* Store the last string in the vector, which is the only logical
         * candidate for processing. */
        frag = tokens.back();
        /* The index where the split is. */
        splitAt = frag.find(splitBy);
        // If we didn't find a new split point...
        if(splitAt == std::string::npos)
        {
            // Break the loop and (implicitly) return.
            break;
        }
        /* Put everything from the left side of the split where the string
         * being processed used to be. */
        tokens.back() = frag.substr(0, splitAt);
        /* Push everything from the right side of the split to the next empty
         * index in the vector. */
        tokens.push_back(frag.substr(splitAt+splitLen, frag.size()-(splitAt+splitLen)));
    }
}


std::vector<double> difference(const std::vector<double>& series) {
    std::vector<double> ret;
    const size_t n = series.size();

    if (n < 2) return ret;
    ret.reserve(n - 1);

    for (size_t t = 1; t < n; ++t) {
        ret.push_back(series[t] - series[t - 1]);
    }

    return ret;
}

double mean(const std::vector<double>& x) {
    if (x.empty()) return 0.0;
    double sum = 0.0;
    for (const double v : x) {
        sum += v;
    }
    return sum / static_cast<double>(x.size());
}

std::vector<double> autocovariances(const std::vector<double>& x, int max_lag, bool unbiased) {
    const int N = static_cast<int>(x.size());
    std::vector<double> gamma(max_lag + 1, 0.0);
    if (N == 0) return gamma;
    
    const double mu = mean(x);
    for (int k = 0; k <= max_lag; ++k) {
        double sum = 0.0;
        const int limit = N - k;
        for (int t = 0; t < limit; ++t) {
            sum += (x[t] - mu) * (x[t + k] - mu);
        }
        const double denom = unbiased ? static_cast<double>(N - k) : static_cast<double>(N);
        gamma[k] = (denom > 0.0 ? (sum / denom) : 0.0);
    }
    return gamma;
}

bool load_configuration(Config& config) {
    std::ifstream config_file("config.txt");
    if (!config_file.is_open()) {
        std::cerr << "Error: Could not open config.txt file\n";
        std::cerr << "Please create config.txt with HOLDOUT_SIZE and STOCK settings\n";
        return false;
    }

    bool holdout_found = false;
    bool stock_found = false;
    std::string line;
    
    while (std::getline(config_file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        const size_t pos = line.find('=');
        if (pos != std::string::npos) {
            const std::string key = line.substr(0, pos);
            const std::string value = line.substr(pos + 1);
            
            if (key == "HOLDOUT_SIZE") {
                config.holdout_size = std::stoi(value);
                holdout_found = true;
            } else if (key == "STOCK") {
                config.stock = value;
                stock_found = true;
            }
        }
    }
    
    if (!holdout_found || !stock_found) {
        std::cerr << "Error: Missing required configuration values\n";
        if (!holdout_found) std::cerr << "  Missing: HOLDOUT_SIZE\n";
        if (!stock_found) std::cerr << "  Missing: STOCK\n";
        return false;
    }
    
#ifdef VERBOSE
    std::cout << "Configuration: HOLDOUT_SIZE=" << config.holdout_size 
              << ", STOCK=" << config.stock << std::endl;
#endif
    
    return true;
}

bool load_stock_data(const std::string& stock_symbol, 
                    std::vector<double>& close_prices, 
                    std::vector<std::string>& dates) {
    const std::string file_path = "data/" + stock_symbol + "_data.csv";
    std::ifstream file(file_path);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file: " << file_path << std::endl;
        std::cerr << "Current working directory should contain 'data/' folder" << std::endl;
        return false;
    }
    
    // Skip header line
    std::string header;
    std::getline(file, header);
    
    std::string line;
    while (std::getline(file, line)) {
        std::vector<std::string> tokens;
        split(line, ",", tokens);
        
        if (tokens.size() >= 5) {
            dates.push_back(tokens[0]);
            close_prices.push_back(std::stod(tokens[4]));
        }
    }
    
    return !close_prices.empty();
}

void write_arima_output(const std::vector<double>& train_data,
                       const std::vector<double>& actual_holdout,
                       const std::vector<double>& predictions,
                       int holdout_size) {
    std::ofstream outfile("data/arima_output.csv");
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not create output file data/arima_output.csv" << std::endl;
        return;
    }
    
    outfile << "index,type,value\n";
    outfile << std::fixed << std::setprecision(4);
    
    // Output training data (last 50 points for visualization)
    constexpr int TRAINING_DATA_DISPLAY_LIMIT = 50;
    const int train_start_idx = std::max(0, 
        static_cast<int>(train_data.size()) - TRAINING_DATA_DISPLAY_LIMIT);
        
    for (int i = train_start_idx; i < static_cast<int>(train_data.size()); ++i) {
        outfile << i << ",training," << train_data[i] << "\n";
    }
    
    // Output actual holdout data
    const int holdout_start_idx = static_cast<int>(train_data.size());
    for (int i = 0; i < holdout_size; ++i) {
        outfile << (holdout_start_idx + i) << ",actual," << actual_holdout[i] << "\n";
    }
    
    // Output predictions
    for (int i = 0; i < static_cast<int>(predictions.size()); ++i) {
        outfile << (holdout_start_idx + i) << ",prediction," << predictions[i] << "\n";
    }
}
