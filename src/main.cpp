#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <cmath>
#include <stdexcept> // std::runtime_error


#include "utils.h"
#include "PACF.h"
#include "ADF.h"

int main() {
  std::ifstream file("/home/daniel/data/stonks/AAPL_data.csv");

  if (!file.is_open()) { 
    throw std::runtime_error("File couldn't be opened");
    //std::fprintf("ERR: File couldn't be opened");
  }

  std::string throwaway;
  std::getline(file, throwaway); // read the first line, column names


  std::string line;
  std::vector<std::string> date;
  std::vector<double> closeP;

  while (std::getline(file, line)) {
    std::vector<std::string> tmp;
    split(line, ",", tmp);

    date.push_back(tmp[0]);
    closeP.push_back(std::stod(tmp[4]));
  }


  // Iteratively difference until stationary
  std::vector<double> curr = closeP;
  int d = 0;
  const int max_d = 2; // cap differencing
  auto result = adf::adfuller(curr, 1);
  while (!result.stationary && curr.size() > 2 && d < max_d) {
    curr = difference(curr);
    ++d;
    result = adf::adfuller(curr, 1);
  }

  if (d > 2) {
    std::cout << "WARNING: TOOK MORE THAN " << d << " DIFFERENCING. DATA MIGHT BE BAD" << std::endl;
  }
  std::cout << "I section" << std::endl;
  std::cout << "  took d differences: " << d << "\n";
  std::cout << "ADF Statistic: " << result.statistic << "\n";
  std::cout << "p-value: " << result.p_value << "\n";
  std::cout << "Stationary (5% crit): " << (result.stationary ? "true" : "false") << "\n";

  //ARIMA(p, d, q);
  // d is complete, now p is needed. this is done using PACF
  double p = pacf_at_m(curr, (int)sqrt(curr.size()), false);
  std::cout << "p is " << p << std::endl;



} 
