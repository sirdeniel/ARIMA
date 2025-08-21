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
  std::ifstream file("/home/daniel/data/stonks/GOOG_data.csv");

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
  std::cout << "d = " << d << "\n";
  /*
  std::cout << "ADF Statistic: " << result.statistic << "\n";
  std::cout << "p-value: " << result.p_value << "\n";
  std::cout << "Stationary  " << (result.stationary ? "true" : "false") << "\n";
  */

  //ARIMA(p, d, q);

  // Choose K ~ sqrt(T)
  const int K = static_cast<int>(std::sqrt(static_cast<double>(curr.size())));
  int p = pick_p_from_pacf(curr, K, false);
  int q = pick_q_from_acf(curr, K, false);
  std::cout << "p = " << p << "\n";
  std::cout << "q = " << q << "\n";




} 
