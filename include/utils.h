#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>

void split(std::string str, std::string splitBy, std::vector<std::string>& tokens);
std::vector<double> difference(const std::vector<double>& series);

#endif
