# How to Run the ARIMA Project

## Prerequisites
- C++ compiler (g++)
- Python 3 with the following packages:
  - pandas
  - matplotlib  
  - statsmodels
  - numpy

## Data Setup
This project assumes the data has been cleaned and preprocessed. I use stock data for S&P 500 companies available on [Kaggle](https://www.kaggle.com/datasets/camnugent/sandp500).

1. Place your stock CSV files in the format: `data/{STOCK_SYMBOL}_data.csv`
2. CSV format should have columns: `Date,Open,High,Low,Close,Volume`

## Configuration
Create a `config.txt` file in the project root with:
```
HOLDOUT_SIZE=20
STOCK=NVDA
```
- `HOLDOUT_SIZE`: Number of data points to reserve for validation
- `STOCK`: Stock symbol (must match the CSV filename without `_data.csv`)

## Running the Project

### Build and Run C++ Implementation
```bash
make && ./bin/arima
```

### Run Python Comparison
```bash
python3 src/test.py
```
## Debugging
For verbose output, compile with debug flags:
```bash
make CFLAGS="-DVERBOSE -DDEBUG"
```
