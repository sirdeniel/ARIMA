#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
import os

HOLDOUT_SIZE = None
company = None

try:
    with open("config.txt", "r") as config_file:
        for line in config_file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                if key == "HOLDOUT_SIZE":
                    HOLDOUT_SIZE = int(value)
                elif key == "STOCK":
                    company = value
except FileNotFoundError:
    print("Error: Could not open ../config.txt file")
    print("Please create config.txt with HOLDOUT_SIZE and STOCK settings")
    exit(1)

# Validate required configuration values
missing_values = []
if HOLDOUT_SIZE is None:
    missing_values.append("HOLDOUT_SIZE")
if company is None:
    missing_values.append("STOCK")

if missing_values:
    print("Error: Missing required configuration values")
    for value in missing_values:
        print(f"  Missing: {value}")
    exit(1)

CSV_PATH = f"data/{company}_data.csv"
CPP_OUTPUT_FILE = "data/arima_output.csv"

print(f"Using configuration: HOLDOUT_SIZE={HOLDOUT_SIZE}, STOCK={company}")

def pick_close(df: pd.DataFrame) -> pd.Series:
    """Extract close price column from dataframe"""
    cols_lower = {c.lower(): c for c in df.columns}
    if "close" in cols_lower:
        s = df[cols_lower["close"]]
    else:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            raise SystemExit("No numeric columns found in CSV")
        s = df[num_cols[0]]
    return pd.to_numeric(s, errors="coerce").dropna()

def choose_d(y: pd.Series, alpha: float = 0.05, max_d: int = 2) -> int:
    """Determine differencing order using ADF test"""
    d = 0
    x = y.copy()
    while d < max_d:
        try:
            _stat, pval, *_ = adfuller(x, autolag="AIC")
        except Exception:
            break
        if pval < alpha:
            break
        x = x.diff().dropna()
        d += 1
        if len(x) < 10:
            break
    return d

def last_significant(series_vals: np.ndarray, band: float) -> int:
    """Find last significant lag above threshold"""
    last = 0
    for k in range(1, len(series_vals)):
        if abs(series_vals[k]) > band:
            last = k
    return last

def load_cpp_results():
    """Load C++ ARIMA results from CSV file"""
    if not os.path.exists(CPP_OUTPUT_FILE):
        raise FileNotFoundError(f"C++ output file '{CPP_OUTPUT_FILE}' not found. Run C++ program first.")
    
    df = pd.read_csv(CPP_OUTPUT_FILE)
    training = df[df['type'] == 'training']
    actual = df[df['type'] == 'actual']
    cpp_predictions = df[df['type'] == 'prediction']
    
    return training, actual, cpp_predictions

def run_python_arima():
    """Run Python ARIMA on the same data split as C++"""
    # Load full dataset
    df = pd.read_csv(CSV_PATH)
    s = pick_close(df).astype("float64").dropna()
    
    # Split data same as C++
    train_data = s.iloc[:-HOLDOUT_SIZE]
    actual_holdout = s.iloc[-HOLDOUT_SIZE:].values
    
    # Determine parameters same as C++
    d = choose_d(train_data, alpha=0.05, max_d=2)
    y = train_data.copy()
    for _ in range(d):
        y = y.diff().dropna()
    
    T = len(y)
    K = max(1, int(np.sqrt(T)))
    band = 1.96 / np.sqrt(T)
    
    # ACF and PACF
    acf_vals = acf(y, nlags=K, fft=True, adjusted=False)
    pacf_vals = pacf(y, nlags=K, method="ywm")
    
    p = last_significant(pacf_vals, band)
    q = last_significant(acf_vals, band)
    
    print(f"Python ARIMA parameters: p={p}, d={d}, q={q}")
    
    # Fit ARIMA and predict
    try:
        model = ARIMA(train_data, order=(p, d, q))
        fitted_model = model.fit()
        python_forecast = fitted_model.forecast(steps=HOLDOUT_SIZE)
        return python_forecast.values, actual_holdout, p, d, q
    except Exception as e:
        print(f"Python ARIMA fitting failed: {e}")
        return None, actual_holdout, p, d, q

def compare_and_plot():
    """Main function to compare C++ and Python ARIMA results"""
    
    # Load C++ results
    training, actual, cpp_predictions = load_cpp_results()
    print("Loaded C++ results successfully")
    
    # Run Python ARIMA
    python_predictions, actual_holdout, p, d, q = run_python_arima()
    
    if python_predictions is None:
        print("Python ARIMA failed, showing only C++ results")
        python_predictions = np.zeros(HOLDOUT_SIZE)
    
    # Extract values for comparison
    cpp_pred_values = cpp_predictions['value'].values
    actual_values = actual['value'].values
    
    # Calculate performance metrics
    print(f"\n=== PERFORMANCE COMPARISON ===")
    print(f"ARIMA({p},{d},{q}) Model Validation")
    print(f"Holdout period: {HOLDOUT_SIZE} steps")
    
    # C++ metrics
    cpp_errors = cpp_pred_values - actual_values
    cpp_mae = np.mean(np.abs(cpp_errors))
    cpp_rmse = np.sqrt(np.mean(cpp_errors**2))
    
    # Python metrics
    py_errors = python_predictions - actual_values
    py_mae = np.mean(np.abs(py_errors))
    py_rmse = np.sqrt(np.mean(py_errors**2))
    
    print(f"\nC++ Implementation:")
    print(f"  MAE:  {cpp_mae:.4f}")
    print(f"  RMSE: {cpp_rmse:.4f}")
    
    print(f"\nPython Implementation:")
    print(f"  MAE:  {py_mae:.4f}")
    print(f"  RMSE: {py_rmse:.4f}")
    
    # Implementation comparison
    impl_diff = np.abs(cpp_pred_values - python_predictions)
    impl_mae = np.mean(impl_diff)
    impl_rmse = np.sqrt(np.mean(impl_diff**2))
    
    print(f"\nImplementation Difference (C++ vs Python):")
    print(f"  MAE:  {impl_mae:.4f}")
    print(f"  RMSE: {impl_rmse:.4f}")
    
    # Detailed step-by-step comparison
    print(f"\n=== DETAILED COMPARISON ===")
    print(f"{'Step':<4} {'Actual':<10} {'C++':<10} {'Python':<10} {'C++ Err':<10} {'Py Err':<10} {'Diff':<10}")
    print("-" * 70)
    for i in range(HOLDOUT_SIZE):
        cpp_err = cpp_pred_values[i] - actual_values[i]
        py_err = python_predictions[i] - actual_values[i]
        diff = cpp_pred_values[i] - python_predictions[i]
        print(f"{i+1:<4} {actual_values[i]:<10.4f} {cpp_pred_values[i]:<10.4f} {python_predictions[i]:<10.4f} "
              f"{cpp_err:<10.4f} {py_err:<10.4f} {diff:<10.4f}")
    
    # Create comprehensive plot
    plt.figure(figsize=(16, 10))
    
    # Main comparison plot
    plt.subplot(2, 2, (1, 2))
    
    # Plot training data (last 50 points)
    train_start = max(0, len(training) - 50)
    train_subset = training.iloc[train_start:]
    plt.plot(train_subset['index'], train_subset['value'], 
             label='Training Data', color='blue', linewidth=2, marker='o', markersize=3)
    
    # Plot actual holdout
    plt.plot(actual['index'], actual['value'], 
             label='Actual (Holdout)', color='green', linewidth=3, marker='o', markersize=6)
    
    # Plot C++ predictions
    plt.plot(cpp_predictions['index'], cpp_predictions['value'], 
             label=f'C++ ARIMA ({cpp_mae:.4f} MAE)', color='red', linewidth=3, marker='s', markersize=6)
    
    # Plot Python predictions
    holdout_indices = cpp_predictions['index'].values
    plt.plot(holdout_indices, python_predictions, 
             label=f'Python ARIMA ({py_mae:.4f} MAE)', color='orange', linewidth=3, marker='^', markersize=6)
    
    plt.title(f'ARIMA({p},{d},{q}) Comparison: C++ vs Python Implementation', fontsize=14, fontweight='bold')
    plt.xlabel('Time Index', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Error comparison plot
    plt.subplot(2, 2, 3)
    steps = range(1, HOLDOUT_SIZE + 1)
    plt.bar([x - 0.2 for x in steps], np.abs(cpp_errors), width=0.4, 
            label='C++ Abs Error', color='red', alpha=0.7)
    plt.bar([x + 0.2 for x in steps], np.abs(py_errors), width=0.4, 
            label='Python Abs Error', color='orange', alpha=0.7)
    plt.xlabel('Prediction Step')
    plt.ylabel('Absolute Error')
    plt.title('Absolute Errors by Step')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Implementation difference plot
    plt.subplot(2, 2, 4)
    plt.bar(steps, impl_diff, color='purple', alpha=0.7)
    plt.xlabel('Prediction Step')
    plt.ylabel('|C++ - Python|')
    plt.title('Implementation Difference')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'cpp_mae': cpp_mae, 'cpp_rmse': cpp_rmse,
        'py_mae': py_mae, 'py_rmse': py_rmse,
        'impl_mae': impl_mae, 'impl_rmse': impl_rmse
    }

if __name__ == "__main__":
    try:
        results = compare_and_plot()
        print(f"\n=== SUMMARY ===")
        if results['cpp_mae'] < results['py_mae']:
            print("🎉 C++ implementation performs better!")
        elif results['py_mae'] < results['cpp_mae']:
            print("📊 Python implementation performs better.")
        else:
            print("🤝 Both implementations perform equally well.")
        print(f"Implementation similarity: {results['impl_mae']:.4f} MAE difference")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to run the C++ program first to generate arima_output.csv")
