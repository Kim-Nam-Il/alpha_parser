# Alpha Parser

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![Code Coverage](https://img.shields.io/badge/coverage-90%25-yellowgreen)]()

A Python library for parsing and analyzing financial Alpha formulas. This library is inspired by the paper "101 Formulaic Alphas" by Zura Kakushadze, which provides a comprehensive set of quantitative trading signals.

## 📦 Installation

```bash
pip install alpha_parser
```

## 🚀 Usage

### Basic Usage
```python
from alpha_parser.lexer import Lexer
from alpha_parser.parser import Parser

# Formula string
formula = "(rank(close) - 0.5)"

# Create lexer
lexer = Lexer(formula)

# Create parser and parse
parser = Parser(lexer)
result = parser.parse()

print(result)
```

Output:
```
ASTNode(BINARY_OP, value=TokenType.MINUS, children=[
    ASTNode(FUNCTION_CALL, value=rank, children=[
        ASTNode(VARIABLE, value=close, children=[])
    ]), 
    ASTNode(NUMBER, value=0.5, children=[])
])
```

This output shows the Abstract Syntax Tree (AST) structure of the formula:
- Root node is a binary operation (MINUS)
- Left child is a function call to `rank` with `close` as its argument
- Right child is a number (0.5)

## ✨ Features

- Expression tokenization (lexing)
- Expression parsing
- Basic arithmetic operations
- Variable and function support
- Parentheses handling
- Time series data processing
- Special function support (rank, delay, correlation, etc.)

## 📚 Supported Functions and Operators

### Basic Functions and Operators
- `abs(x)`: Absolute value (standard definition)
- `log(x)`: Natural logarithm (standard definition)
- `sign(x)`: Sign function (-1, 0, 1) (standard definition)
- Basic operators: `+`, `-`, `*`, `/`, `>`, `<`, `==`, `||`, `x ? y : z` (ternary operator)

### Core Functions
- `rank(x)`: Cross-sectional rank. Calculates the rank of data x across multiple stocks at a specific point in time.
- `delay(x, d)`: Value of x from d days ago.
- `correlation(x, y, d)`: Time-serial correlation between x and y over the past d days.
- `covariance(x, y, d)`: Time-serial covariance between x and y over the past d days.
- `scale(x, a=1)`: Rescales x cross-sectionally so that sum(abs(x)) equals a (default a=1).
- `delta(x, d)`: Difference between today's x value and x value from d days ago (x - delay(x, d)).
- `signedpower(x, a)`: Power with sign preservation: sign(x)×(∣x∣^a).
- `decay_linear(x, d)`: Weighted moving average over the past d days. Weights decrease linearly from d to 1 and are rescaled to sum to 1.
- `indneutralize(x, g)`: Cross-sectionally neutralizes x with respect to group g (e.g., subindustry, industry, sector).
- `ts_min(x, d)`: Time-series minimum over the past d days.
- `ts_max(x, d)`: Time-series maximum over the past d days.
- `ts_argmax(x, d)`: Position/index where ts_max(x, d) occurred in the past d days.
- `ts_argmin(x, d)`: Position/index where ts_min(x, d) occurred in the past d days.
- `ts_rank(x, d)`: Time-series rank over the past d days.
- `min(x, d)`: Alias for ts_min(x, d).
- `max(x, d)`: Alias for ts_max(x, d).
- `sum(x, d)`: Time-series sum over the past d days.
- `product(x, d)`: Time-series product over the past d days.
- `stddev(x, d)`: Moving time-series standard deviation over the past d days.

## 📖 References

- Kakushadze, Z. (2016). 101 Formulaic Alphas. Wilmott, 2016(84), 72-81. [arXiv:1601.00991](https://arxiv.org/abs/1601.00991)

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/add-new-alpha-function` or `git checkout -b fix/correlation-calculation`)
3. Commit your changes (`git commit -m 'Add new alpha function for momentum calculation'`)
4. Push to the branch (`git push origin feature/add-new-alpha-function`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Usage Example

Here's a sample code to test alpha formulas:

```python
from alpha_parser.alpha_lexer import AlphaLexer
from alpha_parser.alpha_parser import Parser
import pandas as pd
import numpy as np

# Generate sample data
def generate_sample_data(size=100):
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=size, freq='D')
    
    data = {
        'date': dates,
        'open': np.random.randn(size) * 10 + 100,
        'high': np.random.randn(size) * 10 + 102,
        'low': np.random.randn(size) * 10 + 98,
        'close': np.random.randn(size) * 10 + 101,
        'volume': np.abs(np.random.randn(size) * 1000000 + 5000000),
        'returns': np.random.randn(size) * 0.02,
        'vwap': np.random.randn(size) * 10 + 100.5,
        'adv20': np.abs(np.random.randn(size) * 800000 + 4000000),
        'cap': np.abs(np.random.randn(size) * 1000000000 + 5000000000),
        'industry': np.random.choice(['tech', 'finance', 'health', 'energy'], size=size),
        'sector': np.random.choice(['A', 'B', 'C', 'D'], size=size)
    }
    return pd.DataFrame(data)

# Parse and evaluate alpha formula
def parse_and_evaluate(expression, variables):
    print(f"\n{'='*50}")
    print(f"Input expression: {expression}")
    if variables is not None:
        print(f"Data shape: {variables.shape}")
    try:
        parser = Parser(AlphaLexer(expression))
        ast = parser.parse()
        print(f"AST structure: {ast}")
        
        # Calculate for each date
        results = []
        positions = []
        for i in range(len(variables)):
            # Use data up to current date
            current_data = {col: variables[col].iloc[:i+1].tolist() for col in variables.columns if col != 'date'}
            result = ast.evaluate(current_data)
            results.append(result[-1] if isinstance(result, list) else result)
            
            # Determine position (long if positive, short if negative)
            position = 1 if result[-1] > 0 else -1 if result[-1] < 0 else 0
            positions.append(position)
        
        # Convert results to DataFrame
        result_df = pd.DataFrame({
            'date': variables['date'],
            'alpha_value': results,
            'position': positions,
            'close': variables['close']
        })
        
        # Calculate PnL
        result_df['returns'] = result_df['close'].pct_change()
        result_df['pnl'] = result_df['position'].shift(1) * result_df['returns']
        result_df['cumulative_pnl'] = result_df['pnl'].cumsum()
        
        print("\nResults:")
        print(result_df[['date', 'alpha_value', 'position', 'close', 'pnl', 'cumulative_pnl']].tail())
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        return False
    finally:
        print(f"{'='*50}")

# Test alpha formula
def test_alpha(formula, data):
    print(f"\nTesting formula: {formula}")
    try:
        result = parse_and_evaluate(formula, data)
        return True
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return False

# Sample alpha formulas to test
alpha_formulas = {
    'Alpha#1': "(rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20): close), 2.), 5)) - 0.5)",
    'Alpha#101': "((close - open) / ((high - low) + .001))"
}

# Prepare data
data = generate_sample_data()

# Test each alpha formula
for name, formula in alpha_formulas.items():
    print(f"\n{'='*50}")
    print(f"Testing: {name}")
    test_alpha(formula, data)

## Alpha Formula Examples and Test Results

### Alpha#1
Formula: `(rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20): close), 2.), 5)) - 0.5)`

Description:
- Uses 20-day standard deviation when returns are negative, otherwise uses close price
- Squares the value and finds the maximum over 5 days
- Ranks the result and subtracts 0.5 to center around zero

Test Results:

Results:
         date  alpha_value  position       close       pnl  cumulative_pnl
95 2023-04-06     0.047368         1   96.308243 -0.146161        0.544046
96 2023-04-07    -0.104167        -1   83.868655 -0.129164        0.414882
97 2023-04-08    -0.304124        -1  114.538724 -0.365692        0.049190
98 2023-04-09    -0.295918        -1   99.854602  0.128202        0.177392
99 2023-04-10     0.500000         1  113.378163 -0.135433        0.041960


### Alpha#101
Formula: `((close - open) / ((high - low) + .001))`

Description:
- Simple price-based formula
- Divides (close-open) by (high-low)
- Adds 0.001 to denominator to prevent division by zero

Test Results:

Results:
         date  alpha_value  position       close       pnl  cumulative_pnl
95 2023-04-06     0.740255         1   96.308243 -0.146161       -1.416476
96 2023-04-07     1.380160         1   83.868655 -0.129164       -1.545640
97 2023-04-08     4.838512         1  114.538724  0.365692       -1.179949
98 2023-04-09     0.055431         1   99.854602 -0.128202       -1.308151
99 2023-04-10    -1.145650        -1  113.378163  0.135433       -1.172718

# CryptoBacktest

A cryptocurrency backtesting framework

## Features

- Alpha formula parsing and evaluation
- Backtesting engine
- Performance analysis and visualization

## Installation

```bash
pip install -r requirements.txt
```

## Usage Examples

### Alpha Backtesting

You can backtest alpha formulas using `examples/alpha_backtest.py`:

```python
from alpha_parser.alpha_lexer import AlphaLexer
from alpha_parser.alpha_parser import Parser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create backtester instance
backtester = AlphaBacktester(lookback_window=20)

# Generate sample data
data = backtester.generate_sample_data(size=100)

# Test alpha formula
alpha_formula = "((close - open) / ((high - low) + .001))"
backtester.test_alpha(alpha_formula, data)
```

### Supported Alpha Formulas

Currently supported alpha formula examples:

1. Alpha#1: `(rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20): close), 2.), 5)) - 0.5)`
2. Alpha#101: `((close - open) / ((high - low) + .001))`

### Backtesting Results

Backtesting results include the following metrics:
- Total Return
- Sharpe Ratio
- Maximum Drawdown

Also provides the following visualizations:
- Cumulative PnL graph
- Alpha value graph
- Position graph
- Daily PnL distribution

## License

MIT License