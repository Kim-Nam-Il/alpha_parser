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
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
