from alpha_parser.alpha_lexer import AlphaLexer
from alpha_parser.alpha_parser import Parser
import pandas as pd
import numpy as np

def parse_and_evaluate(expression, variables=None):
    print(f"\n{'='*50}")
    print(f"Input expression: {expression}")
    if variables:
        print(f"Variable values: {variables}")
    try:
        parser = Parser(AlphaLexer(expression))
        ast = parser.parse()
        print(f"AST structure: {ast}")
        result = ast.evaluate(variables)
        print(f"Calculation result: {result}")
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        return False
    finally:
        print(f"{'='*50}")

# Sample data for testing
def generate_sample_data(size=100):
    np.random.seed(42)
    data = {
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
    return data

# Alpha formulas
alpha_formulas = {
    'Alpha#1': "(rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20): close), 2.), 5)) - 0.5)",
    'Alpha#2': "(-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))",
    'Alpha#3': "(-1 * correlation(rank(open), rank(volume), 10))",
    'Alpha#4': "(-1 * Ts_Rank(rank(low), 9))",
    'Alpha#5': "(rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))",
    'Alpha#6': "(-1 * correlation(open, volume, 10))",
    'Alpha#7': "((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1 * 1))",
    'Alpha#8': "(-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))",
    'Alpha#9': "((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))",
    'Alpha#10': "rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))",
    'Alpha#11': "((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))",
    'Alpha#12': "sign(delta(volume, 1)) * (-1 * delta(close, 1))",
    'Alpha#13': "(-1 * rank(covariance(rank(close), rank(volume), 5)))",
    'Alpha#14': "((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))",
    'Alpha#15': "(-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))",
    'Alpha#16': "(-1 * rank(covariance(rank(high), rank(volume), 5)))",
    'Alpha#17': "(((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank((volume / adv20), 5)))",
    'Alpha#18': "(-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10))))",
    'Alpha#19': "((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))",
    'Alpha#20': "(((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - delay(low, 1))))",
    'Alpha#21': "((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1 * 1) : (((sum(close, 2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 : (((1 < (volume / adv20)) || ((volume / adv20) == 1)) ? 1 : (-1 * 1))))",
    'Alpha#22': "(-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))",
    'Alpha#23': "(((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)",
    'Alpha#24': "((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) || ((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05)) ? (-1 * (close - ts_min(close, 100))) : (-1 * delta(close, 3)))",
    'Alpha#25': "rank(((((-1 * returns) * adv20) * vwap) * (high - close)))"
}

def test_all_alphas():
    data = generate_sample_data()
    
    # Convert data to lists for the parser
    variables = {k: list(v) for k, v in data.items()}
    
    results = {}
    for name, formula in alpha_formulas.items():
        print(f"\nTesting {name}")
        success = parse_and_evaluate(formula, variables)
        results[name] = success
    
    # Print summary
    print("\nTest Summary:")
    print(f"Total alphas tested: {len(results)}")
    print(f"Successful: {sum(results.values())}")
    print(f"Failed: {len(results) - sum(results.values())}")
    
    # Print failed alphas
    failed = [name for name, success in results.items() if not success]
    if failed:
        print("\nFailed Alphas:")
        for name in failed:
            print(f"- {name}")

if __name__ == '__main__':
    test_all_alphas() 