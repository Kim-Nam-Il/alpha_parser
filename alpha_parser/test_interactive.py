from alpha_parser.lexer import Lexer
from alpha_parser.parser import Parser

def parse_and_evaluate(expression, variables=None):
    print(f"\n{'='*50}")
    print(f"Input expression: {expression}")
    if variables:
        print(f"Variable values: {variables}")
    parser = Parser(Lexer(expression))
    ast = parser.parse()
    print(f"AST structure: {ast}")
    result = ast.evaluate(variables)
    print(f"Calculation result: {result}")
    print(f"{'='*50}")
    return result

# Test cases
test_expressions = [
    # Basic arithmetic operations
    ("1 + 2 * 3", None),
    ("(1 + 2) * 3", None),
    ("x + y", {'x': 10, 'y': 5}),
    ("price * quantity", {'price': 100, 'quantity': 3}),
    
    # Time series data tests
    ("close - open", {
        'close': [100, 101, 102],
        'open': [90, 91, 92]
    }),
    ("delta(close, 2)", {
        'close': [100, 101, 102, 103, 104]
    }),
    ("rank(close)", {
        'close': [100, 101, 102, 103, 104]
    }),
    
    # indneutralize tests
    ("indneutralize(returns, industry)", {
        'returns': [0.1, 0.2, 0.3, 0.4, 0.5],
        'industry': ['tech', 'tech', 'finance', 'finance', 'tech']
    }),
    ("indneutralize(volume, sector)", {
        'volume': [1000, 2000, 3000, 4000, 5000],
        'sector': ['A', 'A', 'B', 'B', 'A']
    })
]

if __name__ == '__main__':
    for expr, vars in test_expressions:
        parse_and_evaluate(expr, vars) 