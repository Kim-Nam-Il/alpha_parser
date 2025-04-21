from typing import List, Optional, Union, Dict, Any
from alpha_parser.tokens import Token, TokenType
from alpha_parser.alpha_lexer import AlphaLexer
import math

class ASTNode:
    def __init__(self, type: str, value: Any = None, children: List['ASTNode'] = None):
        self.type = type
        self.value = value
        self.children = children or []
        
    def __repr__(self) -> str:
        return f"ASTNode({self.type}, value={self.value}, children={self.children})"
        
    def evaluate(self, variables: dict = None) -> Any:
        """Evaluate AST node and return the result"""
        if variables is None:
            variables = {}
            
        if self.type == 'NUMBER':
            return float(self.value)
        elif self.type == 'VARIABLE':
            # Convert to list if it's time series data
            value = variables.get(self.value, 0)
            return [value] if not isinstance(value, list) else value
        elif self.type == 'LIST':
            return [node.evaluate(variables) for node in self.children]
        elif self.type == 'BINARY_OP':
            left = self.children[0].evaluate(variables)
            right = self.children[1].evaluate(variables)
            
            # Handle time series data
            if isinstance(left, list) or isinstance(right, list):
                if not isinstance(left, list):
                    left = [left] * len(right)
                if not isinstance(right, list):
                    right = [right] * len(left)
                    
                if len(left) != len(right):
                    raise ValueError("Lists must have the same length for binary operations")
                    
                if self.value == TokenType.PLUS:
                    return [l + r for l, r in zip(left, right)]
                elif self.value == TokenType.MINUS:
                    return [l - r for l, r in zip(left, right)]
                elif self.value == TokenType.MULTIPLY:
                    return [l * r for l, r in zip(left, right)]
                elif self.value == TokenType.DIVIDE:
                    if any(r == 0 for r in right):
                        raise ValueError("Division by zero")
                    return [l / r for l, r in zip(left, right)]
                elif self.value == TokenType.POWER:
                    return [l ** r for l, r in zip(left, right)]
                elif self.value == TokenType.EQUAL:
                    return [l == r for l, r in zip(left, right)]
                elif self.value == TokenType.NOT_EQUAL:
                    return [l != r for l, r in zip(left, right)]
                elif self.value == TokenType.GREATER:
                    return [l > r for l, r in zip(left, right)]
                elif self.value == TokenType.LESS:
                    return [l < r for l, r in zip(left, right)]
                elif self.value == TokenType.GREATER_EQUAL:
                    return [l >= r for l, r in zip(left, right)]
                elif self.value == TokenType.LESS_EQUAL:
                    return [l <= r for l, r in zip(left, right)]
                elif self.value == TokenType.AND:
                    return [l and r for l, r in zip(left, right)]
                elif self.value == TokenType.OR:
                    return [l or r for l, r in zip(left, right)]
            # Handle single values
            else:
                if self.value == TokenType.PLUS:
                    return left + right
                elif self.value == TokenType.MINUS:
                    return left - right
                elif self.value == TokenType.MULTIPLY:
                    return left * right
                elif self.value == TokenType.DIVIDE:
                    if right == 0:
                        raise ValueError("Division by zero")
                    return left / right
                elif self.value == TokenType.POWER:
                    return left ** right
        elif self.type == 'UNARY_OP':
            child = self.children[0].evaluate(variables)
            if isinstance(child, list):
                if self.value == TokenType.PLUS:
                    return child
                elif self.value == TokenType.MINUS:
                    return [-x for x in child]
                elif self.value == TokenType.NOT:
                    return [not x for x in child]
            else:
                if self.value == TokenType.PLUS:
                    return +child
                elif self.value == TokenType.MINUS:
                    return -child
                elif self.value == TokenType.NOT:
                    return not child
        elif self.type == 'FUNCTION_CALL':
            func_name = self.value.lower()
            args = [child.evaluate(variables) for child in self.children]
            
            # Convert all arguments to lists
            args = [[arg] if not isinstance(arg, list) else arg for arg in args]
            
            if func_name == 'rank':
                x = args[0]
                # Store values with their indices
                indexed_values = [(val, i) for i, val in enumerate(x)]
                # Sort by value (ascending)
                sorted_values = sorted(indexed_values, key=lambda x: x[0])
                
                # Calculate ranks (normalized between 0 and 1)
                ranks = [0] * len(x)
                for i, (val, idx) in enumerate(sorted_values):
                    ranks[idx] = i / (len(x) - 1) if len(x) > 1 else 0
                
                return ranks
            elif func_name == 'delay':
                x, n = args
                n = int(n[0])
                if len(x) <= n:
                    return [None] * len(x)
                return [None] * n + x[:-n]
            elif func_name == 'correlation':
                x, y, n = args
                if len(x) != len(y):
                    raise ValueError("Lists must have the same length for correlation")
                n = int(n[0])
                if len(x) < n or len(y) < n:
                    return 0
                x = x[-n:]
                y = y[-n:]
                x_mean = sum(x) / n
                y_mean = sum(y) / n
                cov = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y)) / n
                x_std = (sum((xi - x_mean) ** 2 for xi in x) / n) ** 0.5
                y_std = (sum((yi - y_mean) ** 2 for yi in y) / n) ** 0.5
                return cov / (x_std * y_std) if x_std * y_std != 0 else 0
            elif func_name == 'covariance':
                x, y, n = args
                if len(x) != len(y):
                    raise ValueError("Lists must have the same length for covariance")
                n = int(n[0])
                if len(x) < n or len(y) < n:
                    return 0
                x = x[-n:]
                y = y[-n:]
                x_mean = sum(x) / n
                y_mean = sum(y) / n
                return sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y)) / n
            elif func_name == 'scale':
                x = args[0]
                if not x:
                    raise ValueError("Cannot scale an empty list")
                x_min = min(x)
                x_max = max(x)
                return [(xi - x_min) / (x_max - x_min) for xi in x] if x_max != x_min else [0] * len(x)
            elif func_name == 'delta':
                x, n = args
                n = int(n[0])
                if len(x) <= n:
                    raise ValueError(f"List length must be greater than {n} for delta")
                return x[-1] - x[-n-1]
            elif func_name == 'signedpower':
                x, power = args
                power = float(power[0])
                return [abs(xi) ** power * (1 if xi >= 0 else -1) for xi in x]
            elif func_name == 'decay_linear':
                x, n = args
                n = int(n[0])
                if len(x) < n:
                    raise ValueError(f"List length must be at least {n} for decay_linear")
                weights = [i+1 for i in range(n)]
                total_weight = sum(weights)
                return sum(x[-n+i] * weights[i] for i in range(n)) / total_weight
            elif func_name == 'indneutralize':
                x, g = args
                if len(x) != len(g):
                    raise ValueError("Lists must have the same length for indneutralize")
                if not x or not g:
                    raise ValueError("Empty lists are not allowed for indneutralize")
                    
                # Calculate group means
                group_means = {}
                group_counts = {}
                for i, group in enumerate(g):
                    if group not in group_means:
                        group_means[group] = 0
                        group_counts[group] = 0
                    group_means[group] += x[i]
                    group_counts[group] += 1
                
                # Calculate final means
                for group in group_means:
                    group_means[group] /= group_counts[group]
                
                # Neutralize
                result = []
                for i, group in enumerate(g):
                    result.append(x[i] - group_means[group])
                    
                return result
            elif func_name == 'ts_min':
                x, n = args
                n = int(n[0])
                if len(x) <= n:
                    return [None] * len(x)
                return self.ts_min(x, n)
            elif func_name == 'ts_max':
                x, n = args
                n = int(n[0])
                result = []
                for i in range(len(x)):
                    start = max(0, i - n)          # ← 수정 ⬅︎
                    result.append(max(x[start:i+1]))
                return result
                
            elif func_name == 'ts_argmax':
                x, n = args
                n = int(n[0])
                if len(x) < n:
                    return x.index(max(x))
                return x[-n:].index(max(x[-n:]))
                
            elif func_name == 'ts_argmin':
                x, n = args
                n = int(n[0])
                if len(x) < n:
                    return x.index(min(x))
                return x[-n:].index(min(x[-n:]))
                
            elif func_name == 'ts_rank':
                x, n = args
                n = int(n[0])
                if len(x) < n:
                    return 0
                x = x[-n:]
                # Store values with their indices
                indexed_values = [(val, i) for i, val in enumerate(x)]
                # Sort by value (ascending)
                sorted_values = sorted(indexed_values, key=lambda x: x[0])
                
                # Calculate ranks (normalized between 0 and 1)
                ranks = [0] * len(x)
                for i, (val, idx) in enumerate(sorted_values):
                    ranks[idx] = i / (len(x) - 1) if len(x) > 1 else 0
                
                return ranks[-1]
            else:
                raise ValueError(f"Unknown function: {func_name}")
        else:
            raise ValueError(f"Unknown node type: {self.type}")

    def ts_min(self, data: List[float], window: int) -> List[float]:
        if not data or window <= 0:
            return []
        result = []
        for i in range(len(data)):
            if i < window - 1:
                result.append(None)
            else:
                window_data = data[i - window + 1:i + 1]
                result.append(min(window_data))
        return result

class Parser:
    def __init__(self, lexer: AlphaLexer):
        self.lexer = lexer
        self.current_token = self.lexer.get_next_token()
        
    def error(self, message: str):
        raise Exception(f'Parser error: {message}')
        
    def eat(self, token_type: TokenType):
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            self.error(f'Expected {token_type}, got {self.current_token.type}')
            
    def atom(self) -> ASTNode:
        token = self.current_token
        
        if token.type == TokenType.NUMBER:
            self.eat(TokenType.NUMBER)
            return ASTNode('NUMBER', float(token.value))
            
        elif token.type == TokenType.IDENTIFIER:
            self.eat(TokenType.IDENTIFIER)
            if self.current_token.type == TokenType.LPAREN:
                raise ValueError(f"Unknown function: {token.value}")
            return ASTNode('VARIABLE', token.value)
            
        elif token.type == TokenType.FUNCTION:
            return self.function_call()
            
        elif token.type == TokenType.LPAREN:
            self.eat(TokenType.LPAREN)
            node = self.expr()
            self.eat(TokenType.RPAREN)
            return node
            
        elif token.type == TokenType.LBRACKET:
            self.eat(TokenType.LBRACKET)
            values = []
            if self.current_token.type != TokenType.RBRACKET:
                values.append(self.expr())
                while self.current_token.type == TokenType.COMMA:
                    self.eat(TokenType.COMMA)
                    values.append(self.expr())
            self.eat(TokenType.RBRACKET)
            return ASTNode('LIST', None, values)
            
        elif token.type in [TokenType.PLUS, TokenType.MINUS]:
            self.eat(token.type)
            return ASTNode('UNARY_OP', token.type, [self.factor()])
            
        self.error(f'Unexpected token: {token}')
        
    def function_call(self) -> ASTNode:
        token = self.current_token
        self.eat(TokenType.FUNCTION)
        self.eat(TokenType.LPAREN)
        
        args = []
        if self.current_token.type != TokenType.RPAREN:
            args.append(self.expr())
            while self.current_token.type == TokenType.COMMA:
                self.eat(TokenType.COMMA)
                args.append(self.expr())
                
        self.eat(TokenType.RPAREN)
        return ASTNode('FUNCTION_CALL', token.value, args)
        
    def factor(self) -> ASTNode:
        token = self.current_token
        
        if token.type in [TokenType.PLUS, TokenType.MINUS, TokenType.NOT]:
            self.eat(token.type)
            return ASTNode('UNARY_OP', token.type, [self.factor()])
            
        return self.atom()
        
    def power(self) -> ASTNode:
        """멱승 연산을 처리하는 메서드"""
        node = self.factor()
        
        if self.current_token.type == TokenType.MULTIPLY and self.lexer.peek().type == TokenType.MULTIPLY:
            self.eat(TokenType.MULTIPLY)
            self.eat(TokenType.MULTIPLY)
            node = ASTNode('BINARY_OP', TokenType.POWER, [node, self.factor()])
            
        return node
        
    def term(self) -> ASTNode:
        node = self.power()
        
        while self.current_token.type in [TokenType.MULTIPLY, TokenType.DIVIDE]:
            token = self.current_token
            self.eat(token.type)
            node = ASTNode('BINARY_OP', token.type, [node, self.power()])
            
        return node
        
    def arithmetic_expr(self) -> ASTNode:
        node = self.term()
        
        while self.current_token.type in [TokenType.PLUS, TokenType.MINUS]:
            token = self.current_token
            self.eat(token.type)
            node = ASTNode('BINARY_OP', token.type, [node, self.term()])
            
        return node
        
    def comparison_expr(self) -> ASTNode:
        node = self.arithmetic_expr()
        
        while self.current_token.type in [
            TokenType.EQUAL, TokenType.NOT_EQUAL,
            TokenType.GREATER, TokenType.LESS,
            TokenType.GREATER_EQUAL, TokenType.LESS_EQUAL
        ]:
            token = self.current_token
            self.eat(token.type)
            node = ASTNode('BINARY_OP', token.type, [node, self.arithmetic_expr()])
            
        return node
        
    def and_expr(self) -> ASTNode:
        node = self.comparison_expr()
        
        while self.current_token.type == TokenType.AND:
            token = self.current_token
            self.eat(TokenType.AND)
            node = ASTNode('BINARY_OP', token.type, [node, self.comparison_expr()])
            
        return node
        
    def or_expr(self) -> ASTNode:
        node = self.and_expr()
        
        while self.current_token.type == TokenType.OR:
            token = self.current_token
            self.eat(TokenType.OR)
            node = ASTNode('BINARY_OP', token.type, [node, self.and_expr()])
            
        return node
        
    def conditional_expr(self) -> ASTNode:
        node = self.or_expr()
        
        if self.current_token.type == TokenType.QUESTION:
            self.eat(TokenType.QUESTION)
            true_expr = self.expr()
            self.eat(TokenType.COLON)
            false_expr = self.conditional_expr()
            node = ASTNode('CONDITIONAL', None, [node, true_expr, false_expr])
            
        return node
        
    def expr(self) -> ASTNode:
        return self.conditional_expr()
        
    def parse(self) -> ASTNode:
        node = self.expr()
        if self.current_token.type != TokenType.EOF:
            self.error(f'Unexpected token at end: {self.current_token}')
        return node

class AlphaParser:
    def __init__(self):
        self.lexer = AlphaLexer()
        self.variables = {}
        
    def parse(self, formula: str) -> Any:
        self.lexer.input(formula)
        parser = Parser(self.lexer)
        ast = parser.parse()
        return ast.evaluate(self.variables)

    def evaluate(self, node: ASTNode) -> Union[float, List[float]]:
        if isinstance(node, list):
            return [self.evaluate(item) for item in node]
        
        if node.type == 'NUMBER':
            return float(node.value)
        
        elif node.type == 'VARIABLE':
            if node.value not in self.variables:
                raise ValueError(f"Unknown variable: {node.value}")
            return self.variables[node.value]
        
        elif node.type == 'LIST':
            return [self.evaluate(item) for item in node.children]
        
        elif node.type == 'BINARY_OP':
            left = self.evaluate(node.children[0])
            right = self.evaluate(node.children[1])
            
            # Convert scalar to list if needed
            if isinstance(left, (int, float)) and isinstance(right, list):
                left = [left] * len(right)
            elif isinstance(right, (int, float)) and isinstance(left, list):
                right = [right] * len(left)
            
            # Check list lengths
            if isinstance(left, list) and isinstance(right, list):
                if len(left) != len(right):
                    raise ValueError("Lists must have the same length for binary operations")
            
            if node.value == TokenType.PLUS:
                return [l + r for l, r in zip(left, right)] if isinstance(left, list) else left + right
            elif node.value == TokenType.MINUS:
                return [l - r for l, r in zip(left, right)] if isinstance(left, list) else left - right
            elif node.value == TokenType.MULTIPLY:
                return [l * r for l, r in zip(left, right)] if isinstance(left, list) else left * right
            elif node.value == TokenType.DIVIDE:
                if isinstance(right, list):
                    if any(r == 0 for r in right):
                        raise ValueError("Division by zero")
                elif right == 0:
                    raise ValueError("Division by zero")
                return [l / r for l, r in zip(left, right)] if isinstance(left, list) else left / right
            elif node.value == TokenType.POWER:
                return [l ** r for l, r in zip(left, right)] if isinstance(left, list) else left ** right
                
        elif node.type == 'UNARY_OP':
            value = self.evaluate(node.children[0])
            if isinstance(value, list):
                if node.value == TokenType.MINUS:
                    return [-x for x in value]
                elif node.value == TokenType.NOT:
                    return [not x for x in value]
            else:
                if node.value == TokenType.MINUS:
                    return -value
                elif node.value == TokenType.NOT:
                    return not value
        
        elif node.type == 'FUNCTION_CALL':
            func_name = node.value.lower()
            args = [self.evaluate(arg) for arg in node.children]
            
            if func_name == 'ts_min':
                x, n = args
                n = int(n[0])
                if len(x) < n:
                    raise ValueError(f"List length must be at least {n} for ts_min")
                return self.ts_min(x, n)
            
            elif func_name == 'ts_max':
                x, n = args
                n = int(n[0])
                if len(x) < n:
                    raise ValueError(f"List length must be at least {n} for ts_max")
                result = []
                for i in range(len(x)):
                    if i < n - 1:
                        result.append(max(x[:i+1]))
                    else:
                        result.append(max(x[i-n+1:i+1]))
                return [int(val) if val.is_integer() else val for val in result]
            
            elif func_name == 'delay':
                x, n = args
                n = int(n[0])
                if len(x) <= n:
                    return [None] * len(x)
                return [None] * n + x[:-n]
            
            elif func_name == 'correlation':
                x, y, n = args
                n = int(n[0])
                if len(x) < n or len(y) < n:
                    return 0
                x = x[-n:]
                y = y[-n:]
                x_mean = sum(x) / n
                y_mean = sum(y) / n
                cov = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y)) / n
                x_std = (sum((xi - x_mean) ** 2 for xi in x) / n) ** 0.5
                y_std = (sum((yi - y_mean) ** 2 for yi in y) / n) ** 0.5
                return cov / (x_std * y_std) if x_std * y_std != 0 else 0
            
            elif func_name == 'covariance':
                x, y, n = args
                n = int(n[0])
                if len(x) < n or len(y) < n:
                    return 0
                x = x[-n:]
                y = y[-n:]
                x_mean = sum(x) / n
                y_mean = sum(y) / n
                return sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y)) / n
            
            elif func_name == 'scale':
                x = args[0]
                x_min = min(x)
                x_max = max(x)
                return [(xi - x_min) / (x_max - x_min) for xi in x] if x_max != x_min else [0] * len(x)
            
            elif func_name == 'delta':
                x, n = args
                n = int(n[0])
                if len(x) <= n:
                    return 0
                return x[-1] - x[-n-1]
            
            elif func_name == 'signedpower':
                x, power = args
                power = float(power[0])
                return [abs(xi) ** power * (1 if xi >= 0 else -1) for xi in x]
            
            elif func_name == 'decay_linear':
                x, n = args
                n = int(n[0])
                if len(x) < n:
                    return 0
                weights = [i+1 for i in range(n)]
                total_weight = sum(weights)
                return sum(x[-n+i] * weights[i] for i in range(n)) / total_weight
            
            elif func_name == 'indneutralize':
                x, g = args
                if len(x) != len(g):
                    raise ValueError("Lists must have the same length for indneutralize")
                if not x or not g:
                    raise ValueError("Empty lists are not allowed for indneutralize")
                    
                # Calculate group means
                group_means = {}
                group_counts = {}
                for i, group in enumerate(g):
                    if group not in group_means:
                        group_means[group] = 0
                        group_counts[group] = 0
                    group_means[group] += x[i]
                    group_counts[group] += 1
                
                # Calculate final means
                for group in group_means:
                    group_means[group] /= group_counts[group]
                
                # Neutralize
                result = []
                for i, group in enumerate(g):
                    result.append(x[i] - group_means[group])
                    
                return result
            
            elif func_name == 'rank':
                x = args[0]
                # Store values with their indices
                indexed_values = [(val, i) for i, val in enumerate(x)]
                # Sort by value (ascending)
                sorted_values = sorted(indexed_values, key=lambda x: x[0])
                
                # Calculate ranks (normalized between 0 and 1)
                ranks = [0] * len(x)
                for i, (val, idx) in enumerate(sorted_values):
                    ranks[idx] = i / (len(x) - 1) if len(x) > 1 else 0
                
                return ranks
            
            elif func_name == 'ts_argmax':
                x, n = args
                n = int(n[0])
                if len(x) < n:
                    return x.index(max(x))
                return x[-n:].index(max(x[-n:]))
            
            elif func_name == 'ts_argmin':
                x, n = args
                n = int(n[0])
                if len(x) < n:
                    return x.index(min(x))
                return x[-n:].index(min(x[-n:]))
            
            elif func_name == 'ts_rank':
                x, n = args
                n = int(n[0])
                if len(x) < n:
                    return 0
                x = x[-n:]
                # Store values with their indices
                indexed_values = [(val, i) for i, val in enumerate(x)]
                # Sort by value (ascending)
                sorted_values = sorted(indexed_values, key=lambda x: x[0])
                
                # Calculate ranks (normalized between 0 and 1)
                ranks = [0] * len(x)
                for i, (val, idx) in enumerate(sorted_values):
                    ranks[idx] = i / (len(x) - 1) if len(x) > 1 else 0
                
                return ranks[-1]
            
            else:
                raise ValueError(f"Unknown function: {func_name}") 
        else:
            raise ValueError(f"Unknown node type: {node.type}") 