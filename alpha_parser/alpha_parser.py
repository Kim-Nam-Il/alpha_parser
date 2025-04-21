from typing import List, Optional, Union, Dict, Any
from alpha_parser.tokens import Token, TokenType
from alpha_parser.alpha_lexer import AlphaLexer
import math

class IndClass:
    def __init__(self, level: str):
        self.level = level.lower()
        if self.level not in ['sector', 'industry', 'subindustry']:
            raise ValueError("IndClass level must be one of: sector, industry, subindustry")

    def __eq__(self, other):
        if isinstance(other, IndClass):
            return self.level == other.level
        return False

    def __hash__(self):
        return hash(self.level)

    def __repr__(self):
        return f"IndClass.{self.level}"

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
        elif self.type == 'CONDITIONAL':
            condition = self.children[0].evaluate(variables)
            true_expr = self.children[1]
            false_expr = self.children[2]
            
            # Handle time series data
            if isinstance(condition, list):
                true_val = true_expr.evaluate(variables)
                false_val = false_expr.evaluate(variables)
                
                # Convert scalar values to lists if needed
                if not isinstance(true_val, list):
                    true_val = [true_val] * len(condition)
                if not isinstance(false_val, list):
                    false_val = [false_val] * len(condition)
                    
                return [t if c else f for c, t, f in zip(condition, true_val, false_val)]
            else:
                return true_expr.evaluate(variables) if condition else false_expr.evaluate(variables)
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
                n = int(n[0])
                if len(x) < n or len(y) < n:
                    return [0] * len(x)  # 초기값은 0으로 설정
                
                result = []
                for i in range(len(x)):
                    if i < n - 1:
                        result.append(0)  # n일 이전의 값들은 0으로 설정
                    else:
                        # 최근 n일 데이터 추출
                        x_window = x[i-n+1:i+1]
                        y_window = y[i-n+1:i+1]
                        
                        # 평균 계산
                        x_mean = sum(x_window) / n
                        y_mean = sum(y_window) / n
                        
                        # 공분산과 표준편차 계산
                        cov = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x_window, y_window)) / n
                        x_std = (sum((xi - x_mean) ** 2 for xi in x_window) / n) ** 0.5
                        y_std = (sum((yi - y_mean) ** 2 for yi in y_window) / n) ** 0.5
                        
                        # 상관계수 계산
                        if x_std * y_std != 0:
                            result.append(cov / (x_std * y_std))
                        else:
                            result.append(0)
                
                return result
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
                return [x[i] - x[i-n] for i in range(n, len(x))]
            elif func_name == 'log':
                x = args[0]
                if any(val <= 0 for val in x):
                    raise ValueError("Cannot take log of non-positive values")
                return [math.log(val) for val in x]
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
            elif func_name == 'stddev':
                x, n = args
                n = int(n[0])
                if len(x) < n:
                    raise ValueError(f"List length must be at least {n} for stddev")
                x = x[-n:]
                mean = sum(x) / n
                variance = sum((xi - mean) ** 2 for xi in x) / n
                return (variance ** 0.5)
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
            elif func_name == 'sum':
                x = args[0]
                if len(args) == 1:
                    # 단일 리스트의 합
                    return sum(x)
                elif len(args) == 2:
                    # 특정 기간 동안의 합
                    n = int(args[1][0])
                    if len(x) < n:
                        return sum(x)
                    return sum(x[-n:])
                else:
                    raise ValueError("sum function takes 1 or 2 arguments")
            else:
                raise ValueError(f"Unknown function: {func_name}")
        else:
            raise ValueError(f"Unknown node type: {self.type}")

    def ts_min(self, data: List[float], window: int) -> List[float]:
        """
        시계열 데이터의 이동 최소값을 계산한다.
        현재 값을 포함한 최근 window일의 최소값을 계산
        예: ts_min([1, 2, 3], 2) -> [1, 1, 1]
        """
        if not isinstance(data, list):
            data = [data]
        if not isinstance(window, (int, float)):
            raise ValueError("Window must be a number")
        window = int(window) + 1  # 현재 값을 포함하기 위해 window + 1
        if window <= 0:
            raise ValueError("Window must be positive")
        result = []
        for i in range(len(data)):
            start = max(0, i - window + 1)
            result.append(min(data[start:i + 1]))
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
            return ASTNode('NUMBER', token.value)
        elif token.type == TokenType.IDENTIFIER:
            name = token.value
            self.eat(TokenType.IDENTIFIER)
            if self.current_token.type == TokenType.LPAREN:
                return self.function_call(name)
            return ASTNode('VARIABLE', name)
        elif token.type == TokenType.LPAREN:
            self.eat(TokenType.LPAREN)
            node = self.expr()
            self.eat(TokenType.RPAREN)
            return node
        elif token.type == TokenType.LBRACKET:
            return self.list_expr()
        else:
            self.error(f"Unexpected token: {token.type}")
            
    def list_expr(self) -> ASTNode:
        self.eat(TokenType.LBRACKET)
        elements = []
        
        if self.current_token.type != TokenType.RBRACKET:
            elements.append(self.expr())
            while self.current_token.type == TokenType.COMMA:
                self.eat(TokenType.COMMA)
                elements.append(self.expr())
                
        self.eat(TokenType.RBRACKET)
        return ASTNode('LIST', children=elements)
        
    def function_call(self, name: str) -> ASTNode:
        self.eat(TokenType.LPAREN)
        args = []
        if self.current_token.type != TokenType.RPAREN:
            args.append(self.expr())
            while self.current_token.type == TokenType.COMMA:
                self.eat(TokenType.COMMA)
                args.append(self.expr())
        self.eat(TokenType.RPAREN)
        return ASTNode('FUNCTION_CALL', name, args)
        
    def factor(self) -> ASTNode:
        token = self.current_token
        
        if token.type in [TokenType.PLUS, TokenType.MINUS, TokenType.NOT]:
            self.eat(token.type)
            return ASTNode('UNARY_OP', token.type, [self.factor()])
            
        return self.atom()
        
    def power(self) -> ASTNode:
        """멱승 연산을 처리하는 메서드"""
        node = self.factor()
        # Handle POWER token from lexer
        if self.current_token.type == TokenType.POWER:
            self.eat(TokenType.POWER)
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
        # Only initialize variables; lexer is created per parse call
        self.variables = {}
        
    def parse(self, formula: str) -> Any:
        # Create a fresh lexer for each formula
        lexer = AlphaLexer(formula)
        parser = Parser(lexer)
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
                    return [0] * len(x)  # 초기값은 0으로 설정
                
                result = []
                for i in range(len(x)):
                    if i < n - 1:
                        result.append(0)  # n일 이전의 값들은 0으로 설정
                    else:
                        # 최근 n일 데이터 추출
                        x_window = x[i-n+1:i+1]
                        y_window = y[i-n+1:i+1]
                        
                        # 평균 계산
                        x_mean = sum(x_window) / n
                        y_mean = sum(y_window) / n
                        
                        # 공분산과 표준편차 계산
                        cov = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x_window, y_window)) / n
                        x_std = (sum((xi - x_mean) ** 2 for xi in x_window) / n) ** 0.5
                        y_std = (sum((yi - y_mean) ** 2 for yi in y_window) / n) ** 0.5
                        
                        # 상관계수 계산
                        if x_std * y_std != 0:
                            result.append(cov / (x_std * y_std))
                        else:
                            result.append(0)
                
                return result
            
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
                    raise ValueError(f"List length must be greater than {n} for delta")
                return [x[i] - x[i-n] for i in range(n, len(x))]
            elif func_name == 'log':
                x = args[0]
                if any(val <= 0 for val in x):
                    raise ValueError("Cannot take log of non-positive values")
                return [math.log(val) for val in x]
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
            elif func_name == 'stddev':
                x, n = args
                n = int(n[0])
                if len(x) < n:
                    raise ValueError(f"List length must be at least {n} for stddev")
                x = x[-n:]
                mean = sum(x) / n
                variance = sum((xi - mean) ** 2 for xi in x) / n
                return (variance ** 0.5)
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
            
            elif func_name == 'sum':
                x = args[0]
                if len(args) == 1:
                    # 단일 리스트의 합
                    return sum(x)
                elif len(args) == 2:
                    # 특정 기간 동안의 합
                    n = int(args[1][0])
                    if len(x) < n:
                        return sum(x)
                    return sum(x[-n:])
                else:
                    raise ValueError("sum function takes 1 or 2 arguments")
            
            else:
                raise ValueError(f"Unknown function: {func_name}") 
        else:
            raise ValueError(f"Unknown node type: {node.type}") 

    def delay(self, data: List[float], periods: int) -> List[float]:
        """
        시계열 데이터를 periods만큼 지연시킨다.
        예: delay([1, 2, 3], 1) -> [None, 1, 2]
        """
        if not isinstance(data, list):
            data = [data]
        if not isinstance(periods, (int, float)):
            raise ValueError("Periods must be a number")
        periods = int(periods)
        if periods < 0:
            raise ValueError("Periods must be non-negative")
        result = [None] * periods + data[:-periods] if periods > 0 else data
        return result

    def ts_max(self, data: List[float], window: int) -> List[float]:
        """
        시계열 데이터의 이동 최대값을 계산한다.
        현재 값을 포함한 최근 window일의 최대값을 계산
        예: ts_max([1, 2, 3], 2) -> [1, 2, 3]
        """
        if not isinstance(data, list):
            data = [data]
        if not isinstance(window, (int, float)):
            raise ValueError("Window must be a number")
        window = int(window) + 1  # 현재 값을 포함하기 위해 window + 1
        if window <= 0:
            raise ValueError("Window must be positive")
        result = []
        for i in range(len(data)):
            start = max(0, i - window + 1)
            result.append(max(data[start:i + 1]))
        return result

    def ts_argmax(self, data: List[float], window: int) -> int:
        """
        시계열 데이터의 이동 최대값의 인덱스를 계산한다.
        현재 값을 포함한 최근 window일 중 최대값의 인덱스를 계산
        예: ts_argmax([1, 2, 3], 2) -> 2
        """
        if not isinstance(data, list):
            data = [data]
        if not isinstance(window, (int, float)):
            raise ValueError("Window must be a number")
        window = int(window) + 1  # 현재 값을 포함하기 위해 window + 1
        if window <= 0:
            raise ValueError("Window must be positive")
        result = []
        for i in range(len(data)):
            start = max(0, i - window + 1)
            window_data = data[start:i + 1]
            result.append(start + window_data.index(max(window_data)))
        return result[-1]

    def ts_argmin(self, data: List[float], window: int) -> int:
        """
        시계열 데이터의 이동 최소값의 인덱스를 계산한다.
        현재 값을 포함한 최근 window일 중 최소값의 인덱스를 계산
        예: ts_argmin([1, 2, 3], 2) -> 0
        """
        if not isinstance(data, list):
            data = [data]
        if not isinstance(window, (int, float)):
            raise ValueError("Window must be a number")
        window = int(window) + 1  # 현재 값을 포함하기 위해 window + 1
        if window <= 0:
            raise ValueError("Window must be positive")
        result = []
        for i in range(len(data)):
            start = max(0, i - window + 1)
            window_data = data[start:i + 1]
            result.append(start + window_data.index(min(window_data)))
        return result[-1]

    def ts_rank(self, data: List[float], window: int) -> List[float]:
        """
        시계열 데이터의 이동 순위를 계산한다.
        현재 값을 포함한 최근 window일의 순위를 계산
        예: ts_rank([1, 2, 3], 2) -> [1, 2, 2]
        """
        if not isinstance(data, list):
            data = [data]
        if not isinstance(window, (int, float)):
            raise ValueError("Window must be a number")
        window = int(window) + 1  # 현재 값을 포함하기 위해 window + 1
        if window <= 0:
            raise ValueError("Window must be positive")
        result = []
        for i in range(len(data)):
            start = max(0, i - window + 1)
            window_data = data[start:i + 1]
            sorted_data = sorted(window_data)
            rank = sorted_data.index(data[i]) + 1
            result.append(rank)
        return result 

    def stddev(self, data: List[float], window: int) -> List[float]:
        """
        시계열 데이터의 이동 표준편차를 계산한다.
        현재 값을 포함한 최근 window일의 표준편차를 계산
        예: stddev([1, 2, 3], 2) -> [0, 0.707, 0.707]
        """
        if not isinstance(data, list):
            data = [data]
        if not isinstance(window, (int, float)):
            raise ValueError("Window must be a number")
        window = int(window) + 1  # 현재 값을 포함하기 위해 window + 1
        if window <= 0:
            raise ValueError("Window must be positive")
        
        result = []
        for i in range(len(data)):
            start = max(0, i - window + 1)
            window_data = data[start:i + 1]
            
            # 평균 계산
            mean = sum(window_data) / len(window_data)
            
            # 분산 계산
            variance = sum((x - mean) ** 2 for x in window_data) / len(window_data)
            
            # 표준편차 계산
            std = variance ** 0.5
            result.append(std)
            
        return result 