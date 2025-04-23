import numpy as np
from .tokens import TokenType

class ASTNode:
    def __init__(self, type, value=None, children=None):
        self.type = type
        self.value = value
        self.children = children or []

    def _rank(self, x):
        """Rank the elements in x"""
        return np.argsort(np.argsort(x))

    def _delay(self, x, n):
        """Delay x by n periods"""
        return np.roll(x, n)

    def _correlation(self, x, y, n):
        """Calculate correlation between x and y over n periods"""
        return np.correlate(x, y, mode='valid') / n

    def _covariance(self, x, y, n):
        """Calculate covariance between x and y over n periods"""
        return np.cov(x, y)[0,1]

    def _scale(self, x):
        """Scale x to sum to 1"""
        return x / np.sum(x)

    def _delta(self, x, n):
        """Calculate difference between current and n periods ago"""
        return x - np.roll(x, n)

    def _signedpower(self, x, power):
        """Calculate signed power of x"""
        return np.sign(x) * np.power(np.abs(x), power)

    def _decay_linear(self, x, n):
        """Calculate linear decay of x over n periods"""
        weights = np.linspace(1, 0, n)
        return np.convolve(x, weights, mode='valid')

    def _sum(self, x, n):
        """Calculate sum of x over n periods"""
        return np.convolve(x, np.ones(n), mode='valid')

    def _product(self, x, n):
        """Calculate product of x over n periods"""
        return np.prod(x[-n:])

    def _stddev(self, x, n):
        """Calculate standard deviation of x over n periods"""
        return np.std(x[-n:])

    def _ts_rank(self, x, n):
        """Calculate time series rank of x over n periods"""
        return np.argsort(np.argsort(x[-n:]))[-1]

    def _ts_min(self, x, n):
        """Calculate minimum of x over n periods"""
        return np.min(x[-n:])

    def _ts_max(self, x, n):
        """Calculate maximum of x over n periods"""
        return np.max(x[-n:])

    def _ts_argmax(self, x, n):
        """Calculate index of maximum of x over n periods"""
        return np.argmax(x[-n:])

    def _ts_argmin(self, x, n):
        """Calculate index of minimum of x over n periods"""
        return np.argmin(x[-n:])

    def evaluate(self, variables):
        if self.type == 'NUMBER':
            return self.value
        elif self.type == 'VARIABLE':
            if self.value not in variables:
                raise ValueError(f"Variable {self.value} not found")
            return variables[self.value]
        elif self.type == 'BINARY_OP':
            left = self.children[0].evaluate(variables)
            right = self.children[1].evaluate(variables)
            
            # 안전한 연산을 위한 헬퍼 함수들
            def safe_add(a, b):
                if a is None or b is None:
                    return None
                return a + b
            
            def safe_sub(a, b):
                if a is None or b is None:
                    return None
                return a - b
            
            def safe_mul(a, b):
                if a is None or b is None:
                    return None
                return a * b
            
            def safe_div(a, b):
                if a is None or b is None or b == 0:
                    return None
                return a / b
            
            def safe_lt(a, b):
                if a is None or b is None:
                    return None
                return a < b
            
            def safe_gt(a, b):
                if a is None or b is None:
                    return None
                return a > b
            
            def safe_le(a, b):
                if a is None or b is None:
                    return None
                return a <= b
            
            def safe_ge(a, b):
                if a is None or b is None:
                    return None
                return a >= b
            
            def safe_eq(a, b):
                if a is None or b is None:
                    return None
                return a == b
            
            def safe_ne(a, b):
                if a is None or b is None:
                    return None
                return a != b
            
            # 리스트 연산 처리
            if isinstance(left, list) or isinstance(right, list):
                # 하나라도 리스트면, 둘 다 리스트화 & 길이 맞추기
                if not isinstance(left, list):
                    left = [left] * len(right)
                if not isinstance(right, list):
                    right = [right] * len(left)
                if len(left) != len(right):
                    if len(left) > len(right):
                        left = left[-len(right):]
                    else:
                        right = right[-len(left):]
                
                if self.value == TokenType.PLUS:
                    return [safe_add(l, r) for l, r in zip(left, right)]
                elif self.value == TokenType.MINUS:
                    return [safe_sub(l, r) for l, r in zip(left, right)]
                elif self.value == TokenType.MULTIPLY:
                    return [safe_mul(l, r) for l, r in zip(left, right)]
                elif self.value == TokenType.DIVIDE:
                    return [safe_div(l, r) for l, r in zip(left, right)]
                elif self.value == TokenType.LESS:
                    return [safe_lt(l, r) for l, r in zip(left, right)]
                elif self.value == TokenType.GREATER:
                    return [safe_gt(l, r) for l, r in zip(left, right)]
                elif self.value == TokenType.LESS_EQUAL:
                    return [safe_le(l, r) for l, r in zip(left, right)]
                elif self.value == TokenType.GREATER_EQUAL:
                    return [safe_ge(l, r) for l, r in zip(left, right)]
                elif self.value == TokenType.EQUAL:
                    return [safe_eq(l, r) for l, r in zip(left, right)]
                elif self.value == TokenType.NOT_EQUAL:
                    return [safe_ne(l, r) for l, r in zip(left, right)]
                elif self.value == TokenType.AND:
                    return [l and r for l, r in zip(left, right)]
                elif self.value == TokenType.OR:
                    return [l or r for l, r in zip(left, right)]
            else:
                # 스칼라 연산
                if self.value == TokenType.PLUS:
                    return safe_add(left, right)
                elif self.value == TokenType.MINUS:
                    return safe_sub(left, right)
                elif self.value == TokenType.MULTIPLY:
                    return safe_mul(left, right)
                elif self.value == TokenType.DIVIDE:
                    return safe_div(left, right)
                elif self.value == TokenType.LESS:
                    return safe_lt(left, right)
                elif self.value == TokenType.GREATER:
                    return safe_gt(left, right)
                elif self.value == TokenType.LESS_EQUAL:
                    return safe_le(left, right)
                elif self.value == TokenType.GREATER_EQUAL:
                    return safe_ge(left, right)
                elif self.value == TokenType.EQUAL:
                    return safe_eq(left, right)
                elif self.value == TokenType.NOT_EQUAL:
                    return safe_ne(left, right)
                elif self.value == TokenType.AND:
                    return left and right
                elif self.value == TokenType.OR:
                    return left or right
        elif self.type == 'UNARY_OP':
            value = self.children[0].evaluate(variables)
            if self.value == TokenType.MINUS:
                return -value
            elif self.value == TokenType.NOT:
                return not value
        elif self.type == 'FUNCTION_CALL':
            func_name = self.value.lower()
            args = [child.evaluate(variables) for child in self.children]
            
            if func_name == 'rank':
                return self._rank(args[0])
            elif func_name == 'delay':
                return self._delay(args[0], int(args[1]))
            elif func_name == 'correlation':
                return self._correlation(args[0], args[1], int(args[2]))
            elif func_name == 'covariance':
                return self._covariance(args[0], args[1], int(args[2]))
            elif func_name == 'scale':
                return self._scale(args[0])
            elif func_name == 'delta':
                return self._delta(args[0], int(args[1]))
            elif func_name == 'signedpower':
                return self._signedpower(args[0], args[1])
            elif func_name == 'decay_linear':
                return self._decay_linear(args[0], int(args[1]))
            elif func_name == 'sum':
                return self._sum(args[0], int(args[1]))
            elif func_name == 'product':
                print(f"[DEBUG] product function called with args: {args}")
                if len(args) == 1:
                    # 단일 리스트의 경우 전체 곱 계산
                    x = args[0]
                    if not isinstance(x, list):
                        return x
                    if len(x) == 0:
                        return 0
                    # None 값이 있으면 None 반환
                    if any(v is None for v in x):
                        return None
                    result = 1.0
                    for val in x:
                        result *= val
                    return result
                elif len(args) == 2:
                    # 두 인자의 경우 원소별 곱 계산
                    left = args[0]
                    right = args[1]
                    
                    # 안전한 곱셈을 위한 헬퍼 함수
                    def safe_mul(a, b):
                        if a is None or b is None:
                            return None
                        return a * b
                    
                    # 리스트 연산 처리
                    if isinstance(left, list) or isinstance(right, list):
                        # 하나라도 리스트면, 둘 다 리스트화 & 길이 맞추기
                        if not isinstance(left, list):
                            left = [left] * len(right)
                        if not isinstance(right, list):
                            right = [right] * len(left)
                        if len(left) != len(right):
                            if len(left) > len(right):
                                left = left[-len(right):]
                            else:
                                right = right[-len(left):]
                        
                        return [safe_mul(l, r) for l, r in zip(left, right)]
                    else:
                        # 스칼라 연산
                        return safe_mul(left, right)
                else:
                    raise ValueError("product function expects 1 or 2 arguments")
            elif func_name == 'stddev':
                return self._stddev(args[0], int(args[1]))
            elif func_name == 'ts_rank':
                return self._ts_rank(args[0], int(args[1]))
            elif func_name == 'ts_min':
                return self._ts_min(args[0], int(args[1]))
            elif func_name == 'ts_max':
                return self._ts_max(args[0], int(args[1]))
            elif func_name == 'ts_argmax':
                return self._ts_argmax(args[0], int(args[1]))
            elif func_name == 'ts_argmin':
                return self._ts_argmin(args[0], int(args[1]))
            elif func_name == 'min':
                if len(args) == 1:
                    # 단일 리스트의 경우 전체 최소값 반환
                    if isinstance(args[0], list):
                        return min(args[0])
                    return args[0]
                elif len(args) == 2:
                    # 두 인자의 경우 원소별 최소값 계산
                    left = args[0]
                    right = args[1]
                    
                    # 안전한 min 연산을 위한 헬퍼 함수
                    def safe_min(a, b):
                        if a is None or b is None:
                            return None
                        return min(a, b)
                    
                    # 리스트 연산 처리
                    if isinstance(left, list) or isinstance(right, list):
                        # 하나라도 리스트면, 둘 다 리스트화 & 길이 맞추기
                        if not isinstance(left, list):
                            left = [left] * len(right)
                        if not isinstance(right, list):
                            right = [right] * len(left)
                        if len(left) != len(right):
                            if len(left) > len(right):
                                left = left[-len(right):]
                            else:
                                right = right[-len(left):]
                        
                        return [safe_min(l, r) for l, r in zip(left, right)]
                    else:
                        # 스칼라 연산
                        return safe_min(left, right)
                else:
                    raise ValueError("min function expects 1 or 2 arguments")
            elif func_name == 'abs':
                return np.abs(args[0])
            else:
                raise ValueError(f"Unknown function: {func_name}")
        elif self.type == 'CONDITIONAL':
            condition = self.children[0].evaluate(variables)
            if condition is None:
                return None
            if condition:
                return self.children[1].evaluate(variables)
            else:
                return self.children[2].evaluate(variables)
        else:
            raise ValueError(f"Unknown node type: {self.type}") 