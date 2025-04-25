# alpha_parser/alpha_parser.py
from typing import List, Optional, Union, Dict, Any
from alpha_parser.alpha_tokens import Token, TokenType
from alpha_parser.alpha_lexer import AlphaLexer
import math
import pandas as pd
import numpy as np

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
        
    def evaluate(self, variables: dict = None, depth: int = 0) -> Any:
        """Evaluate AST node and return the result"""
        indent = "  " * depth
        if variables is None:
            variables = {}
            
        if self.type == 'NUMBER':
            val = float(self.value)
            print(f"{indent}[DEBUG] NUMBER => {val}")
            return val
        elif self.type == 'VARIABLE':
            value = variables.get(self.value, 0)
            if isinstance(value, pd.Series):
                value = value.tolist()
            if not isinstance(value, list):
                value = [value]
            print(f"{indent}[DEBUG] VARIABLE {self.value} => shape={len(value)}")
            return value
        elif self.type == 'INDCLASS':
            # IndClass.xxx 패턴 처리
            if self.value.startswith('IndClass.'):
                level = self.value.split('.')[1]
                print(f"{indent}[DEBUG] INDCLASS => {level}")
                # variables에서 해당 레벨의 리스트를 가져옴
                group_list = variables.get(level, [])
                if not isinstance(group_list, list):
                    group_list = [group_list]  # 단일 값이면 리스트로 변환
                print(f"{indent}[DEBUG] INDCLASS {level} => shape={len(group_list)}")
                return group_list
            return self.value
        elif self.type == 'LIST':
            result = [node.evaluate(variables, depth+1) for node in self.children]
            print(f"{indent}[DEBUG] LIST => shape={len(result)}")
            return result
        elif self.type == 'CONDITIONAL':
            condition = self.children[0].evaluate(variables, depth+1)
            true_expr = self.children[1]
            false_expr = self.children[2]
            
            if isinstance(condition, list):
                true_val = true_expr.evaluate(variables, depth+1)
                false_val = false_expr.evaluate(variables, depth+1)
                
                if not isinstance(true_val, list):
                    true_val = [true_val] * len(condition)
                if not isinstance(false_val, list):
                    false_val = [false_val] * len(condition)
                    
                result = [t if c else f for c, t, f in zip(condition, true_val, false_val)]
                print(f"{indent}[DEBUG] CONDITIONAL => shape={len(result)}")
                return result
            else:
                result = true_expr.evaluate(variables, depth+1) if condition else false_expr.evaluate(variables, depth+1)
                print(f"{indent}[DEBUG] CONDITIONAL => scalar={result}")
                return result
        elif self.type == 'BINARY_OP':
            left = self.children[0].evaluate(variables, depth+1)
            right = self.children[1].evaluate(variables, depth+1)
            
            if isinstance(left, list) or isinstance(right, list):
                if not isinstance(left, list):
                    left = [left] * (len(right) if isinstance(right, list) else 1)
                if not isinstance(right, list):
                    right = [right] * (len(left) if isinstance(left, list) else 1)
                
                if len(left) != len(right):
                    left, right = self.align_length(left, right)

                if self.value == TokenType.PLUS:
                    result = [l + r for l, r in zip(left, right)]
                elif self.value == TokenType.MINUS:
                    result = [l - r for l, r in zip(left, right)]
                elif self.value == TokenType.MULTIPLY:
                    result = [l * r for l, r in zip(left, right)]
                elif self.value == TokenType.DIVIDE:
                    result = [l / r if r != 0 else float('nan') for l, r in zip(left, right)]
                elif self.value == TokenType.POWER:
                    result = [l ** r for l, r in zip(left, right)]
                elif self.value == TokenType.EQUAL:
                    result = [l == r for l, r in zip(left, right)]
                elif self.value == TokenType.NOT_EQUAL:
                    result = [l != r for l, r in zip(left, right)]
                elif self.value == TokenType.GREATER:
                    result = [l > r for l, r in zip(left, right)]
                elif self.value == TokenType.LESS:
                    result = [l < r for l, r in zip(left, right)]
                elif self.value == TokenType.GREATER_EQUAL:
                    result = [l >= r for l, r in zip(left, right)]
                elif self.value == TokenType.LESS_EQUAL:
                    result = [l <= r for l, r in zip(left, right)]
                elif self.value == TokenType.AND:
                    result = [l and r for l, r in zip(left, right)]
                elif self.value == TokenType.OR:
                    result = [l or r for l, r in zip(left, right)]
                print(f"{indent}[DEBUG] BINARY_OP {self.value} => shape={len(result)}")
                return result
            else:
                if self.value == TokenType.PLUS:
                    result = left + right
                elif self.value == TokenType.MINUS:
                    result = left - right
                elif self.value == TokenType.MULTIPLY:
                    result = left * right
                elif self.value == TokenType.DIVIDE:
                    result = left / right if right != 0 else float('nan')
                elif self.value == TokenType.POWER:
                    result = left ** right
                elif self.value == TokenType.EQUAL:
                    result = left == right
                elif self.value == TokenType.NOT_EQUAL:
                    result = left != right
                elif self.value == TokenType.GREATER:
                    result = left > right
                elif self.value == TokenType.LESS:
                    result = left < right
                elif self.value == TokenType.GREATER_EQUAL:
                    result = left >= right
                elif self.value == TokenType.LESS_EQUAL:
                    result = left <= right
                elif self.value == TokenType.AND:
                    result = left and right
                elif self.value == TokenType.OR:
                    result = left or right
                print(f"{indent}[DEBUG] BINARY_OP {self.value} => scalar={result}")
                return result
        elif self.type == 'UNARY_OP':
            child = self.children[0].evaluate(variables, depth+1)
            if isinstance(child, list):
                if self.value == TokenType.PLUS:
                    result = child
                elif self.value == TokenType.MINUS:
                    result = [-x for x in child]
                elif self.value == TokenType.NOT:
                    result = [not x for x in child]
                print(f"{indent}[DEBUG] UNARY_OP {self.value} => shape={len(result)}")
                return result
            else:
                if self.value == TokenType.PLUS:
                    result = +child
                elif self.value == TokenType.MINUS:
                    result = -child
                elif self.value == TokenType.NOT:
                    result = not child
                print(f"{indent}[DEBUG] UNARY_OP {self.value} => scalar={result}")
                return result
        elif self.type == 'FUNCTION_CALL':
            func_name = self.value.lower()
            args = [child.evaluate(variables, depth+1) for child in self.children]
            
            # Convert all arguments to lists
            args = [[arg] if not isinstance(arg, list) else arg for arg in args]
            
            # Log function arguments
            for i, arg in enumerate(args):
                print(f"{indent}[DEBUG] FUNCTION {func_name} arg[{i}] => shape={len(arg)}")
            
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
                    return [0] * len(x)
                return [0] * n + x[:-n]
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
                if isinstance(x, list):
                    # 리스트의 경우 각 원소별로 log 적용
                    result = [math.log(xx) if xx > 0 else float('nan') for xx in x]
                    print(f"{indent}[DEBUG] FUNCTION log => shape={len(result)}")
                    return result
                else:
                    # 스칼라의 경우 단순 log 적용
                    result = math.log(x) if x > 0 else float('nan')
                    print(f"{indent}[DEBUG] FUNCTION log => scalar={result}")
                    return result
            elif func_name == 'signedpower':
                x, power = args
                power = float(power[0])
                result = []
                for xi in x:
                    if xi == 0:
                        result.append(0)  # 0의 음수 거듭제곱은 0으로 처리
                    else:
                        result.append(abs(xi) ** power * (1 if xi >= 0 else -1))
                return result
            elif func_name == 'decay_linear':
                x, n = args
                n = int(n[0])
                if len(x) < n:
                    # 데이터가 부족한 경우 NaN으로 채운 리스트 반환
                    return [float('nan')] * len(x)
                weights = [i+1 for i in range(n)]
                total_weight = sum(weights)
                result = []
                for i in range(len(x)):
                    if i < n - 1:
                        # 데이터가 부족한 초기 구간
                        window_data = x[:i+1]
                        window_weights = weights[-len(window_data):]
                        window_total_weight = sum(window_weights)
                        if window_total_weight == 0:
                            result.append(float('nan'))
                        else:
                            result.append(sum(x * w for x, w in zip(window_data, window_weights)) / window_total_weight)
                    else:
                        # 충분한 데이터가 있는 구간
                        window_data = x[i-n+1:i+1]
                        result.append(sum(x * w for x, w in zip(window_data, weights)) / total_weight)
                return result
            elif func_name == 'stddev':
                x, n = args
                n = int(n[0])
                if len(x) < n:
                    return [0] * len(x)  # 데이터가 부족한 경우 0으로 채움
                
                result = []
                for i in range(len(x)):
                    if i < n - 1:
                        # 데이터가 부족한 초기 구간
                        window_data = x[:i+1]
                    else:
                        # 충분한 데이터가 있는 구간
                        window_data = x[i-n+1:i+1]
                        
                    mean = sum(window_data) / len(window_data)
                    variance = sum((xi - mean) ** 2 for xi in window_data) / len(window_data)
                    result.append(variance ** 0.5)
                    
                return result
            elif func_name == 'indneutralize':
                x, g = args
                # x와 g의 길이를 맞춤
                x, g = self.align_length(x, g)
                
                if not x or not g:
                    raise ValueError("Empty lists are not allowed for indneutralize")
                    
                # Calculate group means
                group_means = {}
                group_counts = {}
                for i, group in enumerate(g):
                    if group not in group_means:
                        group_means[group] = 0
                        group_counts[group] = 0
                    if not np.isnan(x[i]):  # NaN 값은 무시
                        group_means[group] += x[i]
                        group_counts[group] += 1
                
                # Calculate final means
                for group in group_means:
                    if group_counts[group] > 0:  # 0으로 나누기 방지
                        group_means[group] /= group_counts[group]
                    else:
                        group_means[group] = 0
                
                # Neutralize
                result = []
                for i, group in enumerate(g):
                    if np.isnan(x[i]):
                        result.append(np.nan)
                    else:
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
                    return [0] * len(x)  # 데이터가 부족한 경우 0으로 채움
                
                result = []
                for i in range(len(x)):
                    if i < n - 1:
                        # 데이터가 부족한 초기 구간
                        window_data = x[:i+1]
                    else:
                        # 충분한 데이터가 있는 구간
                        window_data = x[i-n+1:i+1]
                        
                    max_idx = window_data.index(max(window_data))
                    result.append(max_idx)
                    
                return result
                
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
                    return [0] * len(x)  # 데이터가 부족한 경우 0으로 채움
                
                result = []
                for i in range(len(x)):
                    if i < n - 1:
                        # 데이터가 부족한 초기 구간
                        window_data = x[:i+1]
                    else:
                        # 충분한 데이터가 있는 구간
                        window_data = x[i-n+1:i+1]
                        
                    # Store values with their indices
                    indexed_values = [(val, j) for j, val in enumerate(window_data)]
                    # Sort by value (ascending)
                    sorted_values = sorted(indexed_values, key=lambda x: x[0])
                    
                    # Calculate ranks (normalized between 0 and 1)
                    ranks = [0] * len(window_data)
                    for j, (val, idx) in enumerate(sorted_values):
                        ranks[idx] = j / (len(window_data) - 1) if len(window_data) > 1 else 0
                    
                    result.append(ranks[-1])  # 현재 시점의 순위를 추가
                    
                return result
            elif func_name == 'sum':
                x = args[0]
                if len(args) == 1:
                    # 시계열 합산 (axis=0 방향)
                    if isinstance(x, list):
                        return x  # 원본 시계열 반환
                    return [x]  # 스칼라인 경우 리스트로 변환
                elif len(args) == 2:
                    # 특정 기간 동안의 합을 리스트로 반환
                    n = int(args[1][0])
                    if len(x) < n:
                        return [0] * len(x)
                    result = []
                    for i in range(len(x)):
                        if i < n - 1:
                            # 데이터가 부족한 초기 구간
                            window_data = x[:i+1]
                        else:
                            # 충분한 데이터가 있는 구간
                            window_data = x[i-n+1:i+1]
                        result.append(sum(window_data))
                    return result
                else:
                    raise ValueError("sum function takes 1 or 2 arguments")
            elif func_name == 'abs':
                x = args[0]
                if isinstance(x, list):
                    # 리스트의 경우 각 원소별로 abs 적용
                    result = [abs(xx) for xx in x]
                    print(f"{indent}[DEBUG] FUNCTION abs => shape={len(result)}")
                    return result
                else:
                    # 스칼라의 경우 단순 abs 적용
                    result = abs(x)
                    print(f"{indent}[DEBUG] FUNCTION abs => scalar={result}")
                    return result
            elif func_name == 'sign':
                x = args[0]
                if isinstance(x, list):
                    return [1 if val > 0 else (-1 if val < 0 else 0) for val in x]
                return 1 if x > 0 else (-1 if x < 0 else 0)
            elif func_name == 'min':
                if len(args) == 1:
                    # 단일 리스트의 경우 전체 최소값 계산
                    x = args[0]
                    if not isinstance(x, list):
                        return x
                    if len(x) == 0:
                        return None
                    # None 값이 있으면 None 반환
                    if any(v is None for v in x):
                        return None
                    return min(x)
                elif len(args) == 2:
                    # 두 인자의 경우 원소별 최소값 계산
                    left = args[0]
                    right = args[1]
                    
                    # 안전한 최소값 계산을 위한 헬퍼 함수
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
            elif func_name == 'product':
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
            elif func_name == 'max':
                if len(args) == 1:
                    # 단일 리스트의 경우 전체 최대값 계산
                    x = args[0]
                    if not isinstance(x, list):
                        return x
                    if len(x) == 0:
                        return None
                    # None 값이 있으면 None 반환
                    if any(v is None for v in x):
                        return None
                    return max(x)
                elif len(args) == 2:
                    # 두 인자의 경우 원소별 최대값 계산
                    left = args[0]
                    right = args[1]
                    
                    # 안전한 최대값 계산을 위한 헬퍼 함수
                    def safe_max(a, b):
                        if a is None or b is None:
                            return None
                        return max(a, b)
                    
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
                        
                        return [safe_max(l, r) for l, r in zip(left, right)]
                    else:
                        # 스칼라 연산
                        return safe_max(left, right)
                else:
                    raise ValueError("max function expects 1 or 2 arguments")
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
        window = int(window)
        if window <= 0:
            raise ValueError("Window must be positive")
        result = []
        for i in range(len(data)):
            start = max(0, i - window + 1)
            result.append(min(data[start:i + 1]))
        return result

    def align_length(self, a: list, b: list) -> tuple:
        """
        두 리스트 a, b의 길이가 다르면,
        더 긴 쪽의 앞부분(가장 오래된 구간)을 잘라서 길이를 min_len으로 맞춰줌.
        """
        len_a = len(a)
        len_b = len(b)
        min_len = min(len_a, len_b)
        
        # a의 길이가 더 길면 앞부분(len_a - min_len)을 버림
        if len_a > min_len:
            a = a[-min_len:]
        # b의 길이가 더 길면 앞부분(len_b - min_len)을 버림
        if len_b > min_len:
            b = b[-min_len:]
        
        return a, b

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
            # IndClass.xxx 패턴 처리
            if name.startswith('IndClass.'):
                return ASTNode('INDCLASS', name)
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
                    left, right = self.align_length(left, right)
            
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
                    return [0] * len(x)
                return [0] * n + x[:-n]
            
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
                if isinstance(x, list):
                    # 리스트의 경우 각 원소별로 log 적용
                    result = [math.log(xx) if xx > 0 else float('nan') for xx in x]
                    print(f"{indent}[DEBUG] FUNCTION log => shape={len(result)}")
                    return result
                else:
                    # 스칼라의 경우 단순 log 적용
                    result = math.log(x) if x > 0 else float('nan')
                    print(f"{indent}[DEBUG] FUNCTION log => scalar={result}")
                    return result
            elif func_name == 'signedpower':
                x, power = args
                power = float(power[0])
                result = []
                for xi in x:
                    if xi == 0:
                        result.append(0)  # 0의 음수 거듭제곱은 0으로 처리
                    else:
                        result.append(abs(xi) ** power * (1 if xi >= 0 else -1))
                return result
            elif func_name == 'decay_linear':
                x, n = args
                n = int(n[0])
                if len(x) < n:
                    # 데이터가 부족한 경우 NaN으로 채운 리스트 반환
                    return [float('nan')] * len(x)
                weights = [i+1 for i in range(n)]
                total_weight = sum(weights)
                result = []
                for i in range(len(x)):
                    if i < n - 1:
                        # 데이터가 부족한 초기 구간
                        window_data = x[:i+1]
                        window_weights = weights[-len(window_data):]
                        window_total_weight = sum(window_weights)
                        if window_total_weight == 0:
                            result.append(float('nan'))
                        else:
                            result.append(sum(x * w for x, w in zip(window_data, window_weights)) / window_total_weight)
                    else:
                        # 충분한 데이터가 있는 구간
                        window_data = x[i-n+1:i+1]
                        result.append(sum(x * w for x, w in zip(window_data, weights)) / total_weight)
                return result
            elif func_name == 'stddev':
                x, n = args
                n = int(n[0])
                if len(x) < n:
                    return [0] * len(x)  # 데이터가 부족한 경우 0으로 채움
                
                result = []
                for i in range(len(x)):
                    if i < n - 1:
                        # 데이터가 부족한 초기 구간
                        window_data = x[:i+1]
                    else:
                        # 충분한 데이터가 있는 구간
                        window_data = x[i-n+1:i+1]
                        
                    mean = sum(window_data) / len(window_data)
                    variance = sum((xi - mean) ** 2 for xi in window_data) / len(window_data)
                    result.append(variance ** 0.5)
                    
                return result
            elif func_name == 'indneutralize':
                x, g = args
                # x와 g의 길이를 맞춤
                x, g = self.align_length(x, g)
                
                if not x or not g:
                    raise ValueError("Empty lists are not allowed for indneutralize")
                    
                # Calculate group means
                group_means = {}
                group_counts = {}
                for i, group in enumerate(g):
                    if group not in group_means:
                        group_means[group] = 0
                        group_counts[group] = 0
                    if not np.isnan(x[i]):  # NaN 값은 무시
                        group_means[group] += x[i]
                        group_counts[group] += 1
                
                # Calculate final means
                for group in group_means:
                    if group_counts[group] > 0:  # 0으로 나누기 방지
                        group_means[group] /= group_counts[group]
                    else:
                        group_means[group] = 0
                
                # Neutralize
                result = []
                for i, group in enumerate(g):
                    if np.isnan(x[i]):
                        result.append(np.nan)
                    else:
                        result.append(x[i] - group_means[group])
                    
                return result
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
            elif func_name == 'ts_argmax':
                x, n = args
                n = int(n[0])
                if len(x) < n:
                    return [0] * len(x)  # 데이터가 부족한 경우 0으로 채움
                
                result = []
                for i in range(len(x)):
                    if i < n - 1:
                        # 데이터가 부족한 초기 구간
                        window_data = x[:i+1]
                    else:
                        # 충분한 데이터가 있는 구간
                        window_data = x[i-n+1:i+1]
                        
                    max_idx = window_data.index(max(window_data))
                    result.append(max_idx)
                    
                return result
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
                    return [0] * len(x)  # 데이터가 부족한 경우 0으로 채움
                
                result = []
                for i in range(len(x)):
                    if i < n - 1:
                        # 데이터가 부족한 초기 구간
                        window_data = x[:i+1]
                    else:
                        # 충분한 데이터가 있는 구간
                        window_data = x[i-n+1:i+1]
                        
                    # Store values with their indices
                    indexed_values = [(val, j) for j, val in enumerate(window_data)]
                    # Sort by value (ascending)
                    sorted_values = sorted(indexed_values, key=lambda x: x[0])
                    
                    # Calculate ranks (normalized between 0 and 1)
                    ranks = [0] * len(window_data)
                    for j, (val, idx) in enumerate(sorted_values):
                        ranks[idx] = j / (len(window_data) - 1) if len(window_data) > 1 else 0
                    
                    result.append(ranks[-1])  # 현재 시점의 순위를 추가
                    
                return result
            elif func_name == 'sum':
                x = args[0]
                if len(args) == 1:
                    # 시계열 합산 (axis=0 방향)
                    if isinstance(x, list):
                        return x  # 원본 시계열 반환
                    return [x]  # 스칼라인 경우 리스트로 변환
                elif len(args) == 2:
                    # 특정 기간 동안의 합을 리스트로 반환
                    n = int(args[1][0])
                    if len(x) < n:
                        return [0] * len(x)
                    result = []
                    for i in range(len(x)):
                        if i < n - 1:
                            # 데이터가 부족한 초기 구간
                            window_data = x[:i+1]
                        else:
                            # 충분한 데이터가 있는 구간
                            window_data = x[i-n+1:i+1]
                        result.append(sum(window_data))
                    return result
                else:
                    raise ValueError("sum function takes 1 or 2 arguments")
            elif func_name == 'abs':
                x = args[0]
                if isinstance(x, list):
                    # 리스트의 경우 각 원소별로 abs 적용
                    result = [abs(xx) for xx in x]
                    print(f"{indent}[DEBUG] FUNCTION abs => shape={len(result)}")
                    return result
                else:
                    # 스칼라의 경우 단순 abs 적용
                    result = abs(x)
                    print(f"{indent}[DEBUG] FUNCTION abs => scalar={result}")
                    return result
            elif func_name == 'sign':
                x = args[0]
                if isinstance(x, list):
                    return [1 if val > 0 else (-1 if val < 0 else 0) for val in x]
                return 1 if x > 0 else (-1 if x < 0 else 0)
            elif func_name == 'min':
                if len(args) == 1:
                    # 단일 리스트의 경우 전체 최소값 계산
                    x = args[0]
                    if not isinstance(x, list):
                        return x
                    if len(x) == 0:
                        return None
                    # None 값이 있으면 None 반환
                    if any(v is None for v in x):
                        return None
                    return min(x)
                elif len(args) == 2:
                    # 두 인자의 경우 원소별 최소값 계산
                    left = args[0]
                    right = args[1]
                    
                    # 안전한 최소값 계산을 위한 헬퍼 함수
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
            elif func_name == 'product':
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
            elif func_name == 'max':
                if len(args) == 1:
                    # 단일 리스트의 경우 전체 최대값 계산
                    x = args[0]
                    if not isinstance(x, list):
                        return x
                    if len(x) == 0:
                        return None
                    # None 값이 있으면 None 반환
                    if any(v is None for v in x):
                        return None
                    return max(x)
                elif len(args) == 2:
                    # 두 인자의 경우 원소별 최대값 계산
                    left = args[0]
                    right = args[1]
                    
                    # 안전한 최대값 계산을 위한 헬퍼 함수
                    def safe_max(a, b):
                        if a is None or b is None:
                            return None
                        return max(a, b)
                    
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
                        
                        return [safe_max(l, r) for l, r in zip(left, right)]
                    else:
                        # 스칼라 연산
                        return safe_max(left, right)
                else:
                    raise ValueError("max function expects 1 or 2 arguments")
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
        window = int(window)
        if window <= 0:
            raise ValueError("Window must be positive")
        result = []
        for i in range(len(data)):
            start = max(0, i - window + 1)
            result.append(max(data[start:i + 1]))
        return result

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