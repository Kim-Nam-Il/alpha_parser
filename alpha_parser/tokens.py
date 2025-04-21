from enum import Enum
from typing import Any

class TokenType(Enum):
    # 기본 토큰
    EOF = 'EOF'
    NUMBER = 'NUMBER'
    IDENTIFIER = 'IDENTIFIER'
    
    # 산술 연산자
    PLUS = 'PLUS'
    MINUS = 'MINUS'
    MULTIPLY = 'MULTIPLY'
    DIVIDE = 'DIVIDE'
    POWER = 'POWER'
    
    # 비교 연산자
    EQUAL = 'EQUAL'
    NOT_EQUAL = 'NOT_EQUAL'
    GREATER = 'GREATER'
    LESS = 'LESS'
    GREATER_EQUAL = 'GREATER_EQUAL'
    LESS_EQUAL = 'LESS_EQUAL'
    
    # 논리 연산자
    AND = '&&'
    OR = '||'
    NOT = 'NOT'
    
    # 기타
    LPAREN = 'LPAREN'
    RPAREN = 'RPAREN'
    COMMA = 'COMMA'
    QUESTION = 'QUESTION'
    COLON = 'COLON'
    LBRACKET = 'LBRACKET'
    RBRACKET = 'RBRACKET'

# 알파 공식에서 자주 사용되는 변수들
KNOWN_VARIABLES = {
    'returns', 'open', 'close', 'high', 'low', 'volume', 'vwap', 'cap',
    'close_today', 'sma5', 'sma10', 'sma20', 'sma60',
    'amount', 'turn', 'factor', 'pb', 'pe', 'ps', 'industry'
}

# 알파 공식에서 사용되는 함수들
KNOWN_FUNCTIONS = {
    'rank', 'delay', 'correlation', 'covariance', 'scale', 'delta',
    'signedpower', 'decay_linear', 'indneutralize',
    'ts_min', 'ts_max', 'ts_argmax', 'ts_argmin', 'ts_rank',
    'min', 'max', 'sum', 'product', 'stddev',
    'log', 'abs', 'sign'
}

class Token:
    def __init__(self, type: TokenType, value: Any = None, line: int = 0, column: int = 0):
        self.type = type
        self.value = value
        self.line = line
        self.column = column
        
    def __repr__(self) -> str:
        return f"Token({self.type}, {self.value}, line={self.line}, column={self.column})"
        
    def __eq__(self, other) -> bool:
        if not isinstance(other, Token):
            return False
        return self.type == other.type and self.value == other.value 