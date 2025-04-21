from enum import Enum, auto
from typing import Any

class TokenType(Enum):
    # Basic tokens
    EOF = auto()
    NUMBER = auto()
    IDENTIFIER = auto()
    FUNCTION = auto()
    
    # Operators
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    POWER = auto()
    
    # Comparison operators
    EQUAL = auto()
    NOT_EQUAL = auto()
    GREATER = auto()
    LESS = auto()
    GREATER_EQUAL = auto()
    LESS_EQUAL = auto()
    
    # Logical operators
    AND = auto()
    OR = auto()
    NOT = auto()
    
    # Punctuation
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    COMMA = auto()
    QUESTION = auto()
    COLON = auto()
    
    # Assignment
    ASSIGN = auto()

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
    def __init__(self, type: TokenType, value: Any = None):
        self.type = type
        self.value = value
        
    def __repr__(self) -> str:
        return f"Token({self.type}, {self.value})"
        
    def __eq__(self, other) -> bool:
        if not isinstance(other, Token):
            return False
        return self.type == other.type and self.value == other.value 