from typing import List, Optional
import re
from .tokens import Token, TokenType, KNOWN_FUNCTIONS, KNOWN_VARIABLES

class AlphaLexer:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.current_char = self.text[0] if self.text else None
        
    def advance(self):
        self.pos += 1
        if self.pos < len(self.text):
            self.current_char = self.text[self.pos]
        else:
            self.current_char = None
            
    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()
            
    def number(self):
        result = ''
        while self.current_char is not None and (self.current_char.isdigit() or self.current_char == '.'):
            result += self.current_char
            self.advance()
        return float(result)
    
    def identifier(self):
        result = ''
        while self.current_char is not None and (self.current_char.isalnum() or self.current_char == '_'):
            result += self.current_char
            self.advance()
        return result
    
    def get_next_token(self) -> Token:
        while self.current_char is not None:
            if self.current_char.isspace():
                self.skip_whitespace()
                continue
                
            if self.current_char.isdigit():
                return Token(TokenType.NUMBER, self.number())
                
            if self.current_char.isalpha() or self.current_char == '_':
                ident = self.identifier()
                if ident in ['ts_min', 'ts_max', 'ts_rank', 'ts_sum', 'ts_mean', 'ts_std', 'ts_corr', 'ts_cov']:
                    return Token(TokenType.FUNCTION, ident)
                return Token(TokenType.IDENTIFIER, ident)
                
            if self.current_char == '+':
                self.advance()
                return Token(TokenType.PLUS)
                
            if self.current_char == '-':
                self.advance()
                return Token(TokenType.MINUS)
                
            if self.current_char == '*':
                self.advance()
                if self.current_char == '*':
                    self.advance()
                    return Token(TokenType.POWER)
                return Token(TokenType.MULTIPLY)
                
            if self.current_char == '/':
                self.advance()
                return Token(TokenType.DIVIDE)
                
            if self.current_char == '=':
                self.advance()
                if self.current_char == '=':
                    self.advance()
                    return Token(TokenType.EQUAL)
                return Token(TokenType.EQUAL)
                
            if self.current_char == '!':
                self.advance()
                if self.current_char == '=':
                    self.advance()
                    return Token(TokenType.NOT_EQUAL)
                return Token(TokenType.NOT)
                
            if self.current_char == '>':
                self.advance()
                if self.current_char == '=':
                    self.advance()
                    return Token(TokenType.GREATER_EQUAL)
                return Token(TokenType.GREATER)
                
            if self.current_char == '<':
                self.advance()
                if self.current_char == '=':
                    self.advance()
                    return Token(TokenType.LESS_EQUAL)
                return Token(TokenType.LESS)
                
            if self.current_char == '&':
                self.advance()
                if self.current_char == '&':
                    self.advance()
                    return Token(TokenType.AND)
                
            if self.current_char == '|':
                self.advance()
                if self.current_char == '|':
                    self.advance()
                    return Token(TokenType.OR)
                
            if self.current_char == '(':
                self.advance()
                return Token(TokenType.LPAREN)
                
            if self.current_char == ')':
                self.advance()
                return Token(TokenType.RPAREN)
                
            if self.current_char == '[':
                self.advance()
                return Token(TokenType.LBRACKET)
                
            if self.current_char == ']':
                self.advance()
                return Token(TokenType.RBRACKET)
                
            if self.current_char == ',':
                self.advance()
                return Token(TokenType.COMMA)
                
            if self.current_char == '?':
                self.advance()
                return Token(TokenType.QUESTION)
                
            if self.current_char == ':':
                self.advance()
                return Token(TokenType.COLON)
                
            raise ValueError(f"Invalid character: {self.current_char}")
            
        return Token(TokenType.EOF)
    
    def get_all_tokens(self) -> List[Token]:
        """모든 토큰을 반환"""
        tokens = []
        while True:
            token = self.get_next_token()
            tokens.append(token)
            if token.type == TokenType.EOF:
                break
        return tokens 