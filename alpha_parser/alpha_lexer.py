# alpha_parser/alpha_lexer.py
from typing import List, Optional
import re
from alpha_parser.alpha_tokens import Token, TokenType, KNOWN_FUNCTIONS, KNOWN_VARIABLES

class AlphaLexer:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.current_char = self.text[0] if self.text else None
        self.tokens = self._tokenize()
        self.pos = 0
        
    def _advance(self):
        self.pos += 1
        if self.pos >= len(self.text):
            self.current_char = None
        else:
            self.current_char = self.text[self.pos]
            
    def _peek_char(self) -> Optional[str]:
        peek_pos = self.pos + 1
        if peek_pos >= len(self.text):
            return None
        return self.text[peek_pos]

    def peek(self, k: int = 1) -> "Token":
        """
        현재 위치에서 k번째 앞으로 있는 토큰을 미리본다.
        토큰 시퀀스를 전부 만들고 난 뒤 self.tokens 리스트를 재사용하므로
        인덱스를 벗어나면 EOF 토큰을 돌려준다.
        """
        target = self.pos + k
        if target < len(self.tokens):
            return self.tokens[target]
        return Token(TokenType.EOF, None)

    def _tokenize(self) -> list["Token"]:
        tokens = []
        while self.current_char is not None:
            if self.current_char.isspace():
                self._advance()
            elif self.current_char.isalpha():
                # IndClass 패턴 체크
                match = re.match(r'IndClass\.(sector|industry|subindustry)', self.text[self.pos:])
                if match:
                    value = match.group(0)
                    tokens.append(Token(TokenType.IDENTIFIER, value))
                    self.pos += len(value)
                    self.current_char = self.text[self.pos] if self.pos < len(self.text) else None
                    continue
                else:
                    tokens.append(self._identifier())
            elif self.current_char.isdigit() or (self.current_char == '.' and self._peek_char() is not None and self._peek_char().isdigit()):
                tokens.append(self._number())
            elif self.current_char == '+':
                tokens.append(Token(TokenType.PLUS, '+'))
                self._advance()
            elif self.current_char == '-':
                tokens.append(Token(TokenType.MINUS, '-'))
                self._advance()
            elif self.current_char == '*':
                if self._peek_char() == '*':
                    self._advance()
                    tokens.append(Token(TokenType.POWER, '**'))
                else:
                    tokens.append(Token(TokenType.MULTIPLY, '*'))
                self._advance()
            elif self.current_char == '/':
                tokens.append(Token(TokenType.DIVIDE, '/'))
                self._advance()
            elif self.current_char == '^':
                tokens.append(Token(TokenType.POWER, '^'))
                self._advance()
            elif self.current_char == '%':
                if self._peek_char() == '=':
                    self._advance()
                    tokens.append(Token(TokenType.MODULO_ASSIGN, '%='))
                else:
                    tokens.append(Token(TokenType.MODULO, '%'))
                self._advance()
            elif self.current_char == '(':
                tokens.append(Token(TokenType.LPAREN, '('))
                self._advance()
            elif self.current_char == ')':
                tokens.append(Token(TokenType.RPAREN, ')'))
                self._advance()
            elif self.current_char == ',':
                tokens.append(Token(TokenType.COMMA, ','))
                self._advance()
            elif self.current_char == '[':
                tokens.append(Token(TokenType.LBRACKET, '['))
                self._advance()
            elif self.current_char == ']':
                tokens.append(Token(TokenType.RBRACKET, ']'))
                self._advance()
            elif self.current_char == '<':
                if self._peek_char() == '=':
                    self._advance()
                    tokens.append(Token(TokenType.LESS_EQUAL, '<='))
                else:
                    tokens.append(Token(TokenType.LESS, '<'))
                self._advance()
            elif self.current_char == '>':
                if self._peek_char() == '=':
                    self._advance()
                    tokens.append(Token(TokenType.GREATER_EQUAL, '>='))
                else:
                    tokens.append(Token(TokenType.GREATER, '>'))
                self._advance()
            elif self.current_char == '=':
                if self._peek_char() == '=':
                    self._advance()
                    tokens.append(Token(TokenType.EQUAL, '=='))
                else:
                    raise ValueError(f"Invalid character: {self.current_char}")
                self._advance()
            elif self.current_char == '!':
                if self._peek_char() == '=':
                    self._advance()
                    tokens.append(Token(TokenType.NOT_EQUAL, '!='))
                else:
                    tokens.append(Token(TokenType.NOT, '!'))
                self._advance()
            elif self.current_char == '&':
                if self._peek_char() == '&':
                    self._advance()
                    tokens.append(Token(TokenType.AND, '&&'))
                    self._advance()
                else:
                    raise ValueError(f"Invalid character: {self.current_char} (did you mean && ?)")
            elif self.current_char == '|':
                if self._peek_char() == '|':
                    self._advance()
                    tokens.append(Token(TokenType.OR, '||'))
                    self._advance()
                else:
                    raise ValueError(f"Invalid character: {self.current_char} (did you mean || ?)")
            elif self.current_char == '?':
                tokens.append(Token(TokenType.QUESTION, '?'))
                self._advance()
            elif self.current_char == ':':
                tokens.append(Token(TokenType.COLON, ':'))
                self._advance()
            else:
                raise ValueError(f"Invalid character: {self.current_char}")

        tokens.append(Token(TokenType.EOF, None))
        return tokens

    def _number(self) -> Token:
        result = ''
        # 소수점으로 시작하는 경우
        if self.current_char == '.':
            result = '0'  # 앞에 0 추가
            result += self.current_char
            self._advance()
            while self.current_char is not None and self.current_char.isdigit():
                result += self.current_char
                self._advance()
            return Token(TokenType.NUMBER, float(result))
            
        # 일반적인 숫자 처리
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self._advance()
        if self.current_char == '.':
            result += self.current_char
            self._advance()
            while self.current_char is not None and self.current_char.isdigit():
                result += self.current_char
                self._advance()
        return Token(TokenType.NUMBER, float(result))

    def _identifier(self) -> Token:
        result = ''
        while self.current_char is not None and (self.current_char.isalnum() or self.current_char == '_'):
            result += self.current_char
            self._advance()
        return Token(TokenType.IDENTIFIER, result)

    def get_next_token(self) -> Token:
        if self.pos >= len(self.tokens):
            return Token(TokenType.EOF, None)
        token = self.tokens[self.pos]
        self.pos += 1
        return token
    
    def get_all_tokens(self) -> List[Token]:
        """모든 토큰을 반환"""
        tokens = []
        while True:
            token = self.get_next_token()
            tokens.append(token)
            if token.type == TokenType.EOF:
                break
        return tokens 