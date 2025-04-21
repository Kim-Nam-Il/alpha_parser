# Alpha Parser

금융 Alpha 공식을 파싱하고 분석하기 위한 Python 라이브러리입니다.

## 설치

```bash
pip install alpha_parser
```

## 사용법

```python
from alpha_parser.lexer import Lexer
from alpha_parser.parser import Parser

# 공식 문자열
formula = "(rank(close) - 0.5)"

# 렉서 생성
lexer = Lexer(formula)

# 파서 생성 및 파싱
parser = Parser(lexer)
result = parser.parse()

print(result)
```

## 기능

- 수식 토큰화 (렉싱)
- 수식 파싱
- 기본 산술 연산 지원
- 변수 및 함수 지원
- 괄호 처리

## 라이선스

MIT 