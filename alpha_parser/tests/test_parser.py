import unittest
from alpha_parser.lexer import Lexer
from alpha_parser.parser import Parser

class TestParser(unittest.TestCase):
    def setUp(self):
        self.lexer = None
        self.parser = None
        
    def test_basic_arithmetic(self):
        test_cases = [
            ("1 + 2", 3),
            ("2 * 3", 6),
            ("4 - 2", 2),
            ("6 / 2", 3),
            ("2 + 3 * 4", 14),
            ("(2 + 3) * 4", 20),
        ]
        
        for expression, expected in test_cases:
            self.lexer = Lexer(expression)
            self.parser = Parser(self.lexer)
            result = self.parser.parse()
            self.assertEqual(result.evaluate(), expected)
            
    def test_parentheses(self):
        test_cases = [
            ("(1 + 2) * 3", 9),
            ("1 + (2 * 3)", 7),
            ("((1 + 2) * 3) + 4", 13),
        ]
        
        for expression, expected in test_cases:
            self.lexer = Lexer(expression)
            self.parser = Parser(self.lexer)
            result = self.parser.parse()
            self.assertEqual(result.evaluate(), expected)
            
    def test_invalid_expressions(self):
        invalid_expressions = [
            "1 +",  # 불완전한 표현식
            "* 2",  # 잘못된 시작
            "1 + * 2",  # 잘못된 연산자
            "(1 + 2",  # 닫히지 않은 괄호
            "1 + 2)",  # 열리지 않은 괄호
        ]
        
        for expression in invalid_expressions:
            self.lexer = Lexer(expression)
            self.parser = Parser(self.lexer)
            with self.assertRaises(Exception):
                self.parser.parse()
                
    def test_identifiers(self):
        test_cases = [
            ("x + y", 3),
            ("price * quantity", 50),
            ("a - b", -1),
            ("x / y", 0.5)
        ]
        
        for expr, expected in test_cases:
            parser = Parser(Lexer(expr))
            result = parser.parse()
            self.assertEqual(result.evaluate({'x': 1, 'y': 2, 'price': 10, 'quantity': 5, 'a': 1, 'b': 2}), expected)

if __name__ == '__main__':
    unittest.main() 