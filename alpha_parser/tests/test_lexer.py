import unittest
from alpha_parser.lexer import Lexer
from alpha_parser.alpha_tokens import TokenType

class TestLexer(unittest.TestCase):
    def setUp(self):
        self.lexer = None
        
    def test_basic_tokens(self):
        text = "1 + 2 * 3"
        self.lexer = Lexer(text)
        tokens = self.lexer.get_all_tokens()
        
        expected_types = [
            TokenType.NUMBER,
            TokenType.PLUS,
            TokenType.NUMBER,
            TokenType.MULTIPLY,
            TokenType.NUMBER,
            TokenType.EOF
        ]
        
        self.assertEqual(len(tokens), len(expected_types))
        for token, expected_type in zip(tokens, expected_types):
            self.assertEqual(token.type, expected_type)
            
    def test_identifiers(self):
        text = "x + y * z"
        self.lexer = Lexer(text)
        tokens = self.lexer.get_all_tokens()
        
        expected_types = [
            TokenType.IDENTIFIER,
            TokenType.PLUS,
            TokenType.IDENTIFIER,
            TokenType.MULTIPLY,
            TokenType.IDENTIFIER,
            TokenType.EOF
        ]
        
        self.assertEqual(len(tokens), len(expected_types))
        for token, expected_type in zip(tokens, expected_types):
            self.assertEqual(token.type, expected_type)
            
    def test_parentheses(self):
        text = "(1 + 2) * 3"
        self.lexer = Lexer(text)
        tokens = self.lexer.get_all_tokens()
        
        expected_types = [
            TokenType.LPAREN,
            TokenType.NUMBER,
            TokenType.PLUS,
            TokenType.NUMBER,
            TokenType.RPAREN,
            TokenType.MULTIPLY,
            TokenType.NUMBER,
            TokenType.EOF
        ]
        
        self.assertEqual(len(tokens), len(expected_types))
        for token, expected_type in zip(tokens, expected_types):
            self.assertEqual(token.type, expected_type)
            
    def test_whitespace(self):
        text = "  1  +  2  "
        self.lexer = Lexer(text)
        tokens = self.lexer.get_all_tokens()
        
        expected_types = [
            TokenType.NUMBER,
            TokenType.PLUS,
            TokenType.NUMBER,
            TokenType.EOF
        ]
        
        self.assertEqual(len(tokens), len(expected_types))
        for token, expected_type in zip(tokens, expected_types):
            self.assertEqual(token.type, expected_type)
            
    def test_invalid_character(self):
        text = "1 @ 2"
        self.lexer = Lexer(text)
        with self.assertRaises(Exception):
            self.lexer.get_all_tokens()

if __name__ == '__main__':
    unittest.main() 