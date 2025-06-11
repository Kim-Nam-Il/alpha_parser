#include "alpha_lexer.hpp"
#include <stdexcept>
#include <cctype>
#include <algorithm>

namespace alpha_parser {

AlphaLexer::AlphaLexer(const std::string& text) 
    : text(text), pos(0), token_pos(0) {
    current_char = text.empty() ? '\0' : text[0];
    tokens = tokenize();
    token_pos = 0;
}

void AlphaLexer::advance() {
    pos++;
    if (pos >= text.length()) {
        current_char = '\0';
    } else {
        current_char = text[pos];
    }
}

char AlphaLexer::peek_char(int offset) const {
    size_t peek_pos = pos + offset;
    if (peek_pos >= text.length()) {
        return '\0';
    }
    return text[peek_pos];
}

Token AlphaLexer::number() {
    std::string result;
    
    // Handle numbers starting with decimal point
    if (current_char == '.') {
        result = "0";
        result += current_char;
        advance();
        while (current_char != '\0' && std::isdigit(current_char)) {
            result += current_char;
            advance();
        }
        return Token(TokenType::NUMBER, std::stod(result));
    }
    
    // Handle regular numbers
    while (current_char != '\0' && std::isdigit(current_char)) {
        result += current_char;
        advance();
    }
    
    if (current_char == '.') {
        result += current_char;
        advance();
        while (current_char != '\0' && std::isdigit(current_char)) {
            result += current_char;
            advance();
        }
    }
    
    return Token(TokenType::NUMBER, std::stod(result));
}

Token AlphaLexer::identifier() {
    std::string result;
    
    while (current_char != '\0' && (std::isalnum(current_char) || current_char == '_')) {
        result += current_char;
        advance();
    }
    
    return Token(TokenType::IDENTIFIER, result);
}

std::vector<Token> AlphaLexer::tokenize() {
    std::vector<Token> tokens;
    
    while (current_char != '\0') {
        if (std::isspace(current_char)) {
            advance();
        } else if (std::isalpha(current_char)) {
            // Check for IndClass pattern
            std::regex indclass_pattern("IndClass\\.(sector|industry|subindustry)");
            std::smatch match;
            std::string remaining = text.substr(pos);
            
            if (std::regex_search(remaining, match, indclass_pattern) && match.position() == 0) {
                std::string value = match.str();
                tokens.emplace_back(TokenType::IDENTIFIER, value);
                pos += value.length();
                current_char = (pos < text.length()) ? text[pos] : '\0';
                continue;
            } else {
                tokens.push_back(identifier());
            }
        } else if (std::isdigit(current_char) || (current_char == '.' && std::isdigit(peek_char()))) {
            tokens.push_back(number());
        } else if (current_char == '+') {
            tokens.emplace_back(TokenType::PLUS);
            advance();
        } else if (current_char == '-') {
            tokens.emplace_back(TokenType::MINUS);
            advance();
        } else if (current_char == '*') {
            if (peek_char() == '*') {
                advance();
                tokens.emplace_back(TokenType::POWER);
            } else {
                tokens.emplace_back(TokenType::MULTIPLY);
            }
            advance();
        } else if (current_char == '/') {
            tokens.emplace_back(TokenType::DIVIDE);
            advance();
        } else if (current_char == '^') {
            tokens.emplace_back(TokenType::POWER);
            advance();
        } else if (current_char == '%') {
            tokens.emplace_back(TokenType::MODULO);
            advance();
        } else if (current_char == '(') {
            tokens.emplace_back(TokenType::LPAREN);
            advance();
        } else if (current_char == ')') {
            tokens.emplace_back(TokenType::RPAREN);
            advance();
        } else if (current_char == ',') {
            tokens.emplace_back(TokenType::COMMA);
            advance();
        } else if (current_char == '[') {
            tokens.emplace_back(TokenType::LBRACKET);
            advance();
        } else if (current_char == ']') {
            tokens.emplace_back(TokenType::RBRACKET);
            advance();
        } else if (current_char == '<') {
            if (peek_char() == '=') {
                advance();
                tokens.emplace_back(TokenType::LESS_EQUAL);
            } else {
                tokens.emplace_back(TokenType::LESS);
            }
            advance();
        } else if (current_char == '>') {
            if (peek_char() == '=') {
                advance();
                tokens.emplace_back(TokenType::GREATER_EQUAL);
            } else {
                tokens.emplace_back(TokenType::GREATER);
            }
            advance();
        } else if (current_char == '=') {
            if (peek_char() == '=') {
                advance();
                tokens.emplace_back(TokenType::EQUAL);
            } else {
                throw std::runtime_error("Invalid character: " + std::string(1, current_char));
            }
            advance();
        } else if (current_char == '!') {
            if (peek_char() == '=') {
                advance();
                tokens.emplace_back(TokenType::NOT_EQUAL);
            } else {
                tokens.emplace_back(TokenType::NOT);
            }
            advance();
        } else if (current_char == '&') {
            if (peek_char() == '&') {
                advance();
                tokens.emplace_back(TokenType::AND);
                advance();
            } else {
                throw std::runtime_error("Invalid character: " + std::string(1, current_char) + " (did you mean && ?)");
            }
        } else if (current_char == '|') {
            if (peek_char() == '|') {
                advance();
                tokens.emplace_back(TokenType::OR);
                advance();
            } else {
                throw std::runtime_error("Invalid character: " + std::string(1, current_char) + " (did you mean || ?)");
            }
        } else if (current_char == '?') {
            tokens.emplace_back(TokenType::QUESTION);
            advance();
        } else if (current_char == ':') {
            tokens.emplace_back(TokenType::COLON);
            advance();
        } else {
            throw std::runtime_error("Invalid character: " + std::string(1, current_char));
        }
    }
    
    tokens.emplace_back(TokenType::EOF_TOKEN);
    return tokens;
}

Token AlphaLexer::get_next_token() {
    if (token_pos >= tokens.size()) {
        return Token(TokenType::EOF_TOKEN);
    }
    
    Token token = tokens[token_pos];
    token_pos++;
    return token;
}

Token AlphaLexer::peek(int k) const {
    size_t target = token_pos + k;
    if (target < tokens.size()) {
        return tokens[target];
    }
    return Token(TokenType::EOF_TOKEN);
}

std::vector<Token> AlphaLexer::get_all_tokens() const {
    return tokens;
}

void AlphaLexer::reset() {
    token_pos = 0;
}

} // namespace alpha_parser 