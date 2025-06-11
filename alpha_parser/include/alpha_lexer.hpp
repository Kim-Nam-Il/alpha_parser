#pragma once

#include "alpha_tokens.hpp"
#include <string>
#include <vector>
#include <regex>

namespace alpha_parser {

class AlphaLexer {
private:
    std::string text;
    size_t pos;
    char current_char;
    std::vector<Token> tokens;
    size_t token_pos;
    
    void advance();
    char peek_char(int offset = 1) const;
    Token number();
    Token identifier();
    std::vector<Token> tokenize();
    
public:
    explicit AlphaLexer(const std::string& text);
    
    Token get_next_token();
    Token peek(int k = 1) const;
    std::vector<Token> get_all_tokens() const;
    void reset();
};

} // namespace alpha_parser 