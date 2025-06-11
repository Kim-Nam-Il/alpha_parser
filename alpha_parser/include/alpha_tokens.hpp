#pragma once

#include <string>
#include <unordered_set>
#include <variant>

namespace alpha_parser {

enum class TokenType {
    // Basic tokens
    EOF_TOKEN,
    NUMBER,
    IDENTIFIER,
    
    // Arithmetic operators
    PLUS,
    MINUS,
    MULTIPLY,
    DIVIDE,
    POWER,
    MODULO,
    
    // Comparison operators
    EQUAL,
    NOT_EQUAL,
    GREATER,
    LESS,
    GREATER_EQUAL,
    LESS_EQUAL,
    
    // Logical operators
    AND,
    OR,
    NOT,
    
    // Other tokens
    LPAREN,
    RPAREN,
    COMMA,
    QUESTION,
    COLON,
    LBRACKET,
    RBRACKET
};

struct Token {
    TokenType type;
    std::variant<double, std::string> value;
    int line = 0;
    int column = 0;
    
    // Default constructor
    Token() : type(TokenType::EOF_TOKEN), value(0.0), line(0), column(0) {}
    
    // Explicit constructors to avoid ambiguity
    explicit Token(TokenType t) 
        : type(t), value(0.0), line(0), column(0) {}
    
    Token(TokenType t, double v, int l = 0, int c = 0) 
        : type(t), value(v), line(l), column(c) {}
    
    Token(TokenType t, const std::string& v, int l = 0, int c = 0) 
        : type(t), value(v), line(l), column(c) {}
};

// Known variables in alpha formulas
const std::unordered_set<std::string> KNOWN_VARIABLES = {
    "returns", "open", "close", "high", "low", "volume", "vwap", "cap",
    "close_today", "sma5", "sma10", "sma20", "sma60",
    "amount", "turn", "factor", "pb", "pe", "ps", "industry",
    "adv5", "adv10", "adv15", "adv20", "adv30", "adv40", "adv50", 
    "adv60", "adv81", "adv120", "adv150", "adv180", "sector", "subindustry"
};

// Known functions in alpha formulas
const std::unordered_set<std::string> KNOWN_FUNCTIONS = {
    "rank", "delay", "correlation", "covariance", "scale", "delta",
    "signedpower", "decay_linear", "indneutralize", "sign",
    "ts_min", "ts_max", "ts_argmax", "ts_argmin", "ts_rank",
    "min", "max", "sum", "product", "stddev",
    "log", "abs", "sqrt", "pow"
};

// Industry classification levels
enum class IndClassLevel {
    SECTOR,
    INDUSTRY,
    SUBINDUSTRY
};

struct IndClass {
    IndClassLevel level;
    
    IndClass(const std::string& level_str) {
        if (level_str == "sector") level = IndClassLevel::SECTOR;
        else if (level_str == "industry") level = IndClassLevel::INDUSTRY;
        else if (level_str == "subindustry") level = IndClassLevel::SUBINDUSTRY;
        else throw std::invalid_argument("Invalid IndClass level: " + level_str);
    }
    
    bool operator==(const IndClass& other) const {
        return level == other.level;
    }
};

} // namespace alpha_parser 