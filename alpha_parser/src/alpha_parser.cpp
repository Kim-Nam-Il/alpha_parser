#include "alpha_parser.hpp"
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <numeric>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace alpha_parser {

// ASTNode implementation
VectorXd ASTNode::evaluate(const DataMap& variables, int depth) const {
    switch (type) {
        case ASTNodeType::NUMBER:
            return evaluateNumber();
        case ASTNodeType::VARIABLE:
            return evaluateVariable(variables);
        case ASTNodeType::INDCLASS:
            return evaluateIndClass(variables);
        case ASTNodeType::LIST:
            return evaluateList(variables, depth);
        case ASTNodeType::CONDITIONAL:
            return evaluateConditional(variables, depth);
        case ASTNodeType::BINARY_OP:
            return evaluateBinaryOp(variables, depth);
        case ASTNodeType::UNARY_OP:
            return evaluateUnaryOp(variables, depth);
        case ASTNodeType::FUNCTION_CALL:
            return evaluateFunctionCall(variables, depth);
        default:
            throw std::runtime_error("Unknown AST node type");
    }
}

VectorXd ASTNode::evaluateNumber() const {
    double val = std::get<double>(value);
    return VectorXd::Constant(1, val);
}

VectorXd ASTNode::evaluateVariable(const DataMap& variables) const {
    std::string var_name = std::get<std::string>(value);
    auto it = variables.find(var_name);
    if (it != variables.end()) {
        return it->second;
    }
    // Return zero vector if variable not found
    return VectorXd::Zero(1);
}

VectorXd ASTNode::evaluateIndClass(const DataMap& variables) const {
    std::string indclass_str = std::get<std::string>(value);
    if (indclass_str.find("IndClass.") == 0) {
        std::string level = indclass_str.substr(9); // Remove "IndClass."
        auto it = variables.find(level);
        if (it != variables.end()) {
            return it->second;
        }
    }
    return VectorXd::Zero(1);
}

VectorXd ASTNode::evaluateList(const DataMap& variables, int depth) const {
    if (children.empty()) {
        return VectorXd::Zero(1);
    }
    
    // For lists, we concatenate all child results
    VectorXd result = children[0]->evaluate(variables, depth + 1);
    for (size_t i = 1; i < children.size(); ++i) {
        VectorXd child_result = children[i]->evaluate(variables, depth + 1);
        VectorXd temp(result.size() + child_result.size());
        temp << result, child_result;
        result = temp;
    }
    return result;
}

VectorXd ASTNode::evaluateConditional(const DataMap& variables, int depth) const {
    if (children.size() != 3) {
        throw std::runtime_error("Conditional expression requires 3 children");
    }
    
    VectorXd condition = children[0]->evaluate(variables, depth + 1);
    VectorXd true_expr = children[1]->evaluate(variables, depth + 1);
    VectorXd false_expr = children[2]->evaluate(variables, depth + 1);
    
    // Align all vectors to the same size
    int max_size = std::max({condition.size(), true_expr.size(), false_expr.size()});
    
    if (condition.size() < max_size) {
        condition = VectorXd::Constant(max_size, condition(0));
    }
    if (true_expr.size() < max_size) {
        true_expr = VectorXd::Constant(max_size, true_expr(0));
    }
    if (false_expr.size() < max_size) {
        false_expr = VectorXd::Constant(max_size, false_expr(0));
    }
    
    VectorXd result(max_size);
    for (int i = 0; i < max_size; ++i) {
        result(i) = (condition(i) != 0.0) ? true_expr(i) : false_expr(i);
    }
    
    return result;
}

VectorXd ASTNode::evaluateBinaryOp(const DataMap& variables, int depth) const {
    if (children.size() != 2) {
        throw std::runtime_error("Binary operation requires 2 children");
    }
    
    VectorXd left = children[0]->evaluate(variables, depth + 1);
    VectorXd right = children[1]->evaluate(variables, depth + 1);
    
    auto [aligned_left, aligned_right] = alignVectors(left, right);
    
    TokenType op = std::get<TokenType>(value);
    VectorXd result(aligned_left.size());
    
    // Use SIMD optimized operations for large vectors
    if (aligned_left.size() > 8) {
        switch (op) {
            case TokenType::PLUS:
                AlphaParser::add_simd(aligned_left.data(), aligned_right.data(), result.data(), aligned_left.size());
                break;
            case TokenType::MINUS:
                AlphaParser::sub_simd(aligned_left.data(), aligned_right.data(), result.data(), aligned_left.size());
                break;
            case TokenType::MULTIPLY:
                AlphaParser::mul_simd(aligned_left.data(), aligned_right.data(), result.data(), aligned_left.size());
                break;
            case TokenType::DIVIDE:
                AlphaParser::div_simd(aligned_left.data(), aligned_right.data(), result.data(), aligned_left.size());
                break;
            default:
                // Fall back to element-wise operations for other operators
                for (int i = 0; i < aligned_left.size(); ++i) {
                    switch (op) {
                        case TokenType::POWER:
                            result(i) = std::pow(aligned_left(i), aligned_right(i));
                            break;
                        case TokenType::EQUAL:
                            result(i) = (aligned_left(i) == aligned_right(i)) ? 1.0 : 0.0;
                            break;
                        case TokenType::NOT_EQUAL:
                            result(i) = (aligned_left(i) != aligned_right(i)) ? 1.0 : 0.0;
                            break;
                        case TokenType::GREATER:
                            result(i) = (aligned_left(i) > aligned_right(i)) ? 1.0 : 0.0;
                            break;
                        case TokenType::LESS:
                            result(i) = (aligned_left(i) < aligned_right(i)) ? 1.0 : 0.0;
                            break;
                        case TokenType::GREATER_EQUAL:
                            result(i) = (aligned_left(i) >= aligned_right(i)) ? 1.0 : 0.0;
                            break;
                        case TokenType::LESS_EQUAL:
                            result(i) = (aligned_left(i) <= aligned_right(i)) ? 1.0 : 0.0;
                            break;
                        case TokenType::AND:
                            result(i) = (aligned_left(i) != 0.0 && aligned_right(i) != 0.0) ? 1.0 : 0.0;
                            break;
                        case TokenType::OR:
                            result(i) = (aligned_left(i) != 0.0 || aligned_right(i) != 0.0) ? 1.0 : 0.0;
                            break;
                        default:
                            throw std::runtime_error("Unknown binary operator");
                    }
                }
        }
    } else {
        // For small vectors, use standard operations
        for (int i = 0; i < aligned_left.size(); ++i) {
            switch (op) {
                case TokenType::PLUS:
                    result(i) = aligned_left(i) + aligned_right(i);
                    break;
                case TokenType::MINUS:
                    result(i) = aligned_left(i) - aligned_right(i);
                    break;
                case TokenType::MULTIPLY:
                    result(i) = aligned_left(i) * aligned_right(i);
                    break;
                case TokenType::DIVIDE:
                    result(i) = (aligned_right(i) != 0.0) ? aligned_left(i) / aligned_right(i) : std::numeric_limits<double>::quiet_NaN();
                    break;
                case TokenType::POWER:
                    result(i) = std::pow(aligned_left(i), aligned_right(i));
                    break;
                case TokenType::EQUAL:
                    result(i) = (aligned_left(i) == aligned_right(i)) ? 1.0 : 0.0;
                    break;
                case TokenType::NOT_EQUAL:
                    result(i) = (aligned_left(i) != aligned_right(i)) ? 1.0 : 0.0;
                    break;
                case TokenType::GREATER:
                    result(i) = (aligned_left(i) > aligned_right(i)) ? 1.0 : 0.0;
                    break;
                case TokenType::LESS:
                    result(i) = (aligned_left(i) < aligned_right(i)) ? 1.0 : 0.0;
                    break;
                case TokenType::GREATER_EQUAL:
                    result(i) = (aligned_left(i) >= aligned_right(i)) ? 1.0 : 0.0;
                    break;
                case TokenType::LESS_EQUAL:
                    result(i) = (aligned_left(i) <= aligned_right(i)) ? 1.0 : 0.0;
                    break;
                case TokenType::AND:
                    result(i) = (aligned_left(i) != 0.0 && aligned_right(i) != 0.0) ? 1.0 : 0.0;
                    break;
                case TokenType::OR:
                    result(i) = (aligned_left(i) != 0.0 || aligned_right(i) != 0.0) ? 1.0 : 0.0;
                    break;
                default:
                    throw std::runtime_error("Unknown binary operator");
            }
        }
    }
    
    return result;
}

VectorXd ASTNode::evaluateUnaryOp(const DataMap& variables, int depth) const {
    if (children.size() != 1) {
        throw std::runtime_error("Unary operation requires 1 child");
    }
    
    VectorXd child = children[0]->evaluate(variables, depth + 1);
    TokenType op = std::get<TokenType>(value);
    
    VectorXd result(child.size());
    for (int i = 0; i < child.size(); ++i) {
        switch (op) {
            case TokenType::PLUS:
                result(i) = +child(i);
                break;
            case TokenType::MINUS:
                result(i) = -child(i);
                break;
            case TokenType::NOT:
                result(i) = (child(i) == 0.0) ? 1.0 : 0.0;
                break;
            default:
                throw std::runtime_error("Unknown unary operator");
        }
    }
    
    return result;
}

VectorXd ASTNode::evaluateFunctionCall(const DataMap& variables, int depth) const {
    std::string func_name = std::get<std::string>(value);
    std::transform(func_name.begin(), func_name.end(), func_name.begin(), ::tolower);
    
    std::vector<VectorXd> args;
    for (const auto& child : children) {
        args.push_back(child->evaluate(variables, depth + 1));
    }
    
    // Built-in function implementations
    if (func_name == "rank" && args.size() == 1) {
        return AlphaParser::rank_simd(args[0]);
    } else if (func_name == "delay" && args.size() == 2) {
        int periods = static_cast<int>(args[1](0));
        return AlphaParser::delay_simd(args[0], periods);
    } else if (func_name == "correlation" && args.size() == 3) {
        int window = static_cast<int>(args[2](0));
        return AlphaParser::correlation_simd(args[0], args[1], window);
    } else if (func_name == "ts_max" && args.size() == 2) {
        int window = static_cast<int>(args[1](0));
        return AlphaParser::ts_max_simd(args[0], window);
    } else if (func_name == "ts_min" && args.size() == 2) {
        int window = static_cast<int>(args[1](0));
        return AlphaParser::ts_min_simd(args[0], window);
    } else if (func_name == "ts_mean" && args.size() == 2) {
        int window = static_cast<int>(args[1](0));
        return AlphaParser::ts_mean_simd(args[0], window);
    } else if (func_name == "ts_rank" && args.size() == 2) {
        int window = static_cast<int>(args[1](0));
        return AlphaParser::ts_rank_simd(args[0], window);
    } else if (func_name == "delta" && args.size() >= 1) {
        int periods = (args.size() > 1) ? static_cast<int>(args[1](0)) : 1;
        return AlphaParser::delta_simd(args[0], periods);
    } else if (func_name == "stddev" && args.size() == 2) {
        int window = static_cast<int>(args[1](0));
        return AlphaParser::stddev_simd(args[0], window);
    } else if (func_name == "scale" && args.size() == 1) {
        return AlphaParser::scale_simd(args[0]);
    } else if (func_name == "decay_linear" && args.size() == 2) {
        int window = static_cast<int>(args[1](0));
        return AlphaParser::decay_linear_simd(args[0], window);
    } else if (func_name == "indneutralize" && args.size() == 2) {
        return AlphaParser::indneutralize_simd(args[0], args[1]);
    } else if (func_name == "log" && args.size() == 1) {
        return AlphaParser::log_simd(args[0]);
    } else if (func_name == "abs" && args.size() == 1) {
        return AlphaParser::abs_simd(args[0]);
    } else if (func_name == "sign" && args.size() == 1) {
        return AlphaParser::sign_simd(args[0]);
    // 새로 추가된 함수들
    } else if (func_name == "ts_argmax" && args.size() == 2) {
        int window = static_cast<int>(args[1](0));
        return AlphaParser::ts_argmax_simd(args[0], window);
    } else if (func_name == "ts_argmin" && args.size() == 2) {
        int window = static_cast<int>(args[1](0));
        return AlphaParser::ts_argmin_simd(args[0], window);
    } else if (func_name == "sum" && args.size() == 2) {
        int window = static_cast<int>(args[1](0));
        return AlphaParser::sum_simd(args[0], window);
    } else if (func_name == "product" && args.size() == 2) {
        int window = static_cast<int>(args[1](0));
        return AlphaParser::product_simd(args[0], window);
    } else if (func_name == "min" && args.size() == 2) {
        return AlphaParser::min_simd(args[0], args[1]);
    } else if (func_name == "max" && args.size() == 2) {
        return AlphaParser::max_simd(args[0], args[1]);
    } else if (func_name == "covariance" && args.size() == 3) {
        int window = static_cast<int>(args[2](0));
        return AlphaParser::covariance_simd(args[0], args[1], window);
    } else if (func_name == "signedpower" && args.size() == 2) {
        return AlphaParser::signedpower_simd(args[0], args[1]);
    // ADV functions
    } else if (func_name == "adv20" && args.size() == 1) {
        return AlphaParser::adv_simd(args[0], 20);
    } else if (func_name == "adv30" && args.size() == 1) {
        return AlphaParser::adv_simd(args[0], 30);
    } else if (func_name == "adv40" && args.size() == 1) {
        return AlphaParser::adv_simd(args[0], 40);
    } else if (func_name == "adv50" && args.size() == 1) {
        return AlphaParser::adv_simd(args[0], 50);
    } else if (func_name == "adv60" && args.size() == 1) {
        return AlphaParser::adv_simd(args[0], 60);
    } else if (func_name == "adv81" && args.size() == 1) {
        return AlphaParser::adv_simd(args[0], 81);
    } else if (func_name == "adv120" && args.size() == 1) {
        return AlphaParser::adv_simd(args[0], 120);
    } else if (func_name == "adv150" && args.size() == 1) {
        return AlphaParser::adv_simd(args[0], 150);
    } else if (func_name == "adv180" && args.size() == 1) {
        return AlphaParser::adv_simd(args[0], 180);
    // Market data functions
    } else if (func_name == "cap" && args.size() == 2) {
        return AlphaParser::cap_simd(args[0], args[1]);
    } else if (func_name == "indclass" && args.size() == 2) {
        int target_industry = static_cast<int>(args[1](0));
        return AlphaParser::indclass_simd(args[0], target_industry);
    } else if (func_name == "industry" && args.size() == 1) {
        return args[0]; // Return industry codes as-is
    } else if (func_name == "sector" && args.size() == 2) {
        int target_sector = static_cast<int>(args[1](0));
        return AlphaParser::sector_simd(args[0], target_sector);
    }
    
    throw std::runtime_error("Unknown function: " + func_name);
}

std::pair<VectorXd, VectorXd> ASTNode::alignVectors(const VectorXd& a, const VectorXd& b) {
    if (a.size() == b.size()) {
        return {a, b};
    }
    
    int max_size = std::max(a.size(), b.size());
    VectorXd aligned_a = a;
    VectorXd aligned_b = b;
    
    if (a.size() < max_size) {
        aligned_a = VectorXd::Constant(max_size, a(0));
    }
    if (b.size() < max_size) {
        aligned_b = VectorXd::Constant(max_size, b(0));
    }
    
    return {aligned_a, aligned_b};
}

// Parser implementation
Parser::Parser(AlphaLexer& lexer) : lexer(lexer) {
    current_token = lexer.get_next_token();
}

void Parser::error(const std::string& message) {
    throw std::runtime_error("Parser error: " + message);
}

void Parser::eat(TokenType token_type) {
    if (current_token.type == token_type) {
        current_token = lexer.get_next_token();
    } else {
        error("Expected token type " + std::to_string(static_cast<int>(token_type)));
    }
}

ASTNodePtr Parser::atom() {
    Token token = current_token;
    
    if (token.type == TokenType::NUMBER) {
        eat(TokenType::NUMBER);
        return std::make_unique<ASTNode>(ASTNodeType::NUMBER, std::get<double>(token.value));
    } else if (token.type == TokenType::IDENTIFIER) {
        std::string name = std::get<std::string>(token.value);
        eat(TokenType::IDENTIFIER);
        
        if (current_token.type == TokenType::LPAREN) {
            return function_call(name);
        } else if (name.find("IndClass.") == 0) {
            return std::make_unique<ASTNode>(ASTNodeType::INDCLASS, name);
        } else {
            return std::make_unique<ASTNode>(ASTNodeType::VARIABLE, name);
        }
    } else if (token.type == TokenType::LPAREN) {
        eat(TokenType::LPAREN);
        ASTNodePtr node = expr();
        eat(TokenType::RPAREN);
        return node;
    } else if (token.type == TokenType::LBRACKET) {
        return list_expr();
    } else {
        error("Unexpected token in atom");
        return nullptr;
    }
}

ASTNodePtr Parser::list_expr() {
    eat(TokenType::LBRACKET);
    
    auto list_node = std::make_unique<ASTNode>(ASTNodeType::LIST);
    
    if (current_token.type != TokenType::RBRACKET) {
        list_node->children.push_back(expr());
        
        while (current_token.type == TokenType::COMMA) {
            eat(TokenType::COMMA);
            list_node->children.push_back(expr());
        }
    }
    
    eat(TokenType::RBRACKET);
    return list_node;
}

ASTNodePtr Parser::function_call(const std::string& name) {
    eat(TokenType::LPAREN);
    
    auto func_node = std::make_unique<ASTNode>(ASTNodeType::FUNCTION_CALL, name);
    
    if (current_token.type != TokenType::RPAREN) {
        func_node->children.push_back(expr());
        
        while (current_token.type == TokenType::COMMA) {
            eat(TokenType::COMMA);
            func_node->children.push_back(expr());
        }
    }
    
    eat(TokenType::RPAREN);
    return func_node;
}

ASTNodePtr Parser::factor() {
    Token token = current_token;
    
    if (token.type == TokenType::PLUS) {
        eat(TokenType::PLUS);
        auto unary_node = std::make_unique<ASTNode>(ASTNodeType::UNARY_OP, TokenType::PLUS);
        unary_node->children.push_back(factor());
        return unary_node;
    } else if (token.type == TokenType::MINUS) {
        eat(TokenType::MINUS);
        auto unary_node = std::make_unique<ASTNode>(ASTNodeType::UNARY_OP, TokenType::MINUS);
        unary_node->children.push_back(factor());
        return unary_node;
    } else if (token.type == TokenType::NOT) {
        eat(TokenType::NOT);
        auto unary_node = std::make_unique<ASTNode>(ASTNodeType::UNARY_OP, TokenType::NOT);
        unary_node->children.push_back(factor());
        return unary_node;
    } else {
        return atom();
    }
}

ASTNodePtr Parser::power() {
    ASTNodePtr node = factor();
    
    while (current_token.type == TokenType::POWER) {
        Token token = current_token;
        eat(TokenType::POWER);
        auto binary_node = std::make_unique<ASTNode>(ASTNodeType::BINARY_OP, token.type);
        binary_node->children.push_back(std::move(node));
        binary_node->children.push_back(factor());
        node = std::move(binary_node);
    }
    
    return node;
}

ASTNodePtr Parser::term() {
    ASTNodePtr node = power();
    
    while (current_token.type == TokenType::MULTIPLY || current_token.type == TokenType::DIVIDE) {
        Token token = current_token;
        if (token.type == TokenType::MULTIPLY) {
            eat(TokenType::MULTIPLY);
        } else {
            eat(TokenType::DIVIDE);
        }
        
        auto binary_node = std::make_unique<ASTNode>(ASTNodeType::BINARY_OP, token.type);
        binary_node->children.push_back(std::move(node));
        binary_node->children.push_back(power());
        node = std::move(binary_node);
    }
    
    return node;
}

ASTNodePtr Parser::arithmetic_expr() {
    ASTNodePtr node = term();
    
    while (current_token.type == TokenType::PLUS || current_token.type == TokenType::MINUS) {
        Token token = current_token;
        if (token.type == TokenType::PLUS) {
            eat(TokenType::PLUS);
        } else {
            eat(TokenType::MINUS);
        }
        
        auto binary_node = std::make_unique<ASTNode>(ASTNodeType::BINARY_OP, token.type);
        binary_node->children.push_back(std::move(node));
        binary_node->children.push_back(term());
        node = std::move(binary_node);
    }
    
    return node;
}

ASTNodePtr Parser::comparison_expr() {
    ASTNodePtr node = arithmetic_expr();
    
    while (current_token.type == TokenType::EQUAL || current_token.type == TokenType::NOT_EQUAL ||
           current_token.type == TokenType::GREATER || current_token.type == TokenType::LESS ||
           current_token.type == TokenType::GREATER_EQUAL || current_token.type == TokenType::LESS_EQUAL) {
        Token token = current_token;
        eat(token.type);
        
        auto binary_node = std::make_unique<ASTNode>(ASTNodeType::BINARY_OP, token.type);
        binary_node->children.push_back(std::move(node));
        binary_node->children.push_back(arithmetic_expr());
        node = std::move(binary_node);
    }
    
    return node;
}

ASTNodePtr Parser::and_expr() {
    ASTNodePtr node = comparison_expr();
    
    while (current_token.type == TokenType::AND) {
        Token token = current_token;
        eat(TokenType::AND);
        
        auto binary_node = std::make_unique<ASTNode>(ASTNodeType::BINARY_OP, token.type);
        binary_node->children.push_back(std::move(node));
        binary_node->children.push_back(comparison_expr());
        node = std::move(binary_node);
    }
    
    return node;
}

ASTNodePtr Parser::or_expr() {
    ASTNodePtr node = and_expr();
    
    while (current_token.type == TokenType::OR) {
        Token token = current_token;
        eat(TokenType::OR);
        
        auto binary_node = std::make_unique<ASTNode>(ASTNodeType::BINARY_OP, token.type);
        binary_node->children.push_back(std::move(node));
        binary_node->children.push_back(and_expr());
        node = std::move(binary_node);
    }
    
    return node;
}

ASTNodePtr Parser::conditional_expr() {
    ASTNodePtr node = or_expr();
    
    if (current_token.type == TokenType::QUESTION) {
        eat(TokenType::QUESTION);
        ASTNodePtr true_expr = expr();
        eat(TokenType::COLON);
        ASTNodePtr false_expr = expr();
        
        auto conditional_node = std::make_unique<ASTNode>(ASTNodeType::CONDITIONAL);
        conditional_node->children.push_back(std::move(node));
        conditional_node->children.push_back(std::move(true_expr));
        conditional_node->children.push_back(std::move(false_expr));
        node = std::move(conditional_node);
    }
    
    return node;
}

ASTNodePtr Parser::expr() {
    return conditional_expr();
}

ASTNodePtr Parser::parse() {
    return expr();
}

} // namespace alpha_parser 