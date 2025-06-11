#pragma once

#include "alpha_lexer.hpp"
#include "alpha_tokens.hpp"
#include <Eigen/Dense>
#include <memory>
#include <unordered_map>
#include <vector>
#include <functional>

// Architecture-specific SIMD headers
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
    #include <immintrin.h>
    #define USE_AVX2
#elif defined(__aarch64__) || defined(_M_ARM64)
    #include <arm_neon.h>
    #define USE_NEON
#endif

namespace alpha_parser {

using VectorXd = Eigen::VectorXd;
using MatrixXd = Eigen::MatrixXd;
using DataMap = std::unordered_map<std::string, VectorXd>;

// Forward declaration
class ASTNode;
using ASTNodePtr = std::unique_ptr<ASTNode>;

enum class ASTNodeType {
    NUMBER,
    VARIABLE,
    INDCLASS,
    LIST,
    CONDITIONAL,
    BINARY_OP,
    UNARY_OP,
    FUNCTION_CALL
};

class ASTNode {
public:
    ASTNodeType type;
    std::variant<double, std::string, TokenType> value;
    std::vector<ASTNodePtr> children;
    
    ASTNode(ASTNodeType t) : type(t) {}
    ASTNode(ASTNodeType t, double v) : type(t), value(v) {}
    ASTNode(ASTNodeType t, const std::string& v) : type(t), value(v) {}
    ASTNode(ASTNodeType t, TokenType v) : type(t), value(v) {}
    
    virtual ~ASTNode() = default;
    
    VectorXd evaluate(const DataMap& variables, int depth = 0) const;
    
private:
    VectorXd evaluateNumber() const;
    VectorXd evaluateVariable(const DataMap& variables) const;
    VectorXd evaluateIndClass(const DataMap& variables) const;
    VectorXd evaluateList(const DataMap& variables, int depth) const;
    VectorXd evaluateConditional(const DataMap& variables, int depth) const;
    VectorXd evaluateBinaryOp(const DataMap& variables, int depth) const;
    VectorXd evaluateUnaryOp(const DataMap& variables, int depth) const;
    VectorXd evaluateFunctionCall(const DataMap& variables, int depth) const;
    
    // Helper methods for alignment
    static std::pair<VectorXd, VectorXd> alignVectors(const VectorXd& a, const VectorXd& b);
};

class Parser {
private:
    AlphaLexer& lexer;
    Token current_token;
    
    void error(const std::string& message);
    void eat(TokenType token_type);
    
    ASTNodePtr atom();
    ASTNodePtr list_expr();
    ASTNodePtr function_call(const std::string& name);
    ASTNodePtr factor();
    ASTNodePtr power();
    ASTNodePtr term();
    ASTNodePtr arithmetic_expr();
    ASTNodePtr comparison_expr();
    ASTNodePtr and_expr();
    ASTNodePtr or_expr();
    ASTNodePtr conditional_expr();
    ASTNodePtr expr();
    
public:
    explicit Parser(AlphaLexer& lexer);
    ASTNodePtr parse();
};

class AlphaParser {
public:
    AlphaParser();
    
    void set_variables(const DataMap& vars);
    VectorXd parse_and_evaluate(const std::string& formula);
    
    // Function registry for dynamic function calls
    using FunctionType = std::function<VectorXd(const std::vector<VectorXd>&)>;
    std::unordered_map<std::string, FunctionType> function_registry;
    
    void register_function(const std::string& name, FunctionType func);
    void init_builtin_functions();
    
    // High-performance mathematical functions using SIMD
    static VectorXd rank_simd(const VectorXd& x);
    static VectorXd correlation_simd(const VectorXd& x, const VectorXd& y, int window);
    static VectorXd ts_max_simd(const VectorXd& data, int window);
    static VectorXd ts_min_simd(const VectorXd& data, int window);
    static VectorXd ts_rank_simd(const VectorXd& data, int window);
    static VectorXd delay_simd(const VectorXd& data, int periods);
    static VectorXd delta_simd(const VectorXd& data, int periods = 1);
    static VectorXd stddev_simd(const VectorXd& data, int window);
    static VectorXd scale_simd(const VectorXd& data);
    static VectorXd decay_linear_simd(const VectorXd& data, int window);
    static VectorXd indneutralize_simd(const VectorXd& data, const VectorXd& groups);
    
    // 새로 추가되는 함수들
    static VectorXd ts_mean_simd(const VectorXd& data, int window);
    static VectorXd ts_argmax_simd(const VectorXd& data, int window);
    static VectorXd ts_argmin_simd(const VectorXd& data, int window);
    static VectorXd sum_simd(const VectorXd& data, int window);
    static VectorXd product_simd(const VectorXd& data, int window);
    static VectorXd min_simd(const VectorXd& a, const VectorXd& b);
    static VectorXd max_simd(const VectorXd& a, const VectorXd& b);
    static VectorXd covariance_simd(const VectorXd& x, const VectorXd& y, int window);
    static VectorXd signedpower_simd(const VectorXd& base, const VectorXd& exponent);
    static VectorXd log_simd(const VectorXd& data);
    static VectorXd abs_simd(const VectorXd& data);
    static VectorXd sign_simd(const VectorXd& data);
    
    // ADV and market data functions
    static VectorXd adv_simd(const VectorXd& volume, int window);
    static VectorXd cap_simd(const VectorXd& close, const VectorXd& shares_outstanding);
    static VectorXd indclass_simd(const VectorXd& industry_codes, int target_industry);
    static VectorXd sector_simd(const VectorXd& sector_codes, int target_sector);
    
    // SIMD optimized element-wise operations
    static void add_simd(const double* a, const double* b, double* result, int size);
    static void sub_simd(const double* a, const double* b, double* result, int size);
    static void mul_simd(const double* a, const double* b, double* result, int size);
    static void div_simd(const double* a, const double* b, double* result, int size);

private:
    DataMap variables;
};

// Utility functions for performance optimization
namespace utils {
    // Cache-friendly matrix operations
    template<typename T>
    void prefetch_data(const T* data, int size);
    
    // SIMD-optimized statistical functions
    double mean_simd(const double* data, int size);
    double variance_simd(const double* data, int size);
    double sum_simd(const double* data, int size);
    
    // Memory-aligned allocation for better SIMD performance
    template<typename T>
    T* aligned_alloc(size_t count, size_t alignment = 32);
    
    template<typename T>
    void aligned_free(T* ptr);
}

} // namespace alpha_parser 