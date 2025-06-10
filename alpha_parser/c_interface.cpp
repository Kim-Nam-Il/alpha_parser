#include "../include/alpha_parser.hpp"
#include <cstring>
#include <iostream>

using namespace alpha_parser;

// Global parser instance and variables
static AlphaParser* g_parser = nullptr;
static DataMap g_variables;

extern "C" {
    // Initialize the parser
    int init_parser() {
        try {
            g_parser = new AlphaParser();
            return 0;
        } catch (const std::exception& e) {
            std::cerr << "Error initializing parser: " << e.what() << std::endl;
            return -1;
        }
    }
    
    // Set variable data  
    int set_variable(const char* name, const double* data, int size) {
        if (!g_parser) return -1;
        
        try {
            VectorXd vec(size);
            for (int i = 0; i < size; ++i) {
                vec[i] = data[i];
            }
            
            // 전역 변수맵에 저장
            g_variables[std::string(name)] = vec;
            return 0;
        } catch (const std::exception& e) {
            std::cerr << "Error setting variable: " << e.what() << std::endl;
            return -1;
        }
    }
    
    // Parse and evaluate alpha formula
    int evaluate_alpha(const char* formula, double* result, int* result_size) {
        if (!g_parser) return -1;
        
        try {
            // 저장된 모든 변수를 파서에 설정
            g_parser->set_variables(g_variables);
            
            VectorXd output = g_parser->parse_and_evaluate(std::string(formula));
            
            *result_size = output.size();
            for (int i = 0; i < output.size(); ++i) {
                result[i] = output[i];
            }
            return 0;
        } catch (const std::exception& e) {
            std::cerr << "Error evaluating formula: " << e.what() << std::endl;
            return -1;
        }
    }
    
    // Cleanup
    void cleanup_parser() {
        if (g_parser) {
            delete g_parser;
            g_parser = nullptr;
        }
    }
} 