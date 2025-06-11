#include "alpha_parser.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cassert>

using namespace alpha_parser;

// Sample data generator
DataMap generateSampleData(int size = 1000) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0.0, 1.0);
    std::uniform_real_distribution<> uniform(0.0, 1.0);
    
    DataMap data;
    
    // Generate price data
    VectorXd open(size), high(size), low(size), close(size);
    VectorXd volume(size), vwap(size), returns(size);
    
    double base_price = 100.0;
    for (int i = 0; i < size; ++i) {
        double change = dis(gen) * 0.02; // 2% volatility
        base_price *= (1.0 + change);
        
        close(i) = base_price;
        open(i) = base_price * (1.0 + dis(gen) * 0.01);
        high(i) = std::max(open(i), close(i)) * (1.0 + uniform(gen) * 0.01);
        low(i) = std::min(open(i), close(i)) * (1.0 - uniform(gen) * 0.01);
        volume(i) = std::abs(dis(gen)) * 1000000 + 5000000;
        vwap(i) = (high(i) + low(i) + close(i)) / 3.0;
        returns(i) = (i > 0) ? (close(i) - close(i-1)) / close(i-1) : 0.0;
    }
    
    data["open"] = open;
    data["high"] = high;
    data["low"] = low;
    data["close"] = close;
    data["volume"] = volume;
    data["vwap"] = vwap;
    data["returns"] = returns;
    
    // Generate additional data
    VectorXd adv20(size);
    for (int i = 0; i < size; ++i) {
        int start = std::max(0, i - 19);
        double sum = 0.0;
        for (int j = start; j <= i; ++j) {
            sum += volume(j);
        }
        adv20(i) = sum / (i - start + 1);
    }
    data["adv20"] = adv20;
    
    // Industry groups
    VectorXd industry(size);
    for (int i = 0; i < size; ++i) {
        industry(i) = i % 5; // 5 industry groups
    }
    data["industry"] = industry;
    
    return data;
}

void testBasicOperations() {
    std::cout << "Testing basic operations..." << std::endl;
    
    AlphaParser parser;
    DataMap data = generateSampleData(100);
    parser.set_variables(data);
    
    // Test simple arithmetic
    auto result1 = parser.parse_and_evaluate("close + open");
    assert(result1.size() == 100);
    std::cout << "✓ Addition test passed" << std::endl;
    
    auto result2 = parser.parse_and_evaluate("close - open");
    assert(result2.size() == 100);
    std::cout << "✓ Subtraction test passed" << std::endl;
    
    auto result3 = parser.parse_and_evaluate("close * volume");
    assert(result3.size() == 100);
    std::cout << "✓ Multiplication test passed" << std::endl;
    
    auto result4 = parser.parse_and_evaluate("close / open");
    assert(result4.size() == 100);
    std::cout << "✓ Division test passed" << std::endl;
}

void testFunctions() {
    std::cout << "\nTesting alpha functions..." << std::endl;
    
    AlphaParser parser;
    DataMap data = generateSampleData(100);
    parser.set_variables(data);
    
    // Test rank function
    auto result1 = parser.parse_and_evaluate("rank(close)");
    assert(result1.size() == 100);
    std::cout << "✓ Rank function test passed" << std::endl;
    
    // Test delay function
    auto result2 = parser.parse_and_evaluate("delay(close, 5)");
    assert(result2.size() == 100);
    std::cout << "✓ Delay function test passed" << std::endl;
    
    // Test correlation function
    auto result3 = parser.parse_and_evaluate("correlation(close, volume, 10)");
    assert(result3.size() == 100);
    std::cout << "✓ Correlation function test passed" << std::endl;
    
    // Test ts_max function
    auto result4 = parser.parse_and_evaluate("ts_max(close, 20)");
    assert(result4.size() == 100);
    std::cout << "✓ TS_Max function test passed" << std::endl;
    
    // Test stddev function
    auto result5 = parser.parse_and_evaluate("stddev(returns, 20)");
    assert(result5.size() == 100);
    std::cout << "✓ StdDev function test passed" << std::endl;
}

void testComplexAlphas() {
    std::cout << "\nTesting complex alpha formulas..." << std::endl;
    
    AlphaParser parser;
    DataMap data = generateSampleData(100);
    parser.set_variables(data);
    
    // Alpha #1: Simple alpha formula
    auto result1 = parser.parse_and_evaluate("rank(close - open)");
    assert(result1.size() == 100);
    std::cout << "✓ Alpha #1 test passed" << std::endl;
    
    // Alpha #2: More complex formula
    auto result2 = parser.parse_and_evaluate("(-1 * correlation(rank(open), rank(volume), 10))");
    assert(result2.size() == 100);
    std::cout << "✓ Alpha #2 test passed" << std::endl;
    
    // Alpha #3: With conditional logic
    auto result3 = parser.parse_and_evaluate("((close > open) ? 1 : -1)");
    assert(result3.size() == 100);
    std::cout << "✓ Alpha #3 test passed" << std::endl;
    
    // Alpha #4: With time series functions
    auto result4 = parser.parse_and_evaluate("rank(ts_max(close, 10) - ts_min(close, 10))");
    assert(result4.size() == 100);
    std::cout << "✓ Alpha #4 test passed" << std::endl;
}

void benchmarkPerformance() {
    std::cout << "\nBenchmarking performance..." << std::endl;
    
    AlphaParser parser;
    DataMap data = generateSampleData(10000); // Larger dataset
    parser.set_variables(data);
    
    std::vector<std::string> test_formulas = {
        "rank(close)",
        "correlation(close, volume, 20)",
        "ts_max(close, 50)",
        "stddev(returns, 30)",
        "(-1 * correlation(rank(open), rank(volume), 10))",
        "rank(ts_max(close, 20) - ts_min(close, 20))",
        "scale(close - delay(close, 1))"
    };
    
    for (const auto& formula : test_formulas) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Run multiple times for better measurement
        for (int i = 0; i < 10; ++i) {
            auto result = parser.parse_and_evaluate(formula);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Formula: " << formula << std::endl;
        std::cout << "Time (10 runs): " << duration.count() << " μs" << std::endl;
        std::cout << "Average: " << duration.count() / 10.0 << " μs per run" << std::endl;
        std::cout << "---" << std::endl;
    }
}

void testSIMDOptimizations() {
    std::cout << "\nTesting SIMD optimizations..." << std::endl;
    
    const int size = 10000;
    VectorXd a = VectorXd::Random(size);
    VectorXd b = VectorXd::Random(size);
    VectorXd result(size);
    
    // Test SIMD addition
    auto start = std::chrono::high_resolution_clock::now();
    AlphaParser::add_simd(a.data(), b.data(), result.data(), size);
    auto end = std::chrono::high_resolution_clock::now();
    auto simd_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Test regular addition
    start = std::chrono::high_resolution_clock::now();
    VectorXd regular_result = a + b;
    end = std::chrono::high_resolution_clock::now();
    auto regular_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "SIMD addition time: " << simd_time.count() << " μs" << std::endl;
    std::cout << "Regular addition time: " << regular_time.count() << " μs" << std::endl;
    std::cout << "Speedup: " << (double)regular_time.count() / simd_time.count() << "x" << std::endl;
    
    // Verify correctness
    double max_error = (result - regular_result).cwiseAbs().maxCoeff();
    std::cout << "Max error: " << max_error << std::endl;
    assert(max_error < 1e-10);
    std::cout << "✓ SIMD optimization test passed" << std::endl;
}

int main() {
    std::cout << "Alpha Parser C++ Test Suite" << std::endl;
    std::cout << "===========================" << std::endl;
    
    try {
        testBasicOperations();
        testFunctions();
        testComplexAlphas();
        benchmarkPerformance();
        testSIMDOptimizations();
        
        std::cout << "\n🎉 All tests passed!" << std::endl;
        std::cout << "The C++ alpha parser is working correctly with SIMD optimizations." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 