#include "alpha_parser.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <map>
#include <iomanip>

using namespace alpha_parser;

class AlphaBenchmark {
private:
    AlphaParser parser;
    DataMap data;
    
    // Alpha formulas from the original test file
    std::map<std::string, std::string> alpha_formulas = {
        {"Alpha#1", "(rank(ts_argmax(signedpower(((returns < 0) ? stddev(returns, 20): close), 2.), 5)) - 0.5)"},
        {"Alpha#2", "(-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))"},
        {"Alpha#3", "(-1 * correlation(rank(open), rank(volume), 10))"},
        {"Alpha#4", "(-1 * ts_rank(rank(low), 9))"},
        {"Alpha#5", "(rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))"},
        {"Alpha#6", "(-1 * correlation(open, volume, 10))"},
        {"Alpha#12", "sign(delta(volume, 1)) * (-1 * delta(close, 1))"},
        {"Alpha#13", "(-1 * rank(covariance(rank(close), rank(volume), 5)))"},
        {"Alpha#16", "(-1 * rank(covariance(rank(high), rank(volume), 5)))"},
        {"Alpha#22", "(-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))"},
        {"Alpha#25", "rank(((((-1 * returns) * adv20) * vwap) * (high - close)))"},
        {"Alpha#28", "scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))"},
        {"Alpha#32", "scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5), 230)))"},
        {"Alpha#33", "rank((-1 * ((1 - (open / close))^1)))"},
        {"Alpha#41", "(((high * low)^0.5) - vwap)"},
        {"Alpha#42", "(rank((vwap - close)) / rank((vwap + close)))"},
        {"Alpha#44", "(-1 * correlation(high, rank(volume), 5))"},
        {"Alpha#101", "((close - open) / ((high - low) + .001))"}
    };
    
    // Simplified versions for testing
    std::map<std::string, std::string> simple_formulas = {
        {"Simple#1", "rank(close)"},
        {"Simple#2", "close - open"},
        {"Simple#3", "correlation(close, volume, 10)"},
        {"Simple#4", "ts_max(close, 20)"},
        {"Simple#5", "stddev(returns, 20)"},
        {"Simple#6", "scale(close)"},
        {"Simple#7", "delay(close, 5)"},
        {"Simple#8", "delta(close, 1)"},
        {"Simple#9", "rank(close) * rank(volume)"},
        {"Simple#10", "ts_rank(close, 10)"}
    };
    
public:
    AlphaBenchmark(int data_size = 10000) {
        data = generateSampleData(data_size);
        parser.set_variables(data);
    }
    
    DataMap generateSampleData(int size) {
        std::random_device rd;
        std::mt19937 gen(42); // Fixed seed for reproducibility
        std::normal_distribution<> dis(0.0, 1.0);
        std::uniform_real_distribution<> uniform(0.0, 1.0);
        
        DataMap data;
        
        // Generate comprehensive market data
        VectorXd open(size), high(size), low(size), close(size);
        VectorXd volume(size), vwap(size), returns(size);
        VectorXd adv20(size), cap(size), industry(size);
        
        double base_price = 100.0;
        for (int i = 0; i < size; ++i) {
            double change = dis(gen) * 0.02;
            base_price *= (1.0 + change);
            
            close(i) = base_price;
            open(i) = base_price * (1.0 + dis(gen) * 0.01);
            high(i) = std::max(open(i), close(i)) * (1.0 + uniform(gen) * 0.01);
            low(i) = std::min(open(i), close(i)) * (1.0 - uniform(gen) * 0.01);
            volume(i) = std::abs(dis(gen)) * 1000000 + 5000000;
            vwap(i) = (high(i) + low(i) + close(i)) / 3.0;
            returns(i) = (i > 0) ? (close(i) - close(i-1)) / close(i-1) : 0.0;
            cap(i) = base_price * (uniform(gen) * 1000000000 + 100000000);
            industry(i) = i % 10; // 10 industries
        }
        
        // Calculate ADV20
        for (int i = 0; i < size; ++i) {
            int start = std::max(0, i - 19);
            double sum = 0.0;
            for (int j = start; j <= i; ++j) {
                sum += volume(j);
            }
            adv20(i) = sum / (i - start + 1);
        }
        
        data["open"] = open;
        data["high"] = high;
        data["low"] = low;
        data["close"] = close;
        data["volume"] = volume;
        data["vwap"] = vwap;
        data["returns"] = returns;
        data["adv20"] = adv20;
        data["cap"] = cap;
        data["industry"] = industry;
        
        return data;
    }
    
    void benchmarkFormulas(const std::map<std::string, std::string>& formulas, 
                          const std::string& category, int iterations = 100) {
        std::cout << "\n" << category << " Benchmark Results:" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        std::cout << std::left << std::setw(15) << "Formula" 
                  << std::setw(15) << "Avg Time (μs)" 
                  << std::setw(15) << "Min Time (μs)"
                  << std::setw(15) << "Max Time (μs)" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        
        for (const auto& [name, formula] : formulas) {
            std::vector<double> times;
            bool success = true;
            
            try {
                // Warmup
                for (int i = 0; i < 5; ++i) {
                    parser.parse_and_evaluate(formula);
                }
                
                // Actual benchmark
                for (int i = 0; i < iterations; ++i) {
                    auto start = std::chrono::high_resolution_clock::now();
                    auto result = parser.parse_and_evaluate(formula);
                    auto end = std::chrono::high_resolution_clock::now();
                    
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                    times.push_back(duration.count());
                }
                
            } catch (const std::exception& e) {
                std::cout << std::left << std::setw(15) << name 
                          << "FAILED: " << e.what() << std::endl;
                success = false;
            }
            
            if (success && !times.empty()) {
                double avg = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
                double min_time = *std::min_element(times.begin(), times.end());
                double max_time = *std::max_element(times.begin(), times.end());
                
                std::cout << std::left << std::setw(15) << name 
                          << std::setw(15) << std::fixed << std::setprecision(2) << avg
                          << std::setw(15) << std::fixed << std::setprecision(2) << min_time
                          << std::setw(15) << std::fixed << std::setprecision(2) << max_time << std::endl;
            }
        }
    }
    
    void benchmarkSIMDFunctions() {
        std::cout << "\nSIMD Function Benchmark:" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        
        const int size = 50000;
        const int iterations = 1000;
        
        VectorXd test_data = VectorXd::Random(size) * 100;
        VectorXd test_data2 = VectorXd::Random(size) * 100;
        
        // Test rank function
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            auto result = AlphaParser::rank_simd(test_data);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto rank_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // Test correlation function
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            auto result = AlphaParser::correlation_simd(test_data, test_data2, 20);
        }
        end = std::chrono::high_resolution_clock::now();
        auto corr_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // Test ts_max function
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            auto result = AlphaParser::ts_max_simd(test_data, 20);
        }
        end = std::chrono::high_resolution_clock::now();
        auto tsmax_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // Test stddev function
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            auto result = AlphaParser::stddev_simd(test_data, 20);
        }
        end = std::chrono::high_resolution_clock::now();
        auto stddev_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Function performance (" << iterations << " iterations, " << size << " elements):" << std::endl;
        std::cout << "Rank:        " << rank_time.count() / 1000.0 << " ms total, " 
                  << rank_time.count() / (double)iterations << " μs avg" << std::endl;
        std::cout << "Correlation: " << corr_time.count() / 1000.0 << " ms total, " 
                  << corr_time.count() / (double)iterations << " μs avg" << std::endl;
        std::cout << "TS_Max:      " << tsmax_time.count() / 1000.0 << " ms total, " 
                  << tsmax_time.count() / (double)iterations << " μs avg" << std::endl;
        std::cout << "StdDev:      " << stddev_time.count() / 1000.0 << " ms total, " 
                  << stddev_time.count() / (double)iterations << " μs avg" << std::endl;
    }
    
    void scalabilityTest() {
        std::cout << "\nScalability Test:" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        std::vector<int> sizes = {1000, 5000, 10000, 25000, 50000};
        std::string test_formula = "rank(correlation(close, volume, 20))";
        
        std::cout << std::left << std::setw(15) << "Data Size" 
                  << std::setw(20) << "Time (μs)" 
                  << std::setw(25) << "Time per Element (ns)" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        
        for (int size : sizes) {
            DataMap test_data = generateSampleData(size);
            AlphaParser test_parser;
            test_parser.set_variables(test_data);
            
            // Warmup
            for (int i = 0; i < 3; ++i) {
                test_parser.parse_and_evaluate(test_formula);
            }
            
            // Measure
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < 10; ++i) {
                test_parser.parse_and_evaluate(test_formula);
            }
            auto end = std::chrono::high_resolution_clock::now();
            
            auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            double avg_time = total_time.count() / 10.0;
            double time_per_element = (avg_time * 1000.0) / size; // Convert to nanoseconds
            
            std::cout << std::left << std::setw(15) << size 
                      << std::setw(20) << std::fixed << std::setprecision(2) << avg_time
                      << std::setw(25) << std::fixed << std::setprecision(2) << time_per_element << std::endl;
        }
    }
    
    void run() {
        std::cout << "Alpha Parser C++ Benchmark Suite" << std::endl;
        std::cout << "=================================" << std::endl;
        std::cout << "Data size: " << data["close"].size() << " elements" << std::endl;
        
        // Benchmark simple formulas
        benchmarkFormulas(simple_formulas, "Simple Formulas", 200);
        
        // Benchmark alpha formulas (simplified versions)
        std::map<std::string, std::string> testable_alphas;
        for (const auto& [name, formula] : alpha_formulas) {
            // Skip formulas with unsupported functions for now
            if (formula.find("signedpower") == std::string::npos &&
                formula.find("ts_argmax") == std::string::npos &&
                formula.find("covariance") == std::string::npos &&
                formula.find("sum") == std::string::npos &&
                formula.find("log") == std::string::npos) {
                testable_alphas[name] = formula;
            }
        }
        
        if (!testable_alphas.empty()) {
            benchmarkFormulas(testable_alphas, "Alpha Formulas", 50);
        }
        
        // Benchmark SIMD functions
        benchmarkSIMDFunctions();
        
        // Scalability test
        scalabilityTest();
        
        std::cout << "\nBenchmark completed successfully!" << std::endl;
    }
};

int main() {
    try {
        AlphaBenchmark benchmark(10000);
        benchmark.run();
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 