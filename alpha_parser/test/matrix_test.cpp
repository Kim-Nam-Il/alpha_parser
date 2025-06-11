#include "alpha_parser.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>

using namespace alpha_parser;

class MarketDataGenerator {
private:
    std::mt19937 gen;
    std::normal_distribution<double> price_dis;
    std::normal_distribution<double> volume_dis;
    std::uniform_real_distribution<double> uniform_dis;

public:
    MarketDataGenerator(unsigned seed = 42) 
        : gen(seed), price_dis(0.0, 0.02), volume_dis(0.0, 0.3), uniform_dis(0.0, 1.0) {}

    // N개 주식, M일자의 시장 데이터 생성
    DataMap generateMarketMatrix(int N_stocks, int M_days) {
        std::cout << "📊 Generating " << N_stocks << " stocks × " << M_days << " days market data..." << std::endl;
        
        DataMap market_data;
        
        // 각 주식별로 데이터 생성
        VectorXd all_open(N_stocks * M_days);
        VectorXd all_high(N_stocks * M_days);
        VectorXd all_low(N_stocks * M_days);
        VectorXd all_close(N_stocks * M_days);
        VectorXd all_volume(N_stocks * M_days);
        VectorXd all_returns(N_stocks * M_days);
        VectorXd stock_id(N_stocks * M_days);
        VectorXd shares_outstanding(N_stocks * M_days);
        
        for (int stock = 0; stock < N_stocks; ++stock) {
            double base_price = 50.0 + uniform_dis(gen) * 150.0; // 50~200 기준가
            double shares = 1000000.0 + uniform_dis(gen) * 9000000.0; // 1M~10M 주식수
            
            for (int day = 0; day < M_days; ++day) {
                int idx = stock * M_days + day;
                
                // 주가 변동률 생성 (일일 변동률)
                double daily_return = price_dis(gen);
                if (day > 0) {
                    base_price *= (1.0 + daily_return);
                }
                
                // OHLC 데이터 생성
                double open_price = base_price * (1.0 + price_dis(gen) * 0.3);
                double close_price = base_price * (1.0 + daily_return);
                double high_price = std::max(open_price, close_price) * (1.0 + uniform_dis(gen) * 0.02);
                double low_price = std::min(open_price, close_price) * (1.0 - uniform_dis(gen) * 0.02);
                
                // 거래량 생성 (로그정규분포)
                double volume = std::exp(15.0 + volume_dis(gen)) / 1000.0; // 평균 약 3.3M
                
                all_open(idx) = open_price;
                all_high(idx) = high_price;
                all_low(idx) = low_price;
                all_close(idx) = close_price;
                all_volume(idx) = volume;
                all_returns(idx) = (day > 0) ? daily_return : 0.0;
                stock_id(idx) = stock;
                shares_outstanding(idx) = shares;
            }
        }
        
        market_data["open"] = all_open;
        market_data["high"] = all_high;
        market_data["low"] = all_low;
        market_data["close"] = all_close;
        market_data["volume"] = all_volume;
        market_data["returns"] = all_returns;
        market_data["stock_id"] = stock_id;
        market_data["shares_outstanding"] = shares_outstanding;
        
        // 파생 지표 계산
        VectorXd vwap = (all_high + all_low + all_close) / 3.0;
        market_data["vwap"] = vwap;
        
        // 업종/섹터 분류 (임의로 5개 업종, 3개 섹터)
        VectorXd industry(N_stocks * M_days);
        VectorXd sector(N_stocks * M_days);
        for (int i = 0; i < N_stocks * M_days; ++i) {
            industry(i) = static_cast<int>(stock_id(i)) % 5;
            sector(i) = static_cast<int>(stock_id(i)) % 3;
        }
        market_data["industry"] = industry;
        market_data["sector"] = sector;
        
        std::cout << "✅ Market data generated successfully!" << std::endl;
        return market_data;
    }
    
    // 데이터 요약 정보 출력
    void printDataSummary(const DataMap& data, int N_stocks, int M_days) {
        std::cout << "\n📈 Data Summary:" << std::endl;
        std::cout << "─────────────────" << std::endl;
        std::cout << "Stocks: " << N_stocks << std::endl;
        std::cout << "Days: " << M_days << std::endl;
        std::cout << "Total data points: " << N_stocks * M_days << std::endl;
        
        // 가격 통계
        const auto& close = data.at("close");
        const auto& volume = data.at("volume");
        const auto& returns = data.at("returns");
        
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Close price range: $" << close.minCoeff() << " - $" << close.maxCoeff() << std::endl;
        std::cout << "Average volume: " << volume.mean() / 1000000.0 << "M shares" << std::endl;
        std::cout << "Returns std dev: " << std::sqrt((returns.array() - returns.mean()).square().mean()) * 100 << "%" << std::endl;
    }
};

class AlphaStrategy {
private:
    AlphaParser parser;
    
public:
    AlphaStrategy() {
        // 추가 커스텀 함수 등록 가능
    }
    
    // 새로 구현된 함수들 테스트
    void testNewFunctions(const DataMap& data, int N_stocks, int M_days) {
        std::cout << "\n🆕 Testing New Functions:" << std::endl;
        std::cout << "═══════════════════════════" << std::endl;
        
        parser.set_variables(data);
        
        std::vector<std::pair<std::string, std::string>> new_formulas = {
            // ADV 함수들
            {"ADV20", "adv20(volume)"},
            {"ADV60", "adv60(volume)"},
            {"ADV120", "adv120(volume)"},
            
            // 새로 추가된 수학 함수들
            {"TS_ArgMax", "ts_argmax(close, 10)"},
            {"TS_ArgMin", "ts_argmin(close, 10)"},
            {"Sum", "sum(returns, 20)"},
            {"Product", "product(close / delay(close, 1), 5)"},
            {"Min", "min(open, close)"},
            {"Max", "max(high, low)"},
            {"Covariance", "covariance(close, volume, 20)"},
            {"SignedPower", "signedpower(returns, 2)"},
            
            // 시장 데이터 함수들
            {"Market Cap", "cap(close, shares_outstanding)"},
            {"Industry", "industry"},
            {"IndClass", "indclass(industry, 1)"},
            
            // 복합 알파 공식 (새 함수들 활용)
            {"Complex Alpha #1", "rank(ts_argmax(signedpower(returns, 2), 10))"},
            {"Complex Alpha #2", "correlation(adv20(volume), cap(close, shares_outstanding), 30)"},
            {"Complex Alpha #3", "indneutralize(rank(close - vwap), industry)"},
            {"Complex Alpha #4", "sum(min(returns, 0), 20) / sum(max(returns, 0), 20)"}
        };
        
        for (const auto& [name, formula] : new_formulas) {
            try {
                auto start = std::chrono::high_resolution_clock::now();
                
                VectorXd result = parser.parse_and_evaluate(formula);
                
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                
                // 결과 통계
                double mean_val = result.mean();
                double std_val = std::sqrt((result.array() - mean_val).square().mean());
                double min_val = result.minCoeff();
                double max_val = result.maxCoeff();
                
                // NaN 개수 확인
                int nan_count = 0;
                for (int i = 0; i < result.size(); ++i) {
                    if (std::isnan(result(i))) nan_count++;
                }
                
                std::cout << std::fixed << std::setprecision(4);
                std::cout << "📊 " << name << ":" << std::endl;
                std::cout << "   Formula: " << formula << std::endl;
                std::cout << "   Mean: " << mean_val << ", Std: " << std_val << std::endl;
                std::cout << "   Range: [" << min_val << ", " << max_val << "]" << std::endl;
                std::cout << "   NaN count: " << nan_count << "/" << result.size() << std::endl;
                std::cout << "   Computation time: " << duration.count() << " μs" << std::endl;
                std::cout << "   Performance: " << (N_stocks * M_days * 1000000.0) / duration.count() 
                         << " data points/second" << std::endl;
                std::cout << std::endl;
                
            } catch (const std::exception& e) {
                std::cout << "❌ Error in " << name << ": " << e.what() << std::endl;
                std::cout << std::endl;
            }
        }
    }
    
    // 기존 알파 공식 테스트
    void testAlphaFormulas(const DataMap& data, int N_stocks, int M_days) {
        std::cout << "\n🧮 Testing Alpha Formulas:" << std::endl;
        std::cout << "═══════════════════════════" << std::endl;
        
        parser.set_variables(data);
        
        std::vector<std::pair<std::string, std::string>> alpha_formulas = {
            {"Simple Momentum", "close - delay(close, 5)"},
            {"Price Rank", "rank(close)"},
            {"Volume-Price", "correlation(close, volume, 10)"},
            {"Mean Reversion", "rank(close) - rank(delay(close, 10))"},
            {"Volatility Signal", "stddev(returns, 20)"},
            {"High-Low Range", "rank(high - low)"},
            {"VWAP Signal", "rank(close - vwap)"},
            {"Complex Alpha", "(-1 * correlation(rank(open), rank(volume), 10))"}
        };
        
        for (const auto& [name, formula] : alpha_formulas) {
            try {
                auto start = std::chrono::high_resolution_clock::now();
                
                VectorXd result = parser.parse_and_evaluate(formula);
                
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                
                // 결과 통계
                double mean_val = result.mean();
                double std_val = std::sqrt((result.array() - mean_val).square().mean());
                double min_val = result.minCoeff();
                double max_val = result.maxCoeff();
                
                std::cout << std::fixed << std::setprecision(4);
                std::cout << "📊 " << name << ":" << std::endl;
                std::cout << "   Formula: " << formula << std::endl;
                std::cout << "   Mean: " << mean_val << ", Std: " << std_val << std::endl;
                std::cout << "   Range: [" << min_val << ", " << max_val << "]" << std::endl;
                std::cout << "   Computation time: " << duration.count() << " μs" << std::endl;
                std::cout << "   Performance: " << (N_stocks * M_days * 1000000.0) / duration.count() 
                         << " data points/second" << std::endl;
                std::cout << std::endl;
                
            } catch (const std::exception& e) {
                std::cout << "❌ Error in " << name << ": " << e.what() << std::endl;
            }
        }
    }
    
    // 성능 벤치마크
    void benchmarkPerformance(const DataMap& data, int N_stocks, int M_days) {
        std::cout << "\n⚡ Performance Benchmark:" << std::endl;
        std::cout << "═══════════════════════════" << std::endl;
        
        parser.set_variables(data);
        
        std::vector<std::string> benchmark_formulas = {
            "rank(close)",
            "correlation(close, volume, 20)",
            "stddev(returns, 30)",
            "ts_max(close, 10)",
            "rank(close - delay(close, 5))",
            "adv20(volume)",
            "signedpower(returns, 2)",
            "ts_argmax(close, 20)"
        };
        
        const int iterations = 50;
        
        for (const auto& formula : benchmark_formulas) {
            std::vector<double> times;
            
            // 워밍업
            for (int i = 0; i < 3; ++i) {
                try {
                    parser.parse_and_evaluate(formula);
                } catch (...) {
                    continue;
                }
            }
            
            // 실제 측정
            for (int i = 0; i < iterations; ++i) {
                try {
                    auto start = std::chrono::high_resolution_clock::now();
                    auto result = parser.parse_and_evaluate(formula);
                    auto end = std::chrono::high_resolution_clock::now();
                    
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                    times.push_back(duration.count());
                } catch (...) {
                    continue;
                }
            }
            
            if (times.empty()) {
                std::cout << "❌ " << formula << ": Failed to execute" << std::endl;
                continue;
            }
            
            // 통계 계산
            double mean_time = 0;
            for (double t : times) mean_time += t;
            mean_time /= times.size();
            
            double min_time = *std::min_element(times.begin(), times.end());
            double max_time = *std::max_element(times.begin(), times.end());
            
            std::cout << "🔥 " << formula << ":" << std::endl;
            std::cout << "   Average: " << std::fixed << std::setprecision(1) << mean_time << " μs" << std::endl;
            std::cout << "   Range: " << min_time << " - " << max_time << " μs" << std::endl;
            std::cout << "   Throughput: " << std::fixed << std::setprecision(0) 
                     << (N_stocks * M_days * 1000000.0) / mean_time << " ops/sec" << std::endl;
            std::cout << std::endl;
        }
    }
};

int main() {
    std::cout << "🚀 Alpha Parser Matrix Test - Enhanced Functions" << std::endl;
    std::cout << "════════════════════════════════════════════════" << std::endl;
    
    // 매트릭스 크기 설정
    const int N_STOCKS = 100;   // 100개 주식
    const int M_DAYS = 500;     // 2년 거래일 (약 500일)
    
    std::cout << "Matrix dimensions: " << N_STOCKS << " stocks × " << M_DAYS << " days" << std::endl;
    std::cout << "Total data points: " << N_STOCKS * M_DAYS << std::endl;
    
    try {
        // 1. 시장 데이터 생성
        MarketDataGenerator generator;
        DataMap market_data = generator.generateMarketMatrix(N_STOCKS, M_DAYS);
        generator.printDataSummary(market_data, N_STOCKS, M_DAYS);
        
        // 2. 새로 구현된 함수들 테스트
        AlphaStrategy strategy;
        strategy.testNewFunctions(market_data, N_STOCKS, M_DAYS);
        
        // 3. 기존 알파 전략 테스트
        strategy.testAlphaFormulas(market_data, N_STOCKS, M_DAYS);
        
        // 4. 성능 벤치마크
        strategy.benchmarkPerformance(market_data, N_STOCKS, M_DAYS);
        
        std::cout << "\n✅ All tests completed successfully!" << std::endl;
        std::cout << "\n📋 Summary of Implemented Functions:" << std::endl;
        std::cout << "   • ADV functions: adv20, adv30, adv40, adv50, adv60, adv81, adv120, adv150, adv180" << std::endl;
        std::cout << "   • Time series: ts_argmax, ts_argmin" << std::endl;
        std::cout << "   • Mathematical: sum, product, min, max, covariance, signedpower" << std::endl;
        std::cout << "   • Market data: cap, industry, indclass, sector" << std::endl;
        std::cout << "   • Data processing: indneutralize" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 