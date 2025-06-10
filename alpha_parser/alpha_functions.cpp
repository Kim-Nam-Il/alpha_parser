#include "alpha_parser.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

// Architecture-specific SIMD headers
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
    #include <immintrin.h>
    #define USE_AVX2
#elif defined(__aarch64__) || defined(_M_ARM64)
    #include <arm_neon.h>
    #define USE_NEON
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace alpha_parser {

// SIMD optimized element-wise operations
void AlphaParser::add_simd(const double* a, const double* b, double* result, int size) {
#ifdef USE_AVX2
    int avx_size = size - (size % 4);
    
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < avx_size; i += 4) {
        __m256d va = _mm256_load_pd(&a[i]);
        __m256d vb = _mm256_load_pd(&b[i]);
        __m256d vr = _mm256_add_pd(va, vb);
        _mm256_store_pd(&result[i], vr);
    }
    
    // Handle remaining elements
    for (int i = avx_size; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
#elif defined(USE_NEON)
    int neon_size = size - (size % 2);
    
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < neon_size; i += 2) {
        float64x2_t va = vld1q_f64(&a[i]);
        float64x2_t vb = vld1q_f64(&b[i]);
        float64x2_t vr = vaddq_f64(va, vb);
        vst1q_f64(&result[i], vr);
    }
    
    // Handle remaining elements
    for (int i = neon_size; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
#else
    // Fallback to standard implementation
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
#endif
}

void AlphaParser::sub_simd(const double* a, const double* b, double* result, int size) {
#ifdef USE_AVX2
    int avx_size = size - (size % 4);
    
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < avx_size; i += 4) {
        __m256d va = _mm256_load_pd(&a[i]);
        __m256d vb = _mm256_load_pd(&b[i]);
        __m256d vr = _mm256_sub_pd(va, vb);
        _mm256_store_pd(&result[i], vr);
    }
    
    for (int i = avx_size; i < size; ++i) {
        result[i] = a[i] - b[i];
    }
#elif defined(USE_NEON)
    int neon_size = size - (size % 2);
    
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < neon_size; i += 2) {
        float64x2_t va = vld1q_f64(&a[i]);
        float64x2_t vb = vld1q_f64(&b[i]);
        float64x2_t vr = vsubq_f64(va, vb);
        vst1q_f64(&result[i], vr);
    }
    
    for (int i = neon_size; i < size; ++i) {
        result[i] = a[i] - b[i];
    }
#else
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < size; ++i) {
        result[i] = a[i] - b[i];
    }
#endif
}

void AlphaParser::mul_simd(const double* a, const double* b, double* result, int size) {
#ifdef USE_AVX2
    int avx_size = size - (size % 4);
    
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < avx_size; i += 4) {
        __m256d va = _mm256_load_pd(&a[i]);
        __m256d vb = _mm256_load_pd(&b[i]);
        __m256d vr = _mm256_mul_pd(va, vb);
        _mm256_store_pd(&result[i], vr);
    }
    
    for (int i = avx_size; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
#elif defined(USE_NEON)
    int neon_size = size - (size % 2);
    
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < neon_size; i += 2) {
        float64x2_t va = vld1q_f64(&a[i]);
        float64x2_t vb = vld1q_f64(&b[i]);
        float64x2_t vr = vmulq_f64(va, vb);
        vst1q_f64(&result[i], vr);
    }
    
    for (int i = neon_size; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
#else
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
#endif
}

void AlphaParser::div_simd(const double* a, const double* b, double* result, int size) {
#ifdef USE_AVX2
    int avx_size = size - (size % 4);
    
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < avx_size; i += 4) {
        __m256d va = _mm256_load_pd(&a[i]);
        __m256d vb = _mm256_load_pd(&b[i]);
        __m256d vr = _mm256_div_pd(va, vb);
        _mm256_store_pd(&result[i], vr);
    }
    
    for (int i = avx_size; i < size; ++i) {
        result[i] = (b[i] != 0.0) ? a[i] / b[i] : std::numeric_limits<double>::quiet_NaN();
    }
#elif defined(USE_NEON)
    int neon_size = size - (size % 2);
    
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < neon_size; i += 2) {
        float64x2_t va = vld1q_f64(&a[i]);
        float64x2_t vb = vld1q_f64(&b[i]);
        float64x2_t vr = vdivq_f64(va, vb);
        vst1q_f64(&result[i], vr);
    }
    
    for (int i = neon_size; i < size; ++i) {
        result[i] = (b[i] != 0.0) ? a[i] / b[i] : std::numeric_limits<double>::quiet_NaN();
    }
#else
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < size; ++i) {
        result[i] = (b[i] != 0.0) ? a[i] / b[i] : std::numeric_limits<double>::quiet_NaN();
    }
#endif
}

// High-performance rank function
VectorXd AlphaParser::rank_simd(const VectorXd& x) {
    int n = x.size();
    std::vector<std::pair<double, int>> indexed_values(n);
    
    // Create indexed values
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < n; ++i) {
        indexed_values[i] = {x(i), i};
    }
    
    // Sort by value
    std::sort(indexed_values.begin(), indexed_values.end());
    
    VectorXd ranks(n);
    
    // Calculate normalized ranks
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < n; ++i) {
        int idx = indexed_values[i].second;
        ranks(idx) = (n > 1) ? static_cast<double>(i) / (n - 1) : 0.0;
    }
    
    return ranks;
}

// Delay function with SIMD optimization
VectorXd AlphaParser::delay_simd(const VectorXd& data, int periods) {
    int n = data.size();
    if (periods >= n) {
        return VectorXd::Zero(n);
    }
    
    VectorXd result(n);
    
    // Fill with zeros for delay period
    std::fill_n(result.data(), periods, 0.0);
    
    // Copy delayed data using SIMD
    int copy_size = n - periods;
    
#ifdef USE_AVX2
    int avx_size = copy_size - (copy_size % 4);
    
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < avx_size; i += 4) {
        __m256d vdata = _mm256_loadu_pd(&data.data()[i]);
        _mm256_storeu_pd(&result.data()[i + periods], vdata);
    }
    
    // Handle remaining elements
    for (int i = avx_size; i < copy_size; ++i) {
        result(i + periods) = data(i);
    }
#elif defined(USE_NEON)
    int neon_size = copy_size - (copy_size % 2);
    
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < neon_size; i += 2) {
        float64x2_t vdata = vld1q_f64(&data.data()[i]);
        vst1q_f64(&result.data()[i + periods], vdata);
    }
    
    for (int i = neon_size; i < copy_size; ++i) {
        result(i + periods) = data(i);
    }
#else
    for (int i = 0; i < copy_size; ++i) {
        result(i + periods) = data(i);
    }
#endif
    
    return result;
}

// Delta function
VectorXd AlphaParser::delta_simd(const VectorXd& data, int periods) {
    int n = data.size();
    VectorXd result(n);
    
    // First 'periods' elements are set to 0
    for (int i = 0; i < periods && i < n; ++i) {
        result(i) = 0.0;
    }
    
    // Calculate differences for remaining elements
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = periods; i < n; ++i) {
        result(i) = data(i) - data(i - periods);
    }
    
    return result;
}

// Time series maximum with sliding window
VectorXd AlphaParser::ts_max_simd(const VectorXd& data, int window) {
    int n = data.size();
    VectorXd result(n);
    
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < n; ++i) {
        int start = std::max(0, i - window + 1);
        double max_val = data(start);
        
        for (int j = start + 1; j <= i; ++j) {
            max_val = std::max(max_val, data(j));
        }
        
        result(i) = max_val;
    }
    
    return result;
}

// Time series minimum with sliding window
VectorXd AlphaParser::ts_min_simd(const VectorXd& data, int window) {
    int n = data.size();
    VectorXd result(n);
    
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < n; ++i) {
        int start = std::max(0, i - window + 1);
        double min_val = data(start);
        
        for (int j = start + 1; j <= i; ++j) {
            min_val = std::min(min_val, data(j));
        }
        
        result(i) = min_val;
    }
    
    return result;
}

// Time series mean with sliding window
VectorXd AlphaParser::ts_mean_simd(const VectorXd& data, int window) {
    int n = data.size();
    VectorXd result(n);
    
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < n; ++i) {
        int start = std::max(0, i - window + 1);
        int window_size = i - start + 1;
        
        double sum = 0.0;
        for (int j = start; j <= i; ++j) {
            sum += data(j);
        }
        
        result(i) = sum / window_size;
    }
    
    return result;
}

// Time series rank with sliding window
VectorXd AlphaParser::ts_rank_simd(const VectorXd& data, int window) {
    int n = data.size();
    VectorXd result(n);
    
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < n; ++i) {
        int start = std::max(0, i - window + 1);
        int window_size = i - start + 1;
        
        if (window_size == 1) {
            result(i) = 0.0;
            continue;
        }
        
        // Create window data
        std::vector<double> window_data(window_size);
        for (int j = 0; j < window_size; ++j) {
            window_data[j] = data(start + j);
        }
        
        // Sort and find rank
        std::sort(window_data.begin(), window_data.end());
        
        // Find position of current value
        double current_val = data(i);
        auto it = std::lower_bound(window_data.begin(), window_data.end(), current_val);
        int rank = std::distance(window_data.begin(), it);
        
        result(i) = static_cast<double>(rank) / (window_size - 1);
    }
    
    return result;
}

// Standard deviation with sliding window
VectorXd AlphaParser::stddev_simd(const VectorXd& data, int window) {
    int n = data.size();
    VectorXd result(n);
    
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < n; ++i) {
        int start = std::max(0, i - window + 1);
        int window_size = i - start + 1;
        
        if (window_size < 2) {
            result(i) = 0.0;
            continue;
        }
        
        // Calculate mean
        double sum = 0.0;
        for (int j = start; j <= i; ++j) {
            sum += data(j);
        }
        double mean = sum / window_size;
        
        // Calculate variance
        double variance = 0.0;
        for (int j = start; j <= i; ++j) {
            double diff = data(j) - mean;
            variance += diff * diff;
        }
        variance /= (window_size - 1);
        
        result(i) = std::sqrt(variance);
    }
    
    return result;
}

// Scale function (standardization)
VectorXd AlphaParser::scale_simd(const VectorXd& data) {
    double mean = data.mean();
    double std_dev = std::sqrt((data.array() - mean).square().mean());
    
    if (std_dev == 0.0) {
        return VectorXd::Zero(data.size());
    }
    
    return (data.array() - mean) / std_dev;
}

// Linear decay function
VectorXd AlphaParser::decay_linear_simd(const VectorXd& data, int window) {
    int n = data.size();
    VectorXd result(n);
    
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < n; ++i) {
        int start = std::max(0, i - window + 1);
        int window_size = i - start + 1;
        
        double weighted_sum = 0.0;
        double weight_sum = 0.0;
        
        for (int j = 0; j < window_size; ++j) {
            double weight = j + 1;  // Linear weights
            weighted_sum += data(start + j) * weight;
            weight_sum += weight;
        }
        
        result(i) = (weight_sum > 0.0) ? weighted_sum / weight_sum : 0.0;
    }
    
    return result;
}

// Industry neutralization
VectorXd AlphaParser::indneutralize_simd(const VectorXd& data, const VectorXd& groups) {
    int n = data.size();
    VectorXd result = data;
    
    // Find unique groups
    std::vector<double> unique_groups;
    for (int i = 0; i < groups.size(); ++i) {
        if (std::find(unique_groups.begin(), unique_groups.end(), groups(i)) == unique_groups.end()) {
            unique_groups.push_back(groups(i));
        }
    }
    
    // Neutralize by group
    for (double group : unique_groups) {
        std::vector<int> group_indices;
        for (int i = 0; i < n; ++i) {
            if (groups(i) == group) {
                group_indices.push_back(i);
            }
        }
        
        if (group_indices.size() > 1) {
            // Calculate group mean
            double group_sum = 0.0;
            for (int idx : group_indices) {
                group_sum += data(idx);
            }
            double group_mean = group_sum / group_indices.size();
            
            // Subtract group mean
            for (int idx : group_indices) {
                result(idx) = data(idx) - group_mean;
            }
        }
    }
    
    return result;
}

// Correlation with sliding window
VectorXd AlphaParser::correlation_simd(const VectorXd& x, const VectorXd& y, int window) {
    int n = std::min(x.size(), y.size());
    VectorXd result(n);
    
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < n; ++i) {
        int start = std::max(0, i - window + 1);
        int window_size = i - start + 1;
        
        if (window_size < 2) {
            result(i) = 0.0;
            continue;
        }
        
        // Calculate means
        double sum_x = 0.0, sum_y = 0.0;
        for (int j = start; j <= i; ++j) {
            sum_x += x(j);
            sum_y += y(j);
        }
        double mean_x = sum_x / window_size;
        double mean_y = sum_y / window_size;
        
        // Calculate correlation components
        double numerator = 0.0;
        double sum_sq_x = 0.0, sum_sq_y = 0.0;
        
        for (int j = start; j <= i; ++j) {
            double dx = x(j) - mean_x;
            double dy = y(j) - mean_y;
            numerator += dx * dy;
            sum_sq_x += dx * dx;
            sum_sq_y += dy * dy;
        }
        
        double denominator = std::sqrt(sum_sq_x * sum_sq_y);
        result(i) = (denominator > 0.0) ? numerator / denominator : 0.0;
    }
    
    return result;
}

// AlphaParser implementation
AlphaParser::AlphaParser() {
    init_builtin_functions();
}

void AlphaParser::set_variables(const DataMap& vars) {
    variables = vars;
}

VectorXd AlphaParser::parse_and_evaluate(const std::string& formula) {
    AlphaLexer lexer(formula);
    Parser parser(lexer);
    auto ast = parser.parse();
    return ast->evaluate(variables);
}

void AlphaParser::register_function(const std::string& name, FunctionType func) {
    function_registry[name] = func;
}

void AlphaParser::init_builtin_functions() {
    // Register additional built-in functions here if needed
    // Most functions are handled directly in ASTNode::evaluateFunctionCall
}

// Utility functions implementation
namespace utils {

template<typename T>
void prefetch_data(const T* data, int size) {
#ifdef USE_AVX2
    for (int i = 0; i < size; i += 64 / sizeof(T)) {
        _mm_prefetch(reinterpret_cast<const char*>(&data[i]), _MM_HINT_T0);
    }
#elif defined(USE_NEON)
    for (int i = 0; i < size; i += 64 / sizeof(T)) {
        __builtin_prefetch(&data[i], 0, 3);
    }
#else
    // No prefetch for other architectures
    (void)data;
    (void)size;
#endif
}

double mean_simd(const double* data, int size) {
    double sum = 0.0;
    
#ifdef USE_AVX2
    int avx_size = size - (size % 4);
    
    __m256d sum_vec = _mm256_setzero_pd();
    for (int i = 0; i < avx_size; i += 4) {
        __m256d vals = _mm256_loadu_pd(&data[i]);
        sum_vec = _mm256_add_pd(sum_vec, vals);
    }
    
    // Extract sum from vector
    double sum_array[4];
    _mm256_storeu_pd(sum_array, sum_vec);
    sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
    
    // Handle remaining elements
    for (int i = avx_size; i < size; ++i) {
        sum += data[i];
    }
#elif defined(USE_NEON)
    int neon_size = size - (size % 2);
    
    float64x2_t sum_vec = vdupq_n_f64(0.0);
    for (int i = 0; i < neon_size; i += 2) {
        float64x2_t vals = vld1q_f64(&data[i]);
        sum_vec = vaddq_f64(sum_vec, vals);
    }
    
    // Extract sum from vector
    sum = vgetq_lane_f64(sum_vec, 0) + vgetq_lane_f64(sum_vec, 1);
    
    // Handle remaining elements
    for (int i = neon_size; i < size; ++i) {
        sum += data[i];
    }
#else
    // Fallback implementation
    for (int i = 0; i < size; ++i) {
        sum += data[i];
    }
#endif
    
    return sum / size;
}

double variance_simd(const double* data, int size) {
    double mean = mean_simd(data, size);
    double sum_sq_diff = 0.0;
    
#ifdef USE_AVX2
    int avx_size = size - (size % 4);
    
    __m256d mean_vec = _mm256_set1_pd(mean);
    __m256d sum_vec = _mm256_setzero_pd();
    
    for (int i = 0; i < avx_size; i += 4) {
        __m256d vals = _mm256_loadu_pd(&data[i]);
        __m256d diff = _mm256_sub_pd(vals, mean_vec);
        __m256d sq_diff = _mm256_mul_pd(diff, diff);
        sum_vec = _mm256_add_pd(sum_vec, sq_diff);
    }
    
    // Extract sum from vector
    double sum_array[4];
    _mm256_storeu_pd(sum_array, sum_vec);
    sum_sq_diff = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
    
    // Handle remaining elements
    for (int i = avx_size; i < size; ++i) {
        double diff = data[i] - mean;
        sum_sq_diff += diff * diff;
    }
#elif defined(USE_NEON)
    int neon_size = size - (size % 2);
    
    float64x2_t mean_vec = vdupq_n_f64(mean);
    float64x2_t sum_vec = vdupq_n_f64(0.0);
    
    for (int i = 0; i < neon_size; i += 2) {
        float64x2_t vals = vld1q_f64(&data[i]);
        float64x2_t diff = vsubq_f64(vals, mean_vec);
        float64x2_t sq_diff = vmulq_f64(diff, diff);
        sum_vec = vaddq_f64(sum_vec, sq_diff);
    }
    
    // Extract sum from vector
    sum_sq_diff = vgetq_lane_f64(sum_vec, 0) + vgetq_lane_f64(sum_vec, 1);
    
    // Handle remaining elements
    for (int i = neon_size; i < size; ++i) {
        double diff = data[i] - mean;
        sum_sq_diff += diff * diff;
    }
#else
    // Fallback implementation
    for (int i = 0; i < size; ++i) {
        double diff = data[i] - mean;
        sum_sq_diff += diff * diff;
    }
#endif
    
    return sum_sq_diff / (size - 1);
}

double sum_simd(const double* data, int size) {
    double sum = 0.0;
    
#ifdef USE_AVX2
    int avx_size = size - (size % 4);
    
    __m256d sum_vec = _mm256_setzero_pd();
    for (int i = 0; i < avx_size; i += 4) {
        __m256d vals = _mm256_loadu_pd(&data[i]);
        sum_vec = _mm256_add_pd(sum_vec, vals);
    }
    
    // Extract sum from vector
    double sum_array[4];
    _mm256_storeu_pd(sum_array, sum_vec);
    sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
    
    // Handle remaining elements
    for (int i = avx_size; i < size; ++i) {
        sum += data[i];
    }
#elif defined(USE_NEON)
    int neon_size = size - (size % 2);
    
    float64x2_t sum_vec = vdupq_n_f64(0.0);
    for (int i = 0; i < neon_size; i += 2) {
        float64x2_t vals = vld1q_f64(&data[i]);
        sum_vec = vaddq_f64(sum_vec, vals);
    }
    
    // Extract sum from vector
    sum = vgetq_lane_f64(sum_vec, 0) + vgetq_lane_f64(sum_vec, 1);
    
    // Handle remaining elements
    for (int i = neon_size; i < size; ++i) {
        sum += data[i];
    }
#else
    // Fallback implementation
    for (int i = 0; i < size; ++i) {
        sum += data[i];
    }
#endif
    
    return sum;
}

template<typename T>
T* aligned_alloc(size_t count, size_t alignment) {
    return static_cast<T*>(std::aligned_alloc(alignment, count * sizeof(T)));
}

template<typename T>
void aligned_free(T* ptr) {
    std::free(ptr);
}

// Explicit template instantiations
template void prefetch_data<double>(const double* data, int size);
template double* aligned_alloc<double>(size_t count, size_t alignment);
template void aligned_free<double>(double* ptr);

} // namespace utils

// 새로 추가되는 함수들 구현

// Time series argument of maximum (index of maximum value in window)
VectorXd AlphaParser::ts_argmax_simd(const VectorXd& data, int window) {
    int n = data.size();
    VectorXd result(n);
    
    for (int i = 0; i < n; ++i) {
        int start = std::max(0, i - window + 1);
        int window_size = i - start + 1;
        
        if (window_size <= 0) {
            result(i) = 0;
            continue;
        }
        
        double max_val = data(start);
        int max_idx = 0;
        
        for (int j = 1; j < window_size; ++j) {
            if (data(start + j) > max_val) {
                max_val = data(start + j);
                max_idx = j;
            }
        }
        
        result(i) = max_idx;  // 윈도우 내에서의 상대적 인덱스
    }
    
    return result;
}

// Time series argument of minimum (index of minimum value in window)
VectorXd AlphaParser::ts_argmin_simd(const VectorXd& data, int window) {
    int n = data.size();
    VectorXd result(n);
    
    for (int i = 0; i < n; ++i) {
        int start = std::max(0, i - window + 1);
        int window_size = i - start + 1;
        
        if (window_size <= 0) {
            result(i) = 0;
            continue;
        }
        
        double min_val = data(start);
        int min_idx = 0;
        
        for (int j = 1; j < window_size; ++j) {
            if (data(start + j) < min_val) {
                min_val = data(start + j);
                min_idx = j;
            }
        }
        
        result(i) = min_idx;  // 윈도우 내에서의 상대적 인덱스
    }
    
    return result;
}

// Rolling sum with sliding window
VectorXd AlphaParser::sum_simd(const VectorXd& data, int window) {
    int n = data.size();
    VectorXd result(n);
    
    for (int i = 0; i < n; ++i) {
        int start = std::max(0, i - window + 1);
        int window_size = i - start + 1;
        
        double sum = 0.0;
        for (int j = start; j <= i; ++j) {
            sum += data(j);
        }
        
        result(i) = sum;
    }
    
    return result;
}

// Rolling product with sliding window
VectorXd AlphaParser::product_simd(const VectorXd& data, int window) {
    int n = data.size();
    VectorXd result(n);
    
    for (int i = 0; i < n; ++i) {
        int start = std::max(0, i - window + 1);
        int window_size = i - start + 1;
        
        double product = 1.0;
        for (int j = start; j <= i; ++j) {
            product *= data(j);
        }
        
        result(i) = product;
    }
    
    return result;
}

// Element-wise minimum of two vectors
VectorXd AlphaParser::min_simd(const VectorXd& a, const VectorXd& b) {
    int n = std::max(a.size(), b.size());
    VectorXd result(n);
    
    for (int i = 0; i < n; ++i) {
        double val_a = (i < a.size()) ? a(i) : (a.size() > 0 ? a(0) : 0.0);
        double val_b = (i < b.size()) ? b(i) : (b.size() > 0 ? b(0) : 0.0);
        result(i) = std::min(val_a, val_b);
    }
    
    return result;
}

// Element-wise maximum of two vectors
VectorXd AlphaParser::max_simd(const VectorXd& a, const VectorXd& b) {
    int n = std::max(a.size(), b.size());
    VectorXd result(n);
    
    for (int i = 0; i < n; ++i) {
        double val_a = (i < a.size()) ? a(i) : (a.size() > 0 ? a(0) : 0.0);
        double val_b = (i < b.size()) ? b(i) : (b.size() > 0 ? b(0) : 0.0);
        result(i) = std::max(val_a, val_b);
    }
    
    return result;
}

// Rolling covariance with sliding window
VectorXd AlphaParser::covariance_simd(const VectorXd& x, const VectorXd& y, int window) {
    int n = std::min(x.size(), y.size());
    VectorXd result(n);
    
    for (int i = 0; i < n; ++i) {
        int start = std::max(0, i - window + 1);
        int window_size = i - start + 1;
        
        if (window_size < 2) {
            result(i) = 0.0;
            continue;
        }
        
        // Calculate means
        double sum_x = 0.0, sum_y = 0.0;
        for (int j = start; j <= i; ++j) {
            sum_x += x(j);
            sum_y += y(j);
        }
        double mean_x = sum_x / window_size;
        double mean_y = sum_y / window_size;
        
        // Calculate covariance
        double covar = 0.0;
        for (int j = start; j <= i; ++j) {
            covar += (x(j) - mean_x) * (y(j) - mean_y);
        }
        
        result(i) = covar / (window_size - 1);
    }
    
    return result;
}

// Signed power function: sign(base) * (|base|^exponent)
VectorXd AlphaParser::signedpower_simd(const VectorXd& base, const VectorXd& exponent) {
    int n = std::max(base.size(), exponent.size());
    VectorXd result(n);
    
    for (int i = 0; i < n; ++i) {
        double b = (i < base.size()) ? base(i) : (base.size() > 0 ? base(0) : 0.0);
        double e = (i < exponent.size()) ? exponent(i) : (exponent.size() > 0 ? exponent(0) : 0.0);
        
        if (std::abs(b) < 1e-15) {
            result(i) = 0.0;
        } else {
            double sign_b = (b > 0) ? 1.0 : -1.0;
            double abs_b = std::abs(b);
            result(i) = sign_b * std::pow(abs_b, e);
        }
    }
    
    return result;
}

// Natural logarithm
VectorXd AlphaParser::log_simd(const VectorXd& data) {
    int n = data.size();
    VectorXd result(n);
    
    for (int i = 0; i < n; ++i) {
        if (data(i) > 0) {
            result(i) = std::log(data(i));
        } else {
            result(i) = std::numeric_limits<double>::quiet_NaN();
        }
    }
    
    return result;
}

// Absolute value
VectorXd AlphaParser::abs_simd(const VectorXd& data) {
    int n = data.size();
    VectorXd result(n);
    
    for (int i = 0; i < n; ++i) {
        result(i) = std::abs(data(i));
    }
    
    return result;
}

// Sign function
VectorXd AlphaParser::sign_simd(const VectorXd& data) {
    int n = data.size();
    VectorXd result(n);
    
    for (int i = 0; i < n; ++i) {
        if (data(i) > 0) {
            result(i) = 1.0;
        } else if (data(i) < 0) {
            result(i) = -1.0;
        } else {
            result(i) = 0.0;
        }
    }
    
    return result;
}

// ADV functions (Average Daily Volume)
VectorXd AlphaParser::adv_simd(const VectorXd& volume, int window) {
    int n = volume.size();
    VectorXd result(n);
    
    for (int i = 0; i < n; ++i) {
        if (i < window - 1) {
            result(i) = std::numeric_limits<double>::quiet_NaN();
        } else {
            double sum = 0.0;
            for (int j = i - window + 1; j <= i; ++j) {
                sum += volume(j);
            }
            result(i) = sum / window;
        }
    }
    
    return result;
}

// Market cap proxy (using close price * shares_outstanding)
VectorXd AlphaParser::cap_simd(const VectorXd& close, const VectorXd& shares_outstanding) {
    int n = close.size();
    VectorXd result(n);
    
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < n; ++i) {
        double shares = (i < shares_outstanding.size()) ? shares_outstanding(i) : 1000000.0; // 기본값
        result(i) = close(i) * shares;
    }
    
    return result;
}

// Industry classification function
VectorXd AlphaParser::indclass_simd(const VectorXd& industry_codes, int target_industry) {
    int n = industry_codes.size();
    VectorXd result(n);
    
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < n; ++i) {
        result(i) = (static_cast<int>(industry_codes(i)) == target_industry) ? 1.0 : 0.0;
    }
    
    return result;
}

// Sector classification function  
VectorXd AlphaParser::sector_simd(const VectorXd& sector_codes, int target_sector) {
    int n = sector_codes.size();
    VectorXd result(n);
    
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < n; ++i) {
        result(i) = (static_cast<int>(sector_codes(i)) == target_sector) ? 1.0 : 0.0;
    }
    
    return result;
}

} // namespace alpha_parser 