import sys
import os
import pandas as pd
import numpy as np
from alpha_parser import AlphaParser

# test_alphas.py에서 필요한 함수와 변수를 직접 가져옵니다
def generate_sample_data(size=100):
    np.random.seed(42)
    data = {
        'open': np.random.randn(size) * 10 + 100,
        'high': np.random.randn(size) * 10 + 102,
        'low': np.random.randn(size) * 10 + 98,
        'close': np.random.randn(size) * 10 + 101,
        'volume': np.abs(np.random.randn(size) * 1000000 + 5000000),
        'returns': np.random.randn(size) * 0.02,
        'vwap': np.random.randn(size) * 10 + 100.5,
        'adv20': np.abs(np.random.randn(size) * 800000 + 4000000),
        'cap': np.abs(np.random.randn(size) * 1000000000 + 5000000000),
        'industry': np.random.choice(['tech', 'finance', 'health', 'energy'], size=size),
        'sector': np.random.choice(['A', 'B', 'C', 'D'], size=size)
    }
    return data

# 알파 공식 테스트 함수
def test_alpha_formula(parser, formula, variables):
    try:
        result = parser.parse(formula)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    # 샘플 데이터 생성
    print("\n=== 생성된 샘플 데이터 ===")
    data = generate_sample_data(size=10)  # 10개 샘플만 생성
    df = pd.DataFrame(data)
    
    # 데이터프레임 출력 형식 설정
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_rows', 10)
    pd.set_option('display.precision', 2)
    
    # 각 열의 형식 지정
    for col in df.columns:
        if col in ['industry', 'sector']:
            continue
        if col in ['volume', 'adv20', 'cap']:
            df[col] = df[col].map(lambda x: f'{x:,.0f}')
        else:
            df[col] = df[col].map(lambda x: f'{x:.2f}')
    
    print(df)
    
    # 변수를 리스트로 변환하여 출력
    print("\n=== 리스트 형식의 데이터 ===")
    variables = {k: list(v) for k, v in data.items()}
    for k, v in variables.items():
        if k in ['industry', 'sector']:
            continue
        print(f"{k}: {[float(str(x).replace(',', '')) for x in v[:3]]}...")
    
    # 알파 공식 테스트
    print("\n=== 알파 공식 테스트 ===")
    parser = AlphaParser()
    parser.variables = variables
    
    # 테스트할 알파 공식들
    alpha_formulas = {
        'Alpha#1': '-1 * correlation(rank(delta(log(volume), 1)), rank(((close - open) / open)), 6)',
        'Alpha#8': '-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10)))',
        'Alpha#21': '(((-1 * rank((delta(returns, 3)))) * correlation(open, volume, 10)) + (-1 * delta(close, 3)))'
    }
    
    for name, formula in alpha_formulas.items():
        print(f"\n{name}:")
        print(f"공식: {formula}")
        result = test_alpha_formula(parser, formula, variables)
        if isinstance(result, list):
            print(f"결과 (처음 3개 값): {result[:3]}...")
        else:
            print(f"결과: {result}")

if __name__ == '__main__':
    main() 