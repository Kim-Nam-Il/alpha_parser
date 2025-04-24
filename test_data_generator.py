import numpy as np
import pandas as pd

def generate_test_data(size=100):
    """테스트 데이터를 생성합니다."""
    np.random.seed(42)
    
    # 기본 가격 생성
    base_price = 100
    returns = np.random.normal(0.001, 0.02, size)
    prices = base_price * np.cumprod(1 + returns)
    
    # 고가, 저가, 종가 생성
    high = prices * (1 + np.random.uniform(0, 0.02, size))
    low = prices * (1 - np.random.uniform(0, 0.02, size))
    close = prices
    
    # 시가 생성 (전일 종가 기반)
    open_price = np.roll(close, 1)
    open_price[0] = close[0] * (1 + np.random.normal(0, 0.01))
    
    # 거래량 생성
    volume = np.random.lognormal(10, 1, size) * 1000
    
    return {
        'open': open_price.tolist(),
        'high': high.tolist(),
        'low': low.tolist(),
        'close': close.tolist(),
        'volume': volume.tolist()
    } 