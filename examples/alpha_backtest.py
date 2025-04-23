from alpha_parser.alpha_lexer import AlphaLexer
from alpha_parser.alpha_parser import Parser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
import seaborn as sns
from datetime import datetime, timedelta

class AlphaBacktester:
    def __init__(self, lookback_window: int = 20):
        self.lookback_window = lookback_window
        
    def generate_sample_data(self, size: int = 100) -> pd.DataFrame:
        """
        샘플 데이터를 생성합니다.
        
        Args:
            size: 생성할 데이터 포인트의 수
            
        Returns:
            pd.DataFrame: 생성된 샘플 데이터
        """
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=size, freq='D')
        
        # 기본 가격 데이터 생성
        base_price = 100
        returns = np.random.normal(0, 0.02, size)
        close = pd.Series(base_price * (1 + returns).cumprod())
        
        # 고가, 저가, 시가 생성
        high = pd.Series(close * (1 + np.random.uniform(0, 0.02, size)))
        low = pd.Series(close * (1 - np.random.uniform(0, 0.02, size)))
        open_price = pd.Series(close.shift(1) * (1 + np.random.normal(0, 0.01, size)))
        open_price.iloc[0] = base_price
        
        data = {
            'date': dates,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': pd.Series(np.abs(np.random.normal(1000000, 200000, size))),
            'returns': pd.Series(returns),
            'vwap': pd.Series((high + low + close) / 3),
            'adv20': pd.Series(np.abs(np.random.normal(5000000, 1000000, size))),
            'cap': pd.Series(np.abs(np.random.normal(1000000000, 200000000, size))),
            'industry': pd.Series(np.random.choice(['tech', 'finance', 'health', 'energy'], size=size)),
            'sector': pd.Series(np.random.choice(['A', 'B', 'C', 'D'], size=size)),
            'subindustry': pd.Series(np.random.choice(['X', 'Y', 'Z', 'W'], size=size))
        }
        
        return pd.DataFrame(data)
    
    def parse_and_evaluate(self, expression: str, data: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
        """
        알파 공식을 파싱하고 평가합니다.
        
        Args:
            expression: 평가할 알파 공식
            data: 평가에 사용할 데이터
            
        Returns:
            Tuple[bool, pd.DataFrame]: 성공 여부와 결과 데이터프레임
        """
        print(f"\n{'='*50}")
        print(f"Input expression: {expression}")
        print(f"Data shape: {data.shape}")
        
        try:
            parser = Parser(AlphaLexer(expression))
            ast = parser.parse()
            print(f"AST structure: {ast}")
            
            results = []
            positions = []
            
            for i in range(len(data)):
                if i < self.lookback_window:
                    results.append(0)
                    positions.append(0)
                    continue
                    
                # 현재 시점까지의 데이터를 사용
                current_data = {
                    col: data[col].iloc[:i+1].tolist() 
                    for col in data.columns 
                    if col != 'date'
                }
                
                try:
                    result = ast.evaluate(current_data)
                    value = result[-1] if isinstance(result, list) else result
                    results.append(value)
                    
                    # 포지션 결정 (양수면 롱, 음수면 숏, 0이면 중립)
                    position = 1 if value > 0 else -1 if value < 0 else 0
                    positions.append(position)
                except Exception as e:
                    print(f"Error at index {i}: {str(e)}")
                    results.append(0)
                    positions.append(0)
            
            # 결과를 데이터프레임으로 변환
            result_df = pd.DataFrame({
                'date': data['date'],
                'alpha_value': results,
                'position': positions,
                'close': data['close']
            })
            
            # 수익률 계산
            result_df['returns'] = result_df['close'].pct_change()
            result_df['pnl'] = result_df['position'].shift(1) * result_df['returns']
            result_df['cumulative_pnl'] = result_df['pnl'].cumsum()
            
            # 성과 지표 계산
            total_return = result_df['cumulative_pnl'].iloc[-1]
            sharpe_ratio = result_df['pnl'].mean() / result_df['pnl'].std() * np.sqrt(252)
            max_drawdown = (result_df['cumulative_pnl'].cummax() - result_df['cumulative_pnl']).max()
            
            print("\nPerformance Metrics:")
            print(f"Total Return: {total_return:.2%}")
            print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"Max Drawdown: {max_drawdown:.2%}")
            
            return True, result_df
            
        except Exception as e:
            print(f"Error: {str(e)}")
            return False, pd.DataFrame()
        finally:
            print(f"{'='*50}")
    
    def plot_results(self, result_df: pd.DataFrame, title: str):
        """
        백테스트 결과를 시각화합니다.
        
        Args:
            result_df: 백테스트 결과 데이터프레임
            title: 그래프 제목
        """
        plt.figure(figsize=(15, 10))
        
        # 1. 누적 수익률
        plt.subplot(2, 2, 1)
        plt.plot(result_df['date'], result_df['cumulative_pnl'])
        plt.title('Cumulative PnL')
        plt.grid(True)
        
        # 2. 알파 값
        plt.subplot(2, 2, 2)
        plt.plot(result_df['date'], result_df['alpha_value'])
        plt.title('Alpha Value')
        plt.grid(True)
        
        # 3. 포지션
        plt.subplot(2, 2, 3)
        plt.bar(result_df['date'], result_df['position'])
        plt.title('Position')
        plt.grid(True)
        
        # 4. 일일 수익률 분포
        plt.subplot(2, 2, 4)
        sns.histplot(result_df['pnl'].dropna(), kde=True)
        plt.title('Daily PnL Distribution')
        plt.grid(True)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def test_alpha(self, formula: str, data: pd.DataFrame) -> bool:
        """
        알파 공식을 테스트합니다.
        
        Args:
            formula: 테스트할 알파 공식
            data: 테스트에 사용할 데이터
            
        Returns:
            bool: 테스트 성공 여부
        """
        print(f"\nTesting formula: {formula}")
        success, result_df = self.parse_and_evaluate(formula, data)
        
        if success:
            self.plot_results(result_df, f"Backtest Results: {formula}")
            return True
        return False

def main():
    # 백테스터 인스턴스 생성
    backtester = AlphaBacktester(lookback_window=20)
    
    # 샘플 데이터 생성
    data = backtester.generate_sample_data(size=100)
    
    # 테스트할 알파 공식들
    alpha_formulas = {
        'Alpha#1': "(rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20): close), 2.), 5)) - 0.5)",
        'Alpha#101': "((close - open) / ((high - low) + .001))"
    }
    
    # 각 알파 공식 테스트
    for name, formula in alpha_formulas.items():
        print(f"\n{'='*50}")
        print(f"Testing: {name}")
        backtester.test_alpha(formula, data)

if __name__ == "__main__":
    main() 