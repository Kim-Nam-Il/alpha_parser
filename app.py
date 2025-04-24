from flask import Flask, render_template, request, jsonify
from test_data_generator import generate_test_data
from alpha_parser import AlphaParser
import numpy as np

app = Flask(__name__)

# 알파 수식 목록
ALPHA_FORMULAS = {
    'Alpha#2': '-1 * correlation(rank(delta(log(volume), 2)), rank((close - open) / open), 6)',
    'Alpha#3': '-1 * correlation(rank(open), rank(volume), 10)',
    'Alpha#4': '-1 * rank(low)',
    'Alpha#5': 'rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap))))',
    'Alpha#6': '-1 * correlation(open, volume, 10)',
    'Alpha#7': 'if(adv(volume,20) > volume, -1, 1)'
}

def calculate_signals(formula, data):
    """알파 공식을 파싱하고 신호를 계산합니다."""
    parser = AlphaParser()
    parser.variables = data
    signals = parser.parse(formula)
    # NaN 값을 0으로 대체
    signals = [0 if x is None or np.isnan(x) else x for x in signals]
    return signals

def calculate_pnl(signals, prices):
    """신호를 기반으로 PnL을 계산합니다."""
    # 신호와 가격 데이터의 길이를 맞춥니다
    min_len = min(len(signals), len(prices)-1)
    signals = signals[:min_len]
    prices = prices[:min_len+1]
    
    # 수익률 계산
    returns = np.diff(prices) / prices[:-1]
    
    # PnL 계산
    pnl = np.array(signals) * returns
    cum_pnl = np.cumsum(pnl)
    
    return cum_pnl.tolist()

@app.route('/')
def index():
    return render_template('index.html', formulas=ALPHA_FORMULAS)

@app.route('/simulate', methods=['POST'])
def simulate():
    formula = request.json['formula']
    data = generate_test_data()
    
    try:
        signals = calculate_signals(formula, data)
        pnl = calculate_pnl(signals, data['close'])
        # 날짜 데이터 생성 (인덱스 사용)
        dates = list(range(len(pnl)))
        return jsonify({
            'dates': dates,
            'pnl': pnl
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 