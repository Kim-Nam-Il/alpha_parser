import pytest
from alpha_parser.alpha_parser import AlphaParser

def test_basic_arithmetic():
    parser = AlphaParser()
    
    # 기본 산술 연산 테스트
    assert parser.parse("1 + 2") == 3
    assert parser.parse("3 - 1") == 2
    assert parser.parse("2 * 3") == 6
    assert parser.parse("6 / 2") == 3
    assert parser.parse("2 ** 3") == 8

def test_list_operations():
    parser = AlphaParser()
    
    # 리스트 연산 테스트
    assert parser.parse("[1, 2, 3] + [4, 5, 6]") == [5, 7, 9]
    assert parser.parse("[1, 2, 3] - [1, 1, 1]") == [0, 1, 2]
    assert parser.parse("[1, 2, 3] * 2") == [2, 4, 6]
    assert parser.parse("[4, 6, 8] / 2") == [2, 3, 4]

def test_unary_operations():
    parser = AlphaParser()
    
    # 단항 연산 테스트
    assert parser.parse("-1") == -1
    assert parser.parse("+1") == 1
    assert parser.parse("-[1, 2, 3]") == [-1, -2, -3]

def test_function_calls():
    parser = AlphaParser()
    
    # 함수 호출 테스트
    assert parser.parse("delay([1, 2, 3], 1)") == [None, 1, 2]
    assert parser.parse("ts_min([1, 2, 3], 2)") == [1, 1, 1]
    assert parser.parse("ts_max([1, 2, 3], 2)") == [1, 2, 3]

def test_error_handling():
    parser = AlphaParser()
    
    # 오류 처리 테스트
    with pytest.raises(ValueError):
        parser.parse("[1, 2] + [1, 2, 3]")  # 길이 불일치
    
    with pytest.raises(ValueError):
        parser.parse("1 / 0")  # 0으로 나누기
    
    with pytest.raises(ValueError):
        parser.parse("unknown_function(1)")  # 알 수 없는 함수 