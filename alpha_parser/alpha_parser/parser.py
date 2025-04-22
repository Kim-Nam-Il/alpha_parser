import numpy as np
from .tokens import TokenType

class ASTNode:
    def __init__(self, type, value=None, children=None):
        self.type = type
        self.value = value
        self.children = children or []

    def _rank(self, x):
        """Rank the elements in x"""
        return np.argsort(np.argsort(x))

    def _delay(self, x, n):
        """Delay x by n periods"""
        return np.roll(x, n)

    def _correlation(self, x, y, n):
        """Calculate correlation between x and y over n periods"""
        return np.correlate(x, y, mode='valid') / n

    def _covariance(self, x, y, n):
        """Calculate covariance between x and y over n periods"""
        return np.cov(x, y)[0,1]

    def _scale(self, x):
        """Scale x to sum to 1"""
        return x / np.sum(x)

    def _delta(self, x, n):
        """Calculate difference between current and n periods ago"""
        return x - np.roll(x, n)

    def _signedpower(self, x, power):
        """Calculate signed power of x"""
        return np.sign(x) * np.power(np.abs(x), power)

    def _decay_linear(self, x, n):
        """Calculate linear decay of x over n periods"""
        weights = np.linspace(1, 0, n)
        return np.convolve(x, weights, mode='valid')

    def _sum(self, x, n):
        """Calculate sum of x over n periods"""
        return np.convolve(x, np.ones(n), mode='valid')

    def _product(self, x, n):
        """Calculate product of x over n periods"""
        return np.prod(x[-n:])

    def _stddev(self, x, n):
        """Calculate standard deviation of x over n periods"""
        return np.std(x[-n:])

    def _ts_rank(self, x, n):
        """Calculate time series rank of x over n periods"""
        return np.argsort(np.argsort(x[-n:]))[-1]

    def _ts_min(self, x, n):
        """Calculate minimum of x over n periods"""
        return np.min(x[-n:])

    def _ts_max(self, x, n):
        """Calculate maximum of x over n periods"""
        return np.max(x[-n:])

    def _ts_argmax(self, x, n):
        """Calculate index of maximum of x over n periods"""
        return np.argmax(x[-n:])

    def _ts_argmin(self, x, n):
        """Calculate index of minimum of x over n periods"""
        return np.argmin(x[-n:])

    def evaluate(self, variables):
        if self.type == 'NUMBER':
            return self.value
        elif self.type == 'VARIABLE':
            if self.value not in variables:
                raise ValueError(f"Variable {self.value} not found")
            return variables[self.value]
        elif self.type == 'BINARY_OP':
            left = self.children[0].evaluate(variables)
            right = self.children[1].evaluate(variables)
            if self.value == TokenType.PLUS:
                return left + right
            elif self.value == TokenType.MINUS:
                return left - right
            elif self.value == TokenType.MULTIPLY:
                return left * right
            elif self.value == TokenType.DIVIDE:
                return left / right
            elif self.value == TokenType.LESS_THAN:
                return left < right
            elif self.value == TokenType.GREATER_THAN:
                return left > right
            elif self.value == TokenType.LESS_EQUAL:
                return left <= right
            elif self.value == TokenType.GREATER_EQUAL:
                return left >= right
            elif self.value == TokenType.EQUAL:
                return left == right
            elif self.value == TokenType.NOT_EQUAL:
                return left != right
            elif self.value == TokenType.AND:
                return left and right
            elif self.value == TokenType.OR:
                return left or right
        elif self.type == 'UNARY_OP':
            value = self.children[0].evaluate(variables)
            if self.value == TokenType.MINUS:
                return -value
            elif self.value == TokenType.NOT:
                return not value
        elif self.type == 'FUNCTION_CALL':
            func_name = self.value.lower()
            args = [child.evaluate(variables) for child in self.children]
            
            if func_name == 'rank':
                return self._rank(args[0])
            elif func_name == 'delay':
                return self._delay(args[0], int(args[1]))
            elif func_name == 'correlation':
                return self._correlation(args[0], args[1], int(args[2]))
            elif func_name == 'covariance':
                return self._covariance(args[0], args[1], int(args[2]))
            elif func_name == 'scale':
                return self._scale(args[0])
            elif func_name == 'delta':
                return self._delta(args[0], int(args[1]))
            elif func_name == 'signedpower':
                return self._signedpower(args[0], args[1])
            elif func_name == 'decay_linear':
                return self._decay_linear(args[0], int(args[1]))
            elif func_name == 'sum':
                return self._sum(args[0], int(args[1]))
            elif func_name == 'product':
                return self._product(args[0], int(args[1]))
            elif func_name == 'stddev':
                return self._stddev(args[0], int(args[1]))
            elif func_name == 'ts_rank':
                return self._ts_rank(args[0], int(args[1]))
            elif func_name == 'ts_min':
                return self._ts_min(args[0], int(args[1]))
            elif func_name == 'ts_max':
                return self._ts_max(args[0], int(args[1]))
            elif func_name == 'ts_argmax':
                return self._ts_argmax(args[0], int(args[1]))
            elif func_name == 'ts_argmin':
                return self._ts_argmin(args[0], int(args[1]))
            elif func_name == 'abs':
                return np.abs(args[0])
            else:
                raise ValueError(f"Unknown function: {func_name}")
        elif self.type == 'CONDITIONAL':
            condition = self.children[0].evaluate(variables)
            if condition:
                return self.children[1].evaluate(variables)
            else:
                return self.children[2].evaluate(variables)
        else:
            raise ValueError(f"Unknown node type: {self.type}") 