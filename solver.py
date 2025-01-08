from sympy import *

def solve_param(distance: float = 50.0, rate: float = 100.0, bandwidth: float = 20, ):
    param_B= Symbol('B')
    param_alpha = Symbol('alpha')
    param_R = Symbol('R')

    param_rate = param_B * log(1 + param_alpha / param_R ** 2, 2)

    expr = [
        Eq(param_rate, rate),
        Eq(param_B, bandwidth),
        Eq(param_R, distance)
    ]

    result = solve(expr, [param_alpha, param_R, param_B])

    return result[0][0]

if __name__ == "__main__":
    print(solve_param(50, 100))