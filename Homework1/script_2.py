from script_common import *
import numpy as np


def solve_equation_w_lib(data):
    matrix_a, matrix_b = data
    if np.linalg.det(matrix_a) == 0:
        raise Exception("The determinant is 0!!! Cannot solve the equation")
    solution = np.linalg.solve(np.array(matrix_a), np.array(matrix_b))
    print_solution(solution)

