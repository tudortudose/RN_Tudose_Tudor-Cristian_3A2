from script_common import *


def calculate_determinant(matrix):
    det = 0
    det += matrix[0][0] * matrix[1][1] * matrix[2][2]
    det += matrix[0][1] * matrix[1][2] * matrix[2][0]
    det += matrix[1][0] * matrix[2][1] * matrix[0][2]
    det -= matrix[0][2] * matrix[1][1] * matrix[2][0]
    det -= matrix[0][0] * matrix[1][2] * matrix[2][1]
    det -= matrix[0][1] * matrix[1][0] * matrix[2][2]
    return det


def calculate_transpose_matrix(matrix):
    tran_matrix = [[] for i in range(len(matrix))]
    for i in range(len(matrix)):
        for line in matrix:
            tran_matrix[i].append(line[i])
    return tran_matrix


def calculate_adjunct_matrix(matrix):
    adj_matrix = [[0 for i in range(3)] for j in range(3)]
    tran_matrix = calculate_transpose_matrix(matrix)

    adj_matrix[0][0] = tran_matrix[1][1] * tran_matrix[2][2] - tran_matrix[1][2] * tran_matrix[2][1]
    adj_matrix[0][1] = -(tran_matrix[1][0] * tran_matrix[2][2] - tran_matrix[2][0] * tran_matrix[1][2])
    adj_matrix[0][2] = tran_matrix[1][0] * tran_matrix[2][1] - tran_matrix[2][0] * tran_matrix[1][1]

    adj_matrix[1][0] = -(tran_matrix[0][1] * tran_matrix[2][2] - tran_matrix[2][1] * tran_matrix[0][2])
    adj_matrix[1][1] = tran_matrix[0][0] * tran_matrix[2][2] - tran_matrix[2][0] * tran_matrix[0][2]
    adj_matrix[1][2] = -(tran_matrix[0][0] * tran_matrix[2][1] - tran_matrix[2][0] * tran_matrix[0][1])

    adj_matrix[2][0] = tran_matrix[0][1] * tran_matrix[1][2] - tran_matrix[1][1] * tran_matrix[0][2]
    adj_matrix[2][1] = -(tran_matrix[0][0] * tran_matrix[1][2] - tran_matrix[1][0] * tran_matrix[0][2])
    adj_matrix[2][2] = tran_matrix[0][0] * tran_matrix[1][1] - tran_matrix[1][0] * tran_matrix[0][1]

    return adj_matrix


def calculate_inverted_matrix(matrix):
    adj_matrix = calculate_adjunct_matrix(matrix)
    matrix_det = calculate_determinant(matrix)
    for i, line in enumerate(adj_matrix):
        for j, el in enumerate(line):
            adj_matrix[i][j] = el / matrix_det
    return adj_matrix


def multiply_matrices(matrix_a, matrix_b):
    result_matrix = []
    for line in matrix_a:
        elem = 0
        for i in range(3):
            elem += line[i] * matrix_b[i]
        result_matrix.append(elem)
    return result_matrix


def solve_equation_no_lib(data):
    matrix_a, matrix_b = data
    determinant_a = calculate_determinant(matrix_a)
    if determinant_a == 0:
        raise Exception("The determinant is 0!!! Cannot solve the equation")
    inverted_matrix_a = calculate_inverted_matrix(matrix_a)
    solution = multiply_matrices(inverted_matrix_a, matrix_b)
    print_solution(solution)
