import re

def read_data(file):
    file = open(file, 'r')
    lines = file.readlines()
    matrix_a = []
    matrix_b = []
    for line in lines:
        str_coeffs = re.split('[xyz=\n]', line)
        num_coeffs = []

        for coeff in str_coeffs:
            if coeff == '' or coeff == '+':
                num_coeffs.append(1)
            elif coeff == '-':
                num_coeffs.append(-1)
            else:
                num_coeffs.append(int(coeff))

        matrix_a.append(num_coeffs[0:3])
        matrix_b.append(num_coeffs[4])

    return matrix_a, matrix_b


def print_solution(solution):
    x_var, y_var, z_var = solution
    print("x =", x_var)
    print("y =", y_var)
    print("z =", z_var)
