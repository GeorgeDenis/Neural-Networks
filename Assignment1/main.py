import re
import math


def parse_coeff(group):
    if group is None:
        return 0
    group = group.replace("x", "").replace("y", "").replace("z", "")
    if group == "" or group == "+":
        return 1
    elif group == "-":
        return -1
    else:
        return int(group)


def parse_equations(file_path):
    A = []
    B = []

    pattern = r"([+-]?\d*x)?\s*([+-]?\d*y)?\s*([+-]?\d*z)?\s*=\s*([+-]?\d+)"

    with open(file_path, "r") as f:
        for line in f:
            match = re.match(pattern, line.replace(" ", ""))
            if match:
                a = parse_coeff(match.group(1))
                b = parse_coeff(match.group(2))
                c = parse_coeff(match.group(3))
                d = int(match.group(4))

                A.append([a, b, c])
                B.append(d)

    return A, B


def determinant(matrix):
    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def trace(matrix):
    return matrix[0][0] + matrix[1][1] + matrix[2][2]


def vector_norm(vector):
    return math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)


def transpose(matrix):
    transpose_matrix = []
    first_column = []
    for i in range(3):
        first_column.append(matrix[i][0])
    transpose_matrix.append(first_column)
    first_column = []
    for i in range(3):
        first_column.append(matrix[i][1])
    transpose_matrix.append(first_column)
    first_column = []
    for i in range(3):
        first_column.append(matrix[i][2])
    transpose_matrix.append(first_column)
    return transpose_matrix


def multiply_matrix_vector(matrix, vector):
    dot = []
    for i in range(len(matrix)):
        sum = 0
        for j in range(len(matrix[i])):
            sum += matrix[i][j] * vector[j]
        dot.append(sum)
    return dot


def multiply_matrix_scalar(matrix, scalar):
    new_matrix = [row[:] for row in matrix]
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            new_matrix[i][j] = matrix[i][j] * scalar
    return new_matrix


def replace_column(matrix, vector, column_index):
    new_matrix = [row[:] for row in matrix]
    for i in range(len(matrix)):
        new_matrix[i][column_index] = vector[i]
    return new_matrix


def cramer_rule(matrix, vector):
    det_A = determinant(matrix)
    det_Ax = determinant(replace_column(matrix, vector, 0))
    x = det_Ax / det_A
    det_Ay = determinant(replace_column(matrix, vector, 1))
    y = det_Ay / det_A
    det_Az = determinant(replace_column(matrix, vector, 2))
    z = det_Az / det_A
    return x, y, z


def calculate_det_minor(matrix, row, col):
    new_matrix = []
    coef = (-1) ** (row + col)
    for i in range(len(matrix)):
        line = []
        for j in range(len(matrix[i])):
            if i != row and j != col:
                line.append(matrix[i][j])
        if len(line) > 0:
            new_matrix.append(line)
    det = new_matrix[0][0] * new_matrix[1][1] - new_matrix[1][0] * new_matrix[0][1]
    return coef * det


def calculate_cofactor_matrix(matrix):
    adjugate_matrix = []
    for i in range(len(matrix)):
        line = []
        for j in range(len(matrix[i])):
            line.append(calculate_det_minor(matrix, i, j))
        adjugate_matrix.append(line)
    return adjugate_matrix


def inversion(matrix, B):
    cofactor_matrix = calculate_cofactor_matrix(matrix)
    adjugate_matrix = transpose(cofactor_matrix)
    matrix_determinant = determinant(matrix)
    matrix_inversion = multiply_matrix_scalar(adjugate_matrix, 1 / matrix_determinant)
    solution = multiply_matrix_vector(matrix_inversion, B)
    return solution


if __name__ == "__main__":
    file_path = "input.txt"
    A, B = parse_equations(file_path)
    print("Matrix A (Coefficients):")
    for row in A:
        print(row)

    print("\nVector B (Constants):")
    print(B)

    print(f"Determinant = {determinant(A)}")
    print(f"Trace = {trace(A)}")
    print(f"Norm = {vector_norm(B)}")
    print(f"Transpose = ")
    transpose_matrix = transpose(A)
    for row in transpose_matrix:
        print(row)
    print(f"Multiplication = ")
    dot = multiply_matrix_vector(A, B)
    for row in dot:
        print(row)
    print("Cramer Rule")
    print(cramer_rule(A, B))
    print("Adjugate Matrix")
    print(inversion(A, B))
