import numpy as np
from os.path import isfile
from tempfile import TemporaryFile

def load_data(size):
    #check if already loaded
    path_str_x = "data/array_x_"+str(size)+".npy"
    path_str_y = "data/array_y_"+str(size)+".npy"
    if isfile(path_str_x) and isfile(path_str_y):
        problems = np.load(path_str_x)
        solutions = np.load(path_str_y)
        return problems, solutions

    problems = np.zeros((size, 81), np.int32)
    solutions = np.zeros((size, 81), np.int32)
    for i, line in enumerate(open('data/sudoku.csv', 'r').read().splitlines()[1:]):
        quiz, solution = line.split(",")
        for j, q_s in enumerate(zip(quiz, solution)):
            q, s = q_s
            problems[i, j] = q
            solutions[i, j] = s
        if i == (size-1):
            break
    problems = problems.reshape((-1, 9, 9))
    solutions = solutions.reshape((-1, 9, 9))
    np.save(path_str_x, problems)
    np.save(path_str_y, solutions)
    return problems, solutions
