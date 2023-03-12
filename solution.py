import numpy as np
import multiprocessing as mp
from functools import partial

def readInstance(path):
    with open(path) as f:
        lines = f.readlines()
    nodes = {}
    for i in range(6, 106):
        splitted_line = lines[i].strip().split(' ')
        n_node, x, y = int(splitted_line[0]), int(splitted_line[1]), int(splitted_line[2])
        nodes[n_node] = (x, y)

    length_matrix = []
    n_nodes = 100

    for i in range(1, n_nodes + 1):
        length_matrix.append([])
        for j in range(1, n_nodes + 1):
            length_matrix[i - 1].append(round(np.sqrt((nodes[i][0] - nodes[j][0])**2 + (nodes[i][1] - nodes[j][1])**2)))
    
    return length_matrix

def closest_neighbour_heuristic(length_matrix, starting_node):
    # length_matrix, starting_node = args
    print(starting_node)
    remaining_nodes = list(range(100))
    starting_node_2 = np.argmax(length_matrix[starting_node, :])
    remaining_nodes.remove(starting_node)
    remaining_nodes.remove(starting_node_2)
    cycles = {}
    cycles[0] = [starting_node]
    cycles[1] = [starting_node_2]
    while len(remaining_nodes) > 0:
        for cycle in cycles.keys():
            best_node = remaining_nodes[np.argmin(length_matrix[cycles[cycle][-1], remaining_nodes])]
            cycles[cycle].append(best_node)
            remaining_nodes.remove(best_node)

    return cycles

def score(length_matrix, cycles):
    score = 0
    for cycle in cycles.keys():
        # dodanie pierwszego wierzchołka na koniec cyklu
        cycles[cycle] += [cycles[cycle][0]]
        for i in range(len(cycles[cycle]) - 1):
            score += length_matrix[cycles[cycle][i], cycles[cycle][i + 1]]
        # print(length_matrix[cycles[cycle][i], cycles[cycle][i + 1]] for i in range(len(cycles[cycle] - 1)))
        # print(sum(length_matrix[cycles[cycle][i], cycles[cycle][i + 1]] for i in range(len(cycles[cycle] - 1))))
        # score += sum(length_matrix[cycles[cycle][i], cycles[cycle][i + 1]] for i in range(len(cycles[cycle] - 1)))
    return score



if __name__ == '__main__':
    length_matrix_kroa100 = readInstance('./kroB100.tsp')

    length_matrix_kroa100 = np.array(length_matrix_kroa100)
    args = [(length_matrix_kroa100, i) for i in range(100)]
    solutions = mp.Pool().starmap(closest_neighbour_heuristic, args)
    # solutions = []
    # for i in range(100):
    #     solutions.append(closest_neighbour_heuristic(length_matrix_kroa100, i))

    scores = [score(length_matrix_kroa100, i) for i in solutions]
    best_solution = solutions[np.argmin(scores)]


    print(best_solution)
    print(len(best_solution[0]), len(best_solution[1]))
    # print(length_matrix_kroa100[0][0])
    # print(length_matrix_kroa100[0][1])