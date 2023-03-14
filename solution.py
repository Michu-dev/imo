import numpy as np
import multiprocessing as mp
from functools import partial

def readInstance(path):
    with open(path) as f:
        lines = f.readlines()
    nodes = {}
    for i in range(6, len(lines) - 1):
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

def score_diff(length_matrix, cycle, edge_index, node):
    a, b = cycle[edge_index - 1], cycle[edge_index]
    return length_matrix[a, node] + length_matrix[node, b] - length_matrix[a, b]
    

def greedy_cycle_heuristic(length_matrix, starting_node):
    remaining_nodes = list(range(100))
    starting_node_2 = np.argmax(length_matrix[starting_node, :])
    remaining_nodes.remove(starting_node)
    remaining_nodes.remove(starting_node_2)
    cycles = {}
    cycles[0] = [starting_node]
    cycles[1] = [starting_node_2]
    while len(remaining_nodes) > 0:
        for cycle in cycles.keys():
            scores = np.array([[score_diff(length_matrix, cycles[cycle], i, n) for i in range(len(cycles[cycle]))] for n in remaining_nodes])
            node_to_insert_index, insert_index = np.unravel_index(np.argmin(scores), scores.shape)
            node_to_insert = remaining_nodes[node_to_insert_index]
            cycles[cycle].insert(insert_index, node_to_insert)
            remaining_nodes.remove(node_to_insert)

    return cycles



def find_best_solution(func, length_matrix):
    args = [(length_matrix, i) for i in range(100)]
    # print(args)
    solutions = mp.Pool().starmap(func, args)

    scores = [score(length_matrix, i) for i in solutions]

    return solutions[np.argmin(scores)]

if __name__ == '__main__':
    length_matrix_kroa100 = readInstance('./kroB100.tsp')





    best_solution = find_best_solution(greedy_cycle_heuristic, np.array(length_matrix_kroa100))


    print(best_solution)
    print(len(best_solution[0]), len(best_solution[1]))
    # print(length_matrix_kroa100[0][0])
    # print(length_matrix_kroa100[0][1])