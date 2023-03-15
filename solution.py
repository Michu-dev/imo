import numpy as np
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt
import matplotlib as mpl

def readInstance(path):
    with open(path) as f:
        lines = f.readlines()
    nodes = {}
    for i in range(6, len(lines) - 1):
        splitted_line = lines[i].strip().split(' ')
        n_node, x, y = int(splitted_line[0]), int(splitted_line[1]), int(splitted_line[2])
        nodes[n_node] = (x, y)

    points = []
    for v in nodes.values():
        points.append([v[0], v[1]])

    length_matrix = []
    n_nodes = 100

    for i in range(1, n_nodes + 1):
        length_matrix.append([])
        for j in range(1, n_nodes + 1):
            length_matrix[i - 1].append(round(np.sqrt((nodes[i][0] - nodes[j][0])**2 + (nodes[i][1] - nodes[j][1])**2)))
    
    return length_matrix, np.array(points)

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

def regret_cycle_heuristic(length_matrix, starting_node):
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
            node_to_insert_index = 0
            _, y = scores.shape
            if y == 1:
                node_to_insert_index = np.argmin(scores)
            else:
                regret = np.diff(np.partition(scores, 1)[:, :2]).reshape(-1)
                weight_regret = regret - 0.8 * np.min(scores, axis=1)
                node_to_insert_index = np.argmax(weight_regret)
            
            node_to_insert = remaining_nodes[node_to_insert_index]
            insert_index = np.argmin(scores[node_to_insert_index])
            cycles[cycle].insert(insert_index, node_to_insert)
            remaining_nodes.remove(node_to_insert)
    
    return cycles            



def find_best_solution(func, length_matrix):
    args = [(length_matrix, i) for i in range(100)]
    # print(args)
    solutions = mp.Pool().starmap(func, args)

    scores = [score(length_matrix, i) for i in solutions]

    return solutions[np.argmin(scores)], int(min(scores))

def draw_paths(points, cycles):
    cycle1 = cycles[0]
    cycle2 = cycles[1]
    
    c1 = np.array(points[cycle1,:])
    c2 = np.array(points[cycle2,:])

    plt.scatter(points[:, 0], points[:, 1])

    plt.plot(c1[:, 0], c1[:, 1], color='red')
    plt.plot(c2[:, 0], c2[:, 1], color='blue')

    plt.show()

# def range_with_floats(start, stop, step):
#     while stop > start:
#         yield start
#         start += step

if __name__ == '__main__':
    path = './kroA100.tsp'
    length_matrix_kroa100, points = readInstance(path)

    plt.rc('figure', figsize=(8, 5))
   
    best_solution, best_score = find_best_solution(regret_cycle_heuristic, np.array(length_matrix_kroa100))

    plt.subplots()
    plt.suptitle(f"file: kroA100.tsp, solver: {regret_cycle_heuristic.__name__}, score: {best_score}")

    print(best_solution)
    print(len(best_solution[0]), len(best_solution[1]))
    draw_paths(points, best_solution)
    # print(length_matrix_kroa100[0][0])
    # print(length_matrix_kroa100[0][1])