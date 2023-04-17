import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import pandas as pd
import random
import time
import itertools
from copy import deepcopy
from IPython.display import display

# define constans
SWAP_EDGE = 1
SWAP_NODES = 2

# help functions
def distance(a, b):
    return np.round(np.sqrt(np.sum((a - b)**2)))

def read_instance(path):
    coords = pd.read_csv(path, sep=' ', names=['n','x','y'], skiprows=6, skipfooter=1, engine='python')
    nodes = coords.drop(columns=['n']).values
    ns = np.arange(len(nodes))
    length_matrix = np.array([[distance(nodes[i], nodes[j]) for j in ns] for i in ns])
    return length_matrix, coords


def cycle_score(length_matrix, cycle):
    cycle = cycle + [cycle[0]]
    return sum(length_matrix[cycle[i], cycle[i+1]] for i in range(len(cycle) - 1))

def score(length_matrix, cycles):
    return cycle_score(length_matrix, cycles[0]) + cycle_score(length_matrix, cycles[1])

def delta_insert(length_matrix, cycle, i, node):
    a, b = cycle[i - 1], cycle[i]
    return length_matrix[a, node] + length_matrix[node, b] - length_matrix[a, b]

def find_index_in_cycle(arr, i):
    try:
        return arr.index(i)
    except:
        return None
    
def find_node_index(cycles, elem):
    i = find_index_in_cycle(cycles[0], elem)
    if i is not None:
        return 0, i
    i = find_index_in_cycle(cycles[1], elem)
    if i is not None:
        return 1, i
    assert False, f'Node {elem} has to be in either cycle'

def reverse(arr, i, j):
    n = len(arr)
    d = (j - i) % n
    for k in range(abs(d) // 2 + 1):
        a, b = (i + k) % n, (i + d - k) % n
        arr[a], arr[b] = arr[b], arr[a]
    

# Initial solutions
def regret_cycle_heuristic(args):
    length_matrix, starting_node, N = args
    start_time = time.time()
    remaining_nodes = list(range(N))
    starting_node_2 = np.argmax(length_matrix[starting_node, :])
    remaining_nodes.remove(starting_node)
    remaining_nodes.remove(starting_node_2)
    cycles = [[starting_node], [starting_node_2]]
    while len(remaining_nodes) > 0:
        for cycle in cycles:
            scores = np.array([[delta_insert(length_matrix, cycle, i, n) for i in range(len(cycle))] for n in remaining_nodes])
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
            cycle.insert(insert_index, node_to_insert)
            remaining_nodes.remove(node_to_insert)
    
    return cycles, time.time() - start_time 

def random_solution(n, seed=None):
    remaining = list(range(n))
    random.seed(seed)
    random.shuffle(remaining)
    return remaining[:(n // 2)], remaining[(n // 2):]

def generate_swap_edges_candidates(n):
    return [(i, (i + d) % n) for i in range(n) for d in range(2, n - 1)]

def generate_swap_nodes_candidates(n, m):
    return [(i, j) for i in range(n) for j in range(m)]

def delta_swap_edges(length_matrix, a, b, c, d):
    if a == b or a == c or a == d or b == c or b == d or c == d:
        return 10000000
    return length_matrix[a, c] + length_matrix[b, d] - length_matrix[a, b] - length_matrix[c, d]

def delta_swap_nodes(length_matrix, x1, y1, z1, x2, y2, z2):
    return length_matrix[x1, y2] + length_matrix[z1, y2] - length_matrix[x1, y1] - length_matrix[z1, y1] + \
            length_matrix[x2, y1] + length_matrix[z2, y1] - length_matrix[x2, y2] - length_matrix[z2, y2]

def calculate_swap_edges_delta(length_matrix, cycle, i, j):
    n = len(cycle)
    nodes = cycle[i], cycle[(i + 1) % n], cycle[j], cycle[(j + 1) % n]
    return (delta_swap_edges(length_matrix, *nodes), *nodes)

def calculate_swap_nodes_delta(length_matrix, cycles, c1, c2, i, j):
    cycle1, cycle2 = cycles[c1], cycles[c2]
    n, m = len(cycle1), len(cycle2)
    x1, y1, z1 = cycle1[(i - 1) % n], cycle1[i], cycle1[(i + 1) % n]
    x2, y2, z2 = cycle2[(j - 1) % m], cycle2[j], cycle2[(j + 1) % m]
    delta = delta_swap_nodes(length_matrix, x1, y1, z1, x2, y2, z2)
    move = (SWAP_NODES, delta, c1, c2, x1, y1, z1, x2, y2, z2)
    return delta, move

def initial_moves(length_matrix, cycles):
    moves = []
    for cycle in cycles:
        n = len(cycle)
        for i, j in generate_swap_edges_candidates(n):
            delta, a, b, c, d = calculate_swap_edges_delta(length_matrix, cycle, i, j)
            if delta < 0:
                moves.append((SWAP_EDGE, delta, a, b, c, d))
    for i, j in generate_swap_nodes_candidates(len(cycles[0]), len(cycles[1])):
        delta, move = calculate_swap_nodes_delta(length_matrix, cycles, 0, 1, i, j)
        if delta < 0:
            moves.append(move)
    return moves
    
def apply_move(cycles, move):
    type = move[0]
    if type == SWAP_EDGE:
        _, _, a, _, c, _ = move
        (c1_index, i), (c2_index, j) = find_node_index(cycles, a), find_node_index(cycles, c)
        cycle = cycles[c1_index]
        n = len(cycle)
        reverse(cycle, (i + 1) % n, j)
    elif type == SWAP_NODES:
        _, _, c1_index, c2_index, _, a, _, _, b, _ = move
        i, j = cycles[c1_index].index(a), cycles[c2_index].index(b)
        cycles[c1_index][i], cycles[c2_index][j] = cycles[c2_index][j], cycles[c1_index][i]

def has_edge(cycle, a, b):
    cycle = cycle + [cycle[0]]
    for i in range(len(cycle) - 1):
        x, y = cycle[i], cycle[i + 1]
        if (a, b) == (x, y):
            return 1
        elif (a, b) == (y, x):
            return -1
        
    return 0

def any_has_edge(cycles, a, b):
    for i in range(2):
        status = has_edge(cycles[i], a, b)
        if status != 0:
            return i, status
    return None, 0

def remove_at(arr, sorted_indices):
    for i in reversed(sorted_indices):
        del(arr[i])

class SteepestSearch:
    def __init__(self, length_matrix):
        self.length_matrix = length_matrix

    def __call__(self, cycles):
        cycles = deepcopy(cycles)
        start_time = time.time()
        while True:
            moves = initial_moves(self.length_matrix, cycles)
            if not moves:
                break
            move = min(moves, key=lambda x: x[1])
            apply_move(cycles, move)
        return cycles, time.time() - start_time
                
class CandidatesSearch:
    def __init__(self, length_matrix):
        self.length_matrix = length_matrix
    
    def __call__(self, cycles, k=10):
        n = len(self.length_matrix)
        cycles = deepcopy(cycles)
        start_time = time.time()
        nearest = np.argpartition(self.length_matrix, k+1, axis=1)[:, :k+1]

        while True:
            best_move, best_delta = None, 0
            for a in range(n):
                for b in nearest[a]:
                    if a == b:
                        continue
                    (c1_index, i), (c2_index, j) = find_node_index(cycles, a), find_node_index(cycles, b)
                    move, delta = None, None
                    if c1_index == c2_index:
                        cycle = cycles[c1_index]
                        n = len(cycle)
                        a, b, c, d = a, cycle[(i + 1) % n], b, cycle[(j + 1) % n]
                        delta = delta_swap_edges(self.length_matrix, a, b, c, d)
                        move = SWAP_EDGE, delta, a, b, c, d
                    else:
                        delta, move = calculate_swap_nodes_delta(self.length_matrix, cycles, c1_index, c2_index, i, j)
                    if delta < best_delta:
                        best_delta, best_move = delta, move
            
            if best_move is None:
                break

            apply_move(cycles, best_move)

        return cycles, time.time() - start_time
    
class MemorySearch:
    def __init__(self, length_matrix):
        self.length_matrix = length_matrix

    def next_moves(self, cycles, move):
        type = move[0]
        moves = []
        if type == SWAP_EDGE:
            _, _, a, b, c, d = move
            cycle = cycles[0] if a in cycles[0] else cycles[1]
            n = len(cycle)
            for i, j in generate_swap_edges_candidates(n):
                delta, a, b, c, d = calculate_swap_edges_delta(self.length_matrix, cycle, i, j)
                if delta < 0:
                    moves.append((SWAP_EDGE, delta, a, b, c, d))
        elif type == SWAP_NODES:
            _, _, c1_index, c2_index, _, y1, _, _, y2, _ = move
            i, j = cycles[c1_index].index(y2), cycles[c2_index].index(y1)
            n, m = len(cycles[c1_index]), len(cycles[c2_index])
            for k in range(m):
                delta, move = calculate_swap_nodes_delta(self.length_matrix, cycles, c1_index, c2_index, i, k)
                if delta < 0:
                    moves.append(move)
            for k in range(n):
                delta, move = calculate_swap_nodes_delta(self.length_matrix, cycles, c2_index, c1_index, j, k)
                if delta < 0:
                    moves.append(move)
        
        return moves
    
    def __call__(self, cycles):
        cycles = deepcopy(cycles)
        start_time = time.time()
        moves = sorted(initial_moves(self.length_matrix, cycles), key=lambda x: x[1])

        while True:
            moves_to_delete = []
            best_move = None
            for k, move in enumerate(moves):
                type = move[0]
                if type == SWAP_EDGE:
                    _, _, a, b, c, d = move
                    (c1, s1), (c2, s2) = any_has_edge(cycles, a, b), any_has_edge(cycles, c, d)
                    if c1 != c2 or s1 == 0 or s2 == 0:
                        moves_to_delete.append(k)
                    elif s1 == s2 == 1:
                        moves_to_delete.append(k)
                        best_move = move
                        break
                    elif s1 == s2 == -1:
                        moves_to_delete.append(k)
                        best_move = SWAP_EDGE, move[1], b, a, d, c
                        break
                elif type == SWAP_NODES:
                    _, _, c1, c2, x1, y1, z1, x2, y2, z2 = move
                    s1 = has_edge(cycles[c1], x1, y1)
                    s2 = has_edge(cycles[c1], y1, z1)
                    s3 = has_edge(cycles[c2], x2, y2)
                    s4 = has_edge(cycles[c2], y2, z2)

                    if c1 == c2 or s1 == 0 or s2 == 0 or s3 == 0 or s4 == 0:
                        moves_to_delete.append(k)
                    elif s1 == s2 and s3 == s4:
                        moves_to_delete.append(k)
                        best_move = move
                        break
            
            if best_move is None:
                break
            
            remove_at(moves, moves_to_delete)
            apply_move(cycles, best_move)

            new_moves = self.next_moves(cycles, best_move)
            moves = sorted(list(set(moves).union(set(new_moves))), key=lambda x: x[1])

        return cycles, time.time() - start_time


def draw_path(coords, path, color='blue'):
    cycle = path + [path[0]]
    for i in range(len(cycle) - 1):
        a, b = cycle[i], cycle[i+1]
        plt.plot([coords.x[a], coords.x[b]], [coords.y[a], coords.y[b]], color=color)

def plot_solution(coords, solution):
    path1, path2 = solution
    draw_path(coords, path1, color='green')
    draw_path(coords, path2, color='red')
    plt.scatter(coords.x, coords.y, color='black')
    plt.show()

def show_results(n=200):
    instances = [f'kroA{n}.tsp', f'kroB{n}.tsp']
    score_results = []
    time_results = []

    for file in instances:
        length_matrix, coords = read_instance(file)
        init_solutions, times = zip(*mp.Pool().map(regret_cycle_heuristic, [(length_matrix, i, n) for i in range(n)]))
        scores = [score(length_matrix, x) for x in init_solutions]

        score_results.append(dict(file=file, search="regret", min=int(min(scores)), mean=int(np.mean(scores)), max=int(max(scores))))
        time_results.append(dict(file=file, search="regret", min=float(min(times)), mean=float(np.mean(times)), max=float(max(times))))

        best_solution_index = np.argmin(scores)
        best_solution = init_solutions[best_solution_index]
        print(f"file: {file}, search: regret, score: {scores[best_solution_index]}")
        plot_solution(coords, best_solution)
        
        for local_search_method in [SteepestSearch(length_matrix), MemorySearch(length_matrix), CandidatesSearch(length_matrix)]:
            random_solutions = [random_solution(n) for _ in range(100)]
            solutions, times = zip(*mp.Pool().map(local_search_method, random_solutions))

            scores = [score(length_matrix, x) for x in solutions]
            best_solution_index = np.argmin(scores)
            best_solution = solutions[best_solution_index]
            print(f"file: {file}, search: {type(local_search_method).__name__}, score: {scores[best_solution_index]}")
            plot_solution(coords, best_solution)

            score_results.append(dict(file=file, search=type(local_search_method).__name__, min=int(min(scores)), mean=int(np.mean(scores)), max=int(max(scores))))
            time_results.append(dict(file=file, search=type(local_search_method).__name__, min=float(min(times)), mean=float(np.mean(times)), max=float(max(times))))
    
    return pd.DataFrame(score_results), pd.DataFrame(time_results)


if __name__ == '__main__':
    scores, times = show_results()
    print(scores)
    print(times)