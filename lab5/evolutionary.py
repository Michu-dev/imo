from typing import Any
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import pandas as pd
import random
import time
import itertools
from copy import deepcopy
from IPython.display import display
from tqdm import tqdm

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

def regret_cycle_perturbation(length_matrix, cycles, remaining_nodes):
    start_time = time.time()
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

def get_first_edges_swap(length_matrix, cycles):
    for idx in random.sample(range(2), 2):
        cycle = cycles[idx]
        n = len(cycle)
        candidates = generate_swap_edges_candidates(n)
        random.shuffle(candidates)
        for i, j in candidates:
            delta, a, b, c, d = calculate_swap_edges_delta(length_matrix, cycle, i, j)
            return (SWAP_EDGE, delta, a, b, c, d)
    return None

def get_first_nodes_swap(length_matrix, cycles):
    candidates = generate_swap_nodes_candidates(len(cycles[0]), len(cycles[1]))
    random.shuffle(candidates)
    for i, j in candidates:
        delta, move = calculate_swap_nodes_delta(length_matrix, cycles, 0, 1, i, j)
        return move
    return None

def get_first_move(length_matrix, cycles):
    moves = [get_first_edges_swap, get_first_nodes_swap]
    move_order = random.sample(range(2), 2)
    move = moves[move_order[0]](length_matrix, cycles)
    if move is None:
        move = moves[move_order[1]](length_matrix, cycles)
    return move

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
    
class MSLS:
    def __init__(self, length_matrix, algorithm):
        self.length_matrix = length_matrix
        self.solve = algorithm

    def __call__(self, n_iterations = 100):
        start = time.time()
        n = len(self.length_matrix)
        random_solutions = [random_solution(n) for _ in range(n_iterations)]
        local_cycles, _ = zip(*[self.solve(cycles) for cycles in random_solutions])
        scores = [score(self.length_matrix, cycles) for cycles in local_cycles]
        return local_cycles[np.argmin(scores)], time.time() - start
        
class SmallPerturbation:
    def __init__(self, n_swaps):
        self.n_swaps = n_swaps

    def __call__(self, length_matrix, cycles):
        for _ in range(self.n_swaps):
            move = get_first_move(length_matrix, cycles)
            apply_move(cycles, move)
        return cycles
    
class BigPerturbation:
    def __init__(self, destroy_factor=0.2):
        self.destroy_factor = destroy_factor

    def __call__(self, length_matrix, cycles):
        destroy_elem_number = int(self.destroy_factor * len(length_matrix) / 2)

        remaining_nodes = []
        for cycle in cycles:
            n = len(cycle)
            first_node_to_destroy = random.randint(0, n - 1)
            remaining_nodes.extend(cycle[first_node_to_destroy:(first_node_to_destroy + destroy_elem_number)])
            cycle[first_node_to_destroy:(first_node_to_destroy + destroy_elem_number)] = []

            if first_node_to_destroy + destroy_elem_number > n:
                remaining_nodes.extend(cycle[0:(first_node_to_destroy + destroy_elem_number - n)])
                cycle[0:(first_node_to_destroy + destroy_elem_number - n)] = []

        cycles, _ = regret_cycle_perturbation(length_matrix, cycles, remaining_nodes)

        return cycles


class ILS:
    def __init__(self, length_matrix, perturbation, algorithm, local_search=True):
        self.length_matrix = length_matrix
        self.perturbation = perturbation
        self.solve = algorithm
        self.local_search = local_search

    def __call__(self, time_limit):
        start = time.time()
        random_cycles = random_solution(len(self.length_matrix))
        best_cycles, _  = self.solve(random_cycles)
        best_score = score(self.length_matrix, best_cycles)
        n_iterations = 1
        while time.time() - start < time_limit:
            cycles = deepcopy(best_cycles)
            cycles = self.perturbation(self.length_matrix, cycles)
            if self.local_search:
                new_cycles, _ = self.solve(cycles)
            else:
                new_cycles = cycles
            
            new_score = score(self.length_matrix, new_cycles)
                            
            if new_score < best_score:
                best_cycles = new_cycles
                best_score = new_score
            n_iterations += 1
        
        return best_cycles, time.time() - start, n_iterations
    
def identity(x):
    return x

def argmax(x, f=identity):
    return max(np.arange(len(x)), key=lambda i: f(x[i]))

def argmin(x, f=identity):
    return min(np.arange(len(x)), key=lambda i: f(x[i]))
    
class Evolutionary:
    def __init__(self, length_matrix, algorithm, local_search=True, patience=1000):
        self.length_matrix = length_matrix
        self.solve = algorithm
        self.local_search = local_search
        self.patience = patience

    def combine(self, sol1, sol2):
        sol1, sol2 = deepcopy(sol1), deepcopy(sol2)

        remaining = []
        for cycle1 in sol1:
            n = len(cycle1)
            if n == 1:
                continue
            for i in range(n):
                p, q = cycle1[i], cycle1[(i + 1) % n]
                if p == -1 or q == -1 or p == q:
                    continue
                found = False
                for cycle2 in sol2:
                    m = len(cycle2)
                    for j in range(m):
                        u, v = cycle2[j], cycle2[(j + 1) % m]
                        if (p == u and q == v) or (p == v and q == u):
                            found = True
                            break
                    if found:
                        break
                if not found:
                    remaining.append(cycle1[i])
                    remaining.append(cycle1[(i + 1) % n])
                    cycle1[i] = -1
                    cycle1[(i + 1) % n] = -1
            
            for i in range(1, n):
                x, y, z = cycle1[(i - 1) % n], cycle1[i], cycle1[(i + 1) % n]
                if x == z == -1 and y != -1:
                    remaining.append(y)
                    cycle1[i] = -1

            for i in range(1, n):
                x = cycle1[i]
                if x != -1 and np.random.rand() < 0.2:
                    remaining.append(x)
                    cycle1[i] = -1

        a = [x for x in sol1[0] if x != -1]
        b = [x for x in sol1[1] if x != -1]
        assert len(a) + len(b) + len(remaining) == 200
        return regret_cycle_perturbation(self.length_matrix, (a, b), remaining)[0]

    def __call__(self, time_limit, pop_size=20):
        start = time.time()
        population = []
        n = len(self.length_matrix)
        while len(population) < pop_size:
            solution, _ = self.solve(random_solution(n))
            if solution not in population:
                population.append(solution)
        
        population = [(x, score(self.length_matrix, x)) for x in population]
        n_iterations = last_improvement = best_index = last_mutation = 0

        best_index = argmin(population, lambda x: x[1])
        best_solution, best_score = population[best_index]
        last_best = best_score


        while time.time() - start < time_limit:
            n_iterations += 1

            population_indexes = np.arange(pop_size)
            np.random.shuffle(population_indexes)

            worst_index = argmax(population, lambda x: x[1])
            worst_solution, worst_score = population[worst_index]

            solution1, score1 = population[population_indexes[0]]
            solution2, score2 = population[population_indexes[1]]

            solution = self.combine(solution1, solution2)
            if self.local_search:
                solution = self.solve(solution)[0]
            
            solution_score = score(self.length_matrix, solution)
            print(f'{score1} + {score2} -> {solution_score}')

            if solution not in [p[0] for p in population] and solution_score < worst_score:
                population[worst_index] = solution, solution_score

            best_index = argmin(population, lambda x: x[1])
            best_solution, best_score = population[best_index]
            if best_score < last_best:
                last_best = best_score
                last_improvement = n_iterations

            if n_iterations - last_improvement > self.patience:
                break

        return best_solution, time.time() - start, n_iterations


                

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
    n_iterations = 100
    algorithm_runs = 10
    paths = [f'kroA{n}.tsp', f'kroB{n}.tsp']
    score_results = []
    time_results = []

    for file in paths:
        length_matrix, coords = read_instance(file)
        solve = Evolutionary(length_matrix, SteepestSearch(length_matrix), True, 1000)
        time_limit = 600
        solutions, times, n_iterations_done = zip(*mp.Pool().map(local_search_extension, [time_limit for _ in range(algorithm_runs)]))
        best_solution_index = np.argmin(scores)
        best_solution = solutions[best_solution_index]
        print(f'file: {file}, method: Evolutionary, score: {scores[best_solution_index]}')

        plot_solution(coords, best_solution)

        score_results.append(dict(file=file, method="Evolutionary", min=int(min(scores)), mean=int(np.mean(scores)), max=int(max(scores)), n_iterations=np.mean(n_iterations_done)))
        time_results.append(dict(file=file, method="Evolutionary", min=int(min(times)), mean=int(np.mean(times)), max=int(max(times))))


        

        # scores = [score(length_matrix, cs) for cs in solutions]
        # solve = MSLS(length_matrix, SteepestSearch(length_matrix))
        # msls_solutions, times  = zip(*mp.Pool().map(solve, [n_iterations for _ in range(algorithm_runs)]))
        # scores = [score(length_matrix, cs) for cs in msls_solutions]

        # score_results.append(dict(file=file, method="MSLS", min=int(min(scores)), mean=int(np.mean(scores)), max=int(max(scores)), n_iterations=100))
        # time_results.append(dict(file=file, method="MSLS", min=int(min(times)), mean=int(np.mean(times)), max=int(max(times))))
        # time_limit = np.mean(times)
        # best_solution_index = np.argmin(scores)
        # best_solution = msls_solutions[best_solution_index]
        # print(f'file: {file}, method: MSLS, score: {scores[best_solution_index]}')
        # plot_solution(coords, best_solution)

        # for local_search_extension in [ILS(length_matrix, SmallPerturbation(10), SteepestSearch(length_matrix)), ILS(length_matrix, BigPerturbation(0.2), SteepestSearch(length_matrix), local_search=False), ILS(length_matrix, BigPerturbation(0.2), SteepestSearch(length_matrix))]:
        #     solutions, times, n_iterations_done = zip(*mp.Pool().map(local_search_extension, [time_limit for _ in range(algorithm_runs)]))
        #     scores = [score(length_matrix, cs) for cs in solutions]
        #     best_solution_index = np.argmin(scores)
        #     best_solution = solutions[best_solution_index]
        #     print(f'file: {file}, perturbation: {type(local_search_extension.perturbation).__name__}, method: {type(local_search_extension).__name__}, score: {scores[best_solution_index]}')

        #     plot_solution(coords, best_solution)

        #     score_results.append(dict(file=file, method=type(local_search_extension).__name__+type(local_search_extension.perturbation).__name__, min=int(min(scores)), mean=int(np.mean(scores)), max=int(max(scores)), n_iterations=np.mean(n_iterations_done)))
        #     time_results.append(dict(file=file, method=type(local_search_extension).__name__+type(local_search_extension.perturbation).__name__, min=int(min(times)), mean=int(np.mean(times)), max=int(max(times))))



    return pd.DataFrame(score_results), pd.DataFrame(time_results)

if __name__ == '__main__':
    scores, times = show_results()
    print(scores)
    print(times)