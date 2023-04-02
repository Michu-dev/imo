import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import pandas as pd
import random
import time
import itertools
from copy import deepcopy
from IPython.display import display

def distance(a, b):
    return np.round(np.sqrt(np.sum((a - b)**2)))

def read_instance(path):
    coords = pd.read_csv(path, sep=' ', names=['n','x','y'], skiprows=6, skipfooter=1, engine='python')
    nodes = coords.drop(columns=['n']).values
    ns = np.arange(len(nodes))
    length_matrix = np.array([[distance(nodes[i], nodes[j]) for j in ns] for i in ns])
    return length_matrix, coords


def cycle_score(cities, path):
    cycle = path + [path[0]]
    return sum(cities[cycle[i], cycle[i+1]] for i in range(len(cycle) - 1))

def score(cities, paths):
    return cycle_score(cities, paths[0]) + cycle_score(cities, paths[1])

def delta_insert(cities, path, i, city):
    a, b = path[i - 1], path[i]
    return cities[a, city] + cities[city, b] - cities[a, b]

def score_diff(length_matrix, cycle, edge_index, node):
    a, b = cycle[edge_index - 1], cycle[edge_index]
    return length_matrix[a, node] + length_matrix[node, b] - length_matrix[a, b]


def delta_replace_vertex(length_matrix, cycle, i, point):
    cycle_length = len(cycle)
    a, b, c = cycle[(i - 1) % cycle_length], cycle[i], cycle[(i + 1) % cycle_length]
    return length_matrix[a, point] + length_matrix[point, c] - length_matrix[a, b] - length_matrix[b, c]


def delta_replace_vertices_outside(length_matrix, cycles, i, j):
    return delta_replace_vertex(length_matrix, cycles[0], i, cycles[1][j]) + delta_replace_vertex(length_matrix, cycles[1], j, cycles[0][i])


def delta_replace_vertices_inside(length_matrix, cycle, i, j):
    cycle_length = len(cycle)

    a, b, c = cycle[(i - 1) % cycle_length], cycle[i], cycle[(i + 1) % cycle_length]
    d, e, f = cycle[(j - 1) % cycle_length], cycle[j], cycle[(j + 1) % cycle_length]

    if j - i == 1:
        return length_matrix[a, e] + length_matrix[b, f] - length_matrix[a, b] - length_matrix[e, f]
    elif i == 0 and j == len(cycle) - 1:
        return length_matrix[e, c] + length_matrix[d, b] - length_matrix[b, c] - length_matrix[d, e]
    else:
        return length_matrix[a, e] + length_matrix[e, c] + length_matrix[d, b] + length_matrix[b, f] - length_matrix[a, b] - length_matrix[b, c] - length_matrix[d, e] - length_matrix[e, f]

def delta_replace_edges_inside(length_matrix, cycle, i, j):
    cycle_length = len(cycle)
    if i == 0 and j == len(cycle) - 1:
        a, b, c, d = cycle[i], cycle[(i + 1) % cycle_length], cycle[(j - 1) % cycle_length], cycle[j]
    else:
        a, b, c, d = cycle[(i - 1) % cycle_length], cycle[i], cycle[j], cycle[(j + 1) % cycle_length]
    return length_matrix[a, c] + length_matrix[b, d] - length_matrix[a, b] - length_matrix[c, d]

def outside_candidates(cycles):
    indices = list(range(len(cycles[0]))), list(range(len(cycles[1])))
    return list(itertools.product(*indices))


def inside_candidates(cycle):
    combinations = []
    for i in range(len(cycle)):
        for j in range(i + 1, len(cycle)):
            combinations.append([i, j])

    return combinations

def replace_vertices_outside(cycles, i, j):
    temp = cycles[0][i]
    cycles[0][i] = cycles[1][j]
    cycles[1][j] = temp

def replace_vertices_inside(cycle, i, j):
    temp = cycle[i]
    cycle[i] = cycle[j]
    cycle[j] = temp

def replace_edges_inside(cycle, i, j):
    if i == 0 and j == len(cycle) - 1:
        temp = cycle[i]
        cycle[i] = cycle[j]
        cycle[j] = temp     
    cycle[i:(j + 1)] = reversed(cycle[i:(j + 1)])


# Initial solutions
def regret_cycle_heuristic(args):
    length_matrix, starting_node = args
    remaining_nodes = list(range(100))
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
    
    return cycles      

def random_solution(args):
    length_matrix, _ = args
    n = length_matrix.shape[0]
    remaining = list(range(n))
    random.seed()
    random.shuffle(remaining)
    cycles = [remaining[:(n // 2)], remaining[(n // 2):]]
    return cycles

class GreedySearch(object):
    def __init__(self, length_matrix, variant):
        self.variant = variant
        if variant == "vertices":
            self.delta = delta_replace_vertices_inside
            self.replace = replace_vertices_inside
        else:
            self.delta = delta_replace_edges_inside
            self.replace = replace_edges_inside
        self.length_matrix = length_matrix
        self.moves = [self.outside_vertices_trade_first, self.inside_trade_first]

    def outside_vertices_trade_first(self, length_matrix, cycles):
        random.seed()
        candidates = outside_candidates(cycles)
        random.shuffle(candidates)
        for i, j in candidates:
            score_diff = delta_replace_vertices_outside(length_matrix, cycles, i, j)
            if score_diff < 0:
                replace_vertices_outside(cycles, i, j)
                return score_diff
        return score_diff
    
    def inside_trade_first(self, length_matrix, cycles):
        random.seed()
        cycle_order = random.sample(range(2), 2)
        for index in cycle_order:
            candidates = inside_candidates(cycles[index])
            random.shuffle(candidates)
            for i, j in candidates:
                score_diff = self.delta(length_matrix, cycles[index], i, j)
                if score_diff < 0:
                    self.replace(cycles[index], i, j)
                    return score_diff
        return score_diff
    
    def __call__(self, cycles):
        cycles = deepcopy(cycles)
        random.seed()
        start = time.time()
        while True:
            move_order = random.sample(range(2), 2)
            score = self.moves[move_order[0]](self.length_matrix, cycles)
            if score >= 0:
                score = self.moves[move_order[1]](self.length_matrix, cycles)
                if score >= 0:
                    break
        return time.time() - start, cycles
    

class SteepestSearch(object):
    def __init__(self, length_matrix, variant):
        self.variant = variant
        if variant == 'vertices':
            self.delta = delta_replace_vertices_inside
            self.replace = replace_vertices_inside
        else:
            self.delta = delta_replace_edges_inside
            self.replace = replace_edges_inside
        self.length_matrix = length_matrix
        self.moves = [self.outside_vertices_trade_best, self.inside_trade_best]
        
    def outside_vertices_trade_best(self, length_matrix, cycles):
        candidates = outside_candidates(cycles)
        scores = np.array([delta_replace_vertices_outside(length_matrix, cycles, i, j) for i, j in candidates])
        best_result_idx = np.argmin(scores)
        if scores[best_result_idx] < 0:
            return replace_vertices_outside, (cycles, *candidates[best_result_idx]), scores[best_result_idx]
        return replace_vertices_outside, (cycles, *candidates[best_result_idx]), scores[best_result_idx]
    
    def inside_trade_best(self, length_matrix, cycles):
        combinations = inside_candidates(cycles[0]), inside_candidates(cycles[1])
        scores = np.array([[self.delta(length_matrix, cycles[index], i, j) for i, j in combinations[index]] for index in range(len(cycles))])
        best_cycle_idx, best_combination = np.unravel_index(np.argmin(scores), scores.shape)
        best_score = scores[best_cycle_idx, best_combination]
        if best_score < 0:
            return self.replace, (cycles[best_cycle_idx], *combinations[best_cycle_idx][best_combination]), best_score
        return self.replace, (cycles[best_cycle_idx], *combinations[best_cycle_idx][best_combination]), best_score
    
    def __call__(self, cycles):
        cycles = deepcopy(cycles)
        start = time.time()
        while True:
            replace_func, args, scores = list(zip(*[move(self.length_matrix, cycles) for move in self.moves]))
            best_score_idx = np.argmin(scores)
            if scores[best_score_idx] < 0:
                replace_func[best_score_idx](*args[best_score_idx])
            else:
                break
        return time.time() - start, cycles
    
class RandomSearch(object):
    def __init__(self, length_matrix, time_limit):
        self.length_matrix = length_matrix
        self.time_limit = time_limit
        self.moves = [self.outside_vertices_trade, self.inside_trade_first_vertices, self.inside_trade_first_edges]
        
    def outside_vertices_trade(self, length_marix, cycles):
        random.seed()
        candidates = outside_candidates(cycles)
        i, j = random.choice(candidates)
        replace_vertices_outside(cycles, i, j)
    
    def inside_trade_first_vertices(self, length_matrix, cycles):
        random.seed()
        cycle_idx = random.choice([0, 1])
        candidates = inside_candidates(cycles[cycle_idx])
        i, j = random.choice(candidates)
        replace_vertices_inside(cycles[cycle_idx], i, j)
    
    def inside_trade_first_edges(self, length_matrix, cycles):
        random.seed()
        cycle_idx = random.choice([0, 1])
        candidates = inside_candidates(cycles[cycle_idx])
        i, j = random.choice(candidates)
        replace_edges_inside(cycles[cycle_idx], i, j)

    def __call__(self, cycles):
        best_solution = cycles
        best_score = score(self.length_matrix, cycles)
        cycles = deepcopy(cycles)
        random.seed()
        start = time.time()
        while time.time() - start < self.time_limit:
            move = random.choice(self.moves)
            move(self.length_matrix, cycles)
            new_score = score(self.length_matrix, cycles)
            if new_score < best_score:
                best_solution = deepcopy(cycles)
                best_score = new_score
        return time.time() - start, best_solution
    
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

def show_results():
    instances = ['kroA100.tsp']
    initial_solvers = [regret_cycle_heuristic]
    score_results = []
    time_results = []
    for file in instances:
        cities, coords = read_instance(file)
        local_variants = [RandomSearch(cities, 3.21), GreedySearch(cities, "vertices"), SteepestSearch(cities, "vertices"), GreedySearch(cities, "edges"), SteepestSearch(cities, "edges")]
        for solve in initial_solvers:
            solutions = mp.Pool().map(solve, [(cities, i) for i in range(100)])
            scores = [score(cities, x) for x in solutions]
            score_results.append(dict(file=file, function=solve.__name__, search="none", variant="none", min=int(min(scores)), mean=int(np.mean(scores)), max=int(max(scores))))
            best_idx = np.argmin(scores)
            best = solutions[best_idx]
            print(f'file: {file}, solver: {solve.__name__}, search: None, variant: None, score: {scores[best_idx]}')
            plot_solution(coords, best)
            for local_search in local_variants:
                times, new_solutions = zip(*mp.Pool().map(local_search, solutions))
                new_scores = [score(cities, x) for x in new_solutions]
                best_idx = np.argmin(new_scores)
                best = new_solutions[best_idx]
                if type(local_search).__name__ == 'RandomSearch':
                    print(f'file: {file}, solver: {solve.__name__}, search: {type(local_search).__name__}, score: {new_scores[best_idx]}')
                    plot_solution(coords, best)
                    score_results.append(dict(file=file, function=solve.__name__, search=type(local_search).__name__, min=int(min(new_scores)), mean=int(np.mean(new_scores)), max=int(max(new_scores))))
                    time_results.append(dict(file=file, function=solve.__name__, search=type(local_search).__name__, min=float(min(times)), mean=float(np.mean(times)), max=float(max(times))))
                else:
                    print(f'file: {file}, solver: {solve.__name__}, search: {type(local_search).__name__}, variant: {local_search.variant}, score: {new_scores[best_idx]}')
                    plot_solution(coords, best)
                    score_results.append(dict(file=file, function=solve.__name__, search=type(local_search).__name__, variant=local_search.variant, min=int(min(new_scores)), mean=int(np.mean(new_scores)), max=int(max(new_scores))))
                    time_results.append(dict(file=file, function=solve.__name__, search=type(local_search).__name__,variant=local_search.variant, min=float(min(times)), mean=float(np.mean(times)), max=float(max(times))))
            if solve.__name__ == "solve_random":
                temp_pd = pd.DataFrame(time_results)
                time_limit = max(temp_pd[temp_pd["file"]==file]["mean"])
                random_search = RandomSearch(cities, time_limit)
                random_solutions = mp.Pool().map(random_search, solutions)
                random_scores = [score(cities, x) for x in random_solutions]
                best = random_solutions[best_idx]
                print(f'file: {file}, solver: {solve.__name__}, search: {type(random_search).__name__}, score: {random_scores[best_idx]}')
                plot_solution(coords, best)
                score_results.append(dict(file=file, function=solve.__name__, search=type(random_search).__name__, variant="none", min=int(min(random_scores)), mean=int(np.mean(random_scores)), max=int(max(random_scores))))
    return pd.DataFrame(score_results), pd.DataFrame(time_results)

    # # Random search
    # temp_pd = pd.DataFrame(time_results)
    # time_limit = max(temp_pd[temp_pd["file"]==file]["mean"])
    # random_search = RandomSearch(length_matrix, time_limit)
    # random_solutions = mp.Pool().map(random_search, solutions)
    # random_scores = [score(length_matrix, x) for x in random_solutions]
    # best = random_solutions[best_idx]
    # print(f'file: {file}, solver: {solve.__name__}, search: {type(random_search).__name__}, score: {random_scores[best_idx]}')
    # plot_solution(coords, best)
    # score_results.append(dict(file=file, function=solve.__name__, search=type(random_search).__name__, variant="none", min=int(min(random_scores)), mean=int(np.mean(random_scores)), max=int(max(random_scores))))

if __name__ == '__main__':
    scores, times = show_results()
    print(scores)
    print(times)