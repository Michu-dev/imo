import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

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


# randomowe cykle 
# losowe błądzenie
# 2 żal
# tamte 6