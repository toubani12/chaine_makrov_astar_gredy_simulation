"""
grid.py — Génération et visualisation de grilles 2D avec obstacles
"""

import numpy as np
import random


def generate_grid(rows, cols, obstacle_rate=0.2, seed=42):
    """
    Génère une grille 2D avec obstacles aléatoires.
    0 = libre, 1 = obstacle
    Garantit que start et goal sont libres et connectés.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    grid = np.zeros((rows, cols), dtype=int)
    
    # Place obstacles aléatoirement
    for r in range(rows):
        for c in range(cols):
            if random.random() < obstacle_rate:
                grid[r][c] = 1
    
    # Libère toujours le coin start (0,0) et goal (rows-1, cols-1)
    grid[0][0] = 0
    grid[rows-1][cols-1] = 0
    
    # Vérifie connectivité via BFS, sinon regénère
    if not _is_connected(grid, (0, 0), (rows-1, cols-1)):
        # Créer un corridor simple
        for c in range(cols):
            grid[0][c] = 0
        for r in range(rows):
            grid[r][cols-1] = 0
    
    return grid


def _is_connected(grid, start, goal):
    """BFS pour vérifier si start peut atteindre goal."""
    rows, cols = grid.shape
    visited = set()
    queue = [start]
    visited.add(start)
    
    while queue:
        r, c = queue.pop(0)
        if (r, c) == goal:
            return True
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0 and (nr,nc) not in visited:
                visited.add((nr,nc))
                queue.append((nr,nc))
    return False


def get_neighbors(grid, state):
    """Retourne les voisins accessibles (4-connectivité)."""
    rows, cols = grid.shape
    r, c = state
    neighbors = []
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
            neighbors.append((nr, nc))
    return neighbors


def get_all_free_states(grid):
    """Retourne tous les états libres de la grille."""
    rows, cols = grid.shape
    return [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 0]


# Grilles prédéfinies pour les expériences
GRIDS = {
    'easy': {
        'size': (10, 10),
        'obstacle_rate': 0.15,
        'seed': 42,
        'start': (0, 0),
        'goal': (9, 9)
    },
    'medium': {
        'size': (15, 15),
        'obstacle_rate': 0.25,
        'seed': 7,
        'start': (0, 0),
        'goal': (14, 14)
    },
    'hard': {
        'size': (20, 20),
        'obstacle_rate': 0.30,
        'seed': 123,
        'start': (0, 0),
        'goal': (19, 19)
    }
}
