"""
astar.py — Algorithmes de recherche heuristique : UCS, Greedy, A* (et Weighted A*)
"""

import heapq
import time
import tracemalloc
from grid import get_neighbors


# ─────────────────────────────────────────────
# Heuristiques
# ─────────────────────────────────────────────

def manhattan(p, goal):
    """Heuristique de Manhattan — admissible et cohérente (4-voisins, coût uniforme 1)."""
    return abs(p[0] - goal[0]) + abs(p[1] - goal[1])


def zero_heuristic(p, goal):
    """h = 0 → équivalent à UCS (admissible mais non informée)."""
    return 0


def euclidean(p, goal):
    """Heuristique Euclidienne — admissible mais moins précise que Manhattan sur grille."""
    return ((p[0]-goal[0])**2 + (p[1]-goal[1])**2) ** 0.5


def chebyshev(p, goal):
    """Distance de Chebyshev (8-voisins) — inadmissible sur grille 4-voisins."""
    return max(abs(p[0]-goal[0]), abs(p[1]-goal[1]))


# ─────────────────────────────────────────────
# Algorithme générique de recherche
# ─────────────────────────────────────────────

def search(grid, start, goal, mode='astar', heuristic=manhattan, weight=1.0):
    """
    Algorithme de recherche générique.
    
    Modes:
      - 'ucs'    : f = g  (Uniform Cost Search)
      - 'greedy' : f = h  (Greedy Best-First)
      - 'astar'  : f = g + h  (A*)
      - 'wastar' : f = g + weight*h  (Weighted A*)
    
    Retourne un dict avec: path, cost, nodes_expanded, open_max_size,
                           time_sec, memory_kb
    """
    tracemalloc.start()
    t0 = time.perf_counter()

    # OPEN: heap de (f, g, état)
    open_heap = []
    # Compteur pour départager les égalités (ordre FIFO)
    counter = 0
    
    g_cost = {start: 0.0}
    came_from = {start: None}
    nodes_expanded = 0
    open_max_size = 1

    h0 = heuristic(start, goal)
    f0 = _compute_f(0.0, h0, mode, weight)
    heapq.heappush(open_heap, (f0, counter, start))
    counter += 1

    closed = set()

    while open_heap:
        open_max_size = max(open_max_size, len(open_heap))
        f, _, current = heapq.heappop(open_heap)

        if current in closed:
            continue
        closed.add(current)
        nodes_expanded += 1

        if current == goal:
            path = _reconstruct(came_from, goal)
            elapsed = time.perf_counter() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return {
                'path': path,
                'cost': g_cost[goal],
                'nodes_expanded': nodes_expanded,
                'open_max_size': open_max_size,
                'time_sec': elapsed,
                'memory_kb': peak / 1024,
                'found': True
            }

        for neighbor in get_neighbors(grid, current):
            new_g = g_cost[current] + 1.0  # coût uniforme = 1
            if neighbor not in g_cost or new_g < g_cost[neighbor]:
                g_cost[neighbor] = new_g
                came_from[neighbor] = current
                h = heuristic(neighbor, goal)
                f = _compute_f(new_g, h, mode, weight)
                heapq.heappush(open_heap, (f, counter, neighbor))
                counter += 1

    tracemalloc.stop()
    return {
        'path': [],
        'cost': float('inf'),
        'nodes_expanded': nodes_expanded,
        'open_max_size': open_max_size,
        'time_sec': time.perf_counter() - t0,
        'memory_kb': 0,
        'found': False
    }


def _compute_f(g, h, mode, weight):
    if mode == 'ucs':
        return g
    elif mode == 'greedy':
        return h
    elif mode == 'wastar':
        return g + weight * h
    else:  # astar
        return g + h


def _reconstruct(came_from, goal):
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = came_from[node]
    return list(reversed(path))


# ─────────────────────────────────────────────
# Wrappers pratiques
# ─────────────────────────────────────────────

def ucs(grid, start, goal):
    return search(grid, start, goal, mode='ucs', heuristic=zero_heuristic)

def greedy(grid, start, goal, heuristic=manhattan):
    return search(grid, start, goal, mode='greedy', heuristic=heuristic)

def astar(grid, start, goal, heuristic=manhattan):
    return search(grid, start, goal, mode='astar', heuristic=heuristic)

def weighted_astar(grid, start, goal, weight=1.5, heuristic=manhattan):
    return search(grid, start, goal, mode='wastar', heuristic=heuristic, weight=weight)
