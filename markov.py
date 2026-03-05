"""
markov.py — Construction et analyse des Chaînes de Markov induites par un chemin A*
"""

import numpy as np
from grid import get_neighbors, get_all_free_states


# ─────────────────────────────────────────────
# Construction de la matrice P
# ─────────────────────────────────────────────

def build_policy(path):
    """
    Construit la politique (état → action/next_state) induite par le chemin A*.
    Pour chaque état du chemin, la politique recommande le prochain état.
    """
    policy = {}
    for i in range(len(path) - 1):
        policy[path[i]] = path[i+1]
    return policy


def build_transition_matrix(grid, path, goal, epsilon=0.1, fail_state=None):
    """
    Construit la matrice de transition stochastique P.

    Modèle d'incertitude (ε) :
      - Avec prob (1-ε) : l'agent suit la politique (→ next_state voulu)
      - Avec prob ε/2 chacun : l'agent dévie vers un voisin latéral
      - Si deviation vers obstacle : l'agent reste sur place

    États spéciaux :
      - GOAL (état absorbant) : p(GOAL, GOAL) = 1
      - FAIL (état absorbant optionnel)

    Retourne: P (matrice numpy), state_index (dict), index_state (list)
    """
    free_states = get_all_free_states(grid)
    
    # États spéciaux
    special = ['GOAL']
    if fail_state:
        special.append('FAIL')
    
    all_states = free_states + special
    state_index = {s: i for i, s in enumerate(all_states)}
    n = len(all_states)
    
    P = np.zeros((n, n))
    policy = build_policy(path)
    
    # Directions: haut, bas, gauche, droite
    DIRECTIONS = [(-1,0),(1,0),(0,-1),(0,1)]
    
    for state in free_states:
        i = state_index[state]
        
        # État goal → absorbant
        if state == goal:
            goal_idx = state_index['GOAL']
            P[i][goal_idx] = 1.0
            continue
        
        # Action recommandée par la politique
        if state in policy:
            intended = policy[state]
        else:
            # Hors chemin : rester sur place
            intended = state
        
        # Direction de l'action voulue
        dr = intended[0] - state[0]
        dc = intended[1] - state[1]
        
        # Directions latérales (perpendiculaires)
        if dr != 0:  # mouvement vertical → latéraux = horizontal
            laterals = [(0,-1),(0,1)]
        elif dc != 0:  # mouvement horizontal → latéraux = vertical
            laterals = [(-1,0),(1,0)]
        else:
            laterals = []
        
        # Transition principale (1-ε)
        rows, cols = grid.shape
        
        def _get_dest(from_state, direction):
            nr = from_state[0] + direction[0]
            nc = from_state[1] + direction[1]
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
                return (nr, nc)
            return from_state  # reste sur place si obstacle
        
        main_dest = _get_dest(state, (dr, dc))
        
        # Accumulate transitions
        transitions = {}
        
        # Action principale
        if main_dest == goal:
            dest_key = 'GOAL'
        else:
            dest_key = main_dest
        transitions[dest_key] = transitions.get(dest_key, 0) + (1 - epsilon)
        
        # Déviations latérales
        if laterals and epsilon > 0:
            for lat in laterals:
                lat_dest = _get_dest(state, lat)
                if lat_dest == goal:
                    lk = 'GOAL'
                else:
                    lk = lat_dest
                transitions[lk] = transitions.get(lk, 0) + epsilon / 2
        elif epsilon > 0:
            # Pas de latéraux → reste sur place avec prob ε
            transitions[state] = transitions.get(state, 0) + epsilon
        
        for dest, prob in transitions.items():
            j = state_index[dest]
            P[i][j] += prob
    
    # États absorbants
    for sp in special:
        idx = state_index[sp]
        P[idx][idx] = 1.0
    
    # Normalisation (sécurité numérique)
    for i in range(n):
        row_sum = P[i].sum()
        if row_sum > 0:
            P[i] /= row_sum
    
    return P, state_index, all_states


# ─────────────────────────────────────────────
# Calculs Markov
# ─────────────────────────────────────────────

def compute_pi_n(P, pi0, n):
    """Calcule π(n) = π(0) · P^n (Chapman-Kolmogorov)."""
    Pn = np.linalg.matrix_power(P, n)
    return pi0 @ Pn


def prob_goal_at_step(P, state_index, start, goal_key='GOAL', max_steps=100):
    """
    Calcule la probabilité cumulée d'être dans GOAL à chaque pas t.
    Retourne un array de taille max_steps+1.
    """
    n = len(state_index)
    pi0 = np.zeros(n)
    pi0[state_index[start]] = 1.0
    
    probs = []
    goal_idx = state_index[goal_key]
    
    pi = pi0.copy()
    for t in range(max_steps + 1):
        probs.append(pi[goal_idx])
        pi = pi @ P
    
    return np.array(probs)


def absorption_analysis(P, state_index, all_states, special_states=('GOAL',)):
    """
    Décomposition canonique P = [[Q, R], [0, I]].
    Calcule matrice fondamentale N = (I-Q)^{-1} et probabilités absorption B = N·R.
    
    Retourne: N, B, t_mean (temps moyen absorption), abs_probs
    """
    special_idx = {s: state_index[s] for s in special_states if s in state_index}
    trans_idx = [i for i, s in enumerate(all_states) if s not in special_states and isinstance(s, tuple)]
    abs_idx = list(special_idx.values())
    
    if not trans_idx:
        return None, None, None, None
    
    Q = P[np.ix_(trans_idx, trans_idx)]
    R = P[np.ix_(trans_idx, abs_idx)]
    
    I = np.eye(len(trans_idx))
    try:
        N = np.linalg.inv(I - Q)  # Matrice fondamentale
        B = N @ R                  # Probabilités d'absorption
        t_mean = N.sum(axis=1)     # Temps moyen avant absorption
        return N, B, t_mean, trans_idx
    except np.linalg.LinAlgError:
        return None, None, None, None


def classify_states(P, all_states):
    """
    Identifie les classes de communication et états transitoires/récurrents.
    Retourne un dict: état → 'absorbant', 'transitoire', ou 'récurrent'
    """
    n = len(all_states)
    classification = {}
    
    for i, s in enumerate(all_states):
        if P[i][i] == 1.0 and np.sum(P[i]) == 1.0 and np.count_nonzero(P[i]) == 1:
            classification[s] = 'absorbant'
        else:
            # Vérifie si l'état peut revenir à lui-même
            Pk = P.copy()
            can_return = False
            for _ in range(n):
                if Pk[i][i] > 1e-10:
                    can_return = True
                    break
                Pk = Pk @ P
            classification[s] = 'récurrent' if can_return else 'transitoire'
    
    return classification


# ─────────────────────────────────────────────
# Simulation Monte-Carlo
# ─────────────────────────────────────────────

def monte_carlo_simulation(grid, path, goal, epsilon=0.1, n_simulations=1000, max_steps=500):
    """
    Simule N trajectoires depuis start en suivant la politique stochastique.
    
    Retourne:
      - prob_goal: probabilité empirique d'atteindre GOAL
      - mean_steps: nombre moyen de pas pour atteindre GOAL (parmi succès)
      - steps_distribution: liste des pas pour chaque simulation ayant réussi
      - trajectories: quelques trajectoires exemple
    """
    if not path:
        return 0.0, 0, [], []
    
    policy = build_policy(path)
    start = path[0]
    rows, cols = grid.shape
    
    successes = 0
    steps_list = []
    sample_trajectories = []
    
    for sim in range(n_simulations):
        state = start
        traj = [state]
        steps = 0
        reached_goal = False
        
        for step in range(max_steps):
            if state == goal:
                reached_goal = True
                break
            
            # Action recommandée
            if state in policy:
                intended = policy[state]
            else:
                intended = state
            
            dr = intended[0] - state[0]
            dc = intended[1] - state[1]
            
            # Tirer aléatoirement
            r = np.random.random()
            
            if r < (1 - epsilon):
                move = (dr, dc)
            else:
                # Déviation latérale
                if dr != 0:
                    laterals = [(0,-1),(0,1)]
                elif dc != 0:
                    laterals = [(-1,0),(1,0)]
                else:
                    laterals = [(dr, dc)]
                move = laterals[np.random.randint(0, len(laterals))]
            
            nr, nc = state[0] + move[0], state[1] + move[1]
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
                state = (nr, nc)
            # Sinon reste sur place
            
            traj.append(state)
            steps += 1
        
        if reached_goal or state == goal:
            successes += 1
            steps_list.append(steps)
        
        if sim < 5:  # Garde 5 trajectoires exemple
            sample_trajectories.append(traj)
    
    prob_goal = successes / n_simulations
    mean_steps = np.mean(steps_list) if steps_list else 0
    
    return prob_goal, mean_steps, steps_list, sample_trajectories
