"""
experiments.py — Expériences, métriques et visualisations complètes du projet
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from collections import defaultdict
import time
import os

from grid import generate_grid, GRIDS
from astar import astar, ucs, greedy, weighted_astar, manhattan, zero_heuristic, euclidean
from markov import (build_transition_matrix, prob_goal_at_step,
                    monte_carlo_simulation, absorption_analysis,
                    classify_states, compute_pi_n, build_policy)

OUTPUT_DIR = '/mnt/user-data/outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Style global
plt.rcParams.update({
    'figure.dpi': 150,
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})
PALETTE = ['#2E86AB', '#E84855', '#3BB273', '#F4A261', '#9B5DE5', '#00BBF9']


# ═══════════════════════════════════════════════════════════════════
# FIGURE 1 — Visualisation des grilles et chemins A* / UCS / Greedy
# ═══════════════════════════════════════════════════════════════════

def fig1_grid_paths():
    print("  → Figure 1: Grilles et chemins...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Figure 1 — Chemins trouvés sur grilles (UCS, Greedy, A*)', fontsize=15, fontweight='bold', y=1.01)
    
    grid_cfg = GRIDS['medium']
    grid = generate_grid(*grid_cfg['size'], grid_cfg['obstacle_rate'], grid_cfg['seed'])
    start, goal = grid_cfg['start'], grid_cfg['goal']
    
    algos = [
        ('UCS  (f = g)', ucs(grid, start, goal), '#E84855'),
        ('Greedy  (f = h)', greedy(grid, start, goal), '#3BB273'),
        ('A*  (f = g + h)', astar(grid, start, goal), '#2E86AB'),
    ]
    
    for ax, (title, result, color) in zip(axes, algos):
        # Fond: obstacles en noir, libre en blanc
        display = np.ones((*grid.shape, 3))
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r][c] == 1:
                    display[r, c] = [0.15, 0.15, 0.15]
        
        ax.imshow(display, origin='upper')
        
        # Chemin
        if result['path']:
            path = result['path']
            ys = [p[1] for p in path]
            xs = [p[0] for p in path]
            ax.plot(ys, xs, '-', color=color, linewidth=2.5, alpha=0.9, zorder=3)
            ax.plot(ys[0], xs[0], 's', color='gold', markersize=10, zorder=5, label='Start')
            ax.plot(ys[-1], xs[-1], '*', color='lime', markersize=14, zorder=5, label='Goal')
        
        ax.set_title(f'{title}\nCoût={result["cost"]:.0f} | Nœuds={result["nodes_expanded"]}', fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=8, loc='upper right')
        
        # Annotation métriques
        ax.text(0.02, 0.02, f'Temps: {result["time_sec"]*1000:.1f}ms\nMém: {result["memory_kb"]:.0f}KB',
                transform=ax.transAxes, fontsize=8, va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    path_out = f'{OUTPUT_DIR}/fig1_grid_paths.png'
    plt.savefig(path_out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"    Sauvegardée: {path_out}")
    return path_out


# ═══════════════════════════════════════════════════════════════════
# FIGURE 2 — Comparaison métriques des algorithmes (3 grilles)
# ═══════════════════════════════════════════════════════════════════

def fig2_algorithm_comparison():
    print("  → Figure 2: Comparaison algorithmes...")
    
    grid_names = ['easy', 'medium', 'hard']
    algo_names = ['UCS', 'Greedy', 'A*', 'W-A*(w=1.5)', 'W-A*(w=2.0)']
    metrics = ['nodes_expanded', 'cost', 'time_sec', 'memory_kb']
    metric_labels = ['Nœuds développés', 'Coût du chemin', 'Temps (ms)', 'Mémoire max OPEN (KB)']
    
    results = defaultdict(lambda: defaultdict(dict))
    
    for gname in grid_names:
        cfg = GRIDS[gname]
        grid = generate_grid(*cfg['size'], cfg['obstacle_rate'], cfg['seed'])
        start, goal = cfg['start'], cfg['goal']
        
        results[gname]['UCS'] = ucs(grid, start, goal)
        results[gname]['Greedy'] = greedy(grid, start, goal)
        results[gname]['A*'] = astar(grid, start, goal)
        results[gname]['W-A*(w=1.5)'] = weighted_astar(grid, start, goal, weight=1.5)
        results[gname]['W-A*(w=2.0)'] = weighted_astar(grid, start, goal, weight=2.0)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Figure 2 — Comparaison des algorithmes de recherche (3 niveaux de difficulté)',
                 fontsize=14, fontweight='bold')
    axes = axes.flatten()
    
    x = np.arange(len(grid_names))
    width = 0.15
    
    for ax_idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[ax_idx]
        
        for i, algo in enumerate(algo_names):
            vals = []
            for gname in grid_names:
                v = results[gname][algo].get(metric, 0)
                if metric == 'time_sec':
                    v *= 1000  # → ms
                vals.append(v)
            
            bars = ax.bar(x + i*width - 2*width, vals, width, label=algo,
                          color=PALETTE[i], alpha=0.85, edgecolor='white', linewidth=0.5)
        
        ax.set_title(label, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Facile\n(10×10)', 'Moyenne\n(15×15)', 'Difficile\n(20×20)'])
        ax.legend(fontsize=8, loc='upper left')
        ax.set_ylabel(label)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
    
    plt.tight_layout()
    path_out = f'{OUTPUT_DIR}/fig2_algo_comparison.png'
    plt.savefig(path_out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"    Sauvegardée: {path_out}")
    return path_out, results


# ═══════════════════════════════════════════════════════════════════
# FIGURE 3 — Matrice de transition P (heatmap)
# ═══════════════════════════════════════════════════════════════════

def fig3_transition_matrix():
    print("  → Figure 3: Matrice de transition...")
    
    # Grille petite pour visualisation
    grid = generate_grid(6, 6, 0.15, seed=10)
    start, goal = (0,0), (5,5)
    result = astar(grid, start, goal)
    
    if not result['path']:
        print("    Pas de chemin trouvé, skip")
        return None
    
    P, state_index, all_states = build_transition_matrix(grid, result['path'], goal, epsilon=0.2)
    
    # Affiche uniquement les 20 premiers états pour lisibilité
    n_show = min(20, len(all_states))
    P_show = P[:n_show, :n_show]
    labels_show = [str(s) for s in all_states[:n_show]]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Figure 3 — Matrice de transition stochastique P (ε=0.2)', fontsize=14, fontweight='bold')
    
    # Heatmap P
    ax = axes[0]
    cmap = LinearSegmentedColormap.from_list('blue_white', ['white', '#2E86AB'])
    im = ax.imshow(P_show, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(n_show))
    ax.set_yticks(range(n_show))
    ax.set_xticklabels(labels_show, rotation=90, fontsize=6)
    ax.set_yticklabels(labels_show, fontsize=6)
    ax.set_title(f'Sous-matrice P ({n_show}×{n_show})\n(états du chemin A* + GOAL)')
    plt.colorbar(im, ax=ax, label='Probabilité de transition')
    
    # Vérification stochasticité (somme lignes = 1)
    ax2 = axes[1]
    row_sums = P.sum(axis=1)
    ax2.bar(range(len(row_sums)), row_sums, color=PALETTE[0], alpha=0.8, width=0.8)
    ax2.axhline(1.0, color='red', linestyle='--', linewidth=1.5, label='Somme = 1 (stochastique)')
    ax2.set_xlabel('Index état')
    ax2.set_ylabel('Somme de la ligne')
    ax2.set_title(f'Vérification stochasticité\n(Écart max: {np.max(np.abs(row_sums-1)):.2e})')
    ax2.legend()
    ax2.set_ylim(0.95, 1.05)
    ax2.yaxis.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path_out = f'{OUTPUT_DIR}/fig3_transition_matrix.png'
    plt.savefig(path_out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"    Sauvegardée: {path_out}")
    return path_out


# ═══════════════════════════════════════════════════════════════════
# FIGURE 4 — Évolution π(n) et probabilité d'atteinte GOAL
# ═══════════════════════════════════════════════════════════════════

def fig4_markov_evolution():
    print("  → Figure 4: Évolution Markov π(n)...")
    
    cfg = GRIDS['medium']
    grid = generate_grid(*cfg['size'], cfg['obstacle_rate'], cfg['seed'])
    start, goal = cfg['start'], cfg['goal']
    result = astar(grid, start, goal)
    
    if not result['path']:
        return None
    
    epsilons = [0.0, 0.1, 0.2, 0.3]
    max_steps = 80
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Figure 4 — Évolution de la distribution Markov π(n)', fontsize=14, fontweight='bold')
    
    # Gauche: P(GOAL atteint à t) pour différents ε
    ax = axes[0]
    for eps, color in zip(epsilons, PALETTE):
        P, state_index, all_states = build_transition_matrix(grid, result['path'], goal, epsilon=eps)
        probs = prob_goal_at_step(P, state_index, start, 'GOAL', max_steps)
        ax.plot(range(max_steps+1), probs, color=color, linewidth=2,
                label=f'ε={eps}', marker='o' if eps==0.0 else None, markersize=3)
    
    ax.set_xlabel('Pas de temps n')
    ax.set_ylabel('P(Xₙ = GOAL)')
    ax.set_title('Probabilité d\'être en GOAL au pas n\n(calcul matriciel P^n)')
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_xlim(0, max_steps)
    ax.set_ylim(0, 1.05)
    
    # Droite: Impact de ε sur la probabilité finale
    ax2 = axes[1]
    eps_range = np.linspace(0, 0.5, 25)
    final_probs = []
    
    for eps in eps_range:
        P, state_index, all_states = build_transition_matrix(grid, result['path'], goal, epsilon=eps)
        probs = prob_goal_at_step(P, state_index, start, 'GOAL', max_steps)
        final_probs.append(probs[-1])
    
    ax2.fill_between(eps_range, final_probs, alpha=0.2, color=PALETTE[0])
    ax2.plot(eps_range, final_probs, color=PALETTE[0], linewidth=2.5)
    ax2.set_xlabel('Niveau d\'incertitude ε')
    ax2.set_ylabel(f'P(GOAL) à t={max_steps}')
    ax2.set_title(f'Impact de ε sur la probabilité d\'atteinte\n(A* fixé, t={max_steps})')
    ax2.yaxis.grid(True, alpha=0.3)
    ax2.set_xlim(0, 0.5)
    ax2.set_ylim(0, 1.05)
    ax2.axvline(x=0.2, color='red', linestyle='--', alpha=0.5, label='ε=0.2 (référence)')
    ax2.legend()
    
    plt.tight_layout()
    path_out = f'{OUTPUT_DIR}/fig4_markov_evolution.png'
    plt.savefig(path_out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"    Sauvegardée: {path_out}")
    return path_out


# ═══════════════════════════════════════════════════════════════════
# FIGURE 5 — Monte-Carlo vs calcul matriciel
# ═══════════════════════════════════════════════════════════════════

def fig5_montecarlo_vs_matrix():
    print("  → Figure 5: Monte-Carlo vs Matriciel...")
    
    cfg = GRIDS['medium']
    grid = generate_grid(*cfg['size'], cfg['obstacle_rate'], cfg['seed'])
    start, goal = cfg['start'], cfg['goal']
    result = astar(grid, start, goal)
    
    if not result['path']:
        return None
    
    epsilons = [0.0, 0.1, 0.2, 0.3]
    N_SIM = 2000
    max_steps = 100
    
    matrix_probs = []
    mc_probs = []
    mc_stds = []
    mc_mean_steps_list = []
    
    for eps in epsilons:
        # Matriciel
        P, state_index, all_states = build_transition_matrix(grid, result['path'], goal, epsilon=eps)
        probs = prob_goal_at_step(P, state_index, start, 'GOAL', max_steps)
        matrix_probs.append(probs[-1])
        
        # Monte-Carlo
        prob_mc, mean_steps, steps_dist, _ = monte_carlo_simulation(
            grid, result['path'], goal, epsilon=eps, n_simulations=N_SIM, max_steps=max_steps)
        mc_probs.append(prob_mc)
        mc_stds.append(np.std([1 if s < max_steps else 0 for s in steps_dist + [max_steps]*(N_SIM-len(steps_dist))]) / np.sqrt(N_SIM))
        mc_mean_steps_list.append(mean_steps)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Figure 5 — Validation: Calcul Matriciel vs Simulation Monte-Carlo (N=2000)',
                 fontsize=14, fontweight='bold')
    
    # Comparaison probabilités
    ax = axes[0]
    x = np.arange(len(epsilons))
    w = 0.35
    ax.bar(x - w/2, matrix_probs, w, label='Matriciel (P^n)', color=PALETTE[0], alpha=0.85)
    ax.bar(x + w/2, mc_probs, w, label='Monte-Carlo', color=PALETTE[1], alpha=0.85,
           yerr=np.array(mc_stds)*2, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([f'ε={e}' for e in epsilons])
    ax.set_ylabel(f'P(GOAL) à t={max_steps}')
    ax.set_title('Probabilité d\'atteinte GOAL\nMatriciel vs Monte-Carlo')
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Erreur relative
    ax2 = axes[1]
    errors = [abs(m-mc)*100 for m, mc in zip(matrix_probs, mc_probs)]
    bars = ax2.bar([f'ε={e}' for e in epsilons], errors, color=PALETTE[2], alpha=0.85)
    ax2.axhline(5, color='red', linestyle='--', linewidth=1.5, label='Seuil 5%')
    ax2.set_ylabel('Erreur absolue (%)')
    ax2.set_title('Écart Matriciel - Monte-Carlo\n(validation croisée)')
    ax2.legend()
    ax2.yaxis.grid(True, alpha=0.3)
    for bar, err in zip(bars, errors):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1, f'{err:.2f}%',
                ha='center', va='bottom', fontsize=9)
    
    # Distribution des temps d'atteinte (ε=0.2)
    ax3 = axes[2]
    eps_ref = 0.2
    _, _, steps_dist, _ = monte_carlo_simulation(
        grid, result['path'], goal, epsilon=eps_ref, n_simulations=N_SIM, max_steps=max_steps)
    
    if steps_dist:
        ax3.hist(steps_dist, bins=30, color=PALETTE[3], alpha=0.8, edgecolor='white', density=True)
        mean_s = np.mean(steps_dist)
        ax3.axvline(mean_s, color='red', linestyle='--', linewidth=2, label=f'Moyenne={mean_s:.1f}')
        ax3.axvline(len(result['path'])-1, color='blue', linestyle='--', linewidth=2,
                    label=f'A* prévu={len(result["path"])-1}')
        ax3.set_xlabel('Nombre de pas')
        ax3.set_ylabel('Densité')
        ax3.set_title(f'Distribution du temps d\'atteinte\n(ε={eps_ref}, N={N_SIM})')
        ax3.legend(fontsize=8)
        ax3.yaxis.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path_out = f'{OUTPUT_DIR}/fig5_montecarlo_validation.png'
    plt.savefig(path_out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"    Sauvegardée: {path_out}")
    return path_out, matrix_probs, mc_probs, errors


# ═══════════════════════════════════════════════════════════════════
# FIGURE 6 — Analyse des classes Markov et heatmap d'absorption
# ═══════════════════════════════════════════════════════════════════

def fig6_markov_analysis():
    print("  → Figure 6: Analyse Markov (classes, absorption)...")
    
    cfg = GRIDS['easy']
    grid = generate_grid(*cfg['size'], cfg['obstacle_rate'], cfg['seed'])
    start, goal = cfg['start'], cfg['goal']
    result = astar(grid, start, goal)
    
    if not result['path']:
        return None
    
    eps = 0.2
    P, state_index, all_states = build_transition_matrix(grid, result['path'], goal, epsilon=eps)
    
    # Classification
    classif = classify_states(P, all_states)
    
    # Comptage
    counts = defaultdict(int)
    for v in classif.values():
        counts[v] += 1
    
    # Absorption
    N_mat, B_mat, t_mean, trans_idx = absorption_analysis(P, state_index, all_states, ('GOAL',))
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Figure 6 — Analyse de la Chaîne de Markov : Classes et Absorption',
                 fontsize=14, fontweight='bold')
    
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
    
    # Pie: répartition états
    ax1 = fig.add_subplot(gs[0, 0])
    labels_pie = list(counts.keys())
    sizes_pie = list(counts.values())
    colors_pie = [PALETTE[0], PALETTE[1], PALETTE[2]][:len(labels_pie)]
    wedges, texts, autotexts = ax1.pie(sizes_pie, labels=labels_pie, colors=colors_pie,
                                        autopct='%1.0f%%', startangle=90, pctdistance=0.75)
    ax1.set_title(f'Classification des états\n({len(all_states)} états total, ε={eps})')
    
    # Heatmap grille: colorer par classification
    ax2 = fig.add_subplot(gs[0, 1:])
    rows, cols = grid.shape
    color_map = np.zeros((rows, cols, 3))
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                color_map[r, c] = [0.1, 0.1, 0.1]  # Obstacle
            else:
                state = (r, c)
                cl = classif.get(state, 'inconnu')
                if cl == 'absorbant':
                    color_map[r, c] = [0.18, 0.75, 0.45]  # vert
                elif cl == 'transitoire':
                    color_map[r, c] = [0.17, 0.53, 0.67]  # bleu
                else:
                    color_map[r, c] = [0.9, 0.6, 0.2]  # orange
    
    ax2.imshow(color_map, origin='upper')
    
    if result['path']:
        ys = [p[1] for p in result['path']]
        xs = [p[0] for p in result['path']]
        ax2.plot(ys, xs, 'w--', linewidth=1.5, alpha=0.7, label='Chemin A*')
        ax2.plot(ys[0], xs[0], 's', color='gold', markersize=10, zorder=5)
        ax2.plot(ys[-1], xs[-1], '*', color='white', markersize=14, zorder=5)
    
    legend_patches = [
        mpatches.Patch(color=[0.18, 0.75, 0.45], label='Absorbant (GOAL)'),
        mpatches.Patch(color=[0.17, 0.53, 0.67], label='Transitoire'),
        mpatches.Patch(color=[0.9, 0.6, 0.2], label='Récurrent'),
        mpatches.Patch(color=[0.1, 0.1, 0.1], label='Obstacle'),
    ]
    ax2.legend(handles=legend_patches, loc='upper right', fontsize=8)
    ax2.set_title(f'Carte des classes d\'états\n(grille {rows}×{cols}, ε={eps})', fontsize=11)
    ax2.set_xticks([]); ax2.set_yticks([])
    
    # Temps moyen absorption
    if t_mean is not None and B_mat is not None:
        ax3 = fig.add_subplot(gs[1, 0])
        path_states_idx = [state_index[s] for s in result['path'][:-1] if s in state_index]
        
        if path_states_idx and trans_idx:
            local_means = []
            for pidx in path_states_idx:
                if pidx in trans_idx:
                    loc = trans_idx.index(pidx)
                    local_means.append(t_mean[loc])
                else:
                    local_means.append(0)
            
            step_labels = range(len(local_means))
            ax3.plot(step_labels, local_means, color=PALETTE[3], linewidth=2, marker='o', markersize=4)
            ax3.fill_between(step_labels, local_means, alpha=0.2, color=PALETTE[3])
            ax3.set_xlabel('Position sur le chemin A*')
            ax3.set_ylabel('Temps moyen (pas)')
            ax3.set_title('Temps moyen avant absorption\n(depuis chaque état du chemin)')
            ax3.yaxis.grid(True, alpha=0.3)
        
        # Prob absorption vs position
        ax4 = fig.add_subplot(gs[1, 1])
        if path_states_idx and trans_idx and B_mat is not None:
            probs_abs = []
            for pidx in path_states_idx:
                if pidx in trans_idx:
                    loc = trans_idx.index(pidx)
                    probs_abs.append(B_mat[loc, 0])  # prob vers GOAL
                else:
                    probs_abs.append(1.0 if pidx == state_index.get(goal) else 0.0)
            
            ax4.bar(range(len(probs_abs)), probs_abs, color=PALETTE[0], alpha=0.8)
            ax4.set_xlabel('Position sur le chemin A*')
            ax4.set_ylabel('P(absorption → GOAL)')
            ax4.set_title('Probabilité d\'atteindre GOAL\n(depuis chaque état du chemin)')
            ax4.yaxis.grid(True, alpha=0.3)
            ax4.set_ylim(0, 1.1)
    
    # Matrice fondamentale (diagonale = temps moyen)
    ax5 = fig.add_subplot(gs[1, 2])
    if N_mat is not None:
        n_show = min(15, N_mat.shape[0])
        sns.heatmap(N_mat[:n_show, :n_show], ax=ax5, cmap='Blues',
                    cbar_kws={'label': 'E[visites]'}, linewidths=0.1, linecolor='gray')
        ax5.set_title(f'Matrice fondamentale N=(I-Q)⁻¹\n(sous-matrice {n_show}×{n_show})')
        ax5.set_xticklabels([])
        ax5.set_yticklabels([])
    
    path_out = f'{OUTPUT_DIR}/fig6_markov_analysis.png'
    plt.savefig(path_out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"    Sauvegardée: {path_out}")
    return path_out


# ═══════════════════════════════════════════════════════════════════
# FIGURE 7 — Impact ε sur toutes les métriques (tableau de bord)
# ═══════════════════════════════════════════════════════════════════

def fig7_epsilon_dashboard():
    print("  → Figure 7: Tableau de bord ε...")
    
    cfg = GRIDS['medium']
    grid = generate_grid(*cfg['size'], cfg['obstacle_rate'], cfg['seed'])
    start, goal = cfg['start'], cfg['goal']
    result_astar = astar(grid, start, goal)
    
    if not result_astar['path']:
        return None
    
    epsilons = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    N_SIM = 1500
    max_steps = 120
    
    matrix_final = []
    mc_final = []
    mc_mean_steps = []
    mc_success_rate = []
    
    for eps in epsilons:
        P, state_index, all_states = build_transition_matrix(grid, result_astar['path'], goal, epsilon=eps)
        probs = prob_goal_at_step(P, state_index, start, 'GOAL', max_steps)
        matrix_final.append(probs[-1])
        
        prob_mc, mean_s, steps_dist, _ = monte_carlo_simulation(
            grid, result_astar['path'], goal, epsilon=eps,
            n_simulations=N_SIM, max_steps=max_steps)
        mc_final.append(prob_mc)
        mc_mean_steps.append(mean_s if mean_s > 0 else max_steps)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('Figure 7 — Impact de l\'incertitude ε sur la robustesse du plan A*',
                 fontsize=14, fontweight='bold')
    
    # P(GOAL) vs ε
    ax = axes[0, 0]
    ax.plot(epsilons, matrix_final, 'o-', color=PALETTE[0], linewidth=2.5, label='Matriciel P^n', markersize=6)
    ax.plot(epsilons, mc_final, 's--', color=PALETTE[1], linewidth=2, label='Monte-Carlo', markersize=6)
    ax.fill_between(epsilons, matrix_final, mc_final, alpha=0.1, color='gray')
    ax.set_xlabel('ε (niveau d\'incertitude)')
    ax.set_ylabel(f'P(GOAL) à t={max_steps}')
    ax.set_title('Probabilité d\'atteinte GOAL vs ε')
    ax.legend()
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # Dégradation relative
    ax2 = axes[0, 1]
    ref = matrix_final[0] if matrix_final[0] > 0 else 1
    degradation = [(ref - v)/ref * 100 for v in matrix_final]
    ax2.bar(range(len(epsilons)), degradation, color=[PALETTE[1] if d > 20 else PALETTE[0] for d in degradation],
            alpha=0.85)
    ax2.set_xticks(range(len(epsilons)))
    ax2.set_xticklabels([f'{e}' for e in epsilons], rotation=45)
    ax2.set_xlabel('ε')
    ax2.set_ylabel('Dégradation (%)')
    ax2.set_title('Dégradation de P(GOAL) par rapport à ε=0\n(plan déterministe A*)')
    ax2.axhline(20, color='red', linestyle='--', linewidth=1, label='Seuil 20%')
    ax2.legend()
    ax2.yaxis.grid(True, alpha=0.3)
    
    # Temps moyen d'atteinte
    ax3 = axes[1, 0]
    valid_mc = [(e, s) for e, s in zip(epsilons, mc_mean_steps) if s < max_steps]
    if valid_mc:
        eps_v, steps_v = zip(*valid_mc)
        ax3.plot(eps_v, steps_v, 'D-', color=PALETTE[2], linewidth=2.5, markersize=7)
        ax3.fill_between(eps_v, steps_v, alpha=0.15, color=PALETTE[2])
    ax3.axhline(len(result_astar['path'])-1, color='blue', linestyle='--',
                linewidth=2, label=f'Longueur A*={len(result_astar["path"])-1}')
    ax3.set_xlabel('ε')
    ax3.set_ylabel('Pas moyens pour atteindre GOAL')
    ax3.set_title('Temps moyen d\'atteinte\n(simulations Monte-Carlo réussies)')
    ax3.legend()
    ax3.yaxis.grid(True, alpha=0.3)
    
    # Radar / tableau récapitulatif
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    table_data = []
    headers = ['ε', 'P(GOAL)\nMatriciel', 'P(GOAL)\nMC', 'Écart\n(%)', 'Tps moy.\n(MC)']
    
    eps_show = [0.0, 0.1, 0.2, 0.3, 0.5]
    eps_indices = [epsilons.index(e) for e in eps_show if e in epsilons]
    
    for idx in eps_indices:
        e = epsilons[idx]
        err = abs(matrix_final[idx]-mc_final[idx])*100
        row = [f'{e}', f'{matrix_final[idx]:.3f}', f'{mc_final[idx]:.3f}',
               f'{err:.2f}', f'{mc_mean_steps[idx]:.1f}']
        table_data.append(row)
    
    table = ax4.table(cellText=table_data, colLabels=headers,
                      cellLoc='center', loc='center',
                      bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2E86AB')
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#EEF4FB')
        cell.set_edgecolor('#CCCCCC')
    
    ax4.set_title('Tableau récapitulatif\nImpact de ε', pad=10, fontweight='bold')
    
    plt.tight_layout()
    path_out = f'{OUTPUT_DIR}/fig7_epsilon_dashboard.png'
    plt.savefig(path_out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"    Sauvegardée: {path_out}")
    return path_out


# ═══════════════════════════════════════════════════════════════════
# FIGURE 8 — Trajectoires Monte-Carlo et heuristiques
# ═══════════════════════════════════════════════════════════════════

def fig8_trajectories_heuristics():
    print("  → Figure 8: Trajectoires MC et comparaison heuristiques...")
    
    cfg = GRIDS['medium']
    grid = generate_grid(*cfg['size'], cfg['obstacle_rate'], cfg['seed'])
    start, goal = cfg['start'], cfg['goal']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Figure 8 — Trajectoires stochastiques et comparaison d\'heuristiques',
                 fontsize=14, fontweight='bold')
    
    # Trajectoires MC
    ax = axes[0]
    result = astar(grid, start, goal)
    _, _, _, sample_trajs = monte_carlo_simulation(
        grid, result['path'], goal, epsilon=0.2, n_simulations=5, max_steps=200)
    
    display = np.ones((*grid.shape, 3))
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r][c] == 1:
                display[r, c] = [0.15, 0.15, 0.15]
    ax.imshow(display, origin='upper')
    
    # Chemin A*
    if result['path']:
        ys = [p[1] for p in result['path']]
        xs = [p[0] for p in result['path']]
        ax.plot(ys, xs, 'b-', linewidth=3, alpha=0.9, label='Chemin A*', zorder=4)
    
    # Trajectoires MC
    colors_mc = ['#FF6B6B', '#FFA07A', '#FFD700', '#98FB98', '#DDA0DD']
    for traj, col in zip(sample_trajs, colors_mc):
        if len(traj) > 1:
            ys_t = [p[1] for p in traj]
            xs_t = [p[0] for p in traj]
            ax.plot(ys_t, xs_t, '-', color=col, linewidth=1.2, alpha=0.7)
    
    ax.plot(start[1], start[0], 's', color='gold', markersize=12, zorder=5, label='Start')
    ax.plot(goal[1], goal[0], '*', color='lime', markersize=16, zorder=5, label='Goal')
    ax.set_title('5 trajectoires stochastiques\nvs chemin A* (ε=0.2)', fontsize=11)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xticks([]); ax.set_yticks([])
    
    # Comparaison heuristiques: nœuds développés
    ax2 = axes[1]
    heuristics = {
        'h=0 (UCS)': zero_heuristic,
        'Manhattan': manhattan,
        'Euclidienne': euclidean,
    }
    grids_diff = ['easy', 'medium', 'hard']
    
    x = np.arange(len(grids_diff))
    w = 0.25
    
    for i, (hname, hfunc) in enumerate(heuristics.items()):
        nodes = []
        for gname in grids_diff:
            cfg2 = GRIDS[gname]
            g2 = generate_grid(*cfg2['size'], cfg2['obstacle_rate'], cfg2['seed'])
            res = search_with_h(g2, cfg2['start'], cfg2['goal'], hfunc)
            nodes.append(res['nodes_expanded'])
        ax2.bar(x + (i-1)*w, nodes, w, label=hname, color=PALETTE[i], alpha=0.85)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Facile', 'Moyenne', 'Difficile'])
    ax2.set_ylabel('Nœuds développés')
    ax2.set_title('Comparaison heuristiques admissibles\n(nœuds développés par A*)')
    ax2.legend(fontsize=9)
    ax2.yaxis.grid(True, alpha=0.3)
    ax2.set_axisbelow(True)
    
    # Weighted A* - compromis vitesse/optimalité
    ax3 = axes[2]
    weights = [1.0, 1.2, 1.5, 2.0, 3.0, 5.0]
    cfg3 = GRIDS['hard']
    g3 = generate_grid(*cfg3['size'], cfg3['obstacle_rate'], cfg3['seed'])
    
    wa_nodes = []
    wa_costs = []
    optimal_cost = astar(g3, cfg3['start'], cfg3['goal'])['cost']
    
    for w_val in weights:
        res = weighted_astar(g3, cfg3['start'], cfg3['goal'], weight=w_val)
        wa_nodes.append(res['nodes_expanded'])
        wa_costs.append(res['cost'])
    
    color1, color2 = PALETTE[0], PALETTE[1]
    ax3b = ax3.twinx()
    
    l1, = ax3.plot(weights, wa_nodes, 'o-', color=color1, linewidth=2.5, markersize=7, label='Nœuds (←)')
    l2, = ax3b.plot(weights, [(c/optimal_cost - 1)*100 for c in wa_costs], 's--',
                     color=color2, linewidth=2, markersize=7, label='Surcoût % (→)')
    
    ax3.set_xlabel('Poids w (Weighted A*)')
    ax3.set_ylabel('Nœuds développés', color=color1)
    ax3b.set_ylabel('Surcoût vs A* optimal (%)', color=color2)
    ax3.set_title('Weighted A* : compromis\nvitesse (nœuds) vs optimalité (coût)')
    ax3.tick_params(axis='y', labelcolor=color1)
    ax3b.tick_params(axis='y', labelcolor=color2)
    lines = [l1, l2]
    ax3.legend(lines, [l.get_label() for l in lines], fontsize=9, loc='center right')
    ax3.yaxis.grid(True, alpha=0.2)
    
    plt.tight_layout()
    path_out = f'{OUTPUT_DIR}/fig8_trajectories_heuristics.png'
    plt.savefig(path_out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"    Sauvegardée: {path_out}")
    return path_out


def search_with_h(grid, start, goal, heuristic):
    """Wrapper pour A* avec heuristique spécifique."""
    from astar import search
    return search(grid, start, goal, mode='astar', heuristic=heuristic)


# ═══════════════════════════════════════════════════════════════════
# TABLEAU RÉCAPITULATIF FINAL
# ═══════════════════════════════════════════════════════════════════

def print_summary_table(results_fig2=None):
    print("\n" + "="*80)
    print("  TABLEAU RÉCAPITULATIF — MÉTRIQUES COMPLÈTES")
    print("="*80)
    
    if results_fig2:
        for gname in ['easy', 'medium', 'hard']:
            print(f"\n  Grille: {gname.upper()} ({GRIDS[gname]['size']})")
            print(f"  {'Algo':<15} {'Coût':>8} {'Nœuds':>10} {'Temps(ms)':>12} {'Mém(KB)':>10}")
            print(f"  {'-'*55}")
            for algo in ['UCS', 'Greedy', 'A*', 'W-A*(w=1.5)', 'W-A*(w=2.0)']:
                r = results_fig2[gname][algo]
                found = '✓' if r['found'] else '✗'
                print(f"  {algo:<15} {r['cost']:>7.0f}  {r['nodes_expanded']:>9}  "
                      f"{r['time_sec']*1000:>10.2f}  {r['memory_kb']:>9.1f} {found}")
    print("="*80)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def run_all():
    print("\n" + "="*65)
    print("  PROJET: Planification Robuste A* + Chaînes de Markov")
    print("="*65)
    print("\nGénération de toutes les figures...\n")
    
    paths = {}
    
    paths['fig1'] = fig1_grid_paths()
    paths['fig2'], results = fig2_algorithm_comparison()
    paths['fig3'] = fig3_transition_matrix()
    paths['fig4'] = fig4_markov_evolution()
    ret5 = fig5_montecarlo_vs_matrix()
    if ret5:
        paths['fig5'], mat_probs, mc_probs, errors = ret5
    paths['fig6'] = fig6_markov_analysis()
    paths['fig7'] = fig7_epsilon_dashboard()
    paths['fig8'] = fig8_trajectories_heuristics()
    
    print_summary_table(results)
    
    print("\n✅ Toutes les figures générées avec succès!")
    print(f"   Dossier: {OUTPUT_DIR}")
    
    return paths



run_all()
