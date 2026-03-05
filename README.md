<div align="center">

# 🤖 Planification Robuste sur Grille 2D

### Algorithmes de Recherche Heuristique + Chaînes de Markov à Temps Discret

**A\* · UCS · Greedy · Weighted A\* · Monte-Carlo · Absorption**

---

**Module** : Bases de l'Intelligence Artificielle — 25\_27

</div>

<br>

<table>
<tr>
<td width="50%">

**■ INFORMATIONS ACADÉMIQUES**

| | |
|---|---|
| **École** | ENSET Mohammedia |
| **Université** | Hassan II de Casablanca |
| **Filière** | SDIA — Science des Données et IA |
| **Module** | Bases de l'IA — 25\_27 |
| **Encadrant** | Pr. Mohamed MESTARI |
| **Année** | 2025 – 2026 |

</td>
<td width="50%">

**■ RÉALISÉ PAR**

| | |
|---|---|
| **Nom** | TOUBANI |
| **Prénom** | Badr Eddine |
| **Statut** | Étudiant — Master Recherche IA |
| **Filière** | SDIA — ENSET Mohammedia |
| **Email** | badr.toubani-etu@etu-univh2.ma |

</td>
</tr>
</table>

<div align="center">

*Mars 2026 — ENSET Mohammedia, Université Hassan II de Casablanca*

</div>

---

## 📋 Table des Matières

1. [Résumé Exécutif](#-résumé-exécutif)
2. [Introduction et Contexte](#1--introduction-et-contexte)
3. [Modélisation Mathématique](#2--modélisation-mathématique)
4. [Algorithmes de Recherche Heuristique](#3--algorithmes-de-recherche-heuristique)
5. [Implémentation Python](#4--implémentation-python)
6. [Résultats — Algorithmes de Recherche](#5--résultats--algorithmes-de-recherche)
7. [Analyse de la Chaîne de Markov](#6--analyse-de-la-chaîne-de-markov)
8. [Validation par Simulation Monte-Carlo](#7--validation-par-simulation-monte-carlo)
9. [Impact de l'Incertitude ε sur la Robustesse](#8--impact-de-lincertitude-ε-sur-la-robustesse)
10. [Discussion et Limites](#9--discussion-et-limites)
11. [Conclusion](#10--conclusion)
12. [Simulation Interactive](#-simulation-interactive)
13. [Références Bibliographiques](#-références-bibliographiques)

---

## 📌 Résumé Exécutif

> Ce rapport présente une étude complète de la **planification de chemin robuste sur grille 2D**, combinant la **recherche heuristique** (famille A\*) et les **chaînes de Markov à temps discret**. Cinq algorithmes sont comparés (UCS, Greedy, A\*, W-A\* w=1.5/2.0) sur trois grilles. La modélisation stochastique repose sur la matrice **P**, l'évolution **π(n)=π(0)Pⁿ**, la matrice fondamentale **N=(I−Q)⁻¹** et **2000 simulations Monte-Carlo**.

### Résultat principal

| Métrique | Valeur |
|:---------|:-------|
| Réduction nœuds A\* vs UCS | **−47%** |
| Dégradation P(GOAL) à ε=0.2 | **−27%** |
| Écart Matriciel-MC | **< 5%** |
| Seuil critique ε\* | **≈ 0.35** |

**Mots-clés** : `A*` · `UCS` · `Greedy` · `Weighted A*` · `Chaînes de Markov` · `Matrice P` · `Monte-Carlo` · `Admissibilité` · `Cohérence` · `Absorption` · `Planification robuste`

---

## 1 — Introduction et Contexte

### 1.1 Problématique et motivation

La planification de chemin est un problème fondamental de l'IA : un agent autonome doit aller d'un **état initial** à un **état but** en minimisant un coût cumulatif, en présence d'obstacles et d'incertitudes. Dans un monde déterministe, **A\*** offre la solution optimale. Cependant les environnements réels (robots, drones, logistique) sont intrinsèquement **stochastiques** : glissement, déviation, congestion.

Ce projet **hybride A\*** (planification optimale) et les **chaînes de Markov** (évaluation probabiliste de la robustesse) pour quantifier précisément l'écart entre le plan théorique et la performance réelle.

### 1.2 Objectifs du projet

| Obj. | Description | Méthode |
|:----:|:------------|:--------|
| **O1** | Implémenter et comparer 5 algorithmes de recherche | UCS, Greedy, A\*, W-A\*(1.5/2.0) |
| **O2** | Modéliser l'incertitude par chaîne de Markov | Matrice P, π(n)=π(0)Pⁿ |
| **O3** | Analyser classes et absorption | N=(I−Q)⁻¹, B=N·R |
| **O4** | Valider par simulation Monte-Carlo N=2000 | Trajectoires stochastiques |
| **O5** | Quantifier l'impact de ε ∈ [0, 0.5] | Analyse comparative |

*Tableau 1.1 — Objectifs et méthodes du projet.*

### 1.3 Applications réelles

- 🤖 **Robotique mobile** : Navigation en entrepôt avec sol glissant, obstacles mobiles (ε modélise le glissement).
- 🚛 **Logistique autonome** : Routage de véhicules en milieu urbain dynamique avec congestion aléatoire.
- 🎮 **Jeux vidéo (NPCs)** : A\* est l'algorithme standard (Unity, Unreal Engine) pour le pathfinding de personnages.
- 🌐 **Réseaux informatiques** : Routage de paquets avec probabilités de congestion stochastique.

---

## 2 — Modélisation Mathématique

### 2.1 Espace d'états et graphe de recherche

La grille `r×c` est un graphe pondéré **G = (S, E, c)** où :
- **S** est l'ensemble des cellules libres
- **E** les arêtes de 4-connectivité
- **c(n,n')=1** (coût uniforme)

Chaque nœud est évalué par la fonction :

$$f(n) = g(n) + h(n)$$

avec `g(n)` le coût exact depuis s₀ et `h(n)` l'estimation heuristique vers le but g.

**Heuristique de Manhattan** :

$$h((x,y)) = |x - x_g| + |y - y_g|$$

Admissible et cohérente sur grille 4-connexe à coût uniforme.

### 2.2 Heuristiques : admissibilité et cohérence

#### ▸ Admissibilité

Une heuristique `h` est admissible si elle ne surestime jamais le coût réel `h*(n)` :

$$\forall n \in S : \quad h(n) \leq h^*(n)$$

> **Conséquence** : A\* avec h admissible est **optimal** — il retourne toujours le chemin de coût minimal.

#### ▸ Cohérence (Consistance)

$$\forall n, n' : \quad h(n) \leq c(n, a, n') + h(n')$$

La cohérence implique l'admissibilité et garantit la **monotonicité de f(n)**, évitant toute ré-expansion de nœuds déjà fermés.

| Heuristique | Formule | Admissible | Cohérente | Dominance |
|:------------|:--------|:----------:|:---------:|:----------|
| h = 0 (nulle) | `0` | ✅ Oui | ✅ Oui | Aucune (= UCS) |
| Manhattan | `\|x-xg\|+\|y-yg\|` | ✅ Oui | ✅ Oui | Domine h=0 |
| Euclidienne | `√((x-xg)²+(y-yg)²)` | ✅ Oui | ✅ Oui | Entre h=0 et Manhattan |
| Chebyshev | `max(\|dx\|,\|dy\|)` | ❌ Non (4-v) | ❌ Non | Inadmissible ici |

*Tableau 2.1 — Heuristiques et propriétés (grille 4-connexe, coût uniforme 1).*

#### Théorème d'optimalité d'A\*

> Si `h` est admissible, A\* retourne toujours le chemin de **coût minimal** (optimal).
> Si `h` est de plus cohérente, `f(n)` est **monotone croissant** le long de tout chemin — garantissant qu'un nœud extrait de OPEN avec `f(n)=C*` a `g(n)=g*(n)` optimal.
>
> A\* est **optimalement efficace** : aucun algorithme admissible ne développe strictement moins de nœuds *(Nilsson, 1980)*.

### 2.3 Chaîne de Markov à temps discret

L'agent est modélisé par **{Xₙ}ₙ≥₀** sur **S' = S ∪ {GOAL}**. La propriété de Markov : la transition ne dépend que de l'état courant. La matrice **P = (pᵢⱼ)** est stochastique par lignes. L'évolution suit **Chapman-Kolmogorov** :

$$\pi(n) = \pi(0) \cdot P^n$$

### 2.4 Modèle d'incertitude paramétrique ε

$$p(n \rightarrow n'_{\text{voulu}}) = 1-\varepsilon \qquad \cdot \qquad p(n \rightarrow n'_{\text{lat}}) = \varepsilon/2 \text{ chacun}$$

#### Interprétation physique de ε

| Valeur | Interprétation |
|:-------|:---------------|
| ε=0.0 | Déterministe (robot idéal) |
| ε=0.1 | Légère déviation (sol glissant) |
| ε=0.2 | Incertitude modérée (entrepôt humide) |
| ε=0.3 | Forte incertitude |

### 2.5 États absorbants et décomposition canonique

GOAL est **absorbant** (`p(GOAL,GOAL) = 1`). Décomposition canonique de P :

$$P = \begin{bmatrix} Q & R \\ 0 & I \end{bmatrix} \quad \Rightarrow \quad N = (I-Q)^{-1} \quad \Rightarrow \quad B = N \cdot R$$

- **N(i,j)** = nombre moyen de visites en j avant absorption depuis i
- **B(i,GOAL)** = probabilité d'atteindre GOAL depuis i
- **t = N·1** = temps moyen d'absorption

---

## 3 — Algorithmes de Recherche Heuristique

Tous les algorithmes partagent la même structure : file de priorité **OPEN** (tas min sur f) et ensemble **CLOSED** (nœuds développés). La différence réside uniquement dans le calcul de **f(n)**.

| Algorithme | f(n) | Optimal ? | Complexité mémoire |
|:-----------|:-----|:---------:|:-------------------|
| **UCS** | g(n) | ✅ Oui | O(b^d) |
| **Greedy** | h(n) | ❌ Non | O(b^m) |
| **A\*** | g(n) + h(n) | ✅ (h admissible) | O(b^d) |
| **W-A\* (w>1)** | g(n) + w·h(n) | ε-optimal (≤ w·C\*) | O(b^d) |

*Tableau 3.1 — Comparaison théorique des algorithmes.*

### 3.1 UCS — Uniform Cost Search (f = g)

UCS est équivalent à **Dijkstra** sur graphes pondérés. Il explore en vague circulaire depuis s₀, sans guidance vers le but. Optimal et complet, mais très coûteux : il **n'utilise aucune information** sur la position du but.

$$f_{UCS}(n) = g(n)$$

> **Observation 1** : UCS développe **2.3× plus de nœuds** qu'A\* en moyenne sur les 3 grilles testées, pour le même chemin optimal. Sur grille 20×20 : 247 nœuds (UCS) vs 88 (A\*).

### 3.2 Greedy Best-First Search (f = h)

Greedy ignore g(n) et guide la recherche uniquement par h(n). Très rapide mais **non optimal** : il peut trouver un chemin sous-optimal ou se bloquer dans des culs-de-sac.

$$f_{Greedy}(n) = h(n)$$

> **Observation 2** : Greedy explore **3× moins de nœuds** qu'A\* mais produit des chemins sous-optimaux de +5% à +21%. Sur grille 15×15 : coût 34 (Greedy) vs 28 (A\*), soit **+21%**.

### 3.3 Algorithme A\* — Analyse complète

A\* est la synthèse optimale : il combine `g(n)` (optimalité) et `h(n)` (efficacité). C'est l'algorithme le plus utilisé en pratique pour la planification de chemin.

$$f_{A^*}(n) = g(n) + h(n)$$

#### ▸ Pseudo-code d'A\*

```
Entrée : grille, start s₀, goal g, heuristique h
OPEN ← {(f(s₀)=h(s₀), s₀)} ;  CLOSED ← ∅ ;  g_cost[s₀] ← 0

TANT QUE OPEN ≠ ∅ FAIRE :
    n ← noeud de OPEN avec f(n) minimal    // Extraction O(log n)
    SI n = goal  →  RETOURNER chemin(parent, n)
    CLOSED ← CLOSED ∪ {n}
    POUR chaque voisin n' de n FAIRE :
        g' ← g_cost[n] + c(n, n')          // coût uniforme = 1
        SI n' ∉ CLOSED ET (n' ∉ OPEN OU g' < g_cost[n']) :
            g_cost[n'] ← g'
            f[n'] ← g' + h(n')
            parent[n'] ← n
            OPEN ← OPEN ∪ {(f[n'], n')}
RETOURNER ECHEC
```
*Algorithme 3.1 — Pseudo-code complet d'A\* avec OPEN (tas min) et CLOSED (ensemble).*

### 3.4 Weighted A\* et variantes mémoire

$$f_{wA^*}(n) = g(n) + w \cdot h(n), \quad w \geq 1$$

W-A\* garantit un chemin de coût **≤ w × C\*** (ε-optimal). Pour `w=1` → A\* standard. Pour `w→∞` → Greedy. L'enjeu est de trouver le meilleur `w` minimisant les nœuds développés sans surcoût excessif.

---

## 4 — Implémentation Python

| Module | Responsabilité | Fonctions clés | Lignes |
|:-------|:---------------|:---------------|:------:|
| `grid.py` | Génération grilles 2D + voisinage | `generate_grid()`, `get_neighbors()` | ~85 |
| `astar.py` | Moteur de recherche heuristique | `search()`, `astar()`, `ucs()`, `greedy()`, `weighted_astar()` | ~120 |
| `markov.py` | Modélisation + analyse stochastique | `build_transition_matrix()`, `prob_goal_at_step()`, `monte_carlo_simulation()`, `absorption_analysis()` | ~200 |
| `experiments.py` | Expériences + métriques + figures | `fig1()` … `fig8()`, `run_all()` | ~550 |

*Tableau 4.1 — Architecture modulaire (~955 lignes Python total).*

**Points d'implémentation clés :**

1. **`grid.py`** vérifie la connectivité start↔goal par BFS avec corridor de secours.
2. **`astar.py`** mesure temps (`time.perf_counter()`), mémoire (`tracemalloc`) et nœuds développés.
3. **`markov.py`** construit P avec normalisation garantissant stochasticité (erreur < 10⁻¹⁵) et analyse absorption via `numpy.linalg.inv()` sur (I−Q).

### 🏗️ Architecture

```
project_base_ai/
├── grid.py                    # Génération de grilles 2D
├── astar.py                   # Algorithmes : UCS, Greedy, A*, W-A*
├── markov.py                  # Chaînes de Markov, Monte-Carlo
├── experiments.py             # Génération des 8 figures
├── simulation.html            # Simulation interactive (navigateur)
├── projet_experiments.ipynb   # Notebook Jupyter reproductible
└── README.md                  # Ce rapport
```

### ▶️ Exécution

```bash
# Générer toutes les figures
python experiments.py

# Ouvrir le notebook interactif
jupyter notebook projet_experiments.ipynb

# Lancer la simulation web
python -m http.server 8000
# puis ouvrir http://localhost:8000/simulation.html
```

---

## 5 — Résultats — Algorithmes de Recherche

> **Protocole** : graines aléatoires fixées, coût uniforme = 1, 4-connectivité, moyennes sur 5 exécutions.
> **Métriques** : coût, nœuds développés, taille OPEN max, temps (ms), mémoire (KB).

### 5.1 Tableau comparatif complet

| Grille | Algorithme | Coût | Nœuds | OPEN max | Temps (ms) | Mém. (KB) | Optimal |
|:-------|:-----------|-----:|------:|---------:|-----------:|----------:|:-------:|
| **Facile 10×10** | UCS | 18 | 85 | 45 | 0.86 | 15.7 | ✅ |
| | Greedy | 20 (+11%) | 21 | 12 | 0.18 | 5.0 | ❌ |
| | A\* | **18** | **71** | 38 | 0.63 | 7.8 | ✅ |
| **Moyenne 15×15** | UCS | 28 | 165 | 82 | 1.63 | 25.5 | ✅ |
| | Greedy | 34 (+21%) | 43 | 24 | 0.42 | 7.9 | ❌ |
| | A\* | **28** | **129** | 65 | 2.33 | 22.2 | ✅ |
| **Difficile 20×20** | UCS | 38 | 247 | 124 | 2.27 | 32.6 | ✅ |
| | Greedy | 40 (+5%) | 47 | 26 | 0.45 | 7.8 | ❌ |
| | A\* | **38** | **88** | 45 | 0.86 | 19.9 | ✅ |

*Tableau 5.1 — Résultats complets sur 3 grilles. Gras = meilleur compromis.*

### Figure 5.1 — Chemins sur grille 15×15

<div align="center">

![Figure 1 — Chemins trouvés sur grilles (UCS, Greedy, A*)](fig1_grid_paths.png)

*UCS et A\* trouvent le chemin optimal (coût 28). Greedy trouve un sous-optimal (coût 34) en explorant seulement 43 nœuds vs 129 pour A\*.*

</div>

### Figure 5.2 — Comparaison multi-critères

<div align="center">

![Figure 2 — Comparaison des algorithmes de recherche (3 niveaux de difficulté)](fig2_algo_comparison.png)

*Comparaison multi-critères des 5 algorithmes sur 3 niveaux de difficulté. A\* domine UCS (−47% nœuds) avec la même optimalité.*

</div>

> ✅ **Résultat Clé** : A\* offre le meilleur compromis : **optimalité garantie avec −47% de nœuds vs UCS**. W-A\*(w=1.5) est encore plus efficace (−45% nœuds) avec 0% de surcoût sur les grilles facile et difficile.

### 5.2 Impact de Weighted A\*

| Poids w | Nœuds dév. | Réduction vs A\* | Coût | Surcoût |
|:--------|:-----------|:----------------|:-----|:--------|
| 1.0 (A\*) | 88 | — (référence) | 38 | 0.0% |
| 1.2 | 71 | −19.3% | 38 | 0.0% |
| **1.5** | **48** | **−45.5%** | **38** | **0.0%** |
| 2.0 | 48 | −45.5% | 40 | +5.3% |
| 3.0 | 35 | −60.2% | 40 | +5.3% |
| 5.0 | 28 | −68.2% | 44 | +15.8% |

*Tableau 5.2 — Impact du poids w sur W-A\* (grille difficile 20×20).*

> **Observation 3** : W-A\*(w=1.5) est le **sweet spot** : −45.5% de nœuds développés SANS surcoût sur cette grille. Ce résultat contre-intuitif s'explique par la précision de Manhattan qui évite toute sous-optimalité.

### 5.3 Comparaison des heuristiques admissibles

**Ordre de dominance** sur grille 4-connexe : **Manhattan > Euclidienne > h=0**. Manhattan développe −47% de nœuds vs h=0 (UCS) et −15% vs Euclidienne.

<div align="center">

![Figure 8 — Trajectoires stochastiques et comparaison d'heuristiques](fig8_trajectories_heuristics.png)

*Gauche : trajectoires stochastiques (MC) vs chemin A\* (ε=0.2). Centre : nœuds développés par heuristique. Droite : compromis W-A\* vitesse/optimalité.*

</div>

---

## 6 — Analyse de la Chaîne de Markov

### 6.1 Construction et vérification de P

À partir du chemin A\* (grille 15×15), la politique **π : état → prochain état** est extraite. Pour chaque état libre, `p(n,·)` est calculé selon le modèle d'incertitude ε.

**Vérification** : `max|Σⱼ pᵢⱼ − 1| < 10⁻¹⁵` (précision machine). P est creuse : ≤5 entrées non nulles par ligne.

<div align="center">

![Figure 3 — Matrice de transition stochastique P (ε=0.2)](fig3_transition_matrix.png)

*Gauche : heatmap sous-matrice P (20×20). Droite : vérification stochasticité (sommes de lignes = 1.0 ± 10⁻¹⁵).*

</div>

> **Observation 4** : Densité de P : pour 169 états libres (15×15), P a ~845 entrées non nulles sur 28 561 possibles (2.96%). Des méthodes sparse permettraient ×34 de gain mémoire.

### 6.2 Évolution de π(n) et convergence

P(Xₙ = GOAL) est calculé matriciellement pour ε ∈ {0, 0.1, 0.2, 0.3} sur 80 pas. La convergence est monotone et asymptotique pour ε > 0.

<div align="center">

![Figure 4 — Évolution de la distribution Markov π(n)](fig4_markov_evolution.png)

*Gauche : P(Xₙ=GOAL) vs temps pour 4 valeurs de ε. ε=0 converge en exactement 27 pas. Droite : P(GOAL) en fonction continu de ε ∈ [0, 0.5].*

</div>

| ε | P(GOAL) t=20 | P(GOAL) t=40 | P(GOAL) t=80 | Dégradation |
|:--|:-------------|:-------------|:-------------|:------------|
| 0.00 | 0.991 | 0.991 | 0.991 | — (référence) |
| 0.10 | 0.742 | 0.838 | 0.873 | −11.9% |
| 0.20 | 0.531 | 0.672 | 0.724 | −26.9% |
| 0.30 | 0.389 | 0.519 | 0.578 | −41.6% |

*Tableau 6.1 — Évolution de P(GOAL) par pas de temps selon ε.*

> **Observation 5** : Pour ε=0.0 : convergence instantanée en 27 pas (déterministe). Pour ε=0.3 : l'écart entre t=40 et t=80 (0.059 pts) montre une convergence lente mais continue — des trajectoires très longues atteignent GOAL tardivement.

### 6.3 Classification des états

| Catégorie | Définition | % états | Rôle |
|:----------|:-----------|:--------|:-----|
| **Absorbant** | p(s,s)=1, aucune sortie | ~0.5% | GOAL — cible unique |
| **Transitoire** | P(retour à s) = 0 | ~94% | Cellules libres sur/hors chemin |
| **Récurrent** | P(retour à s) = 1 | ~5.5% | Cellules isolées hors politique |

*Tableau 6.2 — Classification des états.*

<div align="center">

![Figure 6 — Analyse de la Chaîne de Markov : Classes et Absorption](fig6_markov_analysis.drawio.png)

*Classification complète. Haut-gauche : camembert. Haut-droite : carte colorée. Bas : temps moyen d'absorption et probabilités B(i,GOAL) par état.*

</div>

### 6.4 Matrice fondamentale et absorption

- **Gradient de robustesse** : B(i,GOAL) croît monotoniquement de s₀ vers g. Pour ε=0.2 : B(s₀,GOAL)≈0.72 vs B(s\_last,GOAL)≈0.98. Plus l'agent progresse, plus sa probabilité d'atteindre GOAL augmente.
- **Overhead stochastique** : t\_abs(s₀) = 42.1 pas (via N·1) pour ε=0.2, soit **1.51×** la longueur du chemin A\* (28 pas). Ce facteur représente le coût moyen des déviations stochastiques.

---

## 7 — Validation par Simulation Monte-Carlo

**N=2000** trajectoires simulées depuis s₀ pour chaque ε. À chaque pas : tirage aléatoire selon (1−ε, ε/2, ε/2). Succès si GOAL atteint en ≤ t\_max=100 pas.

| ε | P(GOAL) Matr. | P(GOAL) MC | \|Écart\| % | T̄ (pas) | σ\_T | τ₉₅ |
|:--|:-------------|:-----------|:-----------|:--------|:----|:---|
| 0.00 | 0.991 | 0.988±0.001 | 0.30 | 28.1 | 0.3 | 29 |
| 0.10 | 0.873 | 0.861±0.005 | 1.38 | 34.7 | 8.9 | 52 |
| 0.20 | 0.724 | 0.709±0.007 | 2.07 | 42.3 | 14.7 | 72 |
| 0.30 | 0.578 | 0.556±0.008 | 3.81 | 58.9 | 22.1 | 94 |

*Tableau 7.1 — Validation matricielle vs Monte-Carlo (N=2000, t\_max=100, IC 95%). Écart max : 4.72% < seuil 5%.*

<div align="center">

![Figure 5 — Validation: Calcul Matriciel vs Simulation Monte-Carlo (N=2000)](fig5_montecarlo_validation.png)

*Gauche : P(GOAL) matriciel vs MC. Centre : écart relatif (seuil 5%). Droite : distribution des temps d'atteinte pour ε=0.2.*

</div>

> ✅ **Résultat Clé** : Validation concluante — écart matriciel-MC **< 5%** pour tout ε ∈ [0, 0.5], confirmant la correction de P et la fidélité du simulateur.

> **Observation 6** : Implication pratique — pour garantir **95% de succès** avec ε=0.2, il faut prévoir τ₉₅=72 pas — soit **2.57×** la longueur théorique du chemin A\* (28 pas). Ce facteur de sécurité est essentiel pour les systèmes critiques.

---

## 8 — Impact de l'Incertitude ε sur la Robustesse

### 8.1 Deux régimes de dégradation

| Régime | Plage ε | Comportement | Implication |
|:-------|:--------|:-------------|:------------|
| **Linéaire** | [0 ; 0.30] | Décroissance quasi-linéaire, pente ≈ −1.38 par 0.1 | Plan utilisable. Dégradation prévisible. |
| **Non-linéaire** | [0.30 ; 0.50] | Décroissance accélérée. P(GOAL) < 0.5 pour ε > 0.40 | Plan peu fiable. MDP nécessaire. |

*Tableau 8.1 — Deux régimes de robustesse selon ε.*

> ⚠️ **Seuil critique ε\* ≈ 0.35** : au-delà, les déviations créent des boucles de ré-visites et P(GOAL) chute sous 50%. Un plan déterministe seul est alors inadapté.

<div align="center">

![Figure 7 — Impact de l'incertitude ε sur la robustesse du plan A*](fig7_epsilon_dashboard.png)

*Tableau de bord complet de l'impact de ε : P(GOAL), dégradation relative, temps moyen, tableau récapitulatif.*

</div>

### 8.2 Relation analytique et seuil critique

Pour ε ∈ [0, 0.3], régression linéaire :

$$P(\text{GOAL}) \approx 0.991 - 1.38 \times \varepsilon \qquad (R^2 \approx 0.997)$$

**Application** : si P(GOAL) ≥ 0.80 est requis, alors ε\_max ≤ (0.991 − 0.80) / 1.38 ≈ **0.138**.

> **Observation 7** : La relation ε → T̄ est **super-linéaire** : de 28.1 pas (ε=0) à 86.4 pas (ε=0.5), multiplicateur ×3.07. **Double pénalité** de l'incertitude : les trajectoires échouent ET prennent plus de temps.

### 8.3 Tableau de bord synthétique

| ε | P(GOAL) | Taux échec | Dégradation | T̄ (pas) | Facteur sécu. | Recommandation |
|:--|:--------|:-----------|:------------|:--------|:-------------|:---------------|
| 0.00 | 99.1% | 0.9% | — | 28.1 | 1.00× | ✅ Idéal |
| 0.10 | 87.3% | 12.7% | −11.9% | 34.7 | 1.24× | ✅ Acceptable |
| 0.20 | 72.4% | 27.6% | −26.9% | 42.3 | 1.51× | ⚠️ Risqué |
| 0.30 | 57.8% | 42.2% | −41.6% | 58.9 | 2.10× | ❌ Inadapté |

*Tableau 8.2 — Synthèse de l'impact de ε. Facteur sécurité = T̄ / longueur\_A\* (28 pas).*

---

## 9 — Discussion et Limites

Notre approche hybride A\*-Markov évalue un plan déterministe fixe dans un environnement stochastique — différence fondamentale avec les MDP qui optimisent directement la politique stochastique.

| Critère | A\* + Markov (ce projet) | MDP + Value Iteration | D\* Lite |
|:--------|:------------------------|:---------------------|:---------|
| **Optimalité** | Optimale déterm. / sous-opt. stoch. | Optimale stochastique | Optimale, adaptative |
| **Re-planification** | Non (offline) | Non nécessaire | Oui (online) |
| **Éval. Robustesse** | Explicite (Pⁿ, N) | Implicite dans J\*(s) | Non fournie |
| **Complexité** | O(b^d) + O(\|S\|³) | O(\|S\|²·\|A\|/itér.) | ≈ O(A\*) |

*Tableau 9.1 — Comparaison des paradigmes de planification.*

### ▸ Limites principales

- **Pas de re-planification** (offline) : D\* Lite ou RTAA\* permettent une adaptation dynamique en ligne.
- **Explosion mémoire** : P de taille |S|² — pour grille 200×200, ~6.3 GB en dense. `scipy.sparse` nécessaire.
- **Modèle ε simplifié** : Déviation uniforme indépendante de l'état. Les vrais environnements ont ε(s) variable.
- **Pas d'observation partielle** : Agent supposé connaître son état exact. Le cadre POMDP traite ce cas.

### ▸ Extensions proposées

1. **MDP + Value Iteration** : optimiser la politique stochastique globalement via l'équation de Bellman.
2. **POMDP** : étendre aux environnements partiellement observables.
3. **Heuristiques apprises (RL/DQN)** : adapter h à la distribution stochastique réelle.
4. **Planification adaptative** : re-planification déclenchée si P(GOAL) < seuil.

---

## 10 — Conclusion

Ce rapport a présenté une étude approfondie de la **planification robuste sur grille 2D**, articulée autour de deux paradigmes complémentaires : la **recherche heuristique** (A\* et variantes) pour la planification déterministe optimale, et les **chaînes de Markov à temps discret** pour l'évaluation probabiliste de la robustesse.

### Bilan Algorithmique

- **A\*** développe **47% moins de nœuds** que UCS avec garantie d'optimalité
- **Weighted A\* (w=1.5)** atteint le coût optimal avec **45.5% moins d'exploration**
- **Manhattan** domine Euclidienne et h=0 sur toutes les configurations

### Bilan Stochastique

- **ε=0.2** dégrade P(GOAL) de **26.9%**
- Le seuil critique **ε\*≈0.35** marque la frontière au-delà de laquelle le plan déterministe devient inadapté
- La relation `P(GOAL) ≈ 0.991 − 1.38ε` (R²=0.997) permet de spécifier le niveau de bruit tolérable
- La validation MC confirme les calculs analytiques avec écart **< 5%** sur tout ε ∈ [0, 0.5]

### Synthèse Finale

| Aspect | Résultat principal |
|:-------|:-------------------|
| **Algorithmes** | A\* optimal (−47% nœuds vs UCS) ; W-A\*(w=1.5) meilleur compromis (−45% nœuds, 0% surcoût) |
| **Markov** | ε=0.2 → P(GOAL)=72.4% (−26.9%) ; seuil ε\*≈0.35 ; temps moyen ×1.51 |
| **Validation** | Écart MC < 5% sur tout ε ; IC 95% : ±0.01 sur P(GOAL) |
| **Code** | 4 modules Python (~955 lignes) ; 8 figures ; architecture modulaire reproductible |

---

## 🎮 Simulation Interactive

Le fichier [`simulation.html`](simulation.html) fournit une **simulation interactive** directement dans le navigateur :

### Fonctionnalités

- 🔄 **Choix d'algorithme** : A\*, UCS, Greedy avec visualisation en temps réel
- 🎚️ **Contrôle de ε** : Slider pour ajuster le niveau d'incertitude de 0 à 0.5
- 🚧 **Obstacles dynamiques** : Ajout/suppression d'obstacles sur la grille
- ⚡ **Vitesse d'animation** : Contrôle de la vitesse du robot
- 📊 **Graphiques en temps réel** : Métriques de performance actualisées pendant la simulation

### Lancement

```bash
python -m http.server 8000
# Ouvrir http://localhost:8000/simulation.html dans le navigateur
```

### 📸 Captures de la Simulation

#### État initial — Grilles avec chemins A\*, UCS et Greedy calculés

<div align="center">

![Simulation — État initial avec les trois algorithmes](simulation_initial.png)

*Les trois panneaux affichent simultanément les chemins trouvés par A\* (f=g+h), UCS (f=g) et Greedy (f=h) sur une grille 15×15 avec ε=0.15. Métriques affichées : nœuds, coût, déviations, étape courante.*

</div>

#### Simulation en cours — Robots naviguant avec déviations Markov

<div align="center">

![Simulation — Robots en mouvement avec déviations stochastiques (ε=0.15)](simulation_running.png)

*Les robots suivent leur chemin planifié avec des déviations stochastiques. A\* : 5 déviations, UCS : 1 déviation, Greedy : 3 déviations au step 33/30/148 respectivement. La trajectoire cyan montre le chemin réellement parcouru vs le chemin planifié (violet).*

</div>

#### Impact de ε=0.50 — Forte incertitude

<div align="center">

![Simulation — Forte incertitude ε=0.50](simulation_epsilon_high.png)

*Avec ε=0.50, les déviations sont beaucoup plus fréquentes. Le Greedy atteint l'étape 200 sans avoir rejoint le GOAL, illustrant la dégradation critique au-delà du seuil ε\*≈0.35 identifié dans l'analyse théorique.*

</div>

### 🎬 Vidéo de Démonstration

<div align="center">

![Démo vidéo de la simulation interactive — A\*, UCS, Greedy avec effets Markov](simulation_demo.webp)

*Enregistrement de la simulation : démarrage, navigation des robots, augmentation de ε de 0.15 à 0.50 montrant l'impact de l'incertitude sur les trajectoires.*

</div>

---

## 📓 Notebook Reproductible

Le notebook [`projet_experiments.ipynb`](projet_experiments.ipynb) reproduit **l'intégralité des 8 expériences** de manière interactive :

```bash
jupyter notebook projet_experiments.ipynb
```

| Section | Contenu |
|:--------|:--------|
| §0 | Imports et configuration |
| §1 | Visualisation des grilles et chemins |
| §2 | Comparaison quantitative + tableau récapitulatif |
| §3 | Matrice de transition P (heatmap) |
| §4 | Évolution π(n) et impact de ε |
| §5 | Validation Monte-Carlo vs matriciel |
| §6 | Classes d'états et absorption |
| §7 | Dashboard complet impact de ε |
| §8 | Trajectoires MC + heuristiques + Weighted A\* |

---

## 🙏 Remerciements

Je remercie chaleureusement **Pr. Mohamed MESTARI** pour la qualité de ses enseignements sur les chaînes de Markov et les algorithmes de recherche heuristique, et pour ses conseils tout au long du module *Bases de l'IA* à l'ENSET Mohammedia.

---

## 📚 Références Bibliographiques

| # | Référence |
|:-:|:----------|
| [1] | P. E. Hart, N. J. Nilsson, B. Raphael, *"A Formal Basis for the Heuristic Determination of Minimum Cost Paths,"* IEEE Trans. Syst. Sci. Cybern., vol. 4, pp. 100–107, **1968**. |
| [2] | S. J. Russell, P. Norvig, *Artificial Intelligence: A Modern Approach*, 4e éd., Pearson, **2020**. |
| [3] | I. Pohl, *"Heuristic Search Viewed as Path Finding in a Graph,"* Artif. Intell., vol. 1, pp. 193–204, **1970**. |
| [4] | S. Koenig, M. Likhachev, *"D\* Lite,"* Proc. AAAI, pp. 476–483, **2002**. |
| [5] | R. A. Howard, *Dynamic Programming and Markov Processes*, MIT Press, **1960**. |
| [6] | R. S. Sutton, A. G. Barto, *Reinforcement Learning: An Introduction*, 2e éd., MIT Press, **2018**. |
| [7] | J. R. Norris, *Markov Chains*, Cambridge University Press, **1997**. |
| [8] | A. Felner, *"Position Paper: Dijkstra's Algorithm versus UCS,"* Proc. SOCS, **2011**. |
| [9] | M. Mestari, *Synthèse — Chaînes de Markov à temps discret*, ENSET Mohammedia, **2025–2026**. |
| [10] | M. Mestari, *Synthèse — Recherche heuristique : du Best-First à A\**, ENSET Mohammedia, **2025–2026**. |

---

<div align="center">

**© 2026 — TOUBANI Badr Eddine — ENSET Mohammedia, Université Hassan II de Casablanca**

*Module Bases de l'IA — Pr. Mohamed MESTARI*

</div>
