'''Product graph from the association rules and our own Louvain algorithm (lecture of week 7).

A graph is stored as a list of dictionaries: adj[v][w] is the weight of the edge {v, w}.
Every edge is stored in both directions. A loop adj[v][v] holds 2 times the weight inside
a collapsed community (step 2 of the lecture), so the weighted degree of v is sum(adj[v].values()).
'''
import numpy as np


#Product graph

def product_graph(rules, top_m=10):
    '''One node per product, one edge per frequent pair with lift > 1, weight log(lift).
    top_m: an edge is kept if it is among the top_m strongest edges of one of its two ends (None keeps all).
    Returns adj and the product_id of every node.'''
    pairs = rules[(rules['antecedent'] < rules['consequent']) & (rules['lift'] > 1)]
    ids = np.unique(pairs[['antecedent', 'consequent']].to_numpy())
    node_of = {int(p): i for i, p in enumerate(ids)}

    adj = [{} for _ in ids]
    for a, b, lift in zip(pairs['antecedent'], pairs['consequent'], pairs['lift']):
        v, w = node_of[int(a)], node_of[int(b)]
        adj[v][w] = adj[w][v] = float(np.log(lift))

    if top_m is not None:
        #Strongest edges of every node
        strongest = [set(sorted(adj[v], key=adj[v].get, reverse=True)[:top_m]) for v in range(len(adj))]
        adj = [{w: weight for w, weight in adj[v].items() if w in strongest[v] or v in strongest[w]}
               for v in range(len(adj))]
    return adj, ids


def connected_components(adj):
    '''Component of every node; component 0 is the largest.'''
    comp = np.full(len(adj), -1)
    n_comp = 0
    for start in range(len(adj)):
        if comp[start] != -1:
            continue
        comp[start] = n_comp
        stack = [start]
        while stack:
            v = stack.pop()
            for w in adj[v]:
                if comp[w] == -1:
                    comp[w] = n_comp
                    stack.append(w)
        n_comp += 1
    #Renumber from largest to smallest
    order = np.argsort(-np.bincount(comp), kind='stable')
    rank = np.empty(n_comp, dtype=int)
    rank[order] = np.arange(n_comp)
    return rank[comp]


def subgraph(adj, ids, keep):
    '''Graph of the nodes where keep is True.'''
    new_node = -np.ones(len(adj), dtype=int)
    new_node[keep] = np.arange(keep.sum())
    new_adj = [{int(new_node[w]): weight for w, weight in adj[v].items() if keep[w]}
               for v in range(len(adj)) if keep[v]]
    return new_adj, ids[keep]


def n_edges(adj):
    return sum(len(neighbours) for neighbours in adj) // 2


#Modularity

def modularity(adj, labels, gamma=1.0):
    '''Weighted modularity Q = sum over communities of in_c / 2m - gamma * (tot_c / 2m)^2
    in_c: weight inside the community (every edge counted in both directions)
    tot_c: sum of the weighted degrees of the community.'''
    in_c, tot_c = {}, {}
    for v in range(len(adj)):
        c = labels[v]
        for w, weight in adj[v].items():
            tot_c[c] = tot_c.get(c, 0) + weight
            if labels[w] == c:
                in_c[c] = in_c.get(c, 0) + weight
    two_m = sum(tot_c.values())
    return sum(in_c.get(c, 0) / two_m - gamma * (tot_c[c] / two_m) ** 2 for c in tot_c)


#Louvain

def local_moves(adj, order, gamma=1.0):
    '''Step 1: start from singletons and move one vertex at a time to the neighbouring community
    with the largest modularity gain, until no vertex moves.

    Removing v and inserting it in community C changes modularity by
        dQ = 2/2m * (k_vC - gamma * tot_C * k_v / 2m)
    k_vC: weight between v and C, tot_C: degrees of C without v, k_v: degree of v.
    The factor 2/2m is the same for every C, so only the bracket is compared.'''
    n = len(adj)
    degree = [sum(adj[v].values()) for v in range(n)]
    two_m = sum(degree)
    community = list(range(n))
    tot = degree[:]

    improved = False
    moved = True
    while moved:
        moved = False
        for v in order:
            old = community[v]
            #Weight between v and each neighbouring community
            links = {}
            for w, weight in adj[v].items():
                if w != v:
                    links[community[w]] = links.get(community[w], 0) + weight

            #Remove v from its community
            tot[old] -= degree[v]
            best = old
            best_gain = links.get(old, 0) - gamma * tot[old] * degree[v] / two_m

            #Insert v where the gain is largest (it stays if nothing is strictly better)
            for c, k_vc in links.items():
                gain = k_vc - gamma * tot[c] * degree[v] / two_m
                if gain > best_gain + 1e-12:
                    best, best_gain = c, gain

            tot[best] += degree[v]
            if best != old:
                community[v] = best
                moved = True
                improved = True
    return community, improved


def collapse(adj, community):
    '''Step 2: every community becomes one super vertex. Edges between two communities are summed,
    edges inside a community become a loop of 2 times their weight.'''
    new_name = {c: i for i, c in enumerate(dict.fromkeys(community))}
    new_adj = [{} for _ in new_name]
    for v in range(len(adj)):
        cv = new_name[community[v]]
        for w, weight in adj[v].items():
            cw = new_name[community[w]]
            new_adj[cv][cw] = new_adj[cv].get(cw, 0) + weight
    return new_adj, [new_name[c] for c in community]


def louvain(adj, gamma=1.0, seed=0):
    '''Repeats step 1 and step 2 until modularity stops improving.
    seed sets the random order in which the vertices are visited.
    Returns one array per pass with the community of every original node (the hierarchy); the last one is the result.'''
    rng = np.random.default_rng(seed)
    labels = np.arange(len(adj))
    levels = []
    while True:
        order = [int(v) for v in rng.permutation(len(adj))]
        community, improved = local_moves(adj, order, gamma)
        if not improved:
            break
        adj, community = collapse(adj, community)
        labels = np.array(community)[labels]
        levels.append(labels)
    if not levels:
        levels.append(labels)
    return levels


#Description of the communities

def community_table(labels, ids, info, adj, n_examples=5):
    '''Size, most common department and aisle (with their share) and the most connected products of each community.'''
    degree = np.array([sum(neighbours.values()) for neighbours in adj])
    rows = []
    for c in np.argsort(-np.bincount(labels), kind='stable'):
        members = np.flatnonzero(labels == c)
        dept = info.loc[ids[members], 'department'].value_counts(normalize=True)
        aisle = info.loc[ids[members], 'aisle'].value_counts(normalize=True)
        top = members[np.argsort(-degree[members])[:n_examples]]
        rows.append({'community': int(c), 'size': len(members),
                     'top department': f'{dept.index[0]} ({dept.iloc[0]:.0%})',
                     'top aisle': f'{aisle.index[0]} ({aisle.iloc[0]:.0%})',
                     'purity (dept)': dept.iloc[0],
                     'examples': '; '.join(info.loc[ids[top], 'product_name'])})
    return rows
