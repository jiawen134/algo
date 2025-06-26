#!/usr/bin/env python3

import gzip
from collections import defaultdict, Counter
from typing import List, Tuple, Set, Dict


def load_rollernet_data(filename: str) -> List[Tuple[int, int, int]]:
    """
    Charge les données Rollernet depuis le fichier compressé
    Format: <id_sommet1> <id_sommet2> <seconde>
    """
    data = []
    with gzip.open(filename, 'rt') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                node1, node2, timestamp = map(int, parts)
                data.append((node1, node2, timestamp))
    return data

def filter_time_window(data: List[Tuple[int, int, int]], 
                      start_minute: int, end_minute: int) -> List[Tuple[int, int]]:
    """
    Filtre les données pour une fenêtre temporelle donnée (en minutes)
    Retourne les arêtes sous forme (node1, node2)
    
    Approche fonctionnelle avec lambda
    """
    start_second = start_minute * 60
    end_second = end_minute * 60
    
    # Filtrer par fenêtre temporelle avec lambda
    filtered = filter(lambda x: start_second <= x[2] <= end_second, data)
    
    # Extraire seulement les arêtes (sans timestamp) avec map
    edges = map(lambda x: (x[0], x[1]), filtered)
    
    return list(edges)

def create_union_graph(edges: List[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    """
    Crée le graphe union contenant toutes les arêtes uniques
    Utilise une approche fonctionnelle
    """
    # Normaliser les arêtes (ordre canonique pour graphe non-orienté)
    normalized_edges = map(lambda edge: tuple(sorted(edge)), edges)
    
    # Créer l'union en éliminant les doublons
    return set(normalized_edges)

def calculate_degrees_linear(graph: Set[Tuple[int, int]]) -> Dict[int, int]:
    """
    Calcule le degré de tous les sommets en temps linéaire
    Utilise une approche de tri-paniers (bucket sort concept)
    """
    # Compter les degrés avec Counter (implémentation optimisée)
    degree_counter = Counter()
    
    # Pour chaque arête, incrémenter le degré des deux sommets
    for node1, node2 in graph:
        degree_counter[node1] += 1
        degree_counter[node2] += 1
    
    return dict(degree_counter)

def analyze_network_stats(graph: Set[Tuple[int, int]], degrees: Dict[int, int]) -> Dict:
    """
    Calcule des statistiques sur le réseau
    """
    nodes = set()
    for edge in graph:
        nodes.update(edge)
    
    return {
        'num_nodes': len(nodes),
        'num_edges': len(graph),
        'max_degree': max(degrees.values()) if degrees else 0,
        'avg_degree': sum(degrees.values()) / len(degrees) if degrees else 0,
        'total_degree': sum(degrees.values())
    }

def main():
    data = load_rollernet_data('rollernet.dyn.gz')
    
    timestamps = [x[2] for x in data]
    min_time, max_time = min(timestamps), max(timestamps)
    print(f"Plage temporelle: {min_time} - {max_time} secondes")
    print(f"Durée totale: {(max_time - min_time) / 60:.1f} minutes")
    
    # Exercice 1.1: Filtrer pour la fenêtre 20-30 minutes
    print("\n1. Filtrage pour la fenêtre 20-30 minutes...")
    
    start_offset = min_time
    data_normalized = [(n1, n2, t - start_offset) for n1, n2, t in data]
    
    # Filtrer pour 20-30 minutes
    edges_20_30 = filter_time_window(data_normalized, 20, 30)
    union_graph = create_union_graph(edges_20_30)
    
    print(f"Arêtes dans la fenêtre 20-30 min: {len(edges_20_30)}")
    print(f"Arêtes uniques dans le graphe union: {len(union_graph)}")
        
    degrees = calculate_degrees_linear(union_graph)
    
    stats = analyze_network_stats(union_graph, degrees)
    
    print(f"Nombre de sommets: {stats['num_nodes']}")
    print(f"Nombre d'arêtes: {stats['num_edges']}")
    print(f"Degré maximum: {stats['max_degree']}")
    print(f"Degré moyen: {stats['avg_degree']:.2f}")
    
    print("\n3. Top 10 des sommets par nombre de 'followers':")
    sorted_degrees = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    for i, (node, degree) in enumerate(sorted_degrees[:10]):
        print(f"  {i+1}. Sommet {node}: {degree} followers")
    
    print("\n4. Distribution des degrés:")
    degree_distribution = Counter(degrees.values())
    for degree in sorted(degree_distribution.keys())[:10]:
        count = degree_distribution[degree]
        print(f"  Degré {degree}: {count} sommets")
    
    print("\n5. Sauvegarde des résultats...")
    
    with open('graph_union_20_30.txt', 'w') as f:
        f.write("# Graphe union pour la fenêtre 20-30 minutes\n")
        f.write("# Format: node1 node2\n")
        for node1, node2 in sorted(union_graph):
            f.write(f"{node1} {node2}\n")
    
    with open('degrees_20_30.txt', 'w') as f:
        f.write("# Degrés des sommets (nombre de followers)\n")
        f.write("# Format: node degree\n")
        for node, degree in sorted(degrees.items()):
            f.write(f"{node} {degree}\n")
    
    print("Résultats sauvegardés dans:")
    print("  - graph_union_20_30.txt")
    print("  - degrees_20_30.txt")

if __name__ == "__main__":
    main() 