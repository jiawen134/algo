import csv
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def min_max_normalize(values):
    min_v = min(values)
    max_v = max(values)
    if max_v - min_v == 0:
        return [0.0 for _ in values]
    return [round(1000 * (v - min_v) / (max_v - min_v)) / 10.0 for v in values]

def load_graph(filename):
    G = nx.Graph()
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'): 
                u, v = map(int, line.split())
                G.add_edge(u, v)
    return G

def main():
    fichierGraph = "tme4/ex2/graph2030.txt"
    fichierCSV = "tme4/ex2/centrality_results.csv"

    G = load_graph(fichierGraph)
    nodes = sorted(G.nodes())

    degrees = [G.degree(n) for n in nodes]
    closeness = [nx.closeness_centrality(G, n) for n in nodes]
    betweenness_dict = nx.betweenness_centrality(G, normalized=False)
    betweenness = [betweenness_dict[n] for n in nodes]

    degrees_pct = min_max_normalize(degrees)
    closeness_pct = min_max_normalize(closeness)
    betweenness_pct = min_max_normalize(betweenness)

    # Pour vérifier les valeurs
    print("Betweenness (brut) :", betweenness)
    print("Betweenness (normalisé) :", betweenness_pct)

    with open(fichierCSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(["index", "degree", "degree_pct", "closeness_pct", "betweenness_pct"])
        for i, n in enumerate(nodes):
            writer.writerow([n, degrees[i], degrees_pct[i], closeness_pct[i], betweenness_pct[i]])

    print(f"Calculs terminés, résultats dans {fichierCSV}")

    # Scatter plot unique avec les deux séries (format TME4 Exercice 2)
    plt.figure(figsize=(10, 8))
    
    # Série 1: Degree vs Closeness (carrés bleus)
    plt.scatter(degrees_pct, closeness_pct, c='blue', marker='s', s=60, 
               alpha=0.7, label='Degré vs Closeness', edgecolors='navy', linewidth=0.5)
    
    # Série 2: Degree vs Betweenness (losanges orange)
    plt.scatter(degrees_pct, betweenness_pct, c='orange', marker='D', s=60, 
               alpha=0.7, label='Degré vs Betweenness', edgecolors='darkorange', linewidth=0.5)
    
    # Configuration du graphique
    plt.xlabel('Degré (normalisé en %)', fontsize=12, fontweight='bold')
    plt.ylabel('Centralité (normalisée en %)', fontsize=12, fontweight='bold')
    plt.title('Analyse de Centralité: Closeness & Betweenness vs Degré', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    # Calculer et afficher les corrélations
    corr_closeness = np.corrcoef(degrees_pct, closeness_pct)[0, 1]
    corr_betweenness = np.corrcoef(degrees_pct, betweenness_pct)[0, 1]
    
    plt.text(0.02, 0.98, f'Corrélation Closeness-Degré: r = {corr_closeness:.3f}\nCorrellation Betweenness-Degré: r = {corr_betweenness:.3f}',
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('tme4/ex2/centrality_scatter_plot_combined.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Corrélations calculées:")
    print(f"   Closeness-Degré: r = {corr_closeness:.4f}")
    print(f"   Betweenness-Degré: r = {corr_betweenness:.4f}")
    print(f"📁 Graphique sauvegardé: tme4/ex2/centrality_scatter_plot_combined.png")

if __name__ == "__main__":
    main()