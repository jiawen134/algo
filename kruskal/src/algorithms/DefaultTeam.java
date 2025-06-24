package algorithms;

import java.awt.Point;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

public class DefaultTeam {
  public Tree2D calculSteiner(ArrayList<Point> points) {
    /*******************
     * PARTIE A EDITER *
     *******************/
    
    if (points.size() <= 1) {
      return new Tree2D(points.size() == 1 ? points.get(0) : new Point(0, 0), new ArrayList<Tree2D>());
    }
    
    // Créer toutes les arêtes
    ArrayList<double[]> aretes = new ArrayList<double[]>();
    for (int i = 0; i < points.size(); i++) {
      for (int j = i + 1; j < points.size(); j++) {
        Point p1 = points.get(i), p2 = points.get(j);
        double dist = Math.sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
        aretes.add(new double[]{i, j, dist});
      }
    }
    
    // Trier par distance
    Collections.sort(aretes, new Comparator<double[]>() {
      public int compare(double[] a, double[] b) { return Double.compare(a[2], b[2]); }
    });
    
    // Kruskal
    int[] parent = new int[points.size()];
    for (int i = 0; i < points.size(); i++) parent[i] = i;
    
    ArrayList<int[]> mst = new ArrayList<int[]>();
    for (double[] arete : aretes) {
      int p1 = (int)arete[0], p2 = (int)arete[1];
      int r1 = racine(parent, p1), r2 = racine(parent, p2);
      if (r1 != r2) {
        parent[r1] = r2;
        mst.add(new int[]{p1, p2});
        if (mst.size() == points.size() - 1) break;
      }
    }
    
    // Construire l'arbre
    return arbre(points, mst, 0, new boolean[points.size()]);
  }
  
  // Union-find 
  private int racine(int[] parent, int x) {
    return parent[x] == x ? x : (parent[x] = racine(parent, parent[x]));
  }
  
  // Construire Tree2D récursivement
  private Tree2D arbre(ArrayList<Point> points, ArrayList<int[]> mst, int noeud, boolean[] visite) {
    visite[noeud] = true;
    ArrayList<Tree2D> enfants = new ArrayList<Tree2D>();
    
    for (int[] arete : mst) {
      int voisin = arete[0] == noeud ? arete[1] : (arete[1] == noeud ? arete[0] : -1);
      if (voisin != -1 && !visite[voisin]) {
        enfants.add(arbre(points, mst, voisin, visite));
      }
    }
    
    return new Tree2D(points.get(noeud), enfants);
  }
}
