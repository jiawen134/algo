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
    
    // Créer toutes les arêtes possibles
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
    
    // Kruskal MST
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
    
    Tree2D arbre = construireArbre(points, mst, 0, new boolean[points.size()]);
    return optimiserBarycentre(arbre, points, mst);
  }

  private int racine(int[] parent, int x) {
    return parent[x] == x ? x : (parent[x] = racine(parent, parent[x]));
  }
  
  // Construire Tree2D récursivement
  private Tree2D construireArbre(ArrayList<Point> points, ArrayList<int[]> mst, int noeud, boolean[] visite) {
    visite[noeud] = true;
    ArrayList<Tree2D> enfants = new ArrayList<Tree2D>();
    
    for (int[] arete : mst) {
      int voisin = arete[0] == noeud ? arete[1] : (arete[1] == noeud ? arete[0] : -1);
      if (voisin != -1 && !visite[voisin]) {
        enfants.add(construireArbre(points, mst, voisin, visite));
      }
    }
    
    return new Tree2D(points.get(noeud), enfants);
  }
  
  // Optimisation barycentre
  private Tree2D optimiserBarycentre(Tree2D arbre, ArrayList<Point> pointsOriginaux, ArrayList<int[]> mst) {
    for (int i = 0; i < pointsOriginaux.size(); i++) {
      ArrayList<Integer> voisins = new ArrayList<Integer>();
      
      // Trouver voisins de i dans MST
      for (int[] arete : mst) {
        if (arete[0] == i) voisins.add(arete[1]);
        if (arete[1] == i) voisins.add(arete[0]);
      }
      
      if (voisins.size() >= 2) {
        Point a = pointsOriginaux.get(i);
        Point b = pointsOriginaux.get(voisins.get(0));
        Point c = pointsOriginaux.get(voisins.get(1));
        
        Point barycentre = new Point((a.x + b.x + c.x) / 3, (a.y + b.y + c.y) / 3);

        double distOriginale = distance(a, b) + distance(a, c);   
        double distBarycentre = distance(barycentre, b) + distance(barycentre, c) + distance(a, barycentre);
        
        if (distBarycentre < distOriginale) {
          ArrayList<Point> nouveauxPoints = new ArrayList<Point>(pointsOriginaux);
          nouveauxPoints.add(barycentre);
          
                     ArrayList<int[]> nouveauMst = new ArrayList<int[]>(mst);
           final int idx = i;
           final int v0 = voisins.get(0);
           final int v1 = voisins.get(1);
           nouveauMst.removeIf(arete -> 
             (arete[0] == idx && arete[1] == v0) || 
             (arete[1] == idx && arete[0] == v0) ||
             (arete[0] == idx && arete[1] == v1) || 
             (arete[1] == idx && arete[0] == v1)
           );
          
          // Ajouter nouvelles arêtes avec barycentre
          int idxBarycentre = nouveauxPoints.size() - 1;
          nouveauMst.add(new int[]{i, idxBarycentre});
          nouveauMst.add(new int[]{idxBarycentre, voisins.get(0)});
          nouveauMst.add(new int[]{idxBarycentre, voisins.get(1)});
          
          return construireArbre(nouveauxPoints, nouveauMst, 0, new boolean[nouveauxPoints.size()]);
        }
      }
    }
    
    return arbre; 
  }
  
  // distance euclidienne
  private double distance(Point p1, Point p2) {
    return Math.sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
  }
}
