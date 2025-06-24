package algorithms;

import java.awt.Point;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Set;

public class DefaultTeam {
  public Tree2D calculSteiner(ArrayList<Point> points) {
    /*******************
     * PARTIE A EDITER *
     *******************/
    
    if (points.size() <= 1) {
      return new Tree2D(points.size() == 1 ? points.get(0) : new Point(0, 0), new ArrayList<Tree2D>());
    }
    
    // Étape 1: Construire MST avec Kruskal
    ArrayList<int[]> mst = construireMST(points);
    
    // Étape 2: Règle ahdoc - optimisations hiérarchiques
    ArrayList<Point> pointsOptimises = new ArrayList<Point>(points);
    
    // 3 points: règle Torricelli-Fermat classique
    appliquerRegle3Points(pointsOptimises, mst);
    
    // 4 points: configuration en étoile améliorée  
    appliquerRegle4Points(pointsOptimises, mst);
    
    // 5 points: configuration limitée pour éviter explosion combinatoire
    appliquerRegle5Points(pointsOptimises, mst);
    
    return construireArbre(pointsOptimises, mst, 0, new boolean[pointsOptimises.size()]);
  }

  // Construction MST standard avec Kruskal
  private ArrayList<int[]> construireMST(ArrayList<Point> points) {
    ArrayList<double[]> aretes = new ArrayList<double[]>();
    for (int i = 0; i < points.size(); i++) {
      for (int j = i + 1; j < points.size(); j++) {
        double dist = distance(points.get(i), points.get(j));
        aretes.add(new double[]{i, j, dist});
      }
    }
    
    Collections.sort(aretes, new Comparator<double[]>() {
      public int compare(double[] a, double[] b) { return Double.compare(a[2], b[2]); }
    });
    
    int[] parent = new int[points.size()];
    for (int i = 0; i < points.size(); i++) parent[i] = i;
    
    ArrayList<int[]> mst = new ArrayList<int[]>();
    for (double[] arete : aretes) {
      int p1 = (int)arete[0], p2 = (int)arete[1];
      if (union(parent, p1, p2)) {
        mst.add(new int[]{p1, p2});
        if (mst.size() == points.size() - 1) break;
      }
    }
    return mst;
  }
  
  // Règle 3 points: optimiser triangles A-B-C avec point Fermat
  private void appliquerRegle3Points(ArrayList<Point> points, ArrayList<int[]> mst) {
    for (int i = 0; i < points.size(); i++) {
      ArrayList<Integer> voisins = voisinsDans(mst, i);
      
      // Chercher paires de voisins pour former triangle
      for (int j = 0; j < voisins.size() - 1; j++) {
        for (int k = j + 1; k < voisins.size(); k++) {
          Point a = points.get(i);
          Point b = points.get(voisins.get(j));  
          Point c = points.get(voisins.get(k));
          
          // Calculer point Fermat pour triangle ABC
          Point fermat = pointFermat(a, b, c);
          
          // Test d'amélioration
          double coutOriginal = distance(a, b) + distance(a, c);
          double coutFermat = distance(fermat, a) + distance(fermat, b) + distance(fermat, c);
          
          if (coutFermat < coutOriginal - 1.0) { // Seuil d'amélioration
            ajouterPointSteiner(points, mst, i, new int[]{voisins.get(j), voisins.get(k)}, fermat);
            return; // Une amélioration par appel
          }
        }
      }
    }
  }
  
  // Règle 4 points: optimiser étoiles avec centre + 3 branches
  private void appliquerRegle4Points(ArrayList<Point> points, ArrayList<int[]> mst) {
    for (int centre = 0; centre < points.size(); centre++) {
      ArrayList<Integer> voisins = voisinsDans(mst, centre);
      
      if (voisins.size() >= 3) {
        // Prendre premiers 3 voisins pour éviter explosion combinatoire
        Point pCentre = points.get(centre);
        Point p1 = points.get(voisins.get(0));
        Point p2 = points.get(voisins.get(1)); 
        Point p3 = points.get(voisins.get(2));
        
        // Point Steiner = barycentre pondéré par distances inverses
        Point steiner4 = barycentrePondere(pCentre, p1, p2, p3);
        
        // Test d'amélioration
        double coutOriginal = distance(pCentre, p1) + distance(pCentre, p2) + distance(pCentre, p3);
        double coutSteiner = distance(steiner4, pCentre) + distance(steiner4, p1) + 
                            distance(steiner4, p2) + distance(steiner4, p3);
        
        if (coutSteiner < coutOriginal - 2.0) {
          ajouterPointSteiner(points, mst, centre, 
            new int[]{voisins.get(0), voisins.get(1), voisins.get(2)}, steiner4);
          return;
        }
      }
    }
  }
  
  // Règle 5 points: configuration complexe avec limite de tentatives
  private void appliquerRegle5Points(ArrayList<Point> points, ArrayList<int[]> mst) {
    int tentatives = 0;
    final int MAX_TENTATIVES = 20; // Limite pour éviter explosion temps
    
    for (int centre = 0; centre < points.size() && tentatives < MAX_TENTATIVES; centre++) {
      ArrayList<Integer> voisins = voisinsDans(mst, centre);
      
      if (voisins.size() >= 4) {
        // Prendre 4 premiers voisins
        Point pCentre = points.get(centre);
        Point[] points5 = {pCentre, points.get(voisins.get(0)), points.get(voisins.get(1)), 
                          points.get(voisins.get(2)), points.get(voisins.get(3))};
        
        // Configuration 5 points: 2 points Steiner interconnectés
        Point[] steiners = calculerSteiner5Points(points5);
        
        double coutOriginal = 0;
        for (int i = 0; i < 4; i++) {
          coutOriginal += distance(pCentre, points.get(voisins.get(i)));
        }
        
        // Coût approximatif avec points Steiner
        double coutSteiner = distance(steiners[0], steiners[1]); // Liaison entre Steiners
        for (Point s : steiners) {
          for (Point p : points5) {
            coutSteiner += distance(s, p) * 0.2; // Pondération distribué
          }
        }
        
        if (coutSteiner < coutOriginal - 3.0) {
          // Ajout simplifié: un seul point Steiner au lieu de 2
          ajouterPointSteiner(points, mst, centre, 
            new int[]{voisins.get(0), voisins.get(1), voisins.get(2), voisins.get(3)}, steiners[0]);
          return;
        }
        
        tentatives++;
      }
    }
  }
  
  // Calcul point Fermat pour 3 points (géométrie classique)
  private Point pointFermat(Point a, Point b, Point c) {
    // Si un angle >= 120°, le point Fermat est au sommet de cet angle
    if (angleGrand(a, b, c)) return a;
    if (angleGrand(b, a, c)) return b; 
    if (angleGrand(c, a, b)) return c;
    
    // Sinon, point où les 3 distances font des angles de 120°
    // Approximation: itération géométrique simple
    double x = (a.x + b.x + c.x) / 3.0;
    double y = (a.y + b.y + c.y) / 3.0;
    
    // 5 itérations d'amélioration suffisent
    for (int iter = 0; iter < 5; iter++) {
      Point current = new Point((int)x, (int)y);
      double da = distance(current, a);
      double db = distance(current, b); 
      double dc = distance(current, c);
      
      if (da + db + dc < 0.001) break;
      
      // Déplacement vers moyenne pondérée par distance inverse
      x = (a.x/da + b.x/db + c.x/dc) / (1/da + 1/db + 1/dc);
      y = (a.y/da + b.y/db + c.y/dc) / (1/da + 1/db + 1/dc);
    }
    
    return new Point((int)x, (int)y);
  }
  
  // Barycentre pondéré pour 4 points
  private Point barycentrePondere(Point centre, Point p1, Point p2, Point p3) {
    double d1 = Math.max(distance(centre, p1), 0.1);
    double d2 = Math.max(distance(centre, p2), 0.1);
    double d3 = Math.max(distance(centre, p3), 0.1);
    
    // Poids inversement proportionnels aux distances
    double w0 = 2.0; // Poids du centre
    double w1 = 1.0 / d1;
    double w2 = 1.0 / d2; 
    double w3 = 1.0 / d3;
    double wTotal = w0 + w1 + w2 + w3;
    
    double x = (w0 * centre.x + w1 * p1.x + w2 * p2.x + w3 * p3.x) / wTotal;
    double y = (w0 * centre.y + w1 * p1.y + w2 * p2.y + w3 * p3.y) / wTotal;
    
    return new Point((int)x, (int)y);
  }
  
  // Configuration Steiner pour 5 points (approximation rapide)
  private Point[] calculerSteiner5Points(Point[] points5) {
    // Premier point: centroïde général
    Point centroide = new Point(0, 0);
    for (Point p : points5) {
      centroide.x += p.x;
      centroide.y += p.y;
    }
    centroide.x /= 5;
    centroide.y /= 5;
    
    // Deuxième point: barycentre des 3 premiers points
    Point bary3 = new Point(
      (points5[0].x + points5[1].x + points5[2].x) / 3,
      (points5[0].y + points5[1].y + points5[2].y) / 3
    );
    
    return new Point[]{centroide, bary3};
  }
  
  // Ajouter point Steiner et reconfigurer MST
  private void ajouterPointSteiner(ArrayList<Point> points, ArrayList<int[]> mst, 
                                  int centre, int[] voisins, Point steiner) {
    points.add(steiner);
    int idxSteiner = points.size() - 1;
    
    // Supprimer arêtes centre-voisins
    for (int v : voisins) {
      mst.removeIf(arete -> 
        (arete[0] == centre && arete[1] == v) || (arete[1] == centre && arete[0] == v)
      );
    }
    
    // Nouvelles arêtes via point Steiner
    mst.add(new int[]{centre, idxSteiner});
    for (int v : voisins) {
      mst.add(new int[]{v, idxSteiner});
    }
  }
  
  // Utilitaires
  private ArrayList<Integer> voisinsDans(ArrayList<int[]> mst, int noeud) {
    ArrayList<Integer> voisins = new ArrayList<Integer>();
    for (int[] arete : mst) {
      if (arete[0] == noeud) voisins.add(arete[1]);
      if (arete[1] == noeud) voisins.add(arete[0]);
    }
    return voisins;
  }
  
  private boolean angleGrand(Point sommet, Point p1, Point p2) {
    int dx1 = p1.x - sommet.x, dy1 = p1.y - sommet.y;
    int dx2 = p2.x - sommet.x, dy2 = p2.y - sommet.y;
    // Produit scalaire < 0 si angle > 90°, < -0.5*|u||v| si angle > 120°
    return dx1 * dx2 + dy1 * dy2 < -0.5 * Math.sqrt((dx1*dx1 + dy1*dy1) * (dx2*dx2 + dy2*dy2));
  }
  
  private double distance(Point p1, Point p2) {
    return Math.sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
  }
  
  private boolean union(int[] parent, int x, int y) {
    int rx = find(parent, x), ry = find(parent, y);
    if (rx != ry) { parent[rx] = ry; return true; }
    return false;
  }
  
  private int find(int[] parent, int x) {
    return parent[x] == x ? x : (parent[x] = find(parent, parent[x]));
  }
  
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
}
