package algorithms;

import java.awt.Point;
import java.util.ArrayList;
import java.util.Random;

public class DefaultTeam {

  public ArrayList<ArrayList<Point>> calculKMeans(ArrayList<Point> points) {
    ArrayList<Point> rouge = new ArrayList<Point>();
    ArrayList<Point> verte = new ArrayList<Point>();

    for (int i=0;i<points.size()/2;i++){
      rouge.add(points.get(i));
      verte.add(points.get(points.size()-i-1));
    }
    if (points.size()%2==1) rouge.add(points.get(points.size()/2));

    ArrayList<ArrayList<Point>> kmeans = new ArrayList<ArrayList<Point>>();
    kmeans.add(rouge);
    kmeans.add(verte);

    /*******************
     * PARTIE A ECRIRE *
     *******************/
    
    // Algorithme k-means standard avec k=5
    final int K = 5;
    final int MAX_ITERATIONS = 100;
    final double SEUIL_CONVERGENCE = 1.0;
    
    // Initialiser 5 clusters
    ArrayList<ArrayList<Point>> clusters = new ArrayList<>();
    for (int i = 0; i < K; i++) {
      clusters.add(new ArrayList<Point>());
    }
    
    // Initialiser 5 centres aléatoirement
    Point[] centres = new Point[K];
    Random random = new Random();
    for (int i = 0; i < K; i++) {
      centres[i] = new Point(points.get(random.nextInt(points.size())));
    }
    
    boolean converge = false;
    int iterations = 0;
    
    while (!converge && iterations < MAX_ITERATIONS) {
      // Vider tous les clusters
      for (ArrayList<Point> cluster : clusters) {
        cluster.clear();
      }
      
      // Assigner chaque point au centre le plus proche
      for (Point point : points) {
        int centreProche = 0;
        double distanceMin = calculerDistance(point, centres[0]);
        
        for (int i = 1; i < K; i++) {
          double distance = calculerDistance(point, centres[i]);
          if (distance < distanceMin) {
            distanceMin = distance;
            centreProche = i;
          }
        }
        
        clusters.get(centreProche).add(point);
      }
      
      // Calculer les nouveaux centres
      Point[] nouveauxCentres = new Point[K];
      for (int i = 0; i < K; i++) {
        nouveauxCentres[i] = calculerCentre(clusters.get(i));
      }
      
      // Vérifier la convergence
      converge = true;
      for (int i = 0; i < K; i++) {
        double deplacement = calculerDistance(centres[i], nouveauxCentres[i]);
        if (deplacement >= SEUIL_CONVERGENCE) {
          converge = false;
          break;
        }
      }
      
      centres = nouveauxCentres;
      iterations++;
    }
    
    return clusters;
  }
  
  public ArrayList<ArrayList<Point>> calculKMeansBudget(ArrayList<Point> points) {
    ArrayList<Point> rouge = new ArrayList<Point>();
    ArrayList<Point> verte = new ArrayList<Point>();

    for (int i=0;i<points.size()/2;i++){
      rouge.add(points.get(i));
      verte.add(points.get(points.size()-i-1));
    }
    if (points.size()%2==1) rouge.add(points.get(points.size()/2));

    ArrayList<ArrayList<Point>> kmeans = new ArrayList<ArrayList<Point>>();
    kmeans.add(rouge);
    kmeans.add(verte);

    /*******************
     * PARTIE A ECRIRE *
     *******************/
    
    // Algorithme k-means avec contraintes budgétaires
    final int K = 5;
    final double BUDGET_B = 10101.0;
    
    // Initialiser 5 clusters avec les créateurs fondateurs (5 premiers points)
    ArrayList<ArrayList<Point>> clusters = new ArrayList<>();
    Point[] centres = new Point[K];
    
    for (int i = 0; i < K; i++) {
      clusters.add(new ArrayList<Point>());
      if (i < points.size()) {
        centres[i] = new Point(points.get(i)); // s1, s2, s3, s4, s5 comme créateurs
        clusters.get(i).add(points.get(i));    // Ajouter le créateur à son cluster
      } else {
        centres[i] = new Point(0, 0);
      }
    }
    
    // Tenter d'assigner le reste des points en respectant le budget
    for (int j = K; j < points.size(); j++) {
      Point point = points.get(j);
      int meilleurCluster = -1;
      double meilleurDistance = Double.MAX_VALUE;
      
      // Trouver le cluster qui peut accepter ce point sans dépasser le budget
      for (int i = 0; i < K; i++) {
        double distance = calculerDistance(point, centres[i]);
        
        // Calculer le coût total si on ajoute ce point au cluster i
        double coutTotal = calculerCoutCluster(clusters.get(i), centres[i]) + distance;
        
        if (coutTotal <= BUDGET_B && distance < meilleurDistance) {
          meilleurDistance = distance;
          meilleurCluster = i;
        }
      }
      
      // Si un cluster peut accepter ce point, l'ajouter
      if (meilleurCluster != -1) {
        clusters.get(meilleurCluster).add(point);
      }
    }
    
    // Recalculer les centres après l'assignation finale
    for (int i = 0; i < K; i++) {
      if (!clusters.get(i).isEmpty()) {
        centres[i] = calculerCentre(clusters.get(i));
      }
    }
    
    return clusters;
  }
  
  // Méthode auxiliaire pour calculer le centre d'un cluster
  private Point calculerCentre(ArrayList<Point> cluster) {
    if (cluster.isEmpty()) {
      return new Point(0, 0);
    }
    
    int sommeX = 0, sommeY = 0;
    for (Point point : cluster) {
      sommeX += point.x;
      sommeY += point.y;
    }
    
    return new Point(sommeX / cluster.size(), sommeY / cluster.size());
  }
  
  // Méthode auxiliaire pour calculer la distance euclidienne
  private double calculerDistance(Point p1, Point p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return Math.sqrt(dx * dx + dy * dy);
  }
  
  // Méthode auxiliaire pour calculer le coût total d'un cluster
  private double calculerCoutCluster(ArrayList<Point> cluster, Point centre) {
    double coutTotal = 0.0;
    for (Point point : cluster) {
      coutTotal += calculerDistance(point, centre);
    }
    return coutTotal;
  }
}
