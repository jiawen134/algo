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
    
    // Algorithme k-means standard avec k=2
    final int MAX_ITERATIONS = 100;
    final double SEUIL_CONVERGENCE = 1.0;
    
    // Initialisation des centres aléatoirement
    Random random = new Random();
    Point centreRouge = new Point(points.get(random.nextInt(points.size())));
    Point centreVerte = new Point(points.get(random.nextInt(points.size())));
    
    boolean converge = false;
    int iterations = 0;
    
    while (!converge && iterations < MAX_ITERATIONS) {
      // Vider les clusters précédents
      rouge.clear();
      verte.clear();
      
      // Assigner chaque point au centre le plus proche
      for (Point point : points) {
        double distRouge = Math.sqrt(Math.pow(point.x - centreRouge.x, 2) + Math.pow(point.y - centreRouge.y, 2));
        double distVerte = Math.sqrt(Math.pow(point.x - centreVerte.x, 2) + Math.pow(point.y - centreVerte.y, 2));
        
        if (distRouge <= distVerte) {
          rouge.add(point);
        } else {
          verte.add(point);
        }
      }
      
      // Calculer les nouveaux centres
      Point nouveauCentreRouge = calculerCentre(rouge);
      Point nouveauCentreVerte = calculerCentre(verte);
      
      // Vérifier la convergence
      double deplaceRouge = Math.sqrt(Math.pow(centreRouge.x - nouveauCentreRouge.x, 2) + Math.pow(centreRouge.y - nouveauCentreRouge.y, 2));
      double deplaceVerte = Math.sqrt(Math.pow(centreVerte.x - nouveauCentreVerte.x, 2) + Math.pow(centreVerte.y - nouveauCentreVerte.y, 2));
      
      if (deplaceRouge < SEUIL_CONVERGENCE && deplaceVerte < SEUIL_CONVERGENCE) {
        converge = true;
      }
      
      centreRouge = nouveauCentreRouge;
      centreVerte = nouveauCentreVerte;
      iterations++;
    }
    
    // Mettre à jour les clusters finaux
    kmeans.clear();
    kmeans.add(rouge);
    kmeans.add(verte);
    
    return kmeans;
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
    
    // Algorithme k-means avec budget limité (20 itérations max)
    final int BUDGET_ITERATIONS = 20;
    final double SEUIL_CONVERGENCE = 1.0;
    
    // Initialisation des centres aléatoirement
    Random random = new Random();
    Point centreRouge = new Point(points.get(random.nextInt(points.size())));
    Point centreVerte = new Point(points.get(random.nextInt(points.size())));
    
    // Boucle avec budget limité
    for (int iteration = 0; iteration < BUDGET_ITERATIONS; iteration++) {
      // Vider les clusters précédents
      rouge.clear();
      verte.clear();
      
      // Assigner chaque point au centre le plus proche
      for (Point point : points) {
        double distRouge = Math.sqrt(Math.pow(point.x - centreRouge.x, 2) + Math.pow(point.y - centreRouge.y, 2));
        double distVerte = Math.sqrt(Math.pow(point.x - centreVerte.x, 2) + Math.pow(point.y - centreVerte.y, 2));
        
        if (distRouge <= distVerte) {
          rouge.add(point);
        } else {
          verte.add(point);
        }
      }
      
      // Calculer les nouveaux centres
      Point nouveauCentreRouge = calculerCentre(rouge);
      Point nouveauCentreVerte = calculerCentre(verte);
      
      // Vérifier la convergence pour arrêt anticipé
      double deplaceRouge = Math.sqrt(Math.pow(centreRouge.x - nouveauCentreRouge.x, 2) + Math.pow(centreRouge.y - nouveauCentreRouge.y, 2));
      double deplaceVerte = Math.sqrt(Math.pow(centreVerte.x - nouveauCentreVerte.x, 2) + Math.pow(centreVerte.y - nouveauCentreVerte.y, 2));
      
      if (deplaceRouge < SEUIL_CONVERGENCE && deplaceVerte < SEUIL_CONVERGENCE) {
        break; // Convergence atteinte, arrêt anticipé
      }
      
      centreRouge = nouveauCentreRouge;
      centreVerte = nouveauCentreVerte;
    }
    
    // Mettre à jour les clusters finaux
    kmeans.clear();
    kmeans.add(rouge);
    kmeans.add(verte);
    
    return kmeans;
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
}
