package algorithms;

import java.awt.Point;
import java.util.ArrayList;
import java.util.HashSet;

public class DefaultTeam {

  public ArrayList<Point> calculTSP(ArrayList<Point> points) {
    if (points.size()<4) {
      return points;
    }

    ArrayList<Point> p=new ArrayList<Point>();
    p.add(points.get(0));
    p.add(points.get(1));
    p.add(points.get(2));

    /*******************
     * PARTIE A ECRIRE *
     *******************/
    return p;
  }

  // Applique 2-opt à un chemin pour réduire les croisements
  private ArrayList<Point> twoOpt(ArrayList<Point> chemin, double budget) {
    boolean improved = true;
    int n = chemin.size();
    while (improved) {
      improved = false;
      for (int i = 1; i < n - 2; i++) {
        for (int k = i + 1; k < n - 1; k++) {
          ArrayList<Point> newChemin = new ArrayList<>(chemin);
          // Inverser le segment [i, k]
          for (int a = 0; a <= k - i; a++) {
            newChemin.set(i + a, chemin.get(k - a));
          }
          // Calculer la nouvelle distance
          double newDist = 0.0;
          for (int j = 0; j < newChemin.size() - 1; j++) {
            newDist += newChemin.get(j).distance(newChemin.get(j + 1));
          }
          if (newDist <= budget) {
            double oldDist = 0.0;
            for (int j = 0; j < chemin.size() - 1; j++) {
              oldDist += chemin.get(j).distance(chemin.get(j + 1));
            }
            if (newDist < oldDist - 1e-6) { // tolérance numérique
              chemin = newChemin;
              improved = true;
              break;
            }
          }
        }
        if (improved) break;
      }
    }
    return chemin;
  }

  // Attribue chaque point à un secteur angulaire autour de la maison
  private int getSecteur(Point maison, Point p, int nbSecteurs) {
    double angle = Math.atan2(p.y - maison.y, p.x - maison.x);
    if (angle < 0) angle += 2 * Math.PI;
    int secteur = (int) (angle / (2 * Math.PI / nbSecteurs));
    return secteur;
  }

  public ArrayList<ArrayList<Point>> calculCinqVoyageursAvecBudget(Point maison, ArrayList<Point> points) {
    int NB_VOYAGEURS = 5;
    double BUDGET = 1665.0;
    ArrayList<ArrayList<Point>> result = new ArrayList<>();
    ArrayList<ArrayList<Point>> secteurs = new ArrayList<>();
    for (int i = 0; i < NB_VOYAGEURS; i++) secteurs.add(new ArrayList<>());
    ArrayList<Point> pointsRestants = new ArrayList<>();

    // Répartition des points par secteur
    for (Point p : points) {
      if (!p.equals(maison)) {
        int secteur = getSecteur(maison, p, NB_VOYAGEURS);
        secteurs.get(secteur).add(p);
      }
    }

    // Pour chaque secteur, construire le chemin du voyageur
    ArrayList<ArrayList<Point>> chemins = new ArrayList<>();
    ArrayList<Double> distances = new ArrayList<>();
    for (int v = 0; v < NB_VOYAGEURS; v++) {
      ArrayList<Point> chemin = new ArrayList<>();
      chemin.add(maison);
      ArrayList<Point> pointsSecteur = new ArrayList<>(secteurs.get(v));
      double dist = 0.0;
      Point current = maison;
      // Greedy : plus proche voisin sous contrainte de budget
      while (!pointsSecteur.isEmpty()) {
        Point best = null;
        double bestDist = Double.MAX_VALUE;
        for (Point p : pointsSecteur) {
          double d = current.distance(p) + p.distance(maison);
          if (dist + current.distance(p) + p.distance(maison) <= BUDGET && current.distance(p) < bestDist) {
            bestDist = current.distance(p);
            best = p;
          }
        }
        if (best != null) {
          chemin.add(best);
          dist += current.distance(best);
          current = best;
          pointsSecteur.remove(best);
        } else {
          break;
        }
      }
      dist += current.distance(maison);
      chemin.add(maison);
      // 2-opt pour optimiser le chemin
      chemin = twoOpt(chemin, BUDGET);
      chemins.add(chemin);
      distances.add(dist);
    }

    // Récupérer les points non visités
    java.util.HashSet<Point> dejaPris = new java.util.HashSet<>();
    for (ArrayList<Point> chemin : chemins) {
      for (int i = 1; i < chemin.size() - 1; i++) {
        dejaPris.add(chemin.get(i));
      }
    }
    for (Point p : points) {
      if (!p.equals(maison) && !dejaPris.contains(p)) {
        pointsRestants.add(p);
      }
    }

    // Insertion la moins coûteuse pour les points restants
    boolean progress;
    do {
      progress = false;
      Point bestPoint = null;
      int bestVoyageur = -1;
      int bestInsertPos = -1;
      double bestGain = Double.MAX_VALUE;
      for (Point p : new ArrayList<>(pointsRestants)) {
        for (int v = 0; v < NB_VOYAGEURS; v++) {
          ArrayList<Point> chemin = chemins.get(v);
          for (int i = 1; i < chemin.size(); i++) {
            Point prev = chemin.get(i - 1);
            Point next = chemin.get(i);
            double gain = prev.distance(p) + p.distance(next) - prev.distance(next);
            // recalculer la distance totale
            double newDist = 0.0;
            for (int j = 0; j < chemin.size() - 1; j++) {
              newDist += chemin.get(j).distance(chemin.get(j + 1));
            }
            newDist += gain;
            if (newDist <= BUDGET && gain < bestGain) {
              bestGain = gain;
              bestPoint = p;
              bestVoyageur = v;
              bestInsertPos = i;
            }
          }
        }
      }
      if (bestPoint != null && bestVoyageur != -1) {
        chemins.get(bestVoyageur).add(bestInsertPos, bestPoint);
        pointsRestants.remove(bestPoint);
        chemins.set(bestVoyageur, twoOpt(chemins.get(bestVoyageur), BUDGET));
        progress = true;
      }
    } while (progress);

    // Ajouter les chemins au résultat
    for (ArrayList<Point> chemin : chemins) {
      result.add(chemin);
    }
    return result;
  }

  // Méthode main pour tester le score sur input.points
  public static void main(String[] args) throws Exception {
    java.util.List<Point> points = new java.util.ArrayList<>();
    java.io.BufferedReader br = new java.io.BufferedReader(new java.io.FileReader("input.points"));
    String line;
    while ((line = br.readLine()) != null) {
      String[] parts = line.trim().split("\\s+");
      if (parts.length == 2) {
        int x = Integer.parseInt(parts[0]);
        int y = Integer.parseInt(parts[1]);
        points.add(new Point(x, y));
      }
    }
    br.close();
    Point maison = points.get(0); // On suppose que le premier point est la maison
    DefaultTeam team = new DefaultTeam();
    ArrayList<ArrayList<Point>> chemins = team.calculCinqVoyageursAvecBudget(maison, new ArrayList<>(points));
    int total = 0;
    for (ArrayList<Point> chemin : chemins) {
      total += chemin.size() - 2; // On ne compte pas les deux "maison"
    }
    System.out.println("Nombre total de points visités (hors maison) : " + total);
    for (int i = 0; i < chemins.size(); i++) {
      System.out.println("Voyageur " + (i+1) + " : " + (chemins.get(i).size() - 2) + " points");
    }
  }
}