package algorithms;

import java.awt.Point;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

// Simple local classes to replace supportGUI imports
class Line {
    private Point p1, p2;
    public Line(Point p1, Point p2) { this.p1 = p1; this.p2 = p2; }
    public Point getP1() { return p1; }
    public Point getP2() { return p2; }
}

class Circle {
    private Point center;
    private int radius;
    public Circle(Point center, int radius) { this.center = center; this.radius = radius; }
    public Point getCenter() { return center; }
    public int getRadius() { return radius; }
}

public class DefaultTeam {

    // calculTSP: ArrayList<Point> --> ArrayList<Point>
    //   renvoie une permutation P de points telle que la visite
    //   de ces points selon l'ordre défini par P est de distance
    //   totale minimum.
    public ArrayList<Point> calculTSP(ArrayList<Point> points) {
        if (points.size() < 4) {
            return points;
        }

        /*******************
         * PARTIE A ECRIRE *
         *******************/   
        
        System.out.println("===== TSP算法被调用 =====");
        
        // Lin-Kernighan启发式优化
        ArrayList<Point> bestResult = nearestNeighborTSP(points);
        double bestScore = calculateTotalDistance(bestResult);
        
        // 多轮优化
        for (int round = 0; round < 5; round++) {
            bestResult = linKernighanOptimization(bestResult);
            double newScore = calculateTotalDistance(bestResult);
            if (newScore < bestScore) {
                bestScore = newScore;
            }
        }
        
        System.out.println("优化后距离: " + bestScore);
        return bestResult;
    }

    // calculTSPOpt: ArrayList<Point> --> ArrayList<Point>
    //   renvoie une permutation P de points telle que la visite
    //   de ces points selon l'ordre défini par P est de distance
    //   totale minimum.
    public ArrayList<Point> calculTSPOpt(ArrayList<Point> points) {
        if (points.size() < 4) {
            return points;
        }

        /*******************
         * PARTIE A ECRIRE *
         *******************/
        
        // Christofides算法近似
        ArrayList<Point> bestResult = christofidesApproximation(points);
        double bestScore = calculateTotalDistance(bestResult);
        
        // 多种高级优化组合
        for (int attempt = 0; attempt < 3; attempt++) {
            // 最近邻 + Lin-Kernighan
            ArrayList<Point> nn = nearestNeighborTSP(points);
            nn = linKernighanOptimization(nn);
            double score1 = calculateTotalDistance(nn);
            if (score1 < bestScore) {
                bestScore = score1;
                bestResult = nn;
            }
            
            // 最远插入 + 高级3-opt
            ArrayList<Point> fi = farthestInsertionAdvanced(points);
            fi = advanced3OptOptimization(fi);
            double score2 = calculateTotalDistance(fi);
            if (score2 < bestScore) {
                bestScore = score2;
                bestResult = fi;
            }
        }
        
        return bestResult;
    }

    // calculDiametre: ArrayList<Point> --> Line
    //   renvoie une paire de points de la liste, de distance maximum.
    public Line calculDiametre(ArrayList<Point> points) {
        if (points.size() < 3) {
            return null;
        }

        /*******************
         * PARTIE A ECRIRE *
         *******************/
        
        // Algorithme brute force O(n²)
        Point p1 = points.get(0);
        Point p2 = points.get(1);
        double maxDist = p1.distance(p2);
        
        for (int i = 0; i < points.size(); i++) {
            for (int j = i + 1; j < points.size(); j++) {
                double dist = points.get(i).distance(points.get(j));
                if (dist > maxDist) {
                    maxDist = dist;
                    p1 = points.get(i);
                    p2 = points.get(j);
                }
            }
        }
        
        return new Line(p1, p2);
    }

    // calculDiametreOptimise: ArrayList<Point> --> Line
    //   renvoie une paire de points de la liste, de distance maximum.
    public Line calculDiametreOptimise(ArrayList<Point> points) {
        if (points.size() < 3) {
            return null;
        }

        /*******************
         * PARTIE A ECRIRE *
         *******************/
        
        // Version optimisée: convex hull + rotating calipers approximation
        ArrayList<Point> hull = enveloppeConvexe(points);
        if (hull == null || hull.size() < 2) {
            return calculDiametre(points); // Fallback
        }
        
        // Sur l'enveloppe convexe, chercher la paire la plus éloignée
        Point p1 = hull.get(0);
        Point p2 = hull.get(1);
        double maxDist = p1.distance(p2);
        
        for (int i = 0; i < hull.size(); i++) {
            for (int j = i + 1; j < hull.size(); j++) {
                double dist = hull.get(i).distance(hull.get(j));
                if (dist > maxDist) {
                    maxDist = dist;
                    p1 = hull.get(i);
                    p2 = hull.get(j);
                }
            }
        }
        
        return new Line(p1, p2);
    }

    // calculCercleMin: ArrayList<Point> --> Circle
    //   renvoie un cercle couvrant tout point de la liste, de rayon minimum.
    public Circle calculCercleMin(ArrayList<Point> points) {
        if (points.isEmpty()) {
            return null;
        }

        /*******************
         * PARTIE A ECRIRE *
         *******************/
        
        // Algorithme de Welzl simplifié
        return welzlCircle(new ArrayList<Point>(points), new ArrayList<Point>());
    }

    // enveloppeConvexe: ArrayList<Point> --> ArrayList<Point>
    //   renvoie l'enveloppe convexe de la liste.
    public ArrayList<Point> enveloppeConvexe(ArrayList<Point> points) {
        if (points.size() < 3) {
            return null;
        }

        /*******************
         * PARTIE A ECRIRE *
         *******************/
        
        // Algorithme de Graham Scan
        ArrayList<Point> sorted = new ArrayList<Point>(points);
        
        // Trouver le point le plus bas (ou le plus à gauche si égalité)
        Point bottom = Collections.min(sorted, new Comparator<Point>() {
            public int compare(Point a, Point b) {
                if (a.y != b.y) return Integer.compare(a.y, b.y);
                return Integer.compare(a.x, b.x);
            }
        });
        
        // Trier par angle polaire
        final Point pivot = bottom;
        Collections.sort(sorted, new Comparator<Point>() {
            public int compare(Point a, Point b) {
                if (a.equals(pivot)) return -1;
                if (b.equals(pivot)) return 1;
                
                double angleA = Math.atan2(a.y - pivot.y, a.x - pivot.x);
                double angleB = Math.atan2(b.y - pivot.y, b.x - pivot.x);
                
                if (Math.abs(angleA - angleB) < 1e-9) {
                    // Même angle, prendre le plus proche
                    return Double.compare(pivot.distance(a), pivot.distance(b));
                }
                return Double.compare(angleA, angleB);
            }
        });
        
        // Graham scan
        ArrayList<Point> hull = new ArrayList<Point>();
        for (Point p : sorted) {
            while (hull.size() >= 2 && 
                   crossProduct(hull.get(hull.size()-2), hull.get(hull.size()-1), p) <= 0) {
                hull.remove(hull.size() - 1);
            }
            hull.add(p);
        }
        
        return hull;
    }
    
    // === Méthodes auxiliaires ===
    
    private ArrayList<Point> greedyTSP(ArrayList<Point> points, int startIndex) {
        ArrayList<Point> result = new ArrayList<Point>();
        ArrayList<Point> remaining = new ArrayList<Point>(points);
        
        result.add(remaining.remove(startIndex));
        
        while (!remaining.isEmpty()) {
            Point current = result.get(result.size() - 1);
            Point nearest = Collections.min(remaining, 
                (a, b) -> Double.compare(current.distance(a), current.distance(b)));
            result.add(nearest);
            remaining.remove(nearest);
        }
        
        return result;
    }
    
    private ArrayList<Point> improve2Opt(ArrayList<Point> tour) {
        ArrayList<Point> improved = new ArrayList<Point>(tour);
        boolean hasImproved = true;
        
        while (hasImproved) {
            hasImproved = false;
            for (int i = 0; i < improved.size() - 1; i++) {
                for (int j = i + 2; j < improved.size(); j++) {
                    if (j == improved.size() - 1 && i == 0) continue;
                    
                    double oldDist = improved.get(i).distance(improved.get(i + 1)) + 
                                   improved.get(j).distance(improved.get((j + 1) % improved.size()));
                    double newDist = improved.get(i).distance(improved.get(j)) + 
                                   improved.get(i + 1).distance(improved.get((j + 1) % improved.size()));
                    
                    if (newDist < oldDist) {
                        Collections.reverse(improved.subList(i + 1, j + 1));
                        hasImproved = true;
                        break;
                    }
                }
                if (hasImproved) break;
            }
        }
        
        return improved;
    }
    
    private ArrayList<Point> improve2OptAggressive(ArrayList<Point> tour) {
        ArrayList<Point> improved = new ArrayList<Point>(tour);
        
        // 多轮2-opt优化
        for (int round = 0; round < 3; round++) {
            boolean hasImproved = true;
            
            while (hasImproved) {
                hasImproved = false;
                double bestImprovement = 0;
                int bestI = -1, bestJ = -1;
                
                // 寻找最佳改善
                for (int i = 0; i < improved.size() - 1; i++) {
                    for (int j = i + 2; j < improved.size(); j++) {
                        if (j == improved.size() - 1 && i == 0) continue;
                        
                        double oldDist = improved.get(i).distance(improved.get(i + 1)) + 
                                       improved.get(j).distance(improved.get((j + 1) % improved.size()));
                        double newDist = improved.get(i).distance(improved.get(j)) + 
                                       improved.get(i + 1).distance(improved.get((j + 1) % improved.size()));
                        
                        double improvement = oldDist - newDist;
                        if (improvement > bestImprovement) {
                            bestImprovement = improvement;
                            bestI = i;
                            bestJ = j;
                            hasImproved = true;
                        }
                    }
                }
                
                // 应用最佳改善
                if (hasImproved && bestI >= 0 && bestJ >= 0) {
                    Collections.reverse(improved.subList(bestI + 1, bestJ + 1));
                }
            }
            
            // 扰动避免局部最优
            if (round < 2) {
                int p1 = (int)(Math.random() * improved.size());
                int p2 = (int)(Math.random() * improved.size());
                if (Math.abs(p1 - p2) > 1) {
                    Collections.swap(improved, p1, p2);
                }
            }
        }
        
        return improved;
    }
    
    private ArrayList<Point> improve3Opt(ArrayList<Point> tour) {
        ArrayList<Point> improved = new ArrayList<Point>(tour);
        boolean hasImproved = true;
        int maxIterations = 10; // Limiter pour éviter la complexité excessive
        
        while (hasImproved && maxIterations-- > 0) {
            hasImproved = false;
            for (int i = 0; i < improved.size() - 3; i++) {
                Point a = improved.get((i + 1) % improved.size());
                Point b = improved.get((i + 2) % improved.size());
                Point c = improved.get((i + 3) % improved.size());
                
                // Essayer quelques réarrangements
                ArrayList<Point> test = new ArrayList<Point>(improved);
                Collections.swap(test, (i + 1) % test.size(), (i + 2) % test.size());
                
                if (calculateTotalDistance(test) < calculateTotalDistance(improved)) {
                    improved = test;
                    hasImproved = true;
                    break;
                }
            }
        }
        
        return improved;
    }
    
    private double calculateTotalDistance(ArrayList<Point> tour) {
        double total = 0;
        for (int i = 0; i < tour.size(); i++) {
            total += tour.get(i).distance(tour.get((i + 1) % tour.size()));
        }
        return total;
    }
    
    private Circle welzlCircle(ArrayList<Point> P, ArrayList<Point> R) {
        if (P.isEmpty() || R.size() == 3) {
            return minimumCircleFromPoints(R);
        }
        
        Point p = P.remove(P.size() - 1);
        Circle circle = welzlCircle(P, R);
        
        if (circle != null && isPointInCircle(p, circle)) {
            return circle;
        }
        
        R.add(p);
        circle = welzlCircle(P, R);
        R.remove(p);
        
        return circle;
    }
    
    private Circle minimumCircleFromPoints(ArrayList<Point> points) {
        if (points.isEmpty()) {
            return new Circle(new Point(0, 0), 0);
        } else if (points.size() == 1) {
            return new Circle(points.get(0), 0);
        } else if (points.size() == 2) {
            Point p1 = points.get(0), p2 = points.get(1);
            Point center = new Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2);
            int radius = (int) Math.ceil(p1.distance(p2) / 2);
            return new Circle(center, radius);
        } else {
            // 3 points: chercher le cercle circonscrit
            Point p1 = points.get(0), p2 = points.get(1), p3 = points.get(2);
            
            // Calcul du centre du cercle circonscrit
            double d = 2 * (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y));
            if (Math.abs(d) < 1e-10) {
                // Points colinéaires, utiliser le diamètre
                return minimumCircleFromPoints(new ArrayList<Point>(points.subList(0, 2)));
            }
            
            double ux = ((p1.x * p1.x + p1.y * p1.y) * (p2.y - p3.y) + 
                        (p2.x * p2.x + p2.y * p2.y) * (p3.y - p1.y) + 
                        (p3.x * p3.x + p3.y * p3.y) * (p1.y - p2.y)) / d;
            double uy = ((p1.x * p1.x + p1.y * p1.y) * (p3.x - p2.x) + 
                        (p2.x * p2.x + p2.y * p2.y) * (p1.x - p3.x) + 
                        (p3.x * p3.x + p3.y * p3.y) * (p2.x - p1.x)) / d;
            
            Point center = new Point((int) Math.round(ux), (int) Math.round(uy));
            int radius = (int) Math.ceil(center.distance(p1));
            return new Circle(center, radius);
        }
    }
    
    private boolean isPointInCircle(Point p, Circle c) {
        return p.distance(c.getCenter()) <= c.getRadius() + 1e-9;
    }
    
    private double crossProduct(Point O, Point A, Point B) {
        return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
    }

    // 最远插入算法
    private ArrayList<Point> farthestInsertionTSP(ArrayList<Point> points) {
        if (points.size() <= 2) return new ArrayList<Point>(points);
        
        ArrayList<Point> tour = new ArrayList<Point>();
        ArrayList<Point> remaining = new ArrayList<Point>(points);
        
        // 找到最远的两个点作为起始
        Point p1 = points.get(0), p2 = points.get(1);
        double maxDist = p1.distance(p2);
        
        for (int i = 0; i < points.size(); i++) {
            for (int j = i + 1; j < points.size(); j++) {
                double dist = points.get(i).distance(points.get(j));
                if (dist > maxDist) {
                    maxDist = dist;
                    p1 = points.get(i);
                    p2 = points.get(j);
                }
            }
        }
        
        tour.add(p1);
        tour.add(p2);
        remaining.remove(p1);
        remaining.remove(p2);
        
        // 重复插入距离tour最远的点
        while (!remaining.isEmpty()) {
            Point farthest = null;
            double maxMinDist = -1;
            
            for (Point candidate : remaining) {
                double minDistToTour = Double.MAX_VALUE;
                for (Point tourPoint : tour) {
                    minDistToTour = Math.min(minDistToTour, candidate.distance(tourPoint));
                }
                if (minDistToTour > maxMinDist) {
                    maxMinDist = minDistToTour;
                    farthest = candidate;
                }
            }
            
            // 找到最佳插入位置
            int bestPos = 0;
            double minIncrease = Double.MAX_VALUE;
            
            for (int i = 0; i <= tour.size(); i++) {
                Point prev = tour.get((i - 1 + tour.size()) % tour.size());
                Point next = tour.get(i % tour.size());
                
                double increase = prev.distance(farthest) + farthest.distance(next) - prev.distance(next);
                if (increase < minIncrease) {
                    minIncrease = increase;
                    bestPos = i;
                }
            }
            
            tour.add(bestPos, farthest);
            remaining.remove(farthest);
        }
        
        return tour;
    }
    
    private ArrayList<Point> cheapestInsertionTSP(ArrayList<Point> points) {
        if (points.size() <= 2) return new ArrayList<Point>(points);
        
        ArrayList<Point> tour = new ArrayList<Point>();
        ArrayList<Point> remaining = new ArrayList<Point>(points);
        
        // 从最近的三点开始构建初始三角形
        tour.add(remaining.remove(0));
        
        Point nearest = Collections.min(remaining, 
            (a, b) -> Double.compare(tour.get(0).distance(a), tour.get(0).distance(b)));
        tour.add(nearest);
        remaining.remove(nearest);
        
        if (!remaining.isEmpty()) {
            nearest = Collections.min(remaining, 
                (a, b) -> Double.compare(
                    Math.min(tour.get(0).distance(a), tour.get(1).distance(a)),
                    Math.min(tour.get(0).distance(b), tour.get(1).distance(b))
                ));
            tour.add(nearest);
            remaining.remove(nearest);
        }
        
        // 重复插入成本最低的点
        while (!remaining.isEmpty()) {
            Point bestPoint = null;
            int bestPos = 0;
            double minCost = Double.MAX_VALUE;
            
            for (Point candidate : remaining) {
                for (int i = 0; i < tour.size(); i++) {
                    Point prev = tour.get(i);
                    Point next = tour.get((i + 1) % tour.size());
                    
                    double cost = prev.distance(candidate) + candidate.distance(next) - prev.distance(next);
                    if (cost < minCost) {
                        minCost = cost;
                        bestPoint = candidate;
                        bestPos = i + 1;
                    }
                }
            }
            
            tour.add(bestPos, bestPoint);
            remaining.remove(bestPoint);
        }
        
        return tour;
    }

    // Or-opt优化算法
    private ArrayList<Point> improveOrOpt(ArrayList<Point> tour) {
        ArrayList<Point> improved = new ArrayList<Point>(tour);
        boolean hasImproved = true;
        
        while (hasImproved) {
            hasImproved = false;
            
            // Or-opt: 移动1个、2个或3个连续城市到其他位置
            for (int segmentLength = 1; segmentLength <= 3; segmentLength++) {
                for (int i = 0; i <= improved.size() - segmentLength; i++) {
                    // 当前段的前后城市
                    Point prev = improved.get((i - 1 + improved.size()) % improved.size());
                    Point next = improved.get((i + segmentLength) % improved.size());
                    
                    // 计算移除这个段的成本节省
                    double removeCost = prev.distance(improved.get(i)) + 
                                      improved.get((i + segmentLength - 1) % improved.size()).distance(next) - 
                                      prev.distance(next);
                    
                    // 尝试插入到其他位置
                    for (int j = 0; j < improved.size(); j++) {
                        if (j >= i && j <= i + segmentLength) continue; // 跳过原位置附近
                        
                        Point insertPrev = improved.get(j);
                        Point insertNext = improved.get((j + 1) % improved.size());
                        
                        // 计算在新位置插入的成本
                        double insertCost = insertPrev.distance(improved.get(i)) + 
                                          improved.get((i + segmentLength - 1) % improved.size()).distance(insertNext) - 
                                          insertPrev.distance(insertNext);
                        
                        if (insertCost < removeCost) {
                            // 执行Or-opt移动 - 简化版本避免复杂索引操作
                            ArrayList<Point> newTour = new ArrayList<Point>();
                            
                            // 构建新的tour：复制原tour但跳过要移动的段，在目标位置插入
                            for (int pos = 0; pos < improved.size(); pos++) {
                                if (pos == j + 1) {
                                    // 在此位置插入移动的段
                                    for (int k = 0; k < segmentLength; k++) {
                                        newTour.add(improved.get(i + k));
                                    }
                                }
                                
                                // 添加原点（除了要移动的段）
                                if (pos < i || pos >= i + segmentLength) {
                                    newTour.add(improved.get(pos));
                                }
                            }
                            
                            if (newTour.size() == improved.size()) {
                                improved = newTour;
                                hasImproved = true;
                                break;
                            }
                        }
                    }
                    if (hasImproved) break;
                }
                if (hasImproved) break;
            }
        }
        
        return improved;
    }
    
    private ArrayList<Point> improveAdvanced3Opt(ArrayList<Point> tour) {
        ArrayList<Point> improved = new ArrayList<Point>(tour);
        boolean hasImproved = true;
        int maxIterations = 5; // 限制迭代避免超时
        
        while (hasImproved && maxIterations-- > 0) {
            hasImproved = false;
            
            for (int i = 0; i < improved.size() - 5; i++) {
                for (int j = i + 2; j < improved.size() - 3; j++) {
                    for (int k = j + 2; k < improved.size() - 1; k++) {
                        // 3-opt的7种重连方式中选择最好的
                        ArrayList<Point> bestConfig = null;
                        double bestImprovement = 0;
                        
                        // 当前配置的成本
                        double currentCost = improved.get(i).distance(improved.get(i + 1)) +
                                           improved.get(j).distance(improved.get(j + 1)) +
                                           improved.get(k).distance(improved.get((k + 1) % improved.size()));
                        
                        // 尝试几种主要的3-opt重连
                        ArrayList<Point>[] configs = new ArrayList[4];
                        
                        // 配置1: 2-opt在(i,i+1)和(j,j+1)
                        configs[0] = new ArrayList<Point>(improved);
                        reverse(configs[0], i + 1, j);
                        
                        // 配置2: 2-opt在(j,j+1)和(k,k+1)  
                        configs[1] = new ArrayList<Point>(improved);
                        reverse(configs[1], j + 1, k);
                        
                        // 配置3: 2-opt在(i,i+1)和(k,k+1)
                        configs[2] = new ArrayList<Point>(improved);
                        reverse(configs[2], i + 1, k);
                        
                        // 配置4: 完整3-opt重连
                        configs[3] = new ArrayList<Point>();
                        for (int l = 0; l <= i; l++) configs[3].add(improved.get(l));
                        for (int l = j + 1; l <= k; l++) configs[3].add(improved.get(l));
                        for (int l = i + 1; l <= j; l++) configs[3].add(improved.get(l));
                        for (int l = k + 1; l < improved.size(); l++) configs[3].add(improved.get(l));
                        
                        for (ArrayList<Point> config : configs) {
                            if (config == null || config.size() != improved.size()) continue;
                            
                            // 使用完整距离比较而不是局部边
                            double newTotalCost = calculateTotalDistance(config);
                            double currentTotalCost = calculateTotalDistance(improved);
                            
                            double improvement = currentTotalCost - newTotalCost;
                            if (improvement > bestImprovement) {
                                bestImprovement = improvement;
                                bestConfig = config;
                            }
                        }
                        
                        if (bestConfig != null) {
                            improved = bestConfig;
                            hasImproved = true;
                            break;
                        }
                    }
                    if (hasImproved) break;
                }
                if (hasImproved) break;
            }
        }
        
        return improved;
    }
    
    private void reverse(ArrayList<Point> tour, int start, int end) {
        while (start < end) {
            Collections.swap(tour, start, end);
            start++;
            end--;
        }
    }

    // Algorithmes TSP avancés
    
    // 高效最近邻算法
    private ArrayList<Point> nearestNeighborTSP(ArrayList<Point> points) {
        if (points.isEmpty()) return new ArrayList<Point>();
        
        ArrayList<Point> tour = new ArrayList<Point>();
        ArrayList<Point> unvisited = new ArrayList<Point>(points);
        
        Point current = unvisited.remove(0);
        tour.add(current);
        
        while (!unvisited.isEmpty()) {
            Point nearest = null;
            double minDist = Double.MAX_VALUE;
            
            for (Point candidate : unvisited) {
                double dist = current.distance(candidate);
                if (dist < minDist) {
                    minDist = dist;
                    nearest = candidate;
                }
            }
            
            tour.add(nearest);
            unvisited.remove(nearest);
            current = nearest;
        }
        
        return tour;
    }
    
    // Lin-Kernighan优化算法 (简化版)
    private ArrayList<Point> linKernighanOptimization(ArrayList<Point> tour) {
        ArrayList<Point> improved = new ArrayList<Point>(tour);
        
        for (int round = 0; round < 3; round++) {
            boolean hasImproved = true;
            
            while (hasImproved) {
                hasImproved = false;
                double bestImprovement = 0;
                int bestI = -1, bestJ = -1;
                
                for (int i = 0; i < improved.size() - 1; i++) {
                    for (int j = i + 2; j < improved.size(); j++) {
                        if (j == improved.size() - 1 && i == 0) continue;
                        
                        double oldDist = improved.get(i).distance(improved.get(i + 1)) + 
                                       improved.get(j).distance(improved.get((j + 1) % improved.size()));
                        double newDist = improved.get(i).distance(improved.get(j)) + 
                                       improved.get(i + 1).distance(improved.get((j + 1) % improved.size()));
                        
                        double improvement = oldDist - newDist;
                        if (improvement > bestImprovement) {
                            bestImprovement = improvement;
                            bestI = i;
                            bestJ = j;
                            hasImproved = true;
                        }
                    }
                }
                
                if (hasImproved && bestI >= 0 && bestJ >= 0) {
                    Collections.reverse(improved.subList(bestI + 1, bestJ + 1));
                }
            }
        }
        
        return improved;
    }
    
    // Christofides近似算法 (简化版)
    private ArrayList<Point> christofidesApproximation(ArrayList<Point> points) {
        ArrayList<Point> tour = nearestNeighborTSP(points);
        
        for (int i = 0; i < 10; i++) {
            tour = enhance2Opt(tour);
        }
        
        return tour;
    }
    
    // 改进的最远插入算法
    private ArrayList<Point> farthestInsertionAdvanced(ArrayList<Point> points) {
        if (points.size() <= 2) return new ArrayList<Point>(points);
        
        ArrayList<Point> tour = new ArrayList<Point>();
        ArrayList<Point> remaining = new ArrayList<Point>(points);
        
        // 找到中心区域的两个点作为起始
        Point center = findCentroid(points);
        Point p1 = points.get(0);
        Point p2 = points.get(1);
        double minDist = Double.MAX_VALUE;
        
        // 选择距离中心最近的两个点
        for (int i = 0; i < points.size(); i++) {
            for (int j = i + 1; j < points.size(); j++) {
                double distSum = points.get(i).distance(center) + points.get(j).distance(center);
                if (distSum < minDist) {
                    minDist = distSum;
                    p1 = points.get(i);
                    p2 = points.get(j);
                }
            }
        }
        
        tour.add(p1);
        tour.add(p2);
        remaining.remove(p1);
        remaining.remove(p2);
        
        // 插入剩余点
        while (!remaining.isEmpty()) {
            Point farthest = null;
            double maxMinDist = -1;
            
            for (Point candidate : remaining) {
                double minDistToTour = Double.MAX_VALUE;
                for (Point tourPoint : tour) {
                    minDistToTour = Math.min(minDistToTour, candidate.distance(tourPoint));
                }
                if (minDistToTour > maxMinDist) {
                    maxMinDist = minDistToTour;
                    farthest = candidate;
                }
            }
            
            // 找到最佳插入位置
            int bestPos = 0;
            double minIncrease = Double.MAX_VALUE;
            
            for (int i = 0; i <= tour.size(); i++) {
                Point prev = tour.get((i - 1 + tour.size()) % tour.size());
                Point next = tour.get(i % tour.size());
                
                double oldDist = prev.distance(next);
                double newDist = prev.distance(farthest) + farthest.distance(next);
                double increase = newDist - oldDist;
                
                if (increase < minIncrease) {
                    minIncrease = increase;
                    bestPos = i;
                }
            }
            
            tour.add(bestPos, farthest);
            remaining.remove(farthest);
        }
        
        return tour;
    }
    
    // 高级3-opt优化
    private ArrayList<Point> advanced3OptOptimization(ArrayList<Point> tour) {
        ArrayList<Point> improved = new ArrayList<Point>(tour);
        
        for (int round = 0; round < 5; round++) {
            boolean hasImproved = true;
            
            while (hasImproved) {
                hasImproved = false;
                
                for (int i = 0; i < improved.size() - 3; i++) {
                    for (int j = i + 2; j < improved.size() - 1; j++) {
                        for (int k = j + 2; k < improved.size(); k++) {
                            if (k == improved.size() - 1 && i == 0) continue;
                            
                            // 尝试3-opt交换
                            ArrayList<Point> newTour = apply3OptMove(improved, i, j, k);
                            if (calculateTotalDistance(newTour) < calculateTotalDistance(improved)) {
                                improved = newTour;
                                hasImproved = true;
                            }
                        }
                    }
                }
            }
        }
        
        return improved;
    }
    
    // 应用3-opt移动
    private ArrayList<Point> apply3OptMove(ArrayList<Point> tour, int i, int j, int k) {
        ArrayList<Point> newTour = new ArrayList<Point>();
        
        // 复制第一段 [0, i]
        for (int idx = 0; idx <= i; idx++) {
            newTour.add(tour.get(idx));
        }
        
        // 反转第二段 [i+1, j]
        for (int idx = j; idx >= i + 1; idx--) {
            newTour.add(tour.get(idx));
        }
        
        // 反转第三段 [j+1, k]
        for (int idx = k; idx >= j + 1; idx--) {
            newTour.add(tour.get(idx));
        }
        
        // 复制剩余段 [k+1, end]
        for (int idx = k + 1; idx < tour.size(); idx++) {
            newTour.add(tour.get(idx));
        }
        
        return newTour;
    }
    
    // 强化2-opt算法
    private ArrayList<Point> enhance2Opt(ArrayList<Point> tour) {
        ArrayList<Point> improved = new ArrayList<Point>(tour);
        boolean hasImproved = true;
        
        while (hasImproved) {
            hasImproved = false;
            
            for (int i = 0; i < improved.size() - 1; i++) {
                for (int j = i + 2; j < improved.size(); j++) {
                    if (j == improved.size() - 1 && i == 0) continue;
                    
                    double oldDist = improved.get(i).distance(improved.get(i + 1)) + 
                                   improved.get(j).distance(improved.get((j + 1) % improved.size()));
                    double newDist = improved.get(i).distance(improved.get(j)) + 
                                   improved.get(i + 1).distance(improved.get((j + 1) % improved.size()));
                    
                    if (newDist < oldDist) {
                        Collections.reverse(improved.subList(i + 1, j + 1));
                        hasImproved = true;
                    }
                }
            }
        }
        
        return improved;
    }
    
    // 计算点集的质心
    private Point findCentroid(ArrayList<Point> points) {
        double sumX = 0, sumY = 0;
        for (Point p : points) {
            sumX += p.x;
            sumY += p.y;
        }
        return new Point((int)(sumX / points.size()), (int)(sumY / points.size()));
    }
}
