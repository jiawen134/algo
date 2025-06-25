package algorithms;

import java.awt.Point;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class DefaultTeam {

  public int[][] calculShortestPaths(ArrayList<Point> points, int edgeThreshold) {
    int n = points.size();
    double[][] dist = new double[n][n];
    int[][] next = new int[n][n];

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        if (i == j) {
          dist[i][j] = 0;
          next[i][j] = j;
        } else {
          double d = points.get(i).distance(points.get(j));
          if (d <= edgeThreshold) {
            dist[i][j] = d;
            next[i][j] = j;
          } else {
            dist[i][j] = Double.POSITIVE_INFINITY;
            next[i][j] = -1;
          }
        }
      }
    }

    for (int k = 0; k < n; k++) {
      for (int i = 0; i < n; i++) {
        if (dist[i][k] == Double.POSITIVE_INFINITY) continue;
        for (int j = 0; j < n; j++) {
          if (dist[k][j] == Double.POSITIVE_INFINITY) continue;
          if (dist[i][j] > dist[i][k] + dist[k][j]) {
            dist[i][j] = dist[i][k] + dist[k][j];
            next[i][j] = next[i][k];
          }
        }
      }
    }
    return next;
  }

  private ArrayList<Integer> reconstructPath(int[][] next, int u, int v) {
    ArrayList<Integer> path = new ArrayList<>();
    if (next[u][v] == -1) return path;

    path.add(u);
    while (u != v) {
      u = next[u][v];
      path.add(u);
    }
    return path;
  }

  private ArrayList<int[]> primMST(double[][] weights) {
    int n = weights.length;
    boolean[] inMST = new boolean[n];
    double[] key = new double[n];
    int[] parent = new int[n];
    for (int i = 0; i < n; i++) {
      key[i] = Double.POSITIVE_INFINITY;
      parent[i] = -1;
    }
    key[0] = 0;

    for (int count = 0; count < n - 1; count++) {
      double min = Double.POSITIVE_INFINITY;
      int u = -1;
      for (int v = 0; v < n; v++) {
        if (!inMST[v] && key[v] < min) {
          min = key[v];
          u = v;
        }
      }
      inMST[u] = true;

      for (int v = 0; v < n; v++) {
        if (!inMST[v] && weights[u][v] < key[v]) {
          key[v] = weights[u][v];
          parent[v] = u;
        }
      }
    }

    ArrayList<int[]> edges = new ArrayList<>();
    for (int v = 1; v < n; v++) {
      edges.add(new int[] { parent[v], v });
    }
    return edges;
  }
  private double distance(Point a, Point b) {
    return a.distance(b);
  }

  public Tree2D calculSteiner(ArrayList<Point> points, int edgeThreshold, ArrayList<Point> hitPoints) {
    int n = points.size();

    int[][] next = calculShortestPaths(points, edgeThreshold);

    ArrayList<Integer> indicesS = new ArrayList<>();
    for (Point p : hitPoints) {
      for (int i = 0; i < n; i++) {
        if (points.get(i).equals(p)) {
          indicesS.add(i);
          break;
        }
      }
    }

    int sSize = indicesS.size();
    double[][] distK = new double[sSize][sSize];
    for (int i = 0; i < sSize; i++) {
      for (int j = 0; j < sSize; j++) {
        if (i == j) distK[i][j] = 0;
        else {
          int u = indicesS.get(i);
          int v = indicesS.get(j);
          double d = 0;
          ArrayList<Integer> pathUV = reconstructPath(next, u, v);
          for (int k = 0; k < pathUV.size() - 1; k++) {
            d += distance(points.get(pathUV.get(k)), points.get(pathUV.get(k + 1)));
          }
          distK[i][j] = d;
        }
      }
    }

    ArrayList<int[]> mstEdgesK = primMST(distK);
    Set<String> edgesH = new HashSet<>();
    Set<Integer> verticesH = new HashSet<>();

    for (int[] edge : mstEdgesK) {
      int idxUinS = edge[0];
      int idxVinS = edge[1];
      int u = indicesS.get(idxUinS);
      int v = indicesS.get(idxVinS);

      ArrayList<Integer> pathUV = reconstructPath(next, u, v);

      for (int i = 0; i < pathUV.size() - 1; i++) {
        int a = pathUV.get(i);
        int b = pathUV.get(i + 1);
        verticesH.add(a);
        verticesH.add(b);
        String key = (a < b) ? a + "," + b : b + "," + a;
        edgesH.add(key);
      }
    }

    List<Integer> listVerticesH = new ArrayList<>(verticesH);
    Map<Integer, Integer> oldToNewIndex = new HashMap<>();
    for (int i = 0; i < listVerticesH.size(); i++) {
      oldToNewIndex.put(listVerticesH.get(i), i);
    }
    int sizeH = listVerticesH.size();
    double[][] distH = new double[sizeH][sizeH];
    for (int i = 0; i < sizeH; i++) {
      for (int j = 0; j < sizeH; j++) {
        distH[i][j] = Double.POSITIVE_INFINITY;
      }
      distH[i][i] = 0;
    }
    for (String e : edgesH) {
      String[] parts = e.split(",");
      int aOld = Integer.parseInt(parts[0]);
      int bOld = Integer.parseInt(parts[1]);
      int aNew = oldToNewIndex.get(aOld);
      int bNew = oldToNewIndex.get(bOld);
      double d = distance(points.get(aOld), points.get(bOld));
      distH[aNew][bNew] = d;
      distH[bNew][aNew] = d;
    }

    ArrayList<int[]> mstEdgesH = primMST(distH);


    Map<Integer, ArrayList<Integer>> adj = new HashMap<>();
    for (int i = 0; i < sizeH; i++) adj.put(i, new ArrayList<>());
    for (int[] e : mstEdgesH) {
      adj.get(e[0]).add(e[1]);
      adj.get(e[1]).add(e[0]);
    }
    boolean[] visited = new boolean[sizeH];
    return buildTree2D(0, adj, listVerticesH, visited, points);
  }

  private Tree2D buildTree2D(int rootIdx, Map<Integer, ArrayList<Integer>> adj, List<Integer> listVerticesH, boolean[] visited, ArrayList<Point> points) {
    visited[rootIdx] = true;
    ArrayList<Tree2D> subtrees = new ArrayList<>();
    for (int nei : adj.get(rootIdx)) {
      if (!visited[nei]) {
        subtrees.add(buildTree2D(nei, adj, listVerticesH, visited, points));
      }
    }
    return new Tree2D(points.get(listVerticesH.get(rootIdx)), subtrees);
  }
}