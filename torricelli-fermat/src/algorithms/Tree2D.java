package algorithms;

import java.awt.Color;
import java.awt.Point;
import java.util.ArrayList;

public class Tree2D {
  private Point root;
  private ArrayList<Tree2D> subtrees;

  public Tree2D (Point p, ArrayList<Tree2D> trees){
    this.root=p;
    this.subtrees=trees;
  }
  public Point getRoot(){
    return this.root;
  }
  public ArrayList<Tree2D> getSubTrees(){
    return this.subtrees;
  }
  public double distanceRootToSubTrees(){
    double d=0;
    for (int i=0;i<this.subtrees.size();i++){
      d+= Math.sqrt(Math.pow(this.root.getX()-this.subtrees.get(i).getRoot().getX(),2)+Math.pow(this.root.getY()-this.subtrees.get(i).getRoot().getY(),2));
    }
    return d;
  }
}
