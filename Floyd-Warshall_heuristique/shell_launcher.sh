#!/bin/bash
rm -rf beans; mkdir beans
find . -name "*.java" > sourcefiles
javac -cp jars/* -s src/ -d beans/ @sourcefiles
rm sourcefiles
java -cp ./javabeans:./jars/supportGUI.jar supportGUI.ShortestPathsViewer -nbPoints 1000 -edgeThreshold 55
