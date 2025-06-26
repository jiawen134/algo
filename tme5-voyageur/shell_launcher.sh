#!/bin/bash
rm -rf javaBeans; mkdir javaBeans
find . -name "*.java" > sourcefiles
javac -cp jars/* -s src/ -d javaBeans/ @sourcefiles
rm sourcefiles
java -cp javaBeans:jars/supportGUI.jar supportGUI.DiamRace -nbPoints 1000
