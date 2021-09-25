#!/bin/bash

counter=0
path=$1

echo "Changing names..."

for filename in $path/*.jpg; 
do
   name=$(basename -- $filename)
   echo "$name -> $counter.jpg"
   mv $filename ${path}/${counter}.jpg
   counter=$((counter+1))
done

echo "$counter names changed"
