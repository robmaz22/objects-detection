#!/bin/bash

counter=0
path=$1

echo "Changing file_names..."

for filename in $path/*.jpg; 
do
   name=$(basename -- $filename)
   echo "$name -> $counter.jpg"
   mv $filename ${path}/${counter}.jpg
   counter=$((counter+1))
done

echo "$counter file_names changed"
