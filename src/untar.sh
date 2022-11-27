#!/bin/bash
FILES="$(dirname $(pwd))/image_data_2/*/"
for f in $FILES
do
  for k in $f/*.tar
  do
    echo "$k"
    # take action on each file. $f store current file name
    tar -xzf $k -C $f
  done
done