#!/bin/bash
sub_path=$1
sub_90_path=$2
dir=90
for sub in `ls $sub_path`;
do
    echo $sub $dir
    #cp -r ${sub_90_path}/${sub}/diffusion/${dir} ${sub_path}/${sub}/diffusion
    cp -r ${sub_90_path}/${sub}/masks ${sub_path}/${sub}/
done