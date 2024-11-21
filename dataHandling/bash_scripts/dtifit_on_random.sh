#!/bin/bash
sub_path=$1
dgcnn_path=/home/u2hussai/dgcnn/dataHandling/bash_scripts/
dir=6
for sub in `ls $sub_path`;
do
    echo $sub $dir
    mkdir ${sub_path}/${sub}/diffusion/${dir}/dtifit/
    ${dgcnn_path}dtifit_on_subjects.sh ${sub_path}/${sub}/diffusion/${dir}/diffusion/ ${sub_path}/${sub}/diffusion/${dir}/dtifit/
done
        
