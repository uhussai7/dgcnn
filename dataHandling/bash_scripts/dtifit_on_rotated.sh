#!/bin/bash
sub_path=$1
dgcnn_path=/home/u2hussai/dgcnn/dataHandling/bash_scripts/
for sub in `ls $sub_path`;
do
    for dir in {6,90};
    do
            echo $sub $dir
            ${dgcnn_path}dtifit_on_subjects.sh ${sub_path}/${sub}/diffusion/${dir}/diffusion/ ${sub_path}/${sub}/diffusion/${dir}/dtifit/
    done
done
        
