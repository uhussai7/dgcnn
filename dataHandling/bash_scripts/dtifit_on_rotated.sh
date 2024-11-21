#!/bin/bash
sub_path=$1
dgcnn_path=/home/u2hussai/dgcnn/dataHandling/bash_scripts/
for sub in `ls $sub_path`;
do
    for dir in {6,90};
    do
        # echo $sub $dir
        # rm ${sub_path}/${sub}/diffusion/${dir}/dtifit/dtifit_*.nii.gz
        #rm ${sub_path}/${sub}/diffusion/${dir}/dtifit/dtifit_V1.nii.gz
        ${dgcnn_path}dtifit_on_subjects.sh ${sub_path}/${sub}/diffusion/${dir}/diffusion/ ${sub_path}/${sub}/diffusion/${dir}/dtifit/
    done
done
        
# for dir in {15,24,34,43,52,62,71,80};
#     do
#         if [[ "$dir" -ne 6 && "$dir" -ne 90 ]]; then
#             rm -r ${sub_path}/${sub}/diffusion/${dir}/
#         fi
#     done