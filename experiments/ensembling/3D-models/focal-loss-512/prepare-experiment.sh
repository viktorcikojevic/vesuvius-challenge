
#!/bin/bash

# Base directory
BASE_DIR="/home/viktor/Documents/kaggle/vesuvius-challenge/experiments/ensembling/3D-models/bce-dice"

# Folds to create
FOLDS=("1" "2" "3")

# Depths to create
DEPTHS=("18" "34")

# Images to create
IMAGES=("256" "512")

# Script file to be modified
SCRIPT_FILE="vesuvius-challenge-3d-resnet-training-step-2-with-augm.py"

# Iterate over all FOLDS
for fold in ${FOLDS[@]}
do
    # Iterate over all DEPTHS
    for depth in ${DEPTHS[@]}
    do
        # Iterate over all IMAGES
        for image in ${IMAGES[@]}
        do
            # Create new directory path
            NEW_DIR="${BASE_DIR}/fold-${fold}/depth-${depth}/image-${image}"
            
            # Create the directory
            mkdir -p $NEW_DIR
            
            # Copy script file to the new directory
            cp /home/viktor/Documents/kaggle/vesuvius-challenge/experiments/ensembling/3D-models/bce-dice/fold-1/depth-18/image-256/vesuvius-challenge-3d-resnet-training-step-2-with-augm.py ${NEW_DIR}
            cp -r /home/viktor/Documents/kaggle/vesuvius-challenge/experiments/ensembling/3D-models/bce-dice/fold-1/depth-18/image-256/resnet3d ${NEW_DIR}
            
            sed -i "s/FOLD = 1/FOLD = ${fold}/" ${NEW_DIR}/${SCRIPT_FILE}
            sed -i "s/RESNET_DEPTH = 18/RESNET_DEPTH = ${depth}/" ${NEW_DIR}/${SCRIPT_FILE}
            sed -i "s/CROP_SIZE = 256/CROP_SIZE = ${image}/" ${NEW_DIR}/${SCRIPT_FILE}

            cd ${NEW_DIR}
            ipython ${SCRIPT_FILE}
            cd ${BASE_DIR}

        done
    done
done
