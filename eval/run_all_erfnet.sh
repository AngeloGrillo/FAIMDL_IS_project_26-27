#!/bin/bash

DATASETS_PATH="/content/drive/MyDrive/MaskArchitectureAnomaly_CourseProject-main/MaskArchitectureAnomaly_CourseProject-main/Validation_Dataset"
MODELS_PATH="/content/drive/MyDrive/MaskArchitectureAnomaly_CourseProject-main/MaskArchitectureAnomaly_CourseProject-main/trained_models"

for METHOD in msp maxlogit maxentropy; do
    echo "===== METHOD: $METHOD ====="
    
    echo "--- RoadAnomaly21 ---"
    python evalAnomaly_all.py --input "$DATASETS_PATH/RoadAnomaly21/images/*.png" \
        --loadDir "$MODELS_PATH/" --loadWeights "erfnet_pretrained.pth" --method $METHOD

    echo "--- RoadObsticle21 ---"
    python evalAnomaly_all.py --input "$DATASETS_PATH/RoadObsticle21/images/*.webp" \
        --loadDir "$MODELS_PATH/" --loadWeights "erfnet_pretrained.pth" --method $METHOD

    echo "--- fs_static ---"
    python evalAnomaly_all.py --input "$DATASETS_PATH/fs_static/images/*.jpg" \
        --loadDir "$MODELS_PATH/" --loadWeights "erfnet_pretrained.pth" --method $METHOD

    echo "--- FS_LostFound_full ---"
    python evalAnomaly_all.py --input "$DATASETS_PATH/FS_LostFound_full/images/*.png" \
        --loadDir "$MODELS_PATH/" --loadWeights "erfnet_pretrained.pth" --method $METHOD

    echo "--- RoadAnomaly ---"
    python evalAnomaly_all.py --input "$DATASETS_PATH/RoadAnomaly/images/*.jpg" \
        --loadDir "$MODELS_PATH/" --loadWeights "erfnet_pretrained.pth" --method $METHOD
done
