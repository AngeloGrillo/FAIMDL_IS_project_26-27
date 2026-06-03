#!/bin/bash

DATASETS_PATH="/content/drive/MyDrive/MaskArchitectureAnomaly_CourseProject-main/MaskArchitectureAnomaly_CourseProject-main/Validation_Dataset"
MODELS_PATH="/content/drive/MyDrive/MaskArchitectureAnomaly_CourseProject-main/MaskArchitectureAnomaly_CourseProject-main/trained_models"
EOMT_PATH="/content/drive/MyDrive/MaskArchitectureAnomaly_CourseProject-main/MaskArchitectureAnomaly_CourseProject-main/eomt"

CONFIG_CS="$EOMT_PATH/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml"

# Checkpoint 1: EoMT-Cityscapes
echo "=========================================="
echo "CHECKPOINT: EoMT-Cityscapes"
echo "=========================================="
for METHOD in msp maxlogit maxentropy rba; do
    echo "--- METHOD: $METHOD ---"
    for DATASET in "RoadAnomaly21/images/*.png" "RoadObsticle21/images/*.webp" "FS_LostFound_full/images/*.png" "fs_static/images/*.jpg" "RoadAnomaly/images/*.jpg"; do
        DATASET_NAME=$(echo $DATASET | cut -d'/' -f1)
        echo "Dataset: $DATASET_NAME"
        python ../evalAnomaly_eomt.py \
            --input "$DATASETS_PATH/$DATASET" \
            --config "$CONFIG_CS" \
            --ckpt "$MODELS_PATH/eomt_cityscapes.bin" \
            --method $METHOD \
            --img-size 1024 1024 \
            --num-classes 19
    done
done

# Checkpoint 2: EoMT-COCO
echo "=========================================="
echo "CHECKPOINT: EoMT-COCO"
echo "=========================================="
for METHOD in msp maxlogit maxentropy rba; do
    echo "--- METHOD: $METHOD ---"
    for DATASET in "RoadAnomaly21/images/*.png" "RoadObsticle21/images/*.webp" "FS_LostFound_full/images/*.png" "fs_static/images/*.jpg" "RoadAnomaly/images/*.jpg"; do
        DATASET_NAME=$(echo $DATASET | cut -d'/' -f1)
        echo "Dataset: $DATASET_NAME"
        python ../evalAnomaly_eomt.py \
            --input "$DATASETS_PATH/$DATASET" \
            --config "$CONFIG_CS" \
            --ckpt "$MODELS_PATH/eomt_coco.bin" \
            --method $METHOD \
            --img-size 640 640 \
            --num-classes 19 \
            --num-q 200
    done
done

# Checkpoint 3: EoMT-Finetuned
echo "=========================================="
echo "CHECKPOINT: EoMT-Finetuned"
echo "=========================================="
for METHOD in msp maxlogit maxentropy rba; do
    echo "--- METHOD: $METHOD ---"
    for DATASET in "RoadAnomaly21/images/*.png" "RoadObsticle21/images/*.webp" "FS_LostFound_full/images/*.png" "fs_static/images/*.jpg" "RoadAnomaly/images/*.jpg"; do
        DATASET_NAME=$(echo $DATASET | cut -d'/' -f1)
        echo "Dataset: $DATASET_NAME"
        python ../evalAnomaly_eomt.py \
            --input "$DATASETS_PATH/$DATASET" \
            --config "$CONFIG_CS" \
            --ckpt "$MODELS_PATH/eomt_finetuned.ckpt" \
            --method $METHOD \
            --img-size 640 640 \
            --num-classes 19 \
            --num-q 200
    done
done
