#!/bin/bash

DATASETS_PATH="/content/drive/MyDrive/MaskArchitectureAnomaly_CourseProject-main/MaskArchitectureAnomaly_CourseProject-main/Validation_Dataset"
MODELS_PATH="/content/drive/MyDrive/MaskArchitectureAnomaly_CourseProject-main/MaskArchitectureAnomaly_CourseProject-main/trained_models"
EOMT_PATH="/content/drive/MyDrive/MaskArchitectureAnomaly_CourseProject-main/MaskArchitectureAnomaly_CourseProject-main/eomt"
CONFIG_CS="$EOMT_PATH/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml"
CACHE="/content/drive/MyDrive/MaskArchitectureAnomaly_CourseProject-main/MaskArchitectureAnomaly_CourseProject-main/logits_cache"

# EoMT-Cityscapes
for DATASET in "RoadAnomaly21/images/*.png" "RoadObsticle21/images/*.webp" "FS_LostFound_full/images/*.png" "fs_static/images/*.jpg" "RoadAnomaly/images/*.jpg"; do
    DATASET_NAME=$(echo $DATASET | cut -d'/' -f1)
    echo "Cityscapes - $DATASET_NAME"
    python ../save_logits_eomt.py \
        --input "$DATASETS_PATH/$DATASET" \
        --config "$CONFIG_CS" \
        --ckpt "$MODELS_PATH/eomt_cityscapes.bin" \
        --output-dir "$CACHE/cityscapes/$DATASET_NAME" \
        --img-size 1024 1024 \
        --num-classes 19
done

# EoMT-Finetuned
for DATASET in "RoadAnomaly21/images/*.png" "RoadObsticle21/images/*.webp" "FS_LostFound_full/images/*.png" "fs_static/images/*.jpg" "RoadAnomaly/images/*.jpg"; do
    DATASET_NAME=$(echo $DATASET | cut -d'/' -f1)
    echo "Finetuned - $DATASET_NAME"
    python ../save_logits_eomt.py \
        --input "$DATASETS_PATH/$DATASET" \
        --config "$CONFIG_CS" \
        --ckpt "$MODELS_PATH/eomt_finetuned.ckpt" \
        --output-dir "$CACHE/finetuned/$DATASET_NAME" \
        --img-size 640 640 \
        --num-classes 19 \
        --num-q 200
done

# EoMT-COCO
for DATASET in "RoadAnomaly21/images/*.png" "RoadObsticle21/images/*.webp" "FS_LostFound_full/images/*.png" "fs_static/images/*.jpg" "RoadAnomaly/images/*.jpg"; do
    DATASET_NAME=$(echo $DATASET | cut -d'/' -f1)
    echo "COCO - $DATASET_NAME"
    python ../save_logits_eomt.py \
        --input "$DATASETS_PATH/$DATASET" \
        --config "$CONFIG_CS" \
        --ckpt "$MODELS_PATH/eomt_coco.bin" \
        --output-dir "$CACHE/coco/$DATASET_NAME" \
        --img-size 640 640 \
        --num-classes 133 \
        --num-q 200
done
