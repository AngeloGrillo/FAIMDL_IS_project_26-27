# Mask Architecture for Road Scenes
This is the starting repository for two projects:
- Mask Architecture Anomaly Segmentation for Road Scenes  [[Project Description](https://drive.google.com/file/d/1Vz08DHsP_mojpCTAQTR6NHVq-2rEqAZM/view?usp=sharing)]
- Comprehensive Road Scene Understanding for Autonomous Driving  [[Project Description](https://drive.google.com/file/d/1tq5F_j_8O2vlGWbkU1ayPjYvCml1VEwr/view?usp=sharing)]

This repository consists of the code base for training/testing ERFNet on the Cityscapes dataset and perform anomaly segmentation. It also contains some code referring to EoMT. Some of this code may be unnecessary for your project.

## Folders
For instructions, please refer to the README in each folder:

* [eval](eval) contains tools for evaluating/visualizing an ERFNet model's output and performing anomaly segmentation.
* [trained_models](trained_models) Contains the ERFNet trained models for the baseline eval. 
* [eomt](eomt) It is almost the original folder of the EoMT project. Inside it you will find code to train and pretrained checkpoints for EoMT.

## Our Contributions (Project: Comprehensive Road Scene Understanding)

This section documents the work done for **Points 7 and 8** of the project.

### Point 4 — EoMT-Cityscapes vs EoMT-COCO Comparison

We implemented `eval/step4_eval.py` to compare EoMT pretrained on Cityscapes vs COCO on the Cityscapes validation set. Key contributions:
- COCO-to-Cityscapes class mapping via `_COCO_NAME_TO_CS_ID` dictionary
- Per-class IoU evaluation for both models

Results:
- **EoMT-Cityscapes mIoU: 81.68%**
- **EoMT-COCO (no fine-tuning) mIoU: 54.33%**

### Point 5 — Fine-tuning EoMT-COCO on Cityscapes

We fine-tuned EoMT-COCO on Cityscapes using the EoMT training pipeline. Key choices:
- Backbone frozen (`lr_mult=0.0`) to avoid overfitting with limited resources
- `num_q=200`, `img_size=640x640`, 5 epochs on a single GPU
- Checkpoint saved as `trained_models/eomt_finetuned.ckpt`

Results:
- **EoMT-Finetuned mIoU: 78.35%** (+24% over COCO baseline, -3.33% vs Cityscapes)

### Point 7 — Pixel-based Anomaly Detection Baselines (ERFNet)

We extended the original `evalAnomaly.py` script into `eval/evalAnomaly_all.py`, which supports:
- Three post-hoc anomaly scoring methods: **MSP**, **MaxLogit**, **MaxEntropy**
- Evaluation on all 5 anomaly datasets: RoadAnomaly21, RoadObsticle21, FS LostFound, FS Static, RoadAnomaly
- ERFNet mIoU on Cityscapes val: **72.20%**

### Point 8 — Mask-based Anomaly Detection Baselines (EoMT)

We implemented `eval/evalAnomaly_eomt.py` to adapt the EoMT mask architecture for anomaly detection inference. Key contributions:
- Support for all 3 checkpoints: **EoMT-Cityscapes**, **EoMT-COCO**, **EoMT-Finetuned**
- Four anomaly scoring methods: **MSP**, **MaxLogit**, **MaxEntropy**, **RbA**
- Correct preprocessing pipeline (uint8 tensor input for `window_imgs_semantic`)
- Handling of class count differences across checkpoints (19 / 133 classes)

### Temperature Scaling

Following the PRO TIP from the project description, we implemented a two-step pipeline:
- `save_logits_eomt.py`: saves per-pixel logits to disk (one forward pass per image)
- `temp_scaling_eomt.py`: applies temperature scaling offline without GPU, testing T ∈ {0.5, 0.75, 1.0, 1.1, 1.5, 2.0}

### External Resources

Models and datasets are not included in this repository due to size constraints.
They are available at the following Google Drive link: [[INSERT LINK HERE]](https://drive.google.com/drive/folders/1YOOVdyiCvO1BMM8uYJ_ubP_YZ2nRKKs3?usp=drive_link)
