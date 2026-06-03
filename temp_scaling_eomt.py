import os, sys, glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose, Resize
from sklearn.metrics import average_precision_score
from ood_metrics import fpr_at_95_tpr

target_transform = Compose([Resize((512, 1024), Image.NEAREST)])

DATASETS_PATH = "/content/drive/MyDrive/MaskArchitectureAnomaly_CourseProject-main/MaskArchitectureAnomaly_CourseProject-main/Validation_Dataset"
CACHE_PATH = "/content/drive/MyDrive/MaskArchitectureAnomaly_CourseProject-main/MaskArchitectureAnomaly_CourseProject-main/logits_cache"

DATASETS = {
    "RoadAnomaly21": "*.png",
    "RoadObsticle21": "*.webp",
    "FS_LostFound_full": "*.png",
    "fs_static": "*.jpg",
    "RoadAnomaly": "*.jpg",
}

TEMPERATURES = [0.5, 0.75, 1.0, 1.1, 1.5, 2.0]

def get_gt(img_path):
    pathGT = img_path.replace("images", "labels_masks")
    if "RoadObsticle21" in pathGT: pathGT = pathGT.replace("webp", "png")
    if "fs_static" in pathGT: pathGT = pathGT.replace("jpg", "png")
    if "RoadAnomaly" in pathGT: pathGT = pathGT.replace("jpg", "png")
    mask = target_transform(Image.open(pathGT))
    ood_gts = np.array(mask)
    if "RoadAnomaly" in pathGT: ood_gts = np.where((ood_gts == 2), 1, ood_gts)
    if "LostAndFound" in pathGT:
        ood_gts = np.where((ood_gts == 0), 255, ood_gts)
        ood_gts = np.where((ood_gts == 1), 0, ood_gts)
        ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)
    return ood_gts

for checkpoint in ["cityscapes", "finetuned", "coco"]:
    print(f"\n{'='*50}", flush=True)
    print(f"CHECKPOINT: EoMT-{checkpoint}", flush=True)
    print(f"{'='*50}", flush=True)
    for dataset_name, ext in DATASETS.items():
        print(f"Dataset: {dataset_name}", flush=True)
        cache_dir = os.path.join(CACHE_PATH, checkpoint, dataset_name)
        img_paths = sorted(glob.glob(os.path.join(DATASETS_PATH, dataset_name, "images", ext)))

        for T in TEMPERATURES:
            scores, gts = [], []
            for img_path in img_paths:
                fname = os.path.splitext(os.path.basename(img_path))[0] + '.npy'
                logits = torch.from_numpy(np.load(os.path.join(cache_dir, fname))) / T
                probs = F.softmax(logits, dim=0)
                anomaly = 1.0 - probs.max(dim=0).values.numpy()
                gt = get_gt(img_path)
                valid = (gt != 255)
                scores.append(anomaly[valid].ravel())
                gts.append(gt[valid].ravel())

            scores = np.concatenate(scores)
            gts = np.concatenate(gts)
            auprc = average_precision_score(gts, scores) * 100
            fpr = fpr_at_95_tpr(scores, gts) * 100
            print(f"  T={T:.2f}  AUPRC={auprc:.2f}%  FPR95={fpr:.2f}%", flush=True)
