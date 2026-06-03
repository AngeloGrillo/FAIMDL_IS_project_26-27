"""
Point 8 - EoMT Anomaly Segmentation Evaluation
Supports: MSP, MaxLogit, MaxEntropy, RbA
Works with all 3 EoMT checkpoints: COCO, Cityscapes, Finetuned
"""

import sys
sys.path.insert(0, '/content/drive/MyDrive/MaskArchitectureAnomaly_CourseProject-main/MaskArchitectureAnomaly_CourseProject-main/eomt')

import os
import glob
import torch
import random
import warnings
import importlib
import numpy as np
import yaml
from PIL import Image
from argparse import ArgumentParser
from torch.nn import functional as F
from torch.amp.autocast_mode import autocast
from torchvision.transforms import Compose, Resize, ToTensor
from sklearn.metrics import average_precision_score
from ood_metrics import fpr_at_95_tpr

warnings.filterwarnings("ignore", message=r".*Attribute 'network'.*")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Solo resize, NON ToTensor - EoMT vuole PIL Image o uint8
input_transform = Compose([
    Resize((512, 1024), Image.BILINEAR),
])

target_transform = Compose([
    Resize((512, 1024), Image.NEAREST),
])


def load_eomt_model(config_path, ckpt_path, img_size, num_classes,
                    stuff_classes=None, num_q_override=None, device=0):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    enc_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    enc_cls = getattr(importlib.import_module(enc_cfg["class_path"].rsplit(".", 1)[0]),
                      enc_cfg["class_path"].rsplit(".", 1)[1])
    encoder = enc_cls(img_size=img_size, **enc_cfg.get("init_args", {}))

    net_cfg = config["model"]["init_args"]["network"]
    net_cls = getattr(importlib.import_module(net_cfg["class_path"].rsplit(".", 1)[0]),
                      net_cfg["class_path"].rsplit(".", 1)[1])
    net_kwargs = {k: v for k, v in net_cfg["init_args"].items() if k != "encoder"}
    if num_q_override is not None:
        net_kwargs["num_q"] = num_q_override
    network = net_cls(masked_attn_enabled=False, num_classes=num_classes,
                      encoder=encoder, **net_kwargs)

    lit_cfg = config["model"]
    lit_cls = getattr(importlib.import_module(lit_cfg["class_path"].rsplit(".", 1)[0]),
                      lit_cfg["class_path"].rsplit(".", 1)[1])
    model_kwargs = {k: v for k, v in config["model"]["init_args"].items() if k != "network"}
    if stuff_classes is not None:
        model_kwargs["stuff_classes"] = stuff_classes

    model = lit_cls(img_size=img_size, num_classes=num_classes,
                    network=network, **model_kwargs).eval().to(device)

    state_dict = torch.load(ckpt_path, map_location=f"cuda:{device}", weights_only=True)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    print(f"Model loaded from {ckpt_path}")
    return model


@torch.no_grad()
def get_anomaly_score(model, image_tensor, method, device):
    """
    Compute anomaly score for a single image using EoMT semantic inference.
    
    For MSP, MaxLogit, MaxEntropy: use per-pixel logits from semantic inference
    For RbA: use mask logits and class logits directly (mask architecture specific)
    """
    imgs = [image_tensor.to(device)]
    img_sizes = [image_tensor.shape[-2:]]

    with autocast(dtype=torch.float16, device_type="cuda"):
        crops, origins = model.window_imgs_semantic(imgs)
        mask_logits_per_layer, class_logits_per_layer = model(crops)

        # Interpolate mask logits to model img_size
        mask_logits = F.interpolate(
            mask_logits_per_layer[-1], model.img_size, mode="bilinear"
        )

        if method == "rba":
            # RbA: Rejected by All
            # For each pixel, compute the probability that NO query claims it
            # score = product over queries of (1 - sigmoid(mask_logit_i) * p_i(c_i))
            # High score = no query claims the pixel = anomaly
            class_logits = class_logits_per_layer[-1]  # [B, Q, C+1]
            
            # Get per-query class probabilities (excluding no-object class)
            # softmax over classes excluding last (no-object)
            class_probs = F.softmax(class_logits[..., :-1], dim=-1)  # [B, Q, C]
            max_class_probs = class_probs.max(dim=-1).values  # [B, Q]
            
            # Mask sigmoid
            mask_probs = mask_logits.sigmoid()  # [B, Q, H, W]
            
            # RbA score: for each pixel, min over queries of (1 - mask * class_prob)
            # Reshape max_class_probs for broadcasting: [B, Q, 1, 1]
            max_class_probs = max_class_probs.unsqueeze(-1).unsqueeze(-1)
            
            # Combined confidence per query per pixel
            query_confidence = mask_probs * max_class_probs  # [B, Q, H, W]
            
            # RbA: pixel is anomalous if all queries reject it
            # score = 1 - max confidence over queries
            anomaly_score = 1.0 - query_confidence.max(dim=1).values  # [B, H, W]
            
            # RbA: già per-pixel, solo resize a img_size poi float numpy
            anomaly_score = anomaly_score[0].float().cpu().numpy()

        else:
            # Get per-pixel logits via semantic inference
            crop_logits = model.to_per_pixel_logits_semantic(
                mask_logits, class_logits_per_layer[-1]
            )
            logits = model.revert_window_logits_semantic(
                crop_logits, origins, img_sizes
            )  # [1, C, H, W]
            logits = logits[0].float().cpu()  # [C, H, W]

            if method == "maxlogit":
                # Max Logit: 1 - max raw logit
                anomaly_score = 1.0 - logits.max(dim=0).values.numpy()

            elif method == "msp":
                # MSP: 1 - max softmax probability
                probs = F.softmax(logits, dim=0)
                anomaly_score = 1.0 - probs.max(dim=0).values.numpy()

            elif method == "maxentropy":
                # Max Entropy: entropy of softmax distribution
                probs = F.softmax(logits, dim=0)
                log_probs = torch.log(probs + 1e-9)
                entropy = -torch.sum(probs * log_probs, dim=0)
                anomaly_score = entropy.numpy()

    return anomaly_score


def main():
    parser = ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--config", required=True, help="EoMT config yaml path")
    parser.add_argument("--ckpt", required=True, help="EoMT checkpoint path")
    parser.add_argument("--method", default="maxlogit",
                        choices=["msp", "maxlogit", "maxentropy", "rba"])
    parser.add_argument("--img-size", type=int, nargs=2, default=[640, 640])
    parser.add_argument("--num-classes", type=int, default=19)
    parser.add_argument("--num-q", type=int, default=None)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    model = load_eomt_model(
        args.config, args.ckpt, args.img_size, args.num_classes,
        num_q_override=args.num_q, device=args.device
    )

    anomaly_score_list = []
    ood_gts_list = []

    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        print(path)
        image = input_transform(Image.open(path).convert('RGB'))
        # Converti in tensor uint8 [0-255] come si aspetta window_imgs_semantic
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1)

        anomaly_score = get_anomaly_score(model, image, args.method, args.device)

        pathGT = path.replace("images", "labels_masks")
        if "RoadObsticle21" in pathGT:
            pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
            pathGT = pathGT.replace("jpg", "png")
        if "RoadAnomaly" in pathGT:
            pathGT = pathGT.replace("jpg", "png")

        mask = Image.open(pathGT)
        mask = target_transform(mask)
        ood_gts = np.array(mask)

        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts == 2), 1, ood_gts)
        if "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts == 0), 255, ood_gts)
            ood_gts = np.where((ood_gts == 1), 0, ood_gts)
            ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)
        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts == 14), 255, ood_gts)
            ood_gts = np.where((ood_gts < 20), 0, ood_gts)
            ood_gts = np.where((ood_gts == 255), 1, ood_gts)

        if 1 not in np.unique(ood_gts):
            continue

        # Resize anomaly score to match GT if needed
        if anomaly_score.shape != ood_gts.shape:
            from PIL import Image as PILImage
            score_img = PILImage.fromarray(anomaly_score.astype(np.float32))
            score_resized = np.array(score_img.resize(
                (ood_gts.shape[1], ood_gts.shape[0]), PILImage.BILINEAR
            ))
            anomaly_score = score_resized

        ood_gts_list.append(ood_gts)
        anomaly_score_list.append(anomaly_score)

        del anomaly_score, ood_gts, mask
        torch.cuda.empty_cache()

    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((np.zeros(len(ind_out)), np.ones(len(ood_out))))

    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    print(f'AUPRC score: {prc_auc * 100.0:.2f}%')
    print(f'FPR@TPR95: {fpr * 100.0:.2f}%')


if __name__ == '__main__':
    main()
