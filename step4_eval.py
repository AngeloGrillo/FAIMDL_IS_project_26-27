"""
Step 4 - EoMT mIoU Evaluation on Cityscapes val
Evaluates BOTH models on semantic segmentation (mIoU) using 19 Cityscapes classes.
For EoMT-COCO: predictions are remapped via COCO->Cityscapes class mapping.
Same evaluation pipeline for both models.
"""

import argparse
import importlib
import warnings
import time
import yaml
import torch
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
import numpy as np
import sys
sys.path.insert(0, "/content/drive/MyDrive/MaskArchitectureAnomaly_CourseProject-main/MaskArchitectureAnomaly_CourseProject-main/eomt")
from lightning import seed_everything

seed_everything(0, verbose=False)
warnings.filterwarnings("ignore", message=r".*Attribute 'network'.*")

CS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]
NUM_CS_CLASSES = 19
IGNORE_INDEX = 255

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush", "banner", "blanket", "bridge", "cardboard",
    "counter", "curtain", "door-stuff", "floor-wood", "flower", "fruit",
    "gravel", "house", "light", "mirror-stuff", "net", "pillow", "platform",
    "playingfield", "railroad", "river", "road", "roof", "sand", "sea",
    "shelf", "snow", "stairs", "tent", "towel", "wall-brick", "wall-stone",
    "wall-tile", "wall-wood", "water-other", "window-blind", "window-other",
    "tree-merged", "fence-merged", "ceiling-merged", "sky-other-merged",
    "cabinet-merged", "table-merged", "floor-other-merged", "pavement-merged",
    "mountain-merged", "grass-merged", "dirt-merged", "paper-merged",
    "food-other-merged", "building-other-merged", "rock-merged",
    "wall-other-merged", "rug-merged",
]

_COCO_NAME_TO_CS_ID = {
    "person":                11,
    "bicycle":               18,
    "car":                   13,
    "motorcycle":            17,
    "bus":                   15,
    "train":                 16,
    "truck":                 14,
    "traffic light":          6,
    "road":                   0,
    "railroad":               0,
    "pavement-merged":        1,
    "building-other-merged":  2,
    "house":                  2,
    "fence-merged":           4,
    "tree-merged":            8,
    "grass-merged":           9,
    "sky-other-merged":      10,
    "wall-brick":             3,
    "wall-stone":             3,
    "wall-tile":              3,
    "wall-wood":              3,
    "wall-other-merged":      3,
    "stop sign":              7,
}

def build_coco_to_cs_lut(coco_names):
    lut = np.full(len(coco_names) + 1, IGNORE_INDEX, dtype=np.int64)
    for coco_id, name in enumerate(coco_names):
        cs_id = _COCO_NAME_TO_CS_ID.get(name, None)
        if cs_id is not None:
            lut[coco_id] = cs_id
    return lut


class IoUEvaluator:
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.tp = np.zeros(num_classes, dtype=np.float64)
        self.fp = np.zeros(num_classes, dtype=np.float64)
        self.fn = np.zeros(num_classes, dtype=np.float64)

    def update(self, pred, target):
        mask = target != self.ignore_index
        pred = pred[mask]
        target = target[mask]
        for c in range(self.num_classes):
            p = pred == c
            t = target == c
            self.tp[c] += (p & t).sum()
            self.fp[c] += (p & ~t).sum()
            self.fn[c] += (~p & t).sum()

    def iou_per_class(self):
        denom = self.tp + self.fp + self.fn
        return np.where(denom > 0, self.tp / denom, np.nan)

    def miou(self):
        return float(np.nanmean(self.iou_per_class()))


def load_model(config_path, ckpt_path, img_size, num_classes, stuff_classes=None, device=0):
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
    model.load_state_dict(state_dict, strict=False)
    return model


@torch.no_grad()
def infer_semantic(model, img, device):
    imgs = [img.to(device)]
    img_sizes = [img.shape[-2:] for img in imgs]
    with autocast(dtype=torch.float16, device_type="cuda"):
        crops, origins = model.window_imgs_semantic(imgs)
        mask_logits_per_layer, class_logits_per_layer = model(crops)
        mask_logits = F.interpolate(mask_logits_per_layer[-1], model.img_size, mode="bilinear")
        crop_logits = model.to_per_pixel_logits_semantic(mask_logits, class_logits_per_layer[-1])
        logits = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)
    return logits[0].float().cpu()


def evaluate_model(model, val_dataset, device, coco_lut=None, model_name="model"):
    evaluator = IoUEvaluator(NUM_CS_CLASSES, IGNORE_INDEX)
    t0 = time.time()
    for step, (img, target) in enumerate(val_dataset):
        gt = model.to_per_pixel_targets_semantic([target], IGNORE_INDEX)[0].numpy()
        logits = infer_semantic(model, img, device)
        pred = logits.argmax(0).numpy()
        if coco_lut is not None:
            pred = coco_lut[pred]
        if pred.shape != gt.shape:
            pred_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float()
            pred = F.interpolate(pred_t, size=gt.shape, mode="nearest").squeeze().numpy().astype(np.int64)
        evaluator.update(pred, gt)
        if (step + 1) % 50 == 0:
            print(f"  [{model_name}] {step+1}/500  mIoU: {evaluator.miou()*100:.1f}%  ({time.time()-t0:.0f}s)")
    return evaluator


def print_results(eval_cs, eval_coco):
    iou_cs = eval_cs.iou_per_class()
    iou_coco = eval_coco.iou_per_class()
    header = f"{'Class':<25}  {'EoMT-CS':>10}  {'EoMT-COCO':>10}"
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for i, name in enumerate(CS_NAMES):
        cs_str   = f"{iou_cs[i]*100:6.2f}%"   if not np.isnan(iou_cs[i])   else "   N/A  "
        coco_str = f"{iou_coco[i]*100:6.2f}%" if not np.isnan(iou_coco[i]) else "   N/A  "
        print(f"{name:<25}  {cs_str:>10}  {coco_str:>10}")
    print(sep)
    print(f"{'mIoU':<25}  {eval_cs.miou()*100:>9.2f}%  {eval_coco.miou()*100:>9.2f}%")
    print(sep)


def main(args):
    device = args.device

    print("[1/5] Loading Cityscapes val dataset...")
    with open(args.cityscapes_config) as f:
        cs_config = yaml.safe_load(f)
    dm_name, dm_cls_name = cs_config["data"]["class_path"].rsplit(".", 1)
    dm_cls = getattr(importlib.import_module(dm_name), dm_cls_name)
    dm_kwargs = cs_config["data"].get("init_args", {})
    cs_data = dm_cls(path=args.data_path, batch_size=1, num_workers=0,
                     check_empty_targets=False, **dm_kwargs).setup()
    val_dataset = cs_data.val_dataloader().dataset
    print(f"  {len(val_dataset)} images")

    print("\n[2/5] Loading EoMT-Cityscapes...")
    cs_model = load_model(args.cityscapes_config, args.cityscapes_ckpt,
                          cs_data.img_size, cs_data.num_classes, device=device)

    print("\n[3/5] Loading EoMT-COCO...")
    with open(args.coco_config) as f:
        coco_config = yaml.safe_load(f)
    stuff_classes = coco_config["data"]["init_args"].get("stuff_classes", None)
    coco_model = load_model(args.coco_config, args.coco_ckpt,
                            [640, 640], 133,
                            stuff_classes=stuff_classes, device=device)

    print("\n[4/5] Building COCO->Cityscapes mapping...")
    coco_lut = build_coco_to_cs_lut(COCO_NAMES)
    mapped = [(i, COCO_NAMES[i], coco_lut[i]) for i in range(len(COCO_NAMES)) if coco_lut[i] != IGNORE_INDEX]
    print(f"  {len(mapped)} COCO classes mapped to Cityscapes, {len(COCO_NAMES)-len(mapped)} ignored")

    print("\n[5/5] Evaluating...")
    print("\n  --- EoMT-Cityscapes ---")
    eval_cs = evaluate_model(cs_model, val_dataset, device, coco_lut=None, model_name="EoMT-CS")
    print("\n  --- EoMT-COCO ---")
    eval_coco = evaluate_model(coco_model, val_dataset, device, coco_lut=coco_lut, model_name="EoMT-COCO")

    print_results(eval_cs, eval_coco)

    if args.output_file:
        import sys
        orig = sys.stdout
        with open(args.output_file, "w") as f:
            sys.stdout = f
            print_results(eval_cs, eval_coco)
        sys.stdout = orig
        print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cityscapes-config", required=True)
    parser.add_argument("--cityscapes-ckpt",   required=True)
    parser.add_argument("--coco-config",       required=True)
    parser.add_argument("--coco-ckpt",         required=True)
    parser.add_argument("--data-path",         required=True)
    parser.add_argument("--device",            type=int, default=0)
    parser.add_argument("--output-file",       default=None)
    main(parser.parse_args())
