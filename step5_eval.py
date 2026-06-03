"""
Step 5 - Fine-tuned EoMT vs EoMT-Cityscapes evaluation
Compares:
  - EoMT-Cityscapes (original, provided by prof)
  - EoMT-Finetuned (COCO model fine-tuned on Cityscapes)
Same evaluation pipeline as step4_eval.py.
"""

import sys
sys.path.insert(0, '/content/drive/MyDrive/MaskArchitectureAnomaly_CourseProject-main/MaskArchitectureAnomaly_CourseProject-main/eomt')

import argparse
import importlib
import warnings
import time
import yaml
import torch
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
import numpy as np
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


def load_model(config_path, ckpt_path, img_size, num_classes, device=0, num_q_override=None):
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

    model = lit_cls(img_size=img_size, num_classes=num_classes,
                    network=network, **model_kwargs).eval().to(device)

    # Support both .bin and .ckpt checkpoints
    state_dict = torch.load(ckpt_path, map_location=f"cuda:{device}", weights_only=True)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
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


def evaluate_model(model, val_dataset, device, model_name="model"):
    evaluator = IoUEvaluator(NUM_CS_CLASSES, IGNORE_INDEX)
    t0 = time.time()
    for step, (img, target) in enumerate(val_dataset):
        gt = model.to_per_pixel_targets_semantic([target], IGNORE_INDEX)[0].numpy()
        logits = infer_semantic(model, img, device)
        pred = logits.argmax(0).numpy()
        if pred.shape != gt.shape:
            pred_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float()
            pred = F.interpolate(pred_t, size=gt.shape, mode="nearest").squeeze().numpy().astype(np.int64)
        evaluator.update(pred, gt)
        if (step + 1) % 50 == 0:
            print(f"  [{model_name}] {step+1}/500  mIoU: {evaluator.miou()*100:.1f}%  ({time.time()-t0:.0f}s)")
    return evaluator


def print_results(eval_cs, eval_ft):
    iou_cs = eval_cs.iou_per_class()
    iou_ft = eval_ft.iou_per_class()
    header = f"{'Class':<25}  {'EoMT-CS':>10}  {'EoMT-Finetuned':>14}"
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for i, name in enumerate(CS_NAMES):
        cs_str = f"{iou_cs[i]*100:6.2f}%" if not np.isnan(iou_cs[i]) else "   N/A  "
        ft_str = f"{iou_ft[i]*100:6.2f}%" if not np.isnan(iou_ft[i]) else "   N/A  "
        print(f"{name:<25}  {cs_str:>10}  {ft_str:>14}")
    print(sep)
    print(f"{'mIoU':<25}  {eval_cs.miou()*100:>9.2f}%  {eval_ft.miou()*100:>13.2f}%")
    print(sep)


def main(args):
    device = args.device

    print("[1/4] Loading Cityscapes val dataset...")
    with open(args.config) as f:
        cs_config = yaml.safe_load(f)
    dm_name, dm_cls_name = cs_config["data"]["class_path"].rsplit(".", 1)
    dm_cls = getattr(importlib.import_module(dm_name), dm_cls_name)
    dm_kwargs = cs_config["data"].get("init_args", {})
    cs_data = dm_cls(path=args.data_path, batch_size=1, num_workers=0,
                     check_empty_targets=False, **dm_kwargs).setup()
    val_dataset = cs_data.val_dataloader().dataset
    print(f"  {len(val_dataset)} images")

    print("\n[2/4] Loading EoMT-Cityscapes (original)...")
    cs_model = load_model(args.config, args.cityscapes_ckpt,
                          cs_data.img_size, cs_data.num_classes, device=device)

    print("\n[3/4] Loading EoMT-Finetuned...")
    ft_model = load_model(args.config, args.finetuned_ckpt,
                          [640, 640], cs_data.num_classes, device=device, num_q_override=200)

    print("\n[4/4] Evaluating...")
    print("\n  --- EoMT-Cityscapes (original) ---")
    eval_cs = evaluate_model(cs_model, val_dataset, device, model_name="EoMT-CS")
    print("\n  --- EoMT-Finetuned ---")
    eval_ft = evaluate_model(ft_model, val_dataset, device, model_name="EoMT-FT")

    print_results(eval_cs, eval_ft)

    if args.output_file:
        import sys as _sys
        orig = _sys.stdout
        with open(args.output_file, "w") as f:
            _sys.stdout = f
            print_results(eval_cs, eval_ft)
        _sys.stdout = orig
        print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",          required=True, help="Cityscapes semantic config")
    parser.add_argument("--cityscapes-ckpt", required=True, help="EoMT-Cityscapes checkpoint")
    parser.add_argument("--finetuned-ckpt",  required=True, help="EoMT-Finetuned checkpoint")
    parser.add_argument("--data-path",       required=True)
    parser.add_argument("--device",          type=int, default=0)
    parser.add_argument("--output-file",     default=None)
    main(parser.parse_args())
