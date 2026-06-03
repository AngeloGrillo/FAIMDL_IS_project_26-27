import os, sys, glob, argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.amp.autocast_mode import autocast

sys.path.insert(0, '/content/drive/MyDrive/MaskArchitectureAnomaly_CourseProject-main/MaskArchitectureAnomaly_CourseProject-main/eomt')
sys.path.insert(0, '/content/drive/MyDrive/MaskArchitectureAnomaly_CourseProject-main/MaskArchitectureAnomaly_CourseProject-main')
from torchvision.transforms import Compose, Resize
from evalAnomaly_eomt import load_eomt_model

input_transform = Compose([Resize((512, 1024), Image.BILINEAR)])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--img-size', type=int, nargs=2, default=[640, 640])
    parser.add_argument('--num-classes', type=int, default=19)
    parser.add_argument('--num-q', type=int, default=None)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model = load_eomt_model(args.config, args.ckpt, args.img_size, args.num_classes,
                            num_q_override=args.num_q, device=args.device)

    for path in sorted(glob.glob(args.input)):
        print(path)
        image = input_transform(Image.open(path).convert('RGB'))
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1)  # uint8 [C,H,W]
        imgs = [image.to(args.device)]
        img_sizes = [image.shape[-2:]]

        with torch.no_grad():
            with autocast(dtype=torch.float16, device_type="cuda"):
                crops, origins = model.window_imgs_semantic(imgs)
                mask_logits_per_layer, class_logits_per_layer = model(crops)
                mask_logits = F.interpolate(
                    mask_logits_per_layer[-1], model.img_size, mode="bilinear"
                )
                crop_logits = model.to_per_pixel_logits_semantic(
                    mask_logits, class_logits_per_layer[-1]
                )
                logits = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)
                logits = logits[0].float().cpu().numpy()  # [C, H, W]

        fname = os.path.splitext(os.path.basename(path))[0] + '.npy'
        np.save(os.path.join(args.output_dir, fname), logits)
        print(f"  saved {fname}, shape={logits.shape}")

if __name__ == '__main__':
    main()
