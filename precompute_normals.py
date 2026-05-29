"""
Precompute normal maps for a COLMAP dataset using metric3d_vit_small.
Saves normals to <model_path>/normals/<image_name>.png, identical to the
format expected by readColmapCameras in scene/dataset_readers.py.

Usage:
    python precompute_normals.py \
        -s /path/to/dataset \
        -m /path/to/model_output \
        -r 4 \
        [--images images] \
        [--batch_size 1]
"""

import os
import sys
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms

from PIL import Image
from argparse import ArgumentParser
from arguments import ModelParams
from scene.colmap_loader import (
    read_extrinsics_binary,
    read_extrinsics_text,
)


def collect_image_paths(colmap_path, images_folder):
    sparse = os.path.join(colmap_path, "sparse/0")

    if os.path.exists(os.path.join(sparse, "images.bin")):
        cam_extrinsics = read_extrinsics_binary(os.path.join(sparse, "images.bin"))
    else:
        cam_extrinsics = read_extrinsics_text(os.path.join(sparse, "images.txt"))

    entries = []
    for key in cam_extrinsics:
        extr = cam_extrinsics[key]
        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        entries.append((image_path, image_name))

    return entries


def get_inference_size(orig_w, orig_h, resolution, multiple=28):
    """Mirror train.py loadCam resolution logic, then align to multiple of 28."""
    if resolution in [1, 2, 4, 8]:
        new_w = round(orig_w / resolution)
        new_h = round(orig_h / resolution)
    elif resolution == -1:
        if orig_w > 1600:
            scale = orig_w / 1600
        else:
            scale = 1
        new_w = int(orig_w / scale)
        new_h = int(orig_h / scale)
    else:
        scale = orig_w / resolution
        new_w = int(orig_w / scale)
        new_h = int(orig_h / scale)

    # Align to multiple of 28 for ViT
    new_h = max((new_h // multiple) * multiple, multiple)
    new_w = max((new_w // multiple) * multiple, multiple)
    return new_w, new_h


def process_batch(batch_paths, batch_names, model, normal_dir, transform, resolution):
    tensors = []
    orig_sizes = []

    for image_path in batch_paths:
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        orig_sizes.append((orig_h, orig_w))  # (H, W) for interpolate

        inf_w, inf_h = get_inference_size(orig_w, orig_h, resolution)
        t = transform(image).unsqueeze(0).cuda()  # [1, 3, H, W]
        t = torch.nn.functional.interpolate(
            t, size=(inf_h, inf_w), mode='bilinear', align_corners=False
        )
        tensors.append(t)

    batch_tensor = torch.cat(tensors, dim=0)  # [B, 3, inf_H, inf_W] on GPU
    del tensors

    with torch.no_grad():
        _, _, output_dict = model.inference({'input': batch_tensor})
        pred_normals = output_dict['prediction_normal'][:, :3, :, :]  # [B, 3, H', W']

    for i, (image_name, orig_size) in enumerate(zip(batch_names, orig_sizes)):
        normal_path = os.path.join(normal_dir, image_name + ".png")

        pred_resized = torch.nn.functional.interpolate(
            pred_normals[i:i+1], size=orig_size, mode='bilinear', align_corners=False
        )
        normals = pred_resized.squeeze().cpu().numpy()   # [3, H, W]
        normals = np.transpose(normals, (1, 2, 0))       # [H, W, 3]
        normals_vis = ((normals + 1.0) / 2.0).clip(0, 1)
        normals_uint8 = (normals_vis * 255).astype(np.uint8)
        cv2.imwrite(normal_path, cv2.cvtColor(normals_uint8, cv2.COLOR_RGB2BGR))

    del batch_tensor, pred_normals
    torch.cuda.empty_cache()


def main():
    parser = ArgumentParser(description="Precompute normal maps")
    lp = ModelParams(parser)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Images per GPU batch — reduce if you run out of VRAM")
    args = parser.parse_args(sys.argv[1:])
    dataset = lp.extract(args)

    images_folder = os.path.join(dataset.source_path, dataset.images)
    normal_dir = os.path.join(dataset.model_path, "normals")
    os.makedirs(normal_dir, exist_ok=True)

    print(f"Resolution: {dataset.resolution}  (same as -r in train.py)")
    print("Collecting image list from COLMAP sparse model...")
    entries = collect_image_paths(dataset.source_path, images_folder)
    print(f"  Found {len(entries)} images")

    todo = [(p, n) for p, n in entries
            if not os.path.exists(os.path.join(normal_dir, n + ".png"))]
    print(f"  {len(entries) - len(todo)} already cached, {len(todo)} to process")

    if not todo:
        print("All normals already exist. Nothing to do.")
        return

    print("Loading metric3d_vit_small...")
    model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
    model.eval().cuda()

    transform = transforms.Compose([transforms.ToTensor()])

    total = len(todo)
    for start in range(0, total, args.batch_size):
        batch = todo[start : start + args.batch_size]
        batch_paths = [p for p, _ in batch]
        batch_names = [n for _, n in batch]

        sys.stdout.write(f"\rProcessing {start + len(batch)}/{total} ...")
        sys.stdout.flush()

        try:
            process_batch(batch_paths, batch_names, model, normal_dir, transform, dataset.resolution)
        except torch.cuda.OutOfMemoryError:
            print(f"\nOOM on batch starting at {start}. Try --batch_size {max(1, args.batch_size // 2)}")
            raise

    print(f"\nDone. Normals saved to: {normal_dir}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
