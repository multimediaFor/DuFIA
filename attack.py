import os
import argparse
import torch
from tqdm import tqdm
from attacks import DuFIA
from utils import build_model, save_image, resnet_nomalize, clip_normalize
from PIL import Image
import numpy as np

def load_image(image_path, batch_size, target_size=(224, 224)):
    images = []
    filenames = []
    labels = []
    original_sizes = []
    idx = 0

    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    files = os.listdir(image_path)

    for filename in files:
        file_path = os.path.join(image_path, filename)
        if os.path.isfile(file_path) and os.path.splitext(filename)[1].lower() in valid_exts:
            try:
                image = Image.open(file_path).convert('RGB')
                w, h = image.size
                original_sizes.append((h, w))
                image = image.resize(target_size, Image.BILINEAR)  # force resize
                image_np = np.array(image)

                if image_np.ndim == 2:
                    image_np = np.stack([image_np] * 3, axis=-1)

                if image_np.shape != (target_size[1], target_size[0], 3):
                    print(f"Skipping {filename}, unexpected shape: {image_np.shape}")
                    continue

                images.append(image_np)
                filenames.append(filename)
                labels.append(1)  # Fake label, adjust as needed
                idx += 1

                if idx == batch_size:
                    yield np.stack(images), np.array(filenames), np.array(labels), original_sizes
                    images, filenames, labels, original_sizes = [], [], [], []
                    idx = 0
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")

    if idx > 0:
        yield np.stack(images), np.array(filenames), np.array(labels), original_sizes


def run_attack(input_dir, save_dir, args):
    model, _, _, _, _, args.image_size = build_model(args.model_name)
    attacker = DuFIA(args, source_model=model)
    os.makedirs(save_dir, exist_ok=True)

    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    all_files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in valid_exts]
    total_batches = (len(all_files) + args.batch_size - 1) // args.batch_size

    pbar = tqdm(load_image(input_dir, args.batch_size),
                desc=f"Attacking {os.path.basename(os.path.dirname(input_dir))}",
                total=total_batches,
                unit="batch")

    for images, names, labels, original_sizes in pbar:
        images_resized = []
        for img in images:
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize((args.image_size, args.image_size))
            images_resized.append(np.array(pil_img))
        images_resized = np.array(images_resized)

        if 'CLIP:ViT' in args.model_name:
            images_normlize = clip_normalize(images_resized)
        else:
            images_normlize = resnet_nomalize(images_resized)

        ori_img = torch.from_numpy(images_normlize).cuda().permute(0, 3, 1, 2).float()
        label = torch.from_numpy(labels).float().unsqueeze(1).cuda()

        img_adv, _ = attacker(args, ori_img.clone(), label, verbose=False)

        img_adv_resized = []
        for i in range(len(img_adv)):
            adv_img = Image.fromarray(img_adv[i])
            adv_img = adv_img.resize((original_sizes[i][1], original_sizes[i][0]))
            img_adv_resized.append(np.array(adv_img))

        save_image(img_adv_resized, names, labels, input_dir, save_dir)

    print(f"Finished attacking {input_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="./datasets")
    parser.add_argument("--save_root", type=str, default="./outputs")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--model_name", type=str, default="CLIP:ViT-L/14")
    parser.add_argument("--epsilon", type=float, default=8)
    parser.add_argument("--step_size", type=float, default=0.8)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--constraint", type=str, default="linf")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--Integrated_steps", type=int, default=10)

    args = parser.parse_args()

    if args.constraint == "linf":
        args.epsilon = args.epsilon / 255.
        args.step_size = args.step_size / 255.

    if not os.path.isdir(args.dataset_root):
        raise ValueError(f"Dataset root does not exist: {args.dataset_root}")

    main_subfolders = sorted([
        d for d in os.listdir(args.dataset_root)
        if os.path.isdir(os.path.join(args.dataset_root, d))
    ])
    real_fake = ["0_real", "1_fake"]

    all_tasks = [(folder, sub) for folder in main_subfolders for sub in real_fake]
    print(f"\nFound {len(main_subfolders)} dataset domains \n")

    for i, (subfolder, subsub) in enumerate(all_tasks, start=0):
        input_dir = os.path.join(args.dataset_root, subfolder, subsub)
        save_dir = os.path.join(args.save_root, subfolder)
        display_name = f"[{i+1}/{len(main_subfolders)*2}] {subfolder}/{subsub}"

        if not os.path.exists(input_dir):
            print(f"Warning: {input_dir} does not exist, skipping")
            continue

        print(f"\nStarting attack : {input_dir}")
        run_attack(input_dir, save_dir, args)

    
