import torch
import torch.nn as nn
import torchvision.transforms as T
from models.clip_models import CLIPModel
import os
import numpy as np
from PIL import Image


def build_model(model_name):
   

    if model_name != 'CLIP:ViT-L/14':
        raise ValueError(f'Only CLIP:ViT-L/14 is supported for this release, got: {model_name}')

    model = CLIPModel(name="ViT-L/14")
    state_dict = torch.load('./pretrained_weights/fc_weights.pth', map_location='cpu')
    model.fc.load_state_dict(state_dict)
    model.linear = model.fc
    layer = "transformer.resblocks.5"
    size = 224
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    normalize_layer = T.Normalize(mean=mean, std=std)
    model_full = nn.Sequential(normalize_layer, model)

    model_parallel = nn.DataParallel(model_full).cuda()
    model_parallel.eval()

    model.eval()
    model.cuda()

    return model_parallel, None, model, layer, layer, size


def resnet_nomalize(x):
    return x / 255.0


def clip_normalize(images):
    images = images.astype(np.float32) / 255.0
    return images


def load_image(image_path, image_size, batch_size):
    images = []
    filenames = []
    labels = []
    idx = 0

    subfolders = ['1_fake', '0_real']
    for subfolder in subfolders:
        subfolder_path = os.path.join(image_path, subfolder)
        files = os.listdir(subfolder_path)

        for filename in files:
            file_path = os.path.join(subfolder_path, filename)
            if os.path.isfile(file_path):
                image = Image.open(file_path).convert('RGB')
                image = image.resize((image_size, image_size))
                image = np.array(image)

                if image.ndim == 2:  # 灰度转 RGB
                    image = np.stack([image] * 3, axis=-1)

                images.append(image)
                filenames.append(filename)
                labels.append(0 if subfolder == '0_real' else 1)
                idx += 1

                if idx == batch_size:
                    yield np.array(images), np.array(filenames), np.array(labels)
                    images, filenames, labels = [], [], []
                    idx = 0

    if idx > 0:
        yield np.array(images), np.array(filenames), np.array(labels)


def save_image(images_np, names, labels, input_root, save_root):
    for i in range(len(images_np)):
        adv_np = images_np[i]
        if adv_np.dtype != np.uint8:
            adv_np = np.clip(adv_np, 0, 255).astype(np.uint8)

        name = names[i]
        label = int(labels[i])
        subfolder = '0_real' if label == 0 else '1_fake'
        save_dir = os.path.join(save_root, subfolder)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, name)
        Image.fromarray(adv_np).save(save_path)

def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    return np.reshape(categorical, output_shape)


def to_np_uint8(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    x = torch.clamp(x, 0, 1)
    x = x * 255.0
    return x.round().cpu().numpy().astype(np.uint8)


