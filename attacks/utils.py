import numpy as np
import torch


def update_and_clip(ori_img, att_img, grad, epsilon, step_size, norm):
    if norm == "linf":
        att_img = att_img.data + step_size * torch.sign(grad)
        att_img = torch.where(att_img > ori_img + epsilon, ori_img + epsilon, att_img)
        att_img = torch.where(att_img < ori_img - epsilon, ori_img - epsilon, att_img)
        att_img = torch.clamp(att_img, min=0, max=1)
    elif norm == "l2":
        grad = grad / (grad.norm(p=2,dim = (1,2,3), keepdim=True) + 1e-12)
        att_img = att_img + step_size * torch.sign(grad)
        l2_perturb = att_img - ori_img
        l2_perturb = l2_perturb.renorm(p=2, dim = 0, maxnorm=epsilon)
        att_img = ori_img + l2_perturb
        att_img = torch.clamp(att_img, min=0, max=1)
    return att_img

def to_np_uint8(x):
    x = torch.clamp(x, 0, 1)   # 先限制范围
    return (x.detach() * 255).round().cpu().numpy().astype(np.uint8)


def defend_transform(image, tau=0.1):
    """
    image: (B, C, H, W) tensor, in [0,1]
    return: (B, C, H, W) DEFEND-processed image
    """
    B, C, H, W = image.shape
    fft = torch.fft.fft2(image, dim=(-2, -1))
    fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))

    # frequency grid
    fy = torch.fft.fftfreq(H, d=1.0).to(image.device).reshape(1, 1, -1, 1)
    fx = torch.fft.fftfreq(W, d=1.0).to(image.device).reshape(1, 1, 1, -1)
    f = torch.sqrt(fx ** 2 + fy ** 2)

    # kernel k(f) = -0.2f^2 + 0.8f - 0.05
    k = -0.2 * f ** 2 + 0.8 * f - 0.05
    k = torch.clamp(k, min=0)
    w = torch.zeros_like(f)
    mask = f > tau
    w[mask] = (torch.exp(k[mask] / 2) - 1) / (f[mask] + 1e-8)

    weighted_fft = fft_shift * w
    fft_unshift = torch.fft.ifftshift(weighted_fft, dim=(-2, -1))
    defend_img = torch.fft.ifft2(fft_unshift, dim=(-2, -1)).real

    # normalize back to [0, 1]
    defend_img = (defend_img - defend_img.amin(dim=(-2,-1), keepdim=True)) / (defend_img.amax(dim=(-2,-1), keepdim=True) - defend_img.amin(dim=(-2,-1), keepdim=True) + 1e-8)
    return defend_img.clamp(0, 1)