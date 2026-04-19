import torch
import torch.nn.functional as F
import numpy as np
from .utils import update_and_clip, to_np_uint8
from .dct import dct_2d, idct_2d

__all__ = ["DuFIA"]


def get_hook_pd_2(ori_ilout, gamma):
    def hook_pd1(module, grad_input, grad_output):
        modified = [gamma * grad_input[0] + (1 - gamma) * ori_ilout]
        modified += list(grad_input[1:])
        return tuple(modified)

    return hook_pd1


def get_hook_pd_inception_v4(g1, g2, g3, _, gamma):
    def hook_pd1(module, grad_input, grad_output):
        return g1, g2, g3

    return hook_pd1


def get_hook_pd_inception_v3(g1, g2, g3, g4, gamma):
    def hook_pd1(module, grad_input, grad_output):
        return g1, g2, g3, g4

    return hook_pd1


class DuFIA:
    def __init__(self, args, **kwargs):
        print("DuFIA attacking ...")
        self.model = kwargs["source_model"]
        self.coef = getattr(args, "coef", 0.1)
        self.model_name = args.model_name
        self.il_pos = getattr(args, "il_pos", "transformer.resblocks.6")
        self.il_pos2 = getattr(args, "il_pos2", "transformer.resblocks.6")
        self.N = getattr(args, "N", 1)
        self.integrated_step = args.Integrated_steps

        self._select_pos()
        self.hook_backward = None

        self.origin_grad_resemble = None
        self.origin_grad_resemble1 = None
        self.origin_grad_resemble2 = None
        self.origin_grad_resemble3 = None
        self.origin_grad_resemble4 = None

        self.hooks = []

    def _grad_capture_hook(self, module, grad_input, grad_output):
        self.origin_grad_resemble = grad_input[0].clone()

    def _grad_capture_hook_inception_v4(self, module, grad_input, grad_output):
        self.origin_grad_resemble1 = grad_input[0].clone()
        self.origin_grad_resemble2 = grad_input[1].clone()
        self.origin_grad_resemble3 = grad_input[2].clone()

    def _grad_capture_hook_inception_v3(self, module, grad_input, grad_output):
        self.origin_grad_resemble1 = grad_input[0].clone()
        self.origin_grad_resemble2 = grad_input[1].clone()
        self.origin_grad_resemble3 = grad_input[2].clone()
        self.origin_grad_resemble4 = grad_input[3].clone()

    def _select_pos(self):
        if 'CLIP:ViT' in self.model_name:
            clip_model = self.model.module[1]
            visual_model = clip_model.model.visual

            def get_clip_layer(model, path):
                current = model
                for part in path.split('.'):
                    if part.isdigit():
                        current = current[int(part)]
                    else:
                        current = getattr(current, part)
                return current

            self.il_module = get_clip_layer(visual_model, self.il_pos)
            self.il_module2 = get_clip_layer(visual_model, self.il_pos2)
        else:
            self.il_module = eval(f"self.model.module[2].{self.il_pos}")
            self.il_module2 = eval(f"self.model.module[2].{self.il_pos2}")

    def __call__(self, args, ori_img, label, verbose=True):
        if args.epsilon == 0:
            if verbose:
                print("[DuFIA] Skipped: epsilon == 0, return original image.")
            return to_np_uint8(ori_img.permute(0, 2, 3, 1)), ori_img.clone()

        adv_img = ori_img.clone()
        ori_img_copy = ori_img.clone().detach().requires_grad_(True)

        self.origin_grad_resemble = None
        self.origin_grad_resemble1 = None
        self.origin_grad_resemble2 = None
        self.origin_grad_resemble3 = None
        self.origin_grad_resemble4 = None

        self._prep_hook_back_aggragate(args, ori_img_copy, label)

        input_grad = 0
        for _ in range(args.steps):
            for _ in range(self.N):
                adv_img.requires_grad_(True)
                logits = self.model(adv_img)
                loss = F.binary_cross_entropy_with_logits(logits, label)
                grad = torch.autograd.grad(loss, adv_img)[0].data
                input_grad += grad / torch.norm(grad, p=1, dim=(1, 2, 3), keepdim=True)
            input_grad /= self.N
            adv_img = update_and_clip(ori_img, adv_img, input_grad, args.epsilon, args.step_size, args.constraint)

        if self.hook_backward:
            self.hook_backward.remove()
            self.hook_backward = None

        return to_np_uint8(adv_img.permute(0, 2, 3, 1)), adv_img.clone()

    def _prep_hook_back_aggragate(self, args, ori_img, labels):
        ens = self.integrated_step
        img_base = torch.zeros_like(ori_img)

        if self.model_name in ['tv_resnet50', 'tv_resnet152', 'vgg19'] or 'CLIP:ViT' in self.model_name:
            ilout_hook = self.il_module2.register_backward_hook(self._grad_capture_hook)
        elif self.model_name == 'inception_v3':
            ilout_hook = self.il_module2.register_backward_hook(self._grad_capture_hook_inception_v3)
        elif self.model_name == 'inception_v4':
            ilout_hook = self.il_module2.register_backward_hook(self._grad_capture_hook_inception_v4)
        else:
            raise ValueError(f"Unsupported model for DuFIA hook: {self.model_name}")

        self.hooks.append(ilout_hook)

        for i in range(ens):
            img_noise1 = ori_img * (1 - i / ens) + img_base * (i / ens)
            img_noise2 = ori_img + torch.normal(0, 0.5 / 255, ori_img.shape).to(ori_img.device)
            img_noise2 = dct_2d(img_noise2, norm='ortho')
            mask = torch.FloatTensor(ori_img.shape).uniform_(1 - 0.5, 1 + 0.5).to(ori_img.device)
            img_noise2 = idct_2d(img_noise2 * mask, norm='ortho')

            for img_noise in [img_noise1, img_noise2]:
                logits = self.model(img_noise)
                loss = F.binary_cross_entropy_with_logits(logits, labels)
                loss.backward()

        ilout_hook.remove()
        self.hooks.remove(ilout_hook)

        if self.model_name == 'inception_v4':
            hook_func = get_hook_pd_inception_v4(
                self.origin_grad_resemble1 / (2 * ens),
                self.origin_grad_resemble2 / (2 * ens),
                self.origin_grad_resemble3 / (2 * ens),
                0, self.coef)
        elif self.model_name == 'inception_v3':
            hook_func = get_hook_pd_inception_v3(
                self.origin_grad_resemble1 / (2 * ens),
                self.origin_grad_resemble2 / (2 * ens),
                self.origin_grad_resemble3 / (2 * ens),
                self.origin_grad_resemble4 / (2 * ens),
                self.coef)
        else:
            hook_func = get_hook_pd_2(self.origin_grad_resemble / (2 * ens), self.coef)

        self.hook_backward = self.il_module2.register_backward_hook(hook_func)