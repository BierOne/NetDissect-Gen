"""Extensions for pretorched/gans/biggan module."""
import collections
from typing import Any, List, NamedTuple, Sequence, Tuple

import sys
sys.path.append('/home/lyb/ucla/taming-transformers')
sys.path.append('/home/lyb/ucla/difussion/latent-diffusion')

from taming.models import vqgan

import torch
from torch import nn

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


class LDMInputs(NamedTuple):
    """Wraps inputs for sequential BigGAN."""

    # Follow naming conventions from pretorched.
    z: torch.Tensor
    y: torch.Tensor

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def get_model(pretrained='imagenet'):
    if pretrained == 'imagenet':
        config = OmegaConf.load("/home/lyb/ucla/difussion/latent-diffusion/configs/latent-diffusion/cin256-v2.yaml")
        model = load_model_from_config(config, "/home/lyb/ucla/difussion/latent-diffusion/models/ldm/cin256-v2/model.ckpt")
        model.num_classes = 1000
    else:
        print("No model can be find as ", pretrained)
        exit(0)
    return model



def LDM(*args: Any, **kwargs: Any) -> DDIMSampler:
    """Return Diffusion Model."""
    model = get_model(*args, **kwargs)
    sampler = DDIMSampler(model)
    return sampler


# def SeqBigGAN(*args: Any, **kwargs: Any) -> nn.Sequential:
#     """Return BigGAN as a sequential."""
#     generator = biggan.BigGAN(*args, **kwargs)
#
#     modules: List[Tuple[str, nn.Module]] = [
#         ('preprocess', SeqGPreprocess(generator)),
#     ]
#     for index, blocks in enumerate(generator.blocks):
#         assert len(blocks) <= 2, 'should never be more than 2 blocks'
#         for block in blocks:
#             if isinstance(block, biggan.GBlock):
#                 key = 'layer'
#             else:
#                 assert isinstance(block, layers.Attention), 'unknown block'
#                 key = 'attn'
#             key += str(index)
#             modules.append((key, SeqGBlock(block, index)))
#     modules.append(('output', SeqGOutput(generator)))
#
#     return nn.Sequential(collections.OrderedDict(modules))
