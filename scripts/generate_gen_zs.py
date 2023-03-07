"""Generate zs (and/or ys) for pretrained GANs."""
import argparse
import pathlib

from src.deps.pretorched.gans import biggan, utils

import torch

parser = argparse.ArgumentParser(description='generate a bunch of gan inputs')
parser.add_argument('path', type=pathlib.Path, help='write zs and ys here')
parser.add_argument('--dataset',
                    default='imagenet',
                    choices=('imagenet', 'places365'),
                    help='dataset model was trained on')
parser.add_argument('--num-samples',
                    '-n',
                    dest='n',
                    type=int,
                    default=100000,
                    help='number of samples to generate (default: 100k)')
parser.add_argument('--model',
                    '-m',
                    default='biggan',
                    help='biggan or ldm')
args = parser.parse_args()


if args.model == 'biggan':
    model = biggan.BigGAN(pretrained=args.dataset, device='cpu')
    z_size = model.dim_z
else:
    z_size = (3, 64, 64)
print("z size:", z_size)
n_classes = 1000 if args.dataset == 'imagenet' else 365
zs, _ = utils.prepare_z_y(args.n, z_size, n_classes, device='cpu')
zs = torch.from_numpy(zs.numpy())
ys = torch.randint(n_classes, size=(args.n,))

args.path.mkdir(exist_ok=True)
torch.save(zs, args.path / 'zs.pth')
torch.save(ys, args.path / 'ys.pth')
# args.path.parent.mkdir(exist_ok=True, parents=True)

# torch.save((zs, ys), args.path)
