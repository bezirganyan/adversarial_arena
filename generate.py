import argparse
import time
from typing import Union, Optional
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from art.attacks import evasion
from art.estimators.classification import PyTorchClassifier
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from tqdm.auto import tqdm
import torch.distributed as dist
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent

from utils import ImageNetteDataset


def attack_model(model,
                 attack: str,
                 attacker,
                 dataloader: torch.utils.data.DataLoader,
                 norm: int,
                 epsilon: int,
                 device: int = 0,
                 verbose: Optional[bool] = True) -> tuple:
    duration = torch.tensor([0]).long().cuda()
    count = torch.tensor([0]).long().cuda()
    missclassification = torch.tensor([0]).cuda()

    adv_samples = []
    perturbations = []
    for x, y in tqdm(dataloader, leave=False):
        # x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        st = time.time()  # start  time
        x_adv = torch.tensor(attacker.generate(x.detach().numpy())).cuda(non_blocking=True)
        duration += int(time.time() - st)  # end time
        adv_samples.append(x_adv.detach().cpu())
        perturbation = x_adv - x
        perturbations.append(perturbation.detach().cpu())
        count += x.shape[0]
        predictions = model._model.forward(x_adv)
        missclassification += torch.sum(torch.argmax(predictions, dim=1) != y).item()

    adv_samples = torch.cat(adv_samples, dim=0)
    perturbations = torch.cat(perturbations, dim=0)
    adv_path = f'results_art/{attack}_{norm}_{epsilon:.3f}_{device}.pt'
    prt_path = f'results_art/{attack}_{norm}_{epsilon:.3f}_{device}_prt.pt'
    torch.save(adv_samples, adv_path)
    torch.save(perturbations, prt_path)


    group = dist.new_group(list(range(8)))
    dist.all_reduce(missclassification, op=dist.ReduceOp.SUM, group=group, async_op=False)
    dist.all_reduce(count, op=dist.ReduceOp.SUM, group=group, async_op=False)
    dist.all_reduce(duration, op=dist.ReduceOp.SUM, group=group, async_op=False)

    if verbose and device == 0:
        print(f'======  {attack}  ======')
        print(f'Norm: {norm}')
        print(f'Epsilon: {epsilon}')
        print(f'Duration: {(duration / count).item()}')
        print(f'Missclassification rate: {(missclassification / count * 100).item()}%')

    if device == 0:
        res = (attack, norm, epsilon, duration, missclassification, adv_path)
        return res
    else:
        return None


def generate_adversaries(model,
                         attacker,
                         attack: str,
                         dataloader: DataLoader,
                         norms: Union[list, np.array, torch.Tensor],
                         epsilons: Union[list, np.array, torch.Tensor],
                         results: list = None,
                         device: int = 0) -> list:
    if results is None:
        results: list = []
    dtl_path = f'results_art/{attack}_{device}_dtl.pt'
    torch.save(dataloader, dtl_path)
    for n in tqdm(norms):
        for eps in tqdm(epsilons, leave=False):
            attacker = evasion.wasserstein.Wasserstein(model, eps=eps, eps_step=eps / 3, regularization=10000, )
            res = attack_model(model, attack, attacker, dataloader, n, eps, verbose=True, device=device)
            if res:
                results.append(res)
                torch.save(results, '../results_art.pt')

    return results


def craft(device, args, dataset):
    rank = args.nr * args.gpus + device
    dist.init_process_group(
    	backend='nccl',
        init_method='env://',
    	world_size=args.world_size,
    	rank=rank
    )

    torch.manual_seed(args.seed)
    np.random.seed(seed=args.seed)
    batch_size = args.batch_size

    sampler = torch.utils.data.distributed.DistributedSampler(
    	dataset,
    	num_replicas=args.world_size,
    	rank=rank
    )

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=64,
                            shuffle=False,
                            sampler=sampler,
                            pin_memory=True,
                            drop_last=True)

    torch.cuda.set_device(device)
    model = models.vgg16(pretrained=True).cuda(device)
    model.eval()
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device])

    if device == 0:
        accurate = 0
        count = 0
        for x, y in tqdm(dataloader):
            predictions = model.forward(x.cuda(device))
            accurate += torch.sum(torch.argmax(torch.tensor(predictions), dim=1) == y.cuda(device)).item()
            count += x.shape[0]

        accuracy = accurate / count
        print("Accuracy on benign test examples: {}%".format(accuracy * 100))
    results = []
    attacks = ['wasserstein']

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    pt_model = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=x.shape[0],
        nb_classes=1000,
    )
    for attack in attacks:
        results = generate_adversaries(pt_model, None, attack, dataloader, [np.inf],
                                       np.linspace(0.05, 1, 15), results=results, device=device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/imagenette2-320/train',
                        help='Path to source image data in ImageNet format')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to use (default = 42)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch Size (default = 64)')
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')

    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8995'

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    transforms = T.Compose([T.Resize(256), T.CenterCrop(224), normalize])
    dataset = ImageNetteDataset(args.data_path, labels_file='map_clsloc.txt', transforms=transforms)

    mp.spawn(craft, nprocs=args.gpus, args=(args, dataset))
