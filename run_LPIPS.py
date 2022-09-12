import argparse
import os

import numpy as np
from torchmetrics.image.lpip import NoTrainLpips
from torchvision import models
from tqdm.auto import tqdm
import pandas as pd
import torch
import torch.multiprocessing as mp


def normalize(x):
    return 2 * (x - x.min()) / (x.max() - x.min()) - 1

def evaluate(rank: int, files: list, distance_network: str,
             target_network: str, res_path: str, scores_path: str, p_count: int) -> None:
    results = {'eps': [], 'device': [], 'score': [], 'score_pert': [], 'miss_pert': [], 'miss': []}
    network_dict = {'vgg': models.vgg16, 'alex': models.alexnet, 'squeeze': models.squeezenet1_1}

    files = [f for f in files if int(f.split('_')[-1].replace(".pt", "")) == rank]
    for f in tqdm(files):
        fparam = f.split('_')
        device = int(fparam[-1].replace(".pt", ""))
        dataloader = torch.load(os.path.join(res_path, f'{fparam[0]}_{device}_dtl.pt'))
        perturb = torch.load(os.path.join(res_path, f.replace('.pt', '_prt.pt')))
        perturb = perturb.cuda(rank)
        adv = torch.load(os.path.join(res_path, f))
        bs = dataloader.batch_size

        lpips = NoTrainLpips(net=distance_network, verbose=False).cuda(rank)
        lpips_score = lambda x, y: lpips(normalize(x.cuda(rank)), normalize(y.cuda(rank)))
        model = network_dict[target_network](pretrained=True).cuda(rank)
        model.eval()

        for i, (x, y) in tqdm(enumerate(dataloader), leave=False):
            x = x.cuda(rank)
            adv_sample = adv[i * bs:(i + 1) * bs].cuda(rank)

            scores = list(lpips_score(x, adv_sample).detach().cpu().numpy().reshape(-1))
            predictions = model.forward(x)
            pred_adv = model.forward(adv_sample)
            miss = list((torch.argmax(predictions, dim=1) != torch.argmax(pred_adv, dim=1)).detach().cpu().numpy())
            results['score'].extend(scores * p_count)
            results['device'].extend([device] * len(scores) * p_count)
            results['eps'].extend([float(fparam[2])] * len(scores) * p_count)
            results['miss'].extend(miss * p_count)

            for _ in tqdm(range(p_count), leave=False):
                perm = torch.randperm(perturb.size(0))
                idx = perm[0]
                samples = perturb[idx]
                pert_images = x + samples.mean(dim=0)

                pred_adv = model.forward(pert_images)
                miss_pert = list((torch.argmax(predictions, dim=1) != torch.argmax(pred_adv, dim=1)).detach().cpu().numpy())
                results['score_pert'].extend(list(lpips_score(x, pert_images).detach().cpu().numpy().reshape(-1)))
                results['miss_pert'].extend(miss_pert)

        print(f'Eps: {fparam[2]}')
        print(f'scores_mean {np.mean(results["score"])}')
        print(f'pt_scores_mean {np.mean(results["score_pert"])}')
        pd.DataFrame.from_dict(results).to_csv(os.path.join(scores_path, f'results_{rank}.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--res_path', type=str, default='results')
    parser.add_argument('--scores_path', type=str, default='scores')
    parser.add_argument('--distance_network', type=str, default='vgg')
    parser.add_argument('--target_network', type=str, default='vgg')
    parser.add_argument('--p_count', type=int, default=10)
    parser.add_argument('--nprocs', type=int, default=8)

    args = parser.parse_args()

    file_list = os.listdir(args.res_path)
    file_list = [f for f in file_list if not f.endswith('dtl.pt') and not f.endswith('prt.pt') and not f.endswith('dur.pt')]

    p_count = args.p_count
    nprocs = args.nprocs

    mp.spawn(fn=evaluate, args=(file_list, args.distance_network, args.target_network, args.res_path,
                                args.scores_path, p_count), nprocs=nprocs)