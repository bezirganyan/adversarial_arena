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

def evaluate(rank: int, files: list,
             network: str, res_path: str, scores_path: str, p_count: int) -> None:
    results = {'eps': [], 'device': [], 'dur': [], 'miss': []}
    network_dict = {'vgg': models.vgg16, 'alex': models.alexnet, 'squeeze': models.squeezenet1_1}
    dataloader = torch.load(os.path.join(res_path, f'{files[0].split("_")[0]}_{rank}_dtl.pt'))

    files = [f for f in files if int(f.split('_')[-1].replace(".pt", "")) == rank]
    for f in tqdm(files):
        fparam = f.split('_')
        device = int(fparam[-1].replace(".pt", ""))
        adv = torch.load(os.path.join(res_path, f))
        dur = torch.load(os.path.join(res_path, f.replace('.pt', '_dur.pt')))
        bs = dataloader.batch_size
        model = network_dict[network](pretrained=True).cuda(rank)
        model.eval()

        for i, (x, y) in tqdm(enumerate(dataloader), leave=False):
            x = x.cuda(rank)
            adv_sample = adv[i * bs:(i + 1) * bs].cuda(rank)

            predictions = model.forward(x)
            pred_adv = model.forward(adv_sample)
            miss = list((torch.argmax(predictions, dim=1) != torch.argmax(pred_adv, dim=1)).detach().cpu().numpy())
            results['device'].extend([device]*bs)
            results['eps'].extend([float(fparam[2])]*bs)
            results['miss'].extend(miss)
            results['dur'].extend([dur[i]]*bs)

        print(f'Eps: {fparam[2]}')
        print(f'smiss_mean: {np.mean(results["miss"])}')
        pd.DataFrame.from_dict(results).to_csv(os.path.join(scores_path, f'results_{rank}.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--res_path', type=str, default='results')
    parser.add_argument('--scores_path', type=str, default='scores')
    parser.add_argument('--network', type=str, default='vgg')

    args = parser.parse_args()

    file_list = os.listdir(args.res_path)
    file_list = [f for f in file_list if not f.endswith('dtl.pt') and not f.endswith('prt.pt') and not f.endswith('dur.pt')]

    if not os.path.exists(args.scores_path):
        os.makedirs(args.scores_path)

    p_count = 10 #10
    nprocs = 8

    mp.spawn(fn=evaluate, args=(file_list, args.network, args.res_path,
                                args.scores_path, p_count), nprocs=nprocs)