import os

import numpy as np
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity, NoTrainLpips
from torchvision import models
from tqdm.auto import tqdm
import pandas as pd
import torch
from torch.multiprocessing import set_start_method
import torch.multiprocessing as mp


def normalize(x):
    return 2 * (x - x.min()) / (x.max() - x.min()) - 1

def evaluate(rank, files, res_path, scores_path, k_list, p_count):
    results = {'eps': [], 'k':[], 'device': [], 'score': [], 'score_pert': [], 'miss_pert': [], 'miss': []}

    files = [f for f in files if int(f.split('_')[-1].replace(".pt", "")) == rank]
    for f in tqdm(files):
        fparam = f.split('_')
        device = int(fparam[-1].replace(".pt", ""))
        dataloader = torch.load(os.path.join(res_path, f'{fparam[0]}_{device}_dtl.pt'))
        perturb = torch.load(os.path.join(res_path, f.replace('.pt', '_prt.pt')))
        perturb = perturb.cuda(rank)
        adv = torch.load(os.path.join(res_path, f))
        bs = dataloader.batch_size

        lpips = NoTrainLpips(net='squeeze', verbose=False).cuda(rank)
        lpips_score = lambda x, y: lpips(normalize(x.cuda(rank)), normalize(y.cuda(rank)))
        model = models.alexnet(pretrained=True).cuda(rank)
        model.eval()

        for i, (x, y) in tqdm(enumerate(dataloader), leave=False):
            x = x.cuda(rank)
            adv_sample = adv[i * bs:(i + 1) * bs].cuda(rank)

            scores = list(lpips_score(x, adv_sample).detach().cpu().numpy().reshape(-1))
            predictions = model.forward(x)
            pred_adv = model.forward(adv_sample)
            miss = list((torch.argmax(predictions, dim=1) != torch.argmax(pred_adv, dim=1)).detach().cpu().numpy())
            results['score'].extend(scores * p_count * len(k_list))
            results['device'].extend([device] * len(scores) * p_count * len(k_list))
            results['eps'].extend([float(fparam[2])] * len(scores) * p_count * len(k_list))
            results['miss'].extend(miss * p_count * len(k_list))

            for k in tqdm(k_list, leave=False):
                results['k'].extend([k] * len(scores) * p_count)
                for _ in tqdm(range(p_count), leave=False):
                    perm = torch.randperm(perturb.size(0))
                    idx = perm[:k]
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
    res_path = './results'
    scores_path = './scores_alex_alex'

    file_list = os.listdir(res_path)
    file_list = [f for f in file_list if not f.endswith('dtl.pt') and not f.endswith('prt.pt')]



    k_list = [1] #16
    p_count = 10 #10
    nprocs = 8

    mp.spawn(fn=evaluate, args=(file_list, res_path, scores_path, k_list, p_count), nprocs=nprocs, )