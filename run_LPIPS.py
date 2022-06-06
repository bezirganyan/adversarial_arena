import os

import numpy as np
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity, NoTrainLpips
from torchvision import models
from tqdm.auto import tqdm
import pandas as pd
import torch


def normalize(x):
    return (2 * (x - x.min()) / (x.max() - x.min()) - 1).cuda()

if __name__ == '__main__':
    res_path = './results'
    scores_path = './scores_non_red'

    files = os.listdir(res_path)
    files = [f for f in files if not f.endswith('dtl.pt') and not f.endswith('prt.pt')]

    lpips = NoTrainLpips(net='vgg', verbose=False).cuda()
    lpips_score = lambda x, y: lpips(normalize(x), normalize(y))

    k = 1 #16
    p_count = 1 #10
    model = models.vgg16(pretrained=True).cuda()
    model.eval()

    results = {'eps': [], 'device': [], 'score': [], 'score_pert': [], 'miss_pert': [], 'score_gauss': [], 'miss_gauss': [], 'miss': []}

    for f in tqdm(files):
        fparam = f.split('_')
        device = int(fparam[-1].replace(".pt", ""))
        dataloader = torch.load(os.path.join(res_path, f'{fparam[0]}_{device}_dtl.pt'))
        perturb = torch.load(os.path.join(res_path, f.replace('.pt', '_prt.pt')))
        perturb = perturb.cuda()
        adv = torch.load(os.path.join(res_path, f))
        bs = dataloader.batch_size

        for i, (x, y) in tqdm(enumerate(dataloader), leave=False):
            x = x.cuda()
            adv_sample = adv[i*bs:(i+1)*bs].cuda()

            scores = list(lpips_score(x, adv_sample).detach().cpu().numpy().reshape(-1))
            predictions = model.forward(x)
            pred_adv = model.forward(adv_sample)
            miss = list((torch.argmax(predictions, dim=1)  != torch.argmax(pred_adv, dim=1)).detach().cpu().numpy())
            results['score'].extend(scores * p_count)
            results['device'].extend([device] * len(scores) * p_count)
            results['eps'].extend([float(fparam[2])] * len(scores) * p_count)
            results['miss'].extend(miss * p_count)

            for _ in tqdm(range(p_count), leave=False):
                perm = torch.randperm(perturb.size(0))
                idx = perm[:k]
                samples = perturb[idx]
                pert_images = x + samples.mean(dim=0)

                pred_adv = model.forward(pert_images)
                miss_pert = list((torch.argmax(predictions, dim=1) != torch.argmax(pred_adv, dim=1)).detach().cpu().numpy())
                results['score_pert'].extend(list(lpips_score(x, pert_images).detach().cpu().numpy().reshape(-1)))
                results['miss_pert'].extend(miss_pert)

            for _ in tqdm(range(p_count), leave=False):
                noise = torch.normal(mean=0., std=float(fparam[2]), size=x.shape[1:])
                noise = torch.clip(noise, -float(fparam[2]), float(fparam[2])).cuda()
                pert_images = x + noise

                pred_adv = model.forward(pert_images)
                miss_gauss = list((torch.argmax(predictions, dim=1) != torch.argmax(pred_adv, dim=1)).detach().cpu().numpy())
                results['score_gauss'].extend(list(lpips_score(x, pert_images).detach().cpu().numpy().reshape(-1)))
                results['miss_gauss'].extend(miss_gauss)


        print(f'Eps: {fparam[2]}')
        print(f'scores_mean {np.mean(results["score"])}')
        print(f'pt_scores_mean {np.mean(results["score_pert"])}')
        print(f'rn_scores_mean {np.mean(results["score_gauss"])}')

        pd.DataFrame.from_dict(results).to_csv(os.path.join(scores_path, f'results.csv'))
