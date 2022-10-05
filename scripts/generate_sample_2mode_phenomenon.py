#!/usr/bin/env python

# Generate samples from the dual modes distribution found in improved Wasserstein

import numpy as np
from torchmetrics.image.lpip import NoTrainLpips
from torchvision import models
from tqdm.auto import tqdm
import torch
import matplotlib.pyplot as plt
from run_LPIPS import normalize

if __name__ == '__main__':
    mu = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float, device='cpu').unsqueeze(-1).unsqueeze(-1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float, device='cpu').unsqueeze(-1).unsqueeze(-1)
    unnormalize = lambda x: x*std + mu
    rank = 0
    bs = 64
    lpips = NoTrainLpips(net='vgg', verbose=False).cuda()
    lpips_score = lambda x, y: lpips(normalize(x.cuda()), normalize(y.cuda()))
    model = models.vgg16(pretrained=True).cuda()
    model.eval()

    for r in range(1):
        dataloader = torch.load(f'./results_iwass/improved_{r}_dtl.pt')
        adv = torch.load(f'./results_iwass/improved_wasserstein_0.32143_{r}.pt')
        for i, (x, y) in tqdm(enumerate(dataloader), leave=False):
            x = x.cuda()
            adv_sample = adv[i * bs:(i + 1) * bs].cuda()

            scores = lpips_score(x, adv_sample).detach().cpu().numpy().reshape(-1)
            predictions = model.forward(x)
            pred_adv = model.forward(adv_sample)
            miss = list((torch.argmax(predictions, dim=1) != torch.argmax(pred_adv, dim=1)).detach().cpu().numpy())

            if len(np.argwhere((scores < 0.2) & miss)) > 0:
                fig, ax = plt.subplots(2, 8, figsize=(20, 5))
                for k, j in enumerate(np.argwhere((scores < 0.2) & miss)):
                    ax[0, k].imshow(unnormalize(x[j].squeeze(0).detach().cpu()).numpy().transpose(1, 2, 0))
                    # make xaxis invisibel
                    ax[0, k].xaxis.set_visible(False)
                    plt.setp(ax[0, k].spines.values(), visible=False)
                    ax[0, k].tick_params(left=False, labelleft=False)
                    ax[1, k].imshow(adv_sample[j].squeeze(0).detach().cpu().numpy().transpose(1, 2, 0))
                    ax[1, k].xaxis.set_visible(False)
                    plt.setp(ax[1, k].spines.values(), visible=False)
                    ax[1, k].tick_params(left=False, labelleft=False)
                    if k >= 7:
                        break
                ax[0, 0].set_ylabel('Source', fontsize=12)
                ax[1, 0].set_ylabel('Adversarial', fontsize=12)

                fig.savefig('comparison_small_eps.pdf', dpi=300, format='pdf')
                break

    for r in range(1):
        dataloader = torch.load(f'./results_iwass/improved_{r}_dtl.pt')
        adv = torch.load(f'./results_iwass/improved_wasserstein_0.32143_{r}.pt')
        for i, (x, y) in tqdm(enumerate(dataloader), leave=False):
            x = x.cuda()
            adv_sample = adv[i * bs:(i + 1) * bs].cuda()

            scores = lpips_score(x, adv_sample).detach().cpu().numpy().reshape(-1)
            predictions = model.forward(x)
            pred_adv = model.forward(adv_sample)
            miss = list((torch.argmax(predictions, dim=1) != torch.argmax(pred_adv, dim=1)).detach().cpu().numpy())

            if len(np.argwhere((scores > 0.2) & miss)) > 0:
                fig, ax = plt.subplots(2, 8, figsize=(20, 5))
                for k, j in enumerate(np.argwhere((scores > 0.2) & miss)):
                    ax[0, k].imshow(unnormalize(x[j].squeeze(0).detach().cpu()).numpy().transpose(1, 2, 0))
                    # make xaxis invisibel
                    ax[0, k].xaxis.set_visible(False)
                    plt.setp(ax[0, k].spines.values(), visible=False)
                    ax[0, k].tick_params(left=False, labelleft=False)
                    ax[1, k].imshow(adv_sample[j].squeeze(0).detach().cpu().numpy().transpose(1, 2, 0))
                    ax[1, k].xaxis.set_visible(False)
                    plt.setp(ax[1, k].spines.values(), visible=False)
                    ax[1, k].tick_params(left=False, labelleft=False)
                    if k >= 7:
                        break
                ax[0, 0].set_ylabel('Source', fontsize=12)
                ax[1, 0].set_ylabel('Adversarial', fontsize=12)

                fig.savefig('comparison_big_eps.pdf', dpi=300, format='pdf')
                break

