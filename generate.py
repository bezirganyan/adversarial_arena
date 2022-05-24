import time
from typing import Union, Optional

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T
from art.attacks import evasion
from art.estimators.classification import PyTorchClassifier
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils import ImageNetteDataset


def attack_model(model: PyTorchClassifier,
                 attack: str,
                 attacker,
                 dataloader: torch.utils.data.DataLoader,
                 norm: int,
                 epsilon: int,
                 verbose: Optional[bool] = True) -> tuple:

    duration = 0
    count = 0
    missclassification = 0
    adv_samples = []
    perturbations = []
    for x, y in tqdm(dataloader, leave=False):
        st = time.time()  # start  time
        x_adv = attacker.generate(x=x.detach().cpu().numpy())
        duration += time.time() - st  # end time
        adv_samples.append(x_adv)
        perturbation = x_adv - x.detach().cpu().numpy()
        perturbations.append(perturbation)
        count += x.shape[0]
        predictions = model.predict(x_adv)
        missclassification += np.sum(np.argmax(predictions, axis=1) != np.argmax(y))

    adv_samples = torch.cat(adv_samples, dim=0)
    perturbations = torch.cat(perturbations, dim=0)
    missclassification /= count
    adv_path = f'results/{attack}_{norm}_{epsilon}.pt'
    prt_path = f'results/{attack}_{norm}_{epsilon}_pert.pt'
    torch.save(adv_samples, adv_path)
    torch.save(perturbations, prt_path)
    res = (attack, norm, epsilon, duration, missclassification, adv_path)

    if verbose:
        print(f'======  {attack}  ======')
        print(f'Norm: {norm}')
        print(f'Epsilon: {epsilon}')
        print(f'Duration: {duration}')
        print(f'Missclassification rate: {missclassification * 100}%')
    return res


def generate_adversaries(model: PyTorchClassifier,
                         uc_attacker,
                         attack: str,
                         dataloader: DataLoader,
                         norms: Union[list, np.array, torch.Tensor],
                         epsilons: Union[list, np.array, torch.Tensor],
                         results: list = None,
                         batch_size: Optional[int] = 64) -> list:
    if results is None:
        results: list = []
    for n in tqdm(norms):
        for eps in tqdm(epsilons, leave=False):
            try:
                attacker = uc_attacker(norm=n, estimator=model, batch_size=batch_size)
            except ValueError:
                attacker = uc_attacker(norm=int(n), estimator=model, batch_size=batch_size)
            attacker.set_params(eps=eps, eps_step=min(attacker.eps_step, eps / 3))
            res = attack_model(model, attack, attacker, dataloader, n, eps, verbose=True)
            results.append(res)
            torch.save(results, '../results.pt')
    return results


if __name__ == '__main__':
    # TODO - Need to get these from a CLI
    device = 'cpu'
    torch.manual_seed(42)
    np.random.seed(seed=42)
    batch_size = 64
    path = 'data/imagenette2-320/train'

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    transforms = T.Compose([T.Resize(256), T.CenterCrop(224), normalize])

    dataset = ImageNetteDataset(path, labels_file='map_clsloc.txt', transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    model = models.vgg16(pretrained=True)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    pt_model = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 224, 224),
        nb_classes=1000,
        device_type='gpu'
    )

    accurate = 0
    count = 0
    for x, y in tqdm(dataloader):
        predictions = model.forward(x)
        accurate += torch.sum(torch.argmax(torch.tensor(predictions), dim=1) == y).item()
        count += x.shape[0]

    accuracy = accurate / count
    print("Accuracy on benign test examples: {}%".format(accuracy * 100))
    results = []
    attacks = ['pgd']
    attackers = [evasion.FastGradientMethod, evasion.ProjectedGradientDescent, evasion.AutoProjectedGradientDescent]
    for attack, attacker in zip(attacks, attackers):
        results = generate_adversaries(pt_model, attacker, attack, dataloader, ['inf'],
                                       np.linspace(0.01, 0.5, 10), results=results)


