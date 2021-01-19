from typing import Callable, Any, Union, Optional, Tuple
import time

import torch
import numpy as np
from art.attacks import evasion
from art.estimators.classification import PyTorchClassifier
from art.utils import load_cifar10, to_categorical, compute_success
import deeprobust.image.netmodels.resnet as resnet

from resnet import ResNet18


def load_model(path: str) -> torch.nn.Module:
    model = ResNet18()
    if path.endswith(".pth"):
        ckpt = torch.load(path)
        if "net" in ckpt.keys():
            for key in ckpt["net"].keys():
                assert "module" in key
            ckpt["net"] = dict((key[7:], value) for key, value in ckpt["net"].items())
            model.load_state_dict(ckpt["net"])

        elif "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"][0])

        else:
            model.load_state_dict(ckpt['model_state_dict'])

    else:
        ckpt = torch.load(path)
        model.load_state_dict(ckpt['model_state_dict'])
    return model


def load_data(device: str = 'cpu', test_size: int = 1000) \
        -> Tuple[Union[np.array, list],
                 Union[np.array, list],
                 float,
                 float,
                 Callable[[Union[int, np.array]], Union[int, np.array]],
                 Callable[[Union[int, np.array]], Union[int, np.array]]]:
    (_, _), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()
    x_test = torch.tensor(np.moveaxis(x_test, [0, 1, 2, 3], [0, 2, 3, 1])).float()
    perm = torch.randperm(x_test.size(0))
    idx = perm[:test_size]
    x_test = x_test[idx]
    y_test = y_test[idx]
    mu = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float, device=device).unsqueeze(-1).unsqueeze(-1)
    std = torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float, device=device).unsqueeze(-1).unsqueeze(-1)

    normalize = lambda x: (torch.tensor(x, device=device) - mu) / std
    unnormalize = lambda x: torch.tensor(x, device=device) * std + mu

    return x_test, y_test, min_pixel_value, max_pixel_value, normalize, unnormalize


def attack_model(model: PyTorchClassifier,
                 attack: str,
                 attacker,
                 x_test: Union[list, np.array, torch.Tensor],
                 target: Union[int, str],
                 y_test: Union[list, np.array, torch.Tensor],
                 norm: int,
                 epsilon: int,
                 test_size: Optional[int] = 1000,
                 verbose: Optional[bool] = True) -> tuple:
    target_classes = torch.tensor([target] * test_size)
    st = time.time()  # start  time

    x_test_adv = attacker.generate(x=x_test, y=to_categorical(target_classes, nb_classes=10))
    duration = time.time() - st  # end time

    predictions = model.predict(x_test_adv)
    missclassification = np.sum(np.argmax(predictions, axis=1) != np.argmax(y_test, axis=1)) / len(y_test)
    success = compute_success(model, x_test, to_categorical(target_classes, nb_classes=10), x_test_adv, targeted=True)
    save_path = f'../results/{attack}_{norm}_{epsilon}_{target}.pt'
    torch.save(x_test_adv, save_path)
    res = (attack, norm, epsilon, duration, missclassification, success, save_path, target)

    if verbose:
        print(f'======  {attack}  ======')
        print(f'Norm: {norm}')
        print(f'Target: {target}')
        print(f'Epsilon: {epsilon}')
        print(f'Duration: {duration}')
        print(f'Missclassification rate: {missclassification * 100}%')
        print(f'Success Rate: {success * 100}%\n')
    return res


def generate_adversaries(model: PyTorchClassifier,
                         uc_attacker,
                         attack: str,
                         x_test: Union[list, np.array, torch.Tensor],
                         y_test: Union[list, np.array, torch.Tensor],
                         norms: Union[list, np.array, torch.Tensor],
                         epsilons: Union[list, np.array, torch.Tensor],
                         targets: Union[list, np.array] = range(10),
                         results: list = None,
                         batch_size: Optional[int] = 64) -> list:
    if results is None:
        results: list = []
    for n in norms:
        for eps in epsilons:
            for target in targets:
                try:
                    attacker = uc_attacker(norm=n, estimator=model, targeted=True, batch_size=batch_size)
                except ValueError:
                    attacker = uc_attacker(norm=int(n), estimator=model, targeted=True, batch_size=batch_size)
                attacker.set_params(eps=eps, eps_step=min(attacker.eps_step, eps / 3))
                res = attack_model(model, attack, attacker, x_test, target, y_test, n, eps, verbose=True)
                results.append(res)
                torch.save(results, '../results.pt')
    return results


if __name__ == '__main__':
    x_test, y_test, min_pixel_value, max_pixel_value, normalize, unnormalize = load_data(test_size=1000)
    # pt_model = load_model('../cifar_vanilla.pth')

    pt_model = resnet.ResNet18().to('cuda')
    pt_model.load_state_dict(torch.load('../CIFAR10_ResNet18_epoch_100.pt'))
    torch.manual_seed(42)
    np.random.seed(seed=42)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(pt_model.parameters(), lr=0.01)

    model = PyTorchClassifier(
        model=pt_model,
        clip_values=(min_pixel_value, max_pixel_value),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=10,
    )
    predictions = model.predict(x_test)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Accuracy on benign test examples: {}%".format(accuracy * 100))
    results = []
    attacks = ['fgsm', 'pgd', 'auto.pgd']
    attackers = [evasion.FastGradientMethod, evasion.ProjectedGradientDescent, evasion.AutoProjectedGradientDescent]
    for attack, attacker in zip(attacks, attackers):
        results = generate_adversaries(model, attacker, attack, x_test, y_test, [1, 2, 'inf'],
                                       np.linspace(0.01, 0.5, 10), targets=range(10), results=results)

