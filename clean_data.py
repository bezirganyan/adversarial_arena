import os

from torchvision.io import read_image
from tqdm.auto import tqdm

path = 'data/imagenette2-320/train'

counter = 0
categories = os.listdir(path)
for c in tqdm(categories):
    files = os.listdir(os.path.join(path, c))
    for f in tqdm(files, leave=False):
        image = read_image(os.path.join(path, c, f))
        if image.shape[0] != 3:
            counter += 1
            os.remove(os.path.join(path, c, f))

print(f'Operation finished!  {counter} files deleted')