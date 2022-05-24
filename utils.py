import os

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import pandas as pd
from torchvision.io import read_image
from tqdm import tqdm


class ImageNetteDataset(Dataset):
    def __init__(self, path, labels_file, transforms=None):
        self.path = path
        self.transforms = transforms
        category_mapping = pd.read_csv(labels_file, sep=' ', names=['label', 'category']).reset_index()
        category_mapping['label'] = category_mapping['label'].sort_values().values
        categories = os.listdir(path)
        self.file_labels = pd.DataFrame(columns=['file', 'label'])
        concats = []
        for c in tqdm(categories):
            files = os.listdir(os.path.join(path, c))
            for f in files:
                id = int(category_mapping[category_mapping.loc[:, 'label'] == c]['index'])
                # label = category_mapping[category_mapping.loc[:, 'label'] == c]['category'].replace(' ', '_')
                concats.append(pd.Series({'file': os.path.join(c, f), 'id': id,  'label': c}))
        self.file_labels = pd.DataFrame(concats)

    def __len__(self):
        return len(self.file_labels)

    def __getitem__(self, index) -> T_co:
        p = self.file_labels.iloc[index, 0]
        image = read_image(os.path.join(self.path, p))
        if image.shape[0] != 3:
            raise ValueError(f'Image must have 3 dimensions, found {image.shape[0]}')
        image = self.transforms(image / image.max())
        cat_id = self.file_labels.iloc[index, 1]

        return image, cat_id