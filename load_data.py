import glob
import torch
import random
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


hardness_dict = {
        'Agate': [6.5, 7.0], 'Albite': [6.0, 6.5], 'Almandine': [7.0, 7.5],
        'Anglesite': [2.5, 3.0], 'Azurite': [3.5, 4.0], 'Beryl': [7.5, 8.0],
        'Cassiterite': [6.0, 6.5, 7.0], 'Chalcopyrite': [3.5, 4.0],
        'Cinnabar': [2.0, 2.5], 'Copper': [3.0], 'Demantoid': [6.5, 7.0],
        'Diopside': [5.5, 6.0, 6.5], 'Elbaite': [7.5],
        'Epidote': [6.0, 6.5, 7.0], 'Fluorite': [4.0], 'Galena': [3.5, 4.0],
        'Gold': [2.5, 3.0], 'Halite': [2.0, 2.5], 'Hematite': [5.0, 6.0],
        'Magnetite': [5.5, 6.0, 6.5], 'Malachite': [3.5, 4.0],
        'Marcasite': [6.0, 6.5], 'Opal': [5.5, 6.0, 6.5],
        'Orpiment': [1.5, 2.0], 'Pyrite': [6.0, 6.5], 'Quartz': [7.0],
        'Rhodochrosite': [3.5, 4.0], 'Ruby': [9.0], 'Sapphire': [9.0],
        'Schorl': [7.0], 'Sphalerite': [3.5, 4.0], 'Stibnite': [2.0],
        'Sulphur': [1.5, 2.0, 2.5], 'Topaz': [8.0], 'Torbernite': [2.0, 2.5],
        'Wulfenite': [2.5, 3.0]}


def make_weights_for_unbalanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


def make_total_list(root_dir):
    img_paths = glob.glob(root_dir)
    total_list = []
    for img_path in img_paths:
        img_class = img_path.split('/')[-1].split('.')[0].split('_')[1]
        img_class = int(img_class)
        item = (img_path, img_class)
        total_list.append(item)
    return total_list


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class mineral_dataset_hardness(Dataset):
    def __init__(self, root_dir, training=True):

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.files = glob.glob(root_dir)
        self.imgs = make_total_list(root_dir)
        if training is True:
            self.transforms = transforms.Compose([
                transforms.Resize((300, 300)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomAffine(degrees=0, scale=(0.85, 1.15)),
                transforms.ToTensor(),
                normalize
                ])
        else:
            self.transforms = transforms.Compose([
                                        transforms.Resize((300, 300)),
                                        transforms.ToTensor(),
                                        normalize
                                    ])

    def __len__(self):

        return len(self.files)

    def __getitem__(self, index):

        img = pil_loader(self.files[index])
        img = self.transforms(img)

        label = self.files[index].split('/')[-1].split('.')[0].split('_')[1]
        label = int(label)

        label_str = list(hardness_dict.keys())[label]
        # hardness section corresponding to label
        section = hardness_dict[label_str]
        if len(section) == 1:
            hardness = section[0]
        elif len(section) == 2:
            hardness = section[random.randint(0, 1000) % 2]
        elif len(section) == 3:
            hardness = section[random.randint(0, 1000) % 3]
        return img, label, np.double(hardness)


class mineral_dataset(Dataset):
    def __init__(self, root_dir, training=True):

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.files = glob.glob(root_dir)
        self.imgs = make_total_list(root_dir)
        if training is True:
            self.transforms = transforms.Compose([
                transforms.Resize((300, 300)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomAffine(degrees=0, scale=(0.85, 1.15)),
                transforms.ToTensor(),
                normalize
                ])
        else:
            self.transforms = transforms.Compose([
                                        transforms.Resize((300, 300)),
                                        transforms.ToTensor(),
                                        normalize
                                    ])

    def __len__(self):

        return len(self.files)

    def __getitem__(self, index):

        img = pil_loader(self.files[index])
        img = self.transforms(img)

        label = self.files[index].split('/')[-1].split('.')[0].split('_')[1]
        label = int(label)

        return img, label


class valDataProvider:

    def __init__(self, batch_size, val_dataset, distributed):
        self.batch_size = batch_size
        self.dataset = val_dataset
        self.dataiter = None
        self.iteration = 0
        self.epoch = 0
        self.distributed = distributed

    def build(self):
        if not self.distributed:
            dataloader = torch.utils.data.DataLoader(
                    self.dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True,
                    drop_last=True)
        else:
            dataloader = torch.utils.data.DataLoader(
                        self.dataset,
                        batch_size=self.batch_size,
                        shuffle=False,
                        num_workers=4,
                        pin_memory=True,
                        sampler=DistributedSampler(self.dataset),
                        drop_last=True)
        self.dataiter = iter(dataloader)

    def next(self):
        if self.dataiter is None:
            self.build()
        try:
            batch = self.dataiter.next()
            self.iteration += 1
            return batch

        except StopIteration:  # reload after finish a epoch
            self.epoch += 1
            self.build()
            self.iteration = 1  # reset and return the 1st batch

            batch = self.dataiter.next()
            return batch
