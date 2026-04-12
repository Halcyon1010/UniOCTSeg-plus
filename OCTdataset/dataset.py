import os
import numpy as np
from torch.utils.data import Dataset
import albumentations as A

from OCTdataset.class2label import (
    new_layers_generate_v2,
    generate_all_taskonehot,
    layer2onehot_label
)

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'


# =========================
# Utils
# =========================
def get_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.GaussNoise(p=0.5),
        A.MedianBlur(blur_limit=7, p=0.5),
        A.MotionBlur(blur_limit=7, p=0.5),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.5)
    ])


def normalize(img, mean, std):
    return (img - mean) / std


def build_name2idx(layer_names):
    return {name: i for i, name in enumerate(layer_names)}


def map_to_global(onehot_labels, label_names, out_labels, name2idx):
    for i, name in enumerate(label_names):
        if name in name2idx:
            out_labels[name2idx[name]] = onehot_labels[i]
    return out_labels


def is_unlabeled(dataset_name):
    return dataset_name in ['Lab_unlabeled', 'Harvard_30k', 'unlabeled']


# =========================
# Base Dataset
# =========================
class BaseOCTDataset(Dataset):
    def __init__(self, root_path, split):
        self.split = split

        self.norm_mean = 105.07323039
        self.norm_std = 32.7348385

        with open(root_path, 'r') as f:
            self.data_paths = [line.strip() for line in f]

        print(f"{split} dataset num: {len(self.data_paths)}")

    def __len__(self):
        return len(self.data_paths)

    def load_npz(self, path):
        data = np.load(path)
        return data['img'], data.get('label'), str(data['dataset_name'])


# =========================
# Test Dataset
# =========================
class UniversalOCTDataset(BaseOCTDataset):

    def __getitem__(self, index):
        img, label, dataset_name = self.load_npz(self.data_paths[index])

        # normalization
        img = normalize(img, self.norm_mean, self.norm_std)

        # label → onehot
        onehot_labels, label_names, _ = layer2onehot_label(dataset_name, label)

        # global template
        out_layer_names, out_task, out_labels = generate_all_taskonehot()
        name2idx = build_name2idx(out_layer_names)

        # mapping
        out_labels = map_to_global(onehot_labels, label_names, out_labels, name2idx)

        return (
            np.float32(img),
            np.array(out_labels),
            np.array(out_task),
            dataset_name,
            self.data_paths[index]
        )


# =========================
# Train Dataset
# =========================
class UniversalOCTTrainDataset(BaseOCTDataset):

    def __init__(self, root_path, split):
        super().__init__(root_path, split)
        self.transform = get_transforms()

    def __getitem__(self, index):
        img, label, dataset_name = self.load_npz(self.data_paths[index])

        out_layer_names, out_task, out_labels = generate_all_taskonehot()
        name2idx = build_name2idx(out_layer_names)

        if not is_unlabeled(dataset_name):

            # augmentation
            augmented = self.transform(
                image=img.astype(np.uint8),
                mask=label
            )
            img, label = augmented['image'], augmented['mask']

            # onehot
            onehot_labels, label_names, task_onehots = layer2onehot_label(dataset_name, label)

            # generate extra labels
            gen_labels, gen_names, _ = new_layers_generate_v2(
                onehot_labels, label_names, task_onehots
            )

            onehot_labels += gen_labels
            label_names += gen_names

            # mapping
            out_labels = map_to_global(onehot_labels, label_names, out_labels, name2idx)

        else:
            if img.ndim == 3:
                img = img[0]

        img = normalize(img, self.norm_mean, self.norm_std)

        return (
            np.float32(img),
            np.array(out_labels),
            np.array(out_task),
            self.data_paths[index]
        )


# =========================
# Semi Dataset
# =========================
class UniversalOCTTrainDataset_Semi(BaseOCTDataset):

    def __init__(self, root_path, split):
        super().__init__(root_path, split)
        self.transform = get_transforms()

    def __getitem__(self, index):
        img, label, dataset_name = self.load_npz(self.data_paths[index])

        out_layer_names, out_task, out_labels = generate_all_taskonehot()
        name2idx = build_name2idx(out_layer_names)

        if not is_unlabeled(dataset_name):

            augmented = self.transform(
                image=img.astype(np.uint8),
                mask=label
            )
            img, label = augmented['image'], augmented['mask']

            onehot_labels, label_names, task_onehots = layer2onehot_label(dataset_name, label)

            gen_labels, gen_names, gen_task = new_layers_generate_v2(
                onehot_labels, label_names, task_onehots
            )

            onehot_labels += gen_labels
            label_names += gen_names

            out_labels = map_to_global(onehot_labels, label_names, out_labels, name2idx)

        else:
            if img.ndim == 3:
                img = img[0]

        img = normalize(img, self.norm_mean, self.norm_std)

        return (
            np.float32(img),
            np.array(out_labels),
            np.array(out_task),
            self.data_paths[index]
        )