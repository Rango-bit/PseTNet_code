import os
import random
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset


def expand2square(pil_img, background_color=(0)):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class DriveDataset(Dataset):
    def __init__(self, root, image_subfile, mask_subfile, train: bool = False,
                 val: bool = False, transforms = None, train_data_ratio: float = 1.0,
                 dataset_name: str = 'Spineweb-16'):
        super(DriveDataset, self).__init__()
        self.dataset_name = dataset_name

        if train:
            subfile = 'train'
            print('Use dataset:', dataset_name)
        elif val:
            subfile = 'val'
        else:
            subfile = 'test'

        image_path = os.path.join(root, image_subfile, subfile)
        mask_path = os.path.join(root, mask_subfile, subfile)
        self.img_list = os.listdir(image_path)
        if train_data_ratio < 1.0:
            print('Proportion of training set:', train_data_ratio)
            self.img_list = random.sample(self.img_list, int(len(self.img_list) * train_data_ratio))
        self.transforms = transforms
        print(subfile, 'data num:', len(self.img_list))
        self.img_list_path = [os.path.join(image_path, i) for i in self.img_list]
        self.mask_list_path = [os.path.join(mask_path, i) for i in self.img_list]

    def __getitem__(self, idx):
        img = Image.open(self.img_list_path[idx]).convert('L')
        roi_mask = Image.open(self.mask_list_path[idx]).convert('L')
        mask = np.array(roi_mask)

        if self.dataset_name in ['Spineweb-16', 'DFU']:
            mask[mask < 128] = 0
            mask[mask >= 128] = 255
            mask = mask / 255
        elif self.dataset_name == 'ADCD':
            pass
        else:
            raise ValueError('Dataset not supported')

        mask = Image.fromarray(mask)
        if self.dataset_name == 'Spineweb-16':
            img = ImageOps.equalize(img)
            img = expand2square(img, (0))
            mask = expand2square(mask, (0))

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)
            if img.size()[0] == 1:
                img = img.repeat(3, 1, 1)

        return img, mask

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size  # B, C, W, H
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs
