from __future__ import print_function, division
import os
from torch.utils.data import Dataset
from PIL import Image, ImageOps, ImageFilter
from random import random, randint
import warnings

warnings.filterwarnings("ignore")


def make_dataset(root, mode):
    assert mode in ['train', 'val', 'test', 'unlabeled']
    items = []

    if mode == 'train':
        train_img_path = os.path.join(root, 'train', 'Img')
        train_mask_path = os.path.join(root, 'train', 'GT')

        images = os.listdir(train_img_path)
        labels = os.listdir(train_mask_path)

        images.sort()
        labels.sort()

        for it_im, it_gt in zip(images, labels):
            item = (os.path.join(train_img_path, it_im), os.path.join(train_mask_path, it_gt))
            items.append(item)

    # Adding code to load unlabeled images
    elif mode == 'unlabeled':
        unlabeled_img_path = os.path.join(root, 'train', 'Img-Unlabeled')

        images = os.listdir(unlabeled_img_path)
        images.sort()

        for it_im in images:
            item = (os.path.join(unlabeled_img_path, it_im), None)
            items.append(item)

    elif mode == 'val':
        val_img_path = os.path.join(root, 'val', 'Img')
        val_mask_path = os.path.join(root, 'val', 'GT')

        images = os.listdir(val_img_path)
        labels = os.listdir(val_mask_path)

        images.sort()
        labels.sort()

        for it_im, it_gt in zip(images, labels):
            item = (os.path.join(val_img_path, it_im), os.path.join(val_mask_path, it_gt))
            items.append(item)
    else:
        test_img_path = os.path.join(root, 'test', 'Img')
        test_mask_path = os.path.join(root, 'test', 'GT')

        images = os.listdir(test_img_path)
        labels = os.listdir(test_mask_path)

        images.sort()
        labels.sort()

        for it_im, it_gt in zip(images, labels):
            item = (os.path.join(test_img_path, it_im), os.path.join(test_mask_path, it_gt))
            items.append(item)

    return items


class MedicalImageDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, mode, root_dir, transform=None, mask_transform=None, augment=True, equalize=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.imgs = make_dataset(root_dir, mode)
        self.augmentation = augment
        self.equalize = equalize
        self.mode = mode

    def __len__(self):
        return len(self.imgs)

    def apply_flip(self, img, mask, probability):
        if random() > probability:
            img = ImageOps.flip(img)
            mask = ImageOps.flip(mask)
        return img, mask

    def apply_mirror(self, img, mask, probability):
        if random() > probability:
            img = ImageOps.mirror(img)
            mask = ImageOps.mirror(mask)
        return img, mask

    def apply_rotation(self, img, mask, probability, max_angle=30):
        if random() > probability:
            angle = random() * 2 * max_angle - max_angle
            img = img.rotate(angle)
            mask = mask.rotate(angle)
        return img, mask

    def apply_translation(self, img, mask, probability, max_translation=10):
        if random() > probability:
            trans_x = randint(-max_translation, max_translation)
            trans_y = randint(-max_translation, max_translation)
            img = img.transform(img.size, Image.AFFINE, (1, 0, trans_x, 0, 1, trans_y))
            mask = mask.transform(mask.size, Image.AFFINE, (1, 0, trans_x, 0, 1, trans_y))
        return img, mask

    def apply_opening(self, mask, probability):
        if random() > probability:
            mask = mask.filter(ImageFilter.MinFilter(3))
            mask = mask.filter(ImageFilter.MaxFilter(3))
        return mask

    def apply_closing(self, mask, probability):
        if random() > probability:
            mask = mask.filter(ImageFilter.MaxFilter(3))
            mask = mask.filter(ImageFilter.MinFilter(3))
        return mask

    def augment(self, img, mask):
        img, mask = self.apply_flip(img, mask, probability=0.5)
        img, mask = self.apply_mirror(img, mask, probability=0.5)
        img, mask = self.apply_rotation(img, mask, probability=0.5, max_angle=30)
        img, mask = self.apply_translation(img, mask, probability=0.5, max_translation=10)
        mask = self.apply_opening(mask, probability=0.5)
        mask = self.apply_closing(mask, probability=0.5)

        return img, mask

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        img = Image.open(img_path)
        mask = Image.open(mask_path).convert('L') if mask_path else None

        if self.equalize:
            img = ImageOps.equalize(img)

        if self.augmentation and mask is not None:
            img, mask = self.augment(img, mask)

        if self.transform:
            img = self.transform(img)
            mask = self.mask_transform(mask) if mask is not None else None

        return [img, mask, img_path]
