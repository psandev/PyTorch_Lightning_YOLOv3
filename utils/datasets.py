import glob
import random
import os
import sys
import warnings
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from wcmatch.pathlib import Path
import h5py
import pickle

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class ImageFolder(Dataset):
    def __init__(self, folder_path, transform=None):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.transform = transform

    def __getitem__(self, index):

        img_path = self.files[index % len(self.files)]
        img = np.array(
            Image.open(img_path).convert('RGB'),
            dtype=np.uint8)

        # Label Placeholder
        boxes = np.zeros((1, 5))

        # Apply transforms
        if self.transform:
            img, _ = self.transform((img, boxes))

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, image_list_file, cache_path, img_size=416, multiscale=True, transform=None, cache=True):
        with open(image_list_file, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.cache_path = cache_path
        self.cache = cache
        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.transform = transform

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        if self.cache:
            p = Path(img_path)

            try:
                path_fserial = Path(self.cache_path) / p.parts[-2]
                fname_serial = path_fserial / f'{p.stem}.pcl'
            except:
                print(f'{img_path} is bad')
                return
            if fname_serial.is_file():
                # fp = h5py.File(fname_serial, "r")
                # img = fp.get("image")[()]
                # boxes = fp.get("boxes")[()]
                # img_path = fp.attrs['img_path']
                # fp.close()
                try:
                    with open(fname_serial, 'rb') as f:
                        data = pickle.load(f)
                        img = data['image']
                        boxes = data['boxes']
                        img_path = data['img_path']
                except:
                    print(f'{fname_serial.as_posix()} damaged')
                    return
                if self.transform:
                    try:
                        img, bb_targets = self.transform((img, boxes))
                    except:
                        print(f"Could not apply transform")
                return img_path, img, bb_targets
            else:
                try:
                    img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
                except Exception as e:
                    print(f"Could not read image '{img_path}'.")
                    return


        else:
            try:
                img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
            except Exception as e:
                print(f"Could not read image '{img_path}'.")
                return

        # ---------
        #  Label
        # ---------
        try:
            label_path = self.label_files[index % len(self.img_files)].rstrip()

            # Ignore warning if file is empty
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boxes = np.loadtxt(label_path).reshape(-1, 5)
        except Exception as e:
            print(f"Could not read label '{label_path}'.")
            return
        if self.cache:
          with open(fname_serial, 'wb') as f:
              pickle.dump({'image': img, 'boxes': boxes, 'img_path': img_path}, f)



        # -----------
        #  Transform
        # -----------
        if self.transform:
            try:
                img, bb_targets = self.transform((img, boxes))
            except:
                print(f"Could not apply transform.")
                return




        return img_path, img, bb_targets

    def collate_fn(self, batch):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        paths, imgs, bb_targets = list(zip(*batch))

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))

        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        # Add sample index to targets
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)

        return paths, imgs, bb_targets

    def __len__(self):
        return len(self.img_files)
