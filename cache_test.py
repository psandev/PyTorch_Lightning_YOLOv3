from PIL import Image
import cv2
import numpy as np
import msgpack_numpy as m
# import msgpack as m
from loguru import logger
import time
import h5py
import functools
from omegaconf import OmegaConf
from wcmatch.pathlib import Path
import pickle
from tqdm import tqdm

level = 'DEBUG'
logger.opt(record=True).add('log.log', format=' {time:YYYY-MMM HH:mm:ss} {name}:{function}:{line} <lvl>{message}</>',level='DEBUG', rotation='5 MB')
# logger.add(sys.stderr)

n_files=4800
def timeit(func):
    # name = func.__name__

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.opt(colors=True).debug(f"Function {func.__name__} executed in <red>{end - start:f} s</>")
        return result
    return wrapped


conf = OmegaConf.load('config/config.yaml')
with open(conf.data.val, "r") as file:
    img_files = file.readlines()

label_files = [path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in img_files]



serial_dir =  '/media/peter/NVME/coco_serialized'
serial_path =  Path(serial_dir)

@timeit
def load_img_file():
    for idx in tqdm(range(len(img_files))):
        try:
            img_path = img_files[idx % len(img_files)].rstrip()
            img = Image.open(img_path).convert('RGB')
            # if img.size[0] != 416 or img.size[1] != 416:
            #     img = img.resize((416, 416), resample=3)
            img = np.array(img, dtype=np.uint8)
        except:
            logger.opt(colors=True).info(f'{img_path} damaged or non-existent')
            continue
        try:
            label_path = label_files[idx % len(img_files)].rstrip()
            boxes = np.loadtxt(label_path).reshape(-1, 5)
        except Exception as e:
            print(f"Could not read label '{label_path}'.")
            continue

        # dict_to_save = {'image': img, 'bbox': boxes, 'img_path' : img_path}
        p = Path(img_path)
        path_to_save = serial_path/p.parts[-2]
        path_to_save.mkdir(parents=True, exist_ok=True)
        fname = path_to_save/f'{p.stem}.pcl'
        # serial = m.packb(img, default=m.encode)
        with open(fname, 'wb') as f:
            pickle.dump({'image': img, 'boxes': boxes, 'img_path': img_path}, f)

        # file = h5py.File(path_to_save/f'{p.stem}.h5', "w")
        # dataset = file.create_dataset(
        #     "image", np.shape(img), h5py.h5t.STD_U8BE, data=img)
        # label_set = file.create_dataset(
        #     "boxes", np.shape(boxes), h5py.h5t.STD_U8BE, data=boxes)
        # dtype = 'S{0}'.format(len(img_path))
        # file.attrs['img_path'] = img_path
        # file.close()





@timeit
def read_serial():
    for file in list(serial_path.rglob('*.pcl')):
        try:
            # with open(file.as_posix(), 'rb') as f:
            #     imng = m.unpackb(f.read())

            # fp = h5py.File(file, "r")
            # img = fp.get("image")
            # boxes = fp.get("boxes")
            # img_path = fp.attrs['img_path']
            # fp.close()
            with open(file.as_posix(), 'rb') as f:
                data = pickle.load(f)
        except:
            print(f'{file.as_posix()} damaged or non-existannt')

load_img_file()
read_serial()