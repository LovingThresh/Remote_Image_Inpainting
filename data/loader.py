# -*- coding: utf-8 -*-
# @Time    : 2022/11/12 13:54
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : loader.py
# @Software: PyCharm
import os
import re
import logging
import random
from copy import deepcopy
from contextlib import suppress
from functools import partial
from itertools import repeat
from typing import Callable
from abc import abstractmethod
from typing import Dict, List, Optional, Set, Tuple, Union
import torch
import torch.utils.data as data
from torchvision import transforms
import numpy as np

_logger = logging.getLogger(__name__)

IMAGENET_DEFAULT_MEAN = [0]
IMAGENET_DEFAULT_STD = [0]

IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg')  # singleton, kept public for bwd compat use
_IMG_EXTENSIONS_SET = set(IMG_EXTENSIONS)  # set version, private, kept in sync


def get_img_extensions(as_set=False):
    return deepcopy(_IMG_EXTENSIONS_SET if as_set else IMG_EXTENSIONS)


def fast_collate(batch):
    """ A fast collation function optimized for uint8 images (np array or torch) and int64 targets (labels)"""
    assert isinstance(batch[0], tuple)
    batch_size = len(batch)
    if isinstance(batch[0][0], tuple):
        # This branch 'deinterleaves' and flattens tuples of input tensors into one tensor ordered by position
        # such that all tuple of position n will end up in a torch.split(tensor, batch_size) in nth position
        inner_tuple_size = len(batch[0][0])
        flattened_batch_size = batch_size * inner_tuple_size
        targets = torch.zeros(flattened_batch_size, dtype=torch.int64)
        tensor = torch.zeros((flattened_batch_size, *batch[0][0][0].shape), dtype=torch.uint8)
        for i in range(batch_size):
            assert len(batch[i][0]) == inner_tuple_size  # all input tensor tuples must be same length
            for j in range(inner_tuple_size):
                targets[i + j * batch_size] = batch[i][1]
                tensor[i + j * batch_size] += torch.from_numpy(batch[i][0][j])
        return tensor, targets
    elif isinstance(batch[0][0], np.ndarray):
        targets = torch.tensor([b[1] for b in batch], dtype=torch.int64)
        assert len(targets) == batch_size
        tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.uint8)
        for i in range(batch_size):
            tensor[i] += torch.from_numpy(batch[i][0])
        return tensor, targets
    elif isinstance(batch[0][0], torch.Tensor):
        targets = torch.tensor([b[1] for b in batch], dtype=torch.int64)
        assert len(targets) == batch_size
        tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.uint8)
        for i in range(batch_size):
            tensor[i].copy_(batch[i][0])
        return tensor, targets
    else:
        assert False


def adapt_to_chs(x, n):
    if not isinstance(x, (tuple, list)):
        x = tuple(repeat(x, n))
    elif len(x) != n:
        x_mean = np.mean(x).item()
        x = (x_mean,) * n
        _logger.warning(f'Pretrained mean/std different shape than model, using avg value {x}.')
    else:
        assert len(x) == n, 'normalization stats must match image channels'
    return x


def _worker_init(worker_id, worker_seeding='all'):
    worker_info = torch.utils.data.get_worker_info()
    assert worker_info.id == worker_id
    if isinstance(worker_seeding, Callable):
        seed = worker_seeding(worker_info)
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))
    else:
        assert worker_seeding in ('all', 'part')
        # random / torch seed already called in dataloader iter class w/ worker_info.seed
        # to reproduce some old results (same seed + hparam combo), partial seeding is required (skip numpy re-seed)
        if worker_seeding == 'all':
            np.random.seed(worker_info.seed % (2 ** 32 - 1))


def natural_key(string_):
    """See https://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def find_images_and_targets(
        folder: str,
        types: Optional[Union[List, Tuple, Set]] = None,
        class_to_idx: Optional[Dict] = None,
        leaf_name_only: bool = True,
        sort: bool = True
):
    """ Walk folder recursively to discover images and map them to classes by folder names.
    Args:
        folder: root of folder to recrusively search
        types: types (file extensions) to search for in path
        class_to_idx: specify mapping for class (folder name) to class index if set
        leaf_name_only: use only leaf-name of folder walk for class names
        sort: re-sort found images by name (for consistent ordering)
    Returns:
        A list of image and target tuples, class_to_idx mapping
    """
    types = get_img_extensions(as_set=True) if not types else set(types)
    labels = []
    filenames = []
    for root, subdirs, files in os.walk(folder, topdown=False, followlinks=True):
        rel_path = os.path.relpath(root, folder) if (root != folder) else ''
        label = os.path.basename(rel_path) if leaf_name_only else rel_path.replace(os.path.sep, '_')
        for f in files:
            base, ext = os.path.splitext(f)
            if ext.lower() in types:
                filenames.append(os.path.join(root, f))
                labels.append(label)

    if class_to_idx is None:
        # building class index
        unique_labels = set(labels)
        sorted_labels = list(sorted(unique_labels, key=natural_key))
        class_to_idx = {c: idx for idx, c in enumerate(sorted_labels)}
    images_and_targets = [(f, class_to_idx[l]) for f, l in zip(filenames, labels) if l in class_to_idx]
    if sort:
        images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
    return images_and_targets, class_to_idx


class PrefetchLoader:

    def __init__(
            self,
            loader,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
            channels=3,
            device=torch.device('cuda'),
            img_dtype=torch.float32,
            fp16=False,
            re_prob=0.,
            re_mode='const',
            re_count=1,
            re_num_splits=0):

        mean = adapt_to_chs(mean, channels)
        std = adapt_to_chs(std, channels)
        normalization_shape = (1, channels, 1, 1)

        self.loader = loader
        self.device = device
        if fp16:
            # fp16 arg is deprecated, but will override dtype arg if set for bwd compat
            img_dtype = torch.float16
        self.img_dtype = img_dtype
        self.mean = torch.tensor(
            [x * 255 for x in mean], device=device, dtype=img_dtype).view(normalization_shape)
        self.std = torch.tensor(
            [x * 255 for x in std], device=device, dtype=img_dtype).view(normalization_shape)

        self.is_cuda = torch.cuda.is_available() and device.type == 'cuda'

    def __iter__(self):
        first = True
        if self.is_cuda:
            stream = torch.cuda.Stream()
            stream_context = partial(torch.cuda.stream, stream=stream)
        else:
            stream = None
            stream_context = suppress

        for next_input, next_target in self.loader:

            with stream_context():
                next_input = next_input.to(device=self.device, non_blocking=True)
                next_target = next_target.to(device=self.device, non_blocking=True)
                next_input = next_input.to(self.img_dtype).sub_(self.mean).div_(self.std)

            if not first:
                yield input, target
            else:
                first = False

            if stream is not None:
                torch.cuda.current_stream().wait_stream(stream)

            input = next_input
            target = next_target

        yield input, target

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset


def create_transform(
        input_size,
        is_training=False,
        use_prefetcher=False,
        no_aug=False,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        color_jitter=0.4,
        auto_augment=None,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_num_splits=0,
        crop_pct=None,
        tf_preprocessing=False,
        separate=False):
    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
    else:
        img_size = input_size

    if tf_preprocessing and use_prefetcher:
        assert not separate, "Separate transforms not supported for TF preprocessing"
        from timm.data.tf_preprocessing import TfPreprocessTransform
        transform = TfPreprocessTransform(
            is_training=is_training, size=img_size, interpolation=interpolation)
    if is_training:
        transform = transforms.Compose([transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.ToTensor()])

    return transform


def create_loader(
        dataset,
        input_size,
        batch_size,
        is_training=False,
        use_prefetcher=True,
        no_aug=False,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_split=False,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        color_jitter=0.4,
        auto_augment=None,
        num_aug_repeats=0,
        num_aug_splits=0,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        num_workers=1,
        crop_pct=None,
        collate_fn=None,
        pin_memory=False,
        fp16=False,  # deprecated, use img_dtype
        img_dtype=torch.float32,
        device=torch.device('cuda'),
        tf_preprocessing=False,
        use_multi_epochs_loader=False,
        persistent_workers=True,
        worker_seeding='all',
):
    re_num_splits = 0
    if re_split:
        # apply RE to second half of batch if no aug split otherwise line up with aug split
        re_num_splits = num_aug_splits or 2
    dataset.transform = create_transform(
        input_size,
        is_training=is_training,
        use_prefetcher=use_prefetcher,
        no_aug=no_aug,
        scale=scale,
        ratio=ratio,
        hflip=hflip,
        vflip=vflip,
        color_jitter=color_jitter,
        auto_augment=auto_augment,
        interpolation=interpolation,
        mean=mean,
        std=std,
        crop_pct=crop_pct,
        tf_preprocessing=tf_preprocessing,
        re_prob=re_prob,
        re_mode=re_mode,
        re_count=re_count,
        re_num_splits=re_num_splits,
        separate=num_aug_splits > 0,
    )

    sampler = None
    assert num_aug_repeats == 0, "RepeatAugment not currently supported in non-distributed or IterableDataset use"

    if collate_fn is None:
        collate_fn = fast_collate if use_prefetcher else torch.utils.data.dataloader.default_collate

    loader_class = torch.utils.data.DataLoader

    loader_args = dict(
        batch_size=batch_size,
        shuffle=not isinstance(dataset, torch.utils.data.IterableDataset) and sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=is_training,
        worker_init_fn=partial(_worker_init, worker_seeding=worker_seeding),
        persistent_workers=persistent_workers
    )
    try:
        loader = loader_class(dataset, **loader_args)
    except TypeError as e:
        loader_args.pop('persistent_workers')  # only in Pytorch 1.7+
        loader = loader_class(dataset, **loader_args)
    if use_prefetcher:
        prefetch_re_prob = re_prob if is_training and not no_aug else 0.
        loader = PrefetchLoader(
            loader,
            mean=mean,
            std=std,
            channels=input_size[0],
            device=device,
            fp16=fp16,  # deprecated, use img_dtype
            img_dtype=img_dtype,
            re_prob=prefetch_re_prob,
            re_mode=re_mode,
            re_count=re_count,
            re_num_splits=re_num_splits
        )

    return loader


class IterableImageDataset(data.IterableDataset):

    def __init__(
            self,
            root,
            reader=None,
            split='train',
            is_training=False,
            batch_size=None,
            seed=42,
            repeats=0,
            download=False,
            transform=None,
            target_transform=None,
    ):
        assert reader is not None
        if isinstance(reader, str):
            self.reader = create_reader(
                root=root,
                split=split,
                is_training=is_training,
                batch_size=batch_size,
                seed=seed,
                repeats=repeats,
                download=download,
            )
        else:
            self.reader = reader
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

    def __iter__(self):
        for img, target in self.reader:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            yield img, target

    def __len__(self):
        if hasattr(self.reader, '__len__'):
            return len(self.reader)
        else:
            return 0

    def set_epoch(self, count):
        # TFDS and WDS need external epoch count for deterministic cross process shuffle
        if hasattr(self.reader, 'set_epoch'):
            self.reader.set_epoch(count)

    @staticmethod
    def filename(index, basename=False, absolute=False):
        assert False, 'Filename lookup by index not supported, use filenames().'

    def filenames(self, basename=False, absolute=False):
        return self.reader.filenames(basename, absolute)


class Reader:
    def __init__(self):
        pass

    @abstractmethod
    def _filename(self, index, basename=False, absolute=False):
        pass

    def filename(self, index, basename=False, absolute=False):
        return self._filename(index, basename=basename, absolute=absolute)

    def filenames(self, basename=False, absolute=False):
        return [self._filename(index, basename=basename, absolute=absolute) for index in range(len(self))]


class ReaderImageFolder(Reader):

    def __init__(
            self,
            root,
            class_map=''):
        super().__init__()

        self.root = root
        class_to_idx = None
        self.samples, self.class_to_idx = find_images_and_targets(root, class_to_idx=class_to_idx)

    def __getitem__(self, index):
        path, target = self.samples[index]
        return open(path, 'rb'), target

    def __len__(self):
        return len(self.samples)

    def _filename(self, index, basename=False, absolute=False):
        filename = self.samples[index][0]
        if basename:
            filename = os.path.basename(filename)
        elif not absolute:
            filename = os.path.relpath(filename, self.root)
        return filename


def create_reader(root, split='train', **kwargs):

    assert os.path.exists(root)
    reader = ReaderImageFolder(root, **kwargs)
    return reader
