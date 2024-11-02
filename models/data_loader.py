import os
from typing import Dict, Literal, Tuple
from time import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from scipy.interpolate import UnivariateSpline


from models.data_model import BatchDict
from utils import bcolors
from scipy.ndimage import zoom

import random


class CoverDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        file_ext: str,
        dataset_path: str,
        data_split: Literal["train", "val", "test"],
        debug: bool,
        max_len: int,
        config: Dict,

    ) -> None:
        super().__init__()
        self.config = config
        self.augmentations = config["augmentations"]

        self.data_path = data_path
        self.file_ext = file_ext
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.debug = debug
        self.max_len = max_len
        self._load_data()

        self.ram_storage = config.get("store_data_in_ram", False)
        if self.ram_storage:
            self.all_cqt_specs = dict()
            self._load_all_cqt()

        self.rnd_indices = np.random.permutation(len(self.track_ids))
        self.current_index = 0

    def __len__(self) -> int:
        return len(self.track_ids)

    def __getitem__(self, index: int) -> BatchDict:
        track_id = self.track_ids[index]
        anchor_cqt = self._load_cqt(track_id)
        
        if self.data_split == "train":
            clique_id = self.version2clique.loc[track_id, 'clique']
            pos_id, neg_id = self._triplet_sampling(track_id, clique_id)        
            positive_cqt = self._load_cqt(pos_id)
            negative_cqt = self._load_cqt(neg_id)
            neg_clique_id = self.version2clique.loc[neg_id, 'clique']
        else:
            clique_id = -1
            neg_clique_id = -1
            pos_id = torch.empty(0)
            positive_cqt = torch.empty(0)
            neg_id = torch.empty(0)
            negative_cqt = torch.empty(0)
        return dict(
            anchor_id=track_id,
            anchor=anchor_cqt,
            anchor_label=torch.tensor(clique_id, dtype=torch.float),
            positive_id=pos_id,
            positive=positive_cqt,
            negative_id=neg_id,
            negative=negative_cqt,
            negative_label = torch.tensor(neg_clique_id, dtype=torch.float)
        )

    def _make_file_path(self, track_id, file_ext):
        a = track_id % 10
        b = track_id // 10 % 10
        c = track_id // 100 % 10
        return os.path.join(str(c), str(b), str(a), f'{track_id}.{file_ext}')

    def _triplet_sampling(self, track_id: int, clique_id: int) -> Tuple[int, int]:
        versions = self.versions.loc[clique_id, "versions"]
        pos_list = np.setdiff1d(versions, track_id)
        pos_id = np.random.choice(pos_list, 1)[0]
        if self.current_index >= len(self.rnd_indices):
            self.current_index = 0
            self.rnd_indices = np.random.permutation(len(self.track_ids))
        neg_id = self.track_ids[self.rnd_indices[self.current_index]]
        self.current_index += 1
        while neg_id in versions:
            if self.current_index >= len(self.rnd_indices):
                self.current_index = 0
                self.rnd_indices = np.random.permutation(len(self.track_ids))
            neg_id = self.track_ids[self.rnd_indices[self.current_index]]
            self.current_index += 1
        return (pos_id, neg_id)

    def _load_data(self) -> None:
        if self.data_split in ['train', 'val']:
            cliques_subset = np.load(os.path.join(self.data_path, "splits", "{}_cliques.npy".format(self.data_split)))
            self.versions = pd.read_csv(
                os.path.join(self.data_path, "cliques2versions.tsv"), sep='\t', converters={"versions": eval}
            )
            self.versions = self.versions[self.versions["clique"].isin(set(cliques_subset))]
            mapping = {}
            for k, clique in enumerate(sorted(cliques_subset)):
                mapping[clique] = k
            self.versions["clique"] = self.versions["clique"].map(lambda x: mapping[x])
            self.versions.set_index("clique", inplace=True)
            self.version2clique = pd.DataFrame(
                [{'version': version, 'clique': clique} for clique, row in self.versions.iterrows() for version in row['versions']]
            ).set_index('version')
            self.track_ids = self.version2clique.index.to_list()
        else:
            self.track_ids = np.load(os.path.join(self.data_path, "splits", "{}_ids.npy".format(self.data_split)))

    def _roll(self, cqt_spec):
        shift_num = self.config["aug_params"]["roll_shift_num"]
        w, h = np.shape(cqt_spec)
        shift_amount = random.randint(-1, 1) * shift_num
        # for i in range(w):
        #     cqt_spec[i, :] = np.roll(cqt_spec[i, :], shift_amount)
        cqt_spec = np.roll(cqt_spec, shift_amount, axis=1)
        return cqt_spec
    
    def _time_roll(self, cqt_spec):
        min_timeroll_shift_num = self.config["aug_params"]["min_timeroll_shift_num"]
        max_timeroll_shift_num = self.config["aug_params"]["max_timeroll_shift_num"]
        shift_num = random.randint(min_timeroll_shift_num, max_timeroll_shift_num)
        w, h = np.shape(cqt_spec)
        # shift_amount = random.randint(-1, 1) * shift_num
        # for i in range(w):
        #     cqt_spec[i, :] = np.roll(cqt_spec[i, :], shift_amount)
        cqt_spec = np.roll(cqt_spec, shift_num, axis=0)
        return cqt_spec

    def _random_erase(self, cqt_spec):
        region_num = self.config["aug_params"]["mask_region_num"]
        region_size = self.config["aug_params"]["mask_region_size"]
        region_val = random.random() * self.config["aug_params"]["region_val"]
        w, h = np.shape(cqt_spec)
        region_w = int(w * region_size[0])
        region_h = int(h * region_size[1])

        centers = np.random.rand(region_num, 2)
        centers[:, 0] *= (w - region_w)
        centers[:, 1] *= (h - region_h)
        for center_w, center_h in centers:
            cqt_spec[int(center_w - region_w // 2):int(center_w + region_w // 2),
                    int(center_h - region_h // 2):int(center_h + region_h // 2)] = region_val
        # for _ in range(region_num):
        #     center_w = int(random.random() * (w - region_w))
        #     center_h = int(random.random() * (h - region_h))
        #     cqt_spec[center_w - region_w // 2:center_w + region_w // 2,
        #     center_h - region_h // 2:center_h + region_h // 2] = region_val
        return cqt_spec
    
    def _random_time_crop(self, cqt_spec):
        # chunk_len should be equal for all objects in batch.
        chunk_len = random.randint(self.config["aug_params"]["crop_min_chunklen"],
                                   self.config["aug_params"]["crop_max_chunklen"] )   
        cqt_spec_len, h = np.shape(cqt_spec)
        start = int(random.random() * (cqt_spec_len - chunk_len))
        cqt_spec = cqt_spec[start:start + chunk_len]
        return cqt_spec
    
    def _change_volume(self, cqt_spec):
        coef = random.uniform(self.config["aug_params"]["low_volume_coef"], self.config["aug_params"]["high_volume_coef"])
        return cqt_spec * coef
    
    def _time_stretching(self, cqt_spec: np.ndarray) -> np.ndarray:
        """
        Изменяет темп CQT-спектрограммы путем интерполяции по временной оси.

        Args:
            cqt_spec: np.ndarray

        Returns:
            cqt_spectrogram_stretched: np.ndarray
        """
        low_stretch_factor = self.config["aug_params"]["low_stretch_factor"]
        high_stretch_factor = self.config["aug_params"]["high_stretch_factor"]
        stretch_factor = np.random.uniform(low_stretch_factor, high_stretch_factor)

        if stretch_factor <= 0:
            raise ValueError(f"stretch_factor must be positive, but equals {stretch_factor}")

        num_frames, num_bins = cqt_spec.shape
        new_num_frames = int(np.round(num_frames * stretch_factor))
        cqt_stretched = zoom(cqt_spec, (stretch_factor, 1), order=3)
        return cqt_stretched

    
    def _apply_equalize(self, cqt_spec, smoothness=5):
        num_bins = cqt_spec.shape[1]
        x = np.linspace(0, num_bins - 1, num_bins)
        random_points = np.random.uniform(self.config["aug_params"]["equalize_low_factor"], 
                                          self.config["aug_params"]["equalize_high_factor"], size=(smoothness,))
        # print(random_points)
        spline = UnivariateSpline(np.linspace(0, num_bins - 1, smoothness), random_points, s=0)
        eq_curve = spline(x)

        # print(eq_curve)
        # print(cqt_spec.shape)
        return cqt_spec * eq_curve
    
    def _time_mask(self, cqt_spec):
        num_time_masks = self.config["aug_params"]["num_time_masks"] 
        max_mask_size = self.config["aug_params"]["max_time_mask_size"]
        region_val = self.config["aug_params"]["region_val"]

        w, h = np.shape(cqt_spec)

        for _ in range(num_time_masks):
            mask_start = random.randint(0, w - max_mask_size)
            mask_length = random.randint(1, max_mask_size)
            cqt_spec[mask_start:mask_start + mask_length, :] = region_val

        return cqt_spec
    
    def _apply_padding(self, cqt_spec):
        w, h = np.shape(cqt_spec)
        padding_value = self.config["aug_params"]["pad_value"]
        
        if w > self.max_len:
            cqt_spec = cqt_spec[:self.max_len, :]
        elif w < self.max_len:
            padding_width = self.max_len - w
            cqt_spec = np.pad(cqt_spec, ((0, padding_width), (0, 0)), 'constant', constant_values=padding_value)
        
        return cqt_spec

    def _mask_silence(self, cqt_spec):
        # probability should be low (0.1)
        # if random.random() > p_threshold:
        #     return feat

        w, h = np.shape(cqt_spec)
        for i in range(w):
            if random.random() < 0.1:
                cqt_spec[i, :] = self.config["aug_params"]["pad_value"]
        for i in range(h):
            if random.random() < 0.1:
                cqt_spec[:, i] = self.config["aug_params"]["pad_value"]

        return cqt_spec

    
    def _apply_augmentations(self, cqt_spec: np.ndarray) -> np.ndarray:
        """
        Args:
            cqt_spectrogram: np.ndarray

        Transformations:
            time_stretching - slight compression or stretching along the time axis

        Returns:
            cqt_spectrogram_augmented: np.ndarray

        """
        # add frequency masking?
        if "mask_silence" in self.config["aug_params"] and self.config["aug_params"]["mask_silence"]:
            p = random.random()
            if p <= self.config["aug_params"]["mask_silence_prob"]:
                # print("mask_silence")
                cqt_spec = self._mask_silence(cqt_spec)

        if "time_stretching" in self.config["aug_params"] and self.config["aug_params"]["time_stretching"]:
            p = random.random()
            if p <= self.config["aug_params"]["time_stretching_prob"]:
                # print("time_stretching")
                cqt_spec = self._time_stretching(cqt_spec)


        # seems like works bad ... ?
        if "roll_pitch" in self.config["aug_params"] and self.config["aug_params"]["roll_pitch"]:
            p = random.random()
            if p <= self.config["aug_params"]["roll_pitch_prob"]:
                # print("roll")
                cqt_spec = self._roll(cqt_spec)

        # works good
        if "time_roll" in self.config["aug_params"] and self.config["aug_params"]["time_roll"]:
            p = random.random()
            if p <= self.config["aug_params"]["time_roll_prob"]:
                # print("timeroll")
                cqt_spec = self._time_roll(cqt_spec)

        # works good 
        # low_volume_coef: 0.5
        # high_volume_coef: 1.0
        if "volume" in self.config["aug_params"] and self.config["aug_params"]["volume"]:
            # print("volume")
            p = random.random()
            if p <= self.config["aug_params"]["volume_prob"]:
                # print("roll")
                cqt_spec = self._change_volume(cqt_spec)

        # works good 
        # equalize_low_factor: 0.70
        # equalize_high_factor: 1.15
        if "equalize" in self.config["aug_params"] and self.config["aug_params"]["equalize"]:
            # print("equalize")
            p = random.random()
            if p <= self.config["aug_params"]["equalize_prob"]:
                cqt_spec = self._apply_equalize(cqt_spec)


        # if "add_noise" in self.config['aug_params'].keys():
        #     None

        if "time_masking" in self.config["aug_params"] and self.config["aug_params"]["time_masking"]:
            p = random.random()
            if p <= self.config["aug_params"]["time_mask_prob"]:
                # print("time_masking")
                cqt_spec = self._time_mask(cqt_spec)

        if "random_time_crop" in self.config["aug_params"] and self.config["aug_params"]["random_time_crop"]:
            p = random.random()
            if p <= self.config["aug_params"]["random_timecrop_prob"]:
                # print("random_time_crop")
                cqt_spec = self._random_time_crop(cqt_spec)
        
        if "random_erase" in self.config["aug_params"] and self.config["aug_params"]["random_erase"]:
            p = random.random()
            if p <= self.config["aug_params"]["random_erase_prob"]:
                # print("random_erase")
                cqt_spec = self._random_erase(cqt_spec)

        cqt_spec = self._apply_padding(cqt_spec)

        # time_stretching - легкое сужение или растяжение по временной оси
        return cqt_spec

    def _load_all_cqt(self):
        for track_id in self.track_ids:
            filename = os.path.join(self.dataset_path, self._make_file_path(track_id, self.file_ext))
            cqt_spectrogram = np.load(filename).transpose(1, 0)
            self.all_cqt_specs[track_id] = cqt_spectrogram
        
    def _load_cqt(self, track_id: str) -> torch.Tensor:
        if self.ram_storage:
            cqt_spectrogram = self.all_cqt_specs[track_id]
        else:
            filename = os.path.join(self.dataset_path, self._make_file_path(track_id, self.file_ext))
            cqt_spectrogram = np.load(filename)
            cqt_spectrogram = cqt_spectrogram.transpose(1, 0)

        if self.augmentations and self.data_split == "train":
            # print("!!!!!!!!!!!!!!USE AUGMENTATIONS!!!!!!!!!!!!!!")
            cqt_spectrogram = self._apply_augmentations(cqt_spectrogram)

        return torch.from_numpy(cqt_spectrogram).float()




from typing import Literal, Dict
import random

def cover_dataloader(
    data_path: str,
    file_ext: str,
    dataset_path: str,
    data_split: Literal["train", "val", "test"],
    debug: bool,
    max_len: int,
    batch_size: int,
    config: Dict,
) -> DataLoader:
    return DataLoader(
        CoverDataset(data_path, file_ext, dataset_path,
                      data_split, debug, max_len=max_len, config=config),
        batch_size=batch_size if max_len > 0 else 1,
        num_workers=config[data_split]["num_workers"],
        shuffle=config[data_split]["shuffle"],
        drop_last=config[data_split]["drop_last"]
    )


# from torch.utils.data import RandomSampler, SequentialSampler

# class RandomChunkBatchSampler(torch.utils.data.BatchSampler):
#     def __init__(self, dataset, sampler, batch_size, drop_last, min_chunk, max_chunk):
#         super().__init__(sampler, batch_size, drop_last)
#         self.dataset = dataset
#         self.min_chunk = min_chunk
#         self.max_chunk = max_chunk

#     def __iter__(self):
#         for batch_indices in super().__iter__():
#             # Устанавливаем случайный chunk_len для текущего батча
#             chunk_len = random.randint(self.min_chunk, self.max_chunk)
#             self.dataset.set_chunk_len(chunk_len)
#             yield batch_indices


# def cover_dataloader(
#     data_path: str,
#     file_ext: str,
#     dataset_path: str,
#     data_split: Literal["train", "val", "test"],
#     debug: bool,
#     max_len: int,
#     batch_size: int,
#     config: Dict,
# ) -> DataLoader:
#     dataset = CoverDataset(data_path, file_ext, dataset_path,
#                       data_split, debug, max_len=max_len, config = config)
#     min_chunk, max_chunk = config["aug_params"]["min_chunk"], config["aug_params"]["max_chunk"]

#     base_sampler = RandomSampler(dataset) if data_split == "train" else SequentialSampler(dataset)
    
#     # Создаем кастомный BatchSampler с переменной длиной фрагмента
#     batch_sampler = RandomChunkBatchSampler(
#         dataset=dataset,
#         sampler=base_sampler,
#         batch_size=batch_size,
#         drop_last=True,
#         min_chunk=config["aug_params"]["min_chunk"],
#         max_chunk=config["aug_params"]["max_chunk"],
#     )

#     # def collate_fn(batch):
#     #     chunk_len = random.randint(config["aug_params"]["min_chunk"], config["aug_params"]["max_chunk"])
#     #     dataset.set_chunk_len(chunk_len)

#     #     batch_data = {
#     #         "anchor_id": [item["anchor_id"] for item in batch],
#     #         "anchor": torch.stack([item["anchor"] for item in batch]),
#     #         "anchor_label": torch.stack([item["anchor_label"] for item in batch]),
#     #         "positive_id": [item["positive_id"] for item in batch],
#     #         "positive": torch.stack([item["positive"] for item in batch]),
#     #         "negative_id": [item["negative_id"] for item in batch],
#     #         "negative": torch.stack([item["negative"] for item in batch]),
#     #         "negative_label": torch.stack([item["negative_label"] for item in batch]),
#     #     }

#     #     return batch_data
    
#     return DataLoader(
#         dataset=dataset,
#         batch_size=batch_size if max_len > 0 else 1,
#         num_workers=config[data_split]["num_workers"],
#         # shuffle=config[data_split]["shuffle"],
#         drop_last=config[data_split]["drop_last"],
#         sampler=batch_sampler
#     )