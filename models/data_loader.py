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

    def add_frequency_weighted_noise(self, cqt_spec, shift_amount=10, boundary_content=None):
        """
        Добавляет частотно-зависимый шум к CQT-спектрограмме.
        
        Параметры:
        - cqt: np.array, входная CQT-спектрограмма.
        - shift_amount: int, количество частотных шагов для добавления шума.
        - boundary_content: np.array, значение для учета границ.
        """
        time_steps, freq_bins = cqt_spec.shape
        noise = np.zeros((time_steps, shift_amount))

        # Рассчитаем локальные статистики для амплитуды
        local_maxes = np.max(cqt_spec, axis=1, keepdims=True)
        local_means = np.mean(cqt_spec, axis=1, keepdims=True)
        local_stds = np.std(cqt_spec, axis=1, keepdims=True)

        # Итерируем по shift_amount, чтобы добавить шум
        for i in range(shift_amount):
            # Вычисляем вес для частотного шага, чтобы сгладить падение
            freq_weight = np.cos(np.pi * i / (2 * shift_amount))  # Плавный спад

            # Ограничения для амплитуды
            max_allowed = local_maxes * freq_weight
            min_allowed = local_means - local_stds

            # Генерируем начальный шум
            base_noise = np.random.normal(
                loc=local_means,
                scale=local_stds * (1 - freq_weight * 0.7),  # Снижение вариации рядом с границей
                size=(time_steps, 1)
            )

            # Ограничиваем шум, чтобы он не превышал локальные амплитуды
            base_noise = np.clip(base_noise, min_allowed, max_allowed)

            # Смешиваем шум с boundary_content на основе частотной позиции
            if boundary_content is None:
                boundary_content = cqt[:, :shift_amount]  # Используем первые частоты, если нет конкретного значения

            noise[:, i] = freq_weight * boundary_content[:, i] + (1 - freq_weight) * base_noise[:, 0]

        # Добавляем шум к оригинальной спектрограмме
        noisy_cqt = cqt_spec.copy()
        noisy_cqt[:, :shift_amount] += noise  # Добавляем шум к shift_amount частотных диапазонов

        return noisy_cqt

    def _time_mask(self, cqt_spec):
        # num_time_masks = self.config["aug_params"]["num_time_masks"]
        min_region, max_region = self.config["aug_params"]["time_mask_regionsize"]
        region = random.uniform(min_region, max_region)
        region_val = random.random() * self.config["aug_params"]["region_val"]
        w, h = np.shape(cqt_spec)
        region_w = int(w * region)

        start_pos = random.randint(0, w - region_w)
        cqt_spec[start_pos:start_pos + region_w, :] = region_val
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

    def _low_shift(self, feat, noise_scale=1.1):
        """
        Alternative to original _roll() method, better for strongly
        melodic music that appears at the bottom of the CQT frequency range.

        Don't shift lower in case meaningful melodic content is closer than
        shift_num to the bottom of the sample.

        Shift the spectrogram between 1 and shift_num higher in frequency bins.
        Fill missing data at the bottom with smoothed noise that simulates the
        local noise in each sample, and discard top-range frequency content
        that spills outside the sample's frequency range.

        Includes optional debugging / research code to visualize the resulting
        spectrograms. Uncomment those lines if you like.

        Args:
            feat (np.array): The input spectrogram of shape (time_steps, frequency_bins).
            shift_num (int): The maximum number of frequency bins to shift.
            noise_scale (float): Multiplier for amplitude of spectral texture in empty region
                Note that amplitude is negative, so quieter = higher noise_scale

        Returns:
            np.array: The pitch-transposed spectrogram.
        """

        min_shift_num, max_shift_num = self.config["aug_params"]["low_shift_values"]
        shift_num = random.randint(min_shift_num, max_shift_num)

        time_steps, freq_bins = np.shape(feat)
        shift_amount = random.randint(1, shift_num)

        if shift_amount >= freq_bins:
            return feat

        # Uncomment this line if you also uncomment the visualization code at the
        # end of this script
        original_feat = feat.copy()

        # Debugging: Print statistics before shift
        #        print(f"Before shift - Min: {np.min(feat)}, Max: {np.max(feat)}, Mean: {np.mean(feat)}")
        #        print(f"shift {self.current_perf} by {shift_amount} up to {fill_max} above {min_amplitude}")

        # 1. Shift the spectrogram upwards in place
        feat[:, shift_amount:] = feat[:, : freq_bins - shift_amount]

        # 2. Analyze transition boundary and local statistics
        boundary_content = feat[:, shift_amount]
        reference_height = min(shift_amount * 2, freq_bins - shift_amount)
        reference_region = feat[
            :, shift_amount : shift_amount + reference_height
        ]

        # Calculate per-time-step statistics with context
        window_size = 3
        padded_region = np.pad(
            reference_region,
            ((window_size // 2, window_size // 2), (0, 0)),
            mode="edge",
        )
        local_means = np.zeros(time_steps)
        local_stds = np.zeros(time_steps)
        local_maxes = np.zeros(time_steps)

        for i in range(time_steps):
            window = padded_region[i : i + window_size]
            local_means[i] = np.mean(window)
            local_stds[i] = np.std(window)
            local_maxes[i] = np.max(boundary_content[i : i + 1])

        # 3. Generate noise base with gradual transition
        noise = np.zeros((time_steps, shift_amount))

        for i in range(shift_amount):
            # Calculate frequency-dependent weights
            freq_weight = np.cos(
                np.pi * i / (2 * shift_amount)
            )  # Smoother falloff

            # Calculate amplitude bounds relative to boundary
            max_allowed = local_maxes * freq_weight
            min_allowed = local_means - local_stds

            # Generate initial noise
            base_noise = np.random.normal(
                loc=local_means,
                scale=local_stds
                * (1 - freq_weight * 0.7),  # Reduce variation near boundary
                size=time_steps,
            )

            # Clip noise to ensure it doesn't exceed local amplitude bounds
            base_noise = np.clip(base_noise, min_allowed, max_allowed)

            # Blend with boundary content based on frequency position
            noise[:, i] = (
                freq_weight * boundary_content + (1 - freq_weight) * base_noise
            )

        # 4. Apply temporal smoothing while preserving local variation
        smoothed_noise = np.zeros_like(noise)
        for i in range(time_steps):
            start = max(0, i - window_size // 2)
            end = min(time_steps, i + window_size // 2 + 1)
            for j in range(shift_amount):
                # Weight the smoothing based on frequency position
                smooth_weight = 0.8 + 0.2 * (
                    j / shift_amount
                )  # Less smoothing near boundary
                current = noise[i, j]
                neighborhood = noise[start:end, j]
                smoothed = np.mean(neighborhood)
                smoothed_noise[i, j] = (
                    smooth_weight * current + (1 - smooth_weight) * smoothed
                )

        # 5. Ensure explicit amplitude matching at boundary
        transition_width = min(3, shift_amount)
        for i in range(transition_width):
            # Use boundary amplitudes directly for top rows
            amplitude_factor = np.minimum(
                1.0,
                np.abs(smoothed_noise[:, i])
                / (np.abs(boundary_content) + 1e-8),
            )
            smoothed_noise[:, i] = np.minimum(
                smoothed_noise[:, i], boundary_content * amplitude_factor
            )

        # 6. Scale and apply the processed noise
        feat[:, :shift_amount] = smoothed_noise * noise_scale

        # Debugging: Print statistics after shift
        #            print(f"After shift - Min: {np.min(feat)}, Max: {np.max(feat)}, Mean: {np.mean(feat)}")

        # Uncomment these lines if you want to generate visualizations
        # CAUTION: If you are using an MPS device, you must also temporarily
        # disable the line "mp.set_start_method("fork")" in whichever
        # tools.train_... script you are using. Otherwise matplotlib will
        # cause a crash.
        # if not np.array_equal(original_feat, feat):
        #     print(f"augmented {self.current_perf} by {shift_amount}")
        #     if self.device == "mps":
        #         self._save_spectrograms(
        #             original_feat, feat, "shift_melody_low"
        #         )
        #     else:
        #         self._save_spectrograms(
        #             original_feat, feat, "shift_melody_low"
        #         )

        return feat

    
    import numpy as np

    def _gaussian_noise(self, cqt):
        """
        Добавляет гауссовский шум ко всей CQT-спектрограмме.
        
        Параметры:
        - cqt: np.array, входная CQT-спектрограмма.
        - noise_level: float, уровень шума (среднеквадратическое отклонение).
        
        Возвращает:
        - noisy_cqt: np.array, спектрограмма с добавленным гауссовским шумом.
        """
        min_noise_level, max_noise_level = self.config["aug_params"]["noise_levels"]
        noise_level = random.uniform(min_noise_level, max_noise_level)
        gaussian_noise = np.random.normal(0, noise_level, cqt.shape)
        
        noisy_cqt = cqt + gaussian_noise
        
        return noisy_cqt
    
    def _duplicate(self, cqt_spec):
        w, h = np.shape(cqt_spec)
        feat_aug = cqt_spec
        for i in range(1, w):
            if random.random() < 0.06:
                feat_aug[i, :] = feat_aug[i - 1, :]
        for i in range(1, h):
            if random.random() < 0.06:
                feat_aug[:, i] = feat_aug[:, i - 1]
        return feat_aug

    def _apply_augmentations(self, cqt_spec: np.ndarray) -> np.ndarray:
        """
        Args:
            cqt_spectrogram: np.ndarray

        Transformations:
            time_stretching - slight compression or stretching along the time axis

        Returns:
            cqt_spectrogram_augmented: np.ndarray

        """

        # # seems like works bad ... ?
        # if "roll_pitch" in self.config["aug_params"] and self.config["aug_params"]["roll_pitch"]:
        #     p = random.random()
        #     if p <= self.config["aug_params"]["roll_pitch_prob"]:
        #         # print("roll")

        if "duplicate" in self.config["aug_params"] and self.config["aug_params"]["duplicate"]:
            p = random.random()
            if p <= self.config["aug_params"]["duplicate_prob"]:
                print("duplicate")
                cqt_spec = self._duplicate(cqt_spec)

        # works good
        if "time_roll" in self.config["aug_params"] and self.config["aug_params"]["time_roll"]:
            p = random.random()
            if p <= self.config["aug_params"]["time_roll_prob"]:
                print("timeroll")
                cqt_spec = self._time_roll(cqt_spec)

        # works good 
        # low_volume_coef: 0.5
        # high_volume_coef: 1.0
        if "volume" in self.config["aug_params"] and self.config["aug_params"]["volume"]:
            # print("volume")
            p = random.random()
            if p <= self.config["aug_params"]["volume_prob"]:
                print("volume")
                cqt_spec = self._change_volume(cqt_spec)

        # works good 
        # equalize_low_factor: 0.70
        # equalize_high_factor: 1.15
        if "equalize" in self.config["aug_params"] and self.config["aug_params"]["equalize"]:
            # print("equalize")
            p = random.random()
            if p <= self.config["aug_params"]["equalize_prob"]:
                print("equalize")
                cqt_spec = self._apply_equalize(cqt_spec)

        if "gaussian_noise" in self.config["aug_params"] and self.config["aug_params"]["gaussian_noise"]:
            p = random.random()
            if p <= self.config["aug_params"]["gaussian_noise_prob"]:
                print("gaussian_noise")
                cqt_spec = self._gaussian_noise(cqt_spec)

        # works slow and seems like bad
        # if "low_shift" in self.config["aug_params"] and self.config["aug_params"]["low_shift"]:
        #     p = random.random()
        #     if p <= self.config["aug_params"]["low_shift_prob"]:
        #         # print("low_shift")
        #         cqt_spec = self._low_shift(cqt_spec)

        # works good
        if "mask_silence" in self.config["aug_params"] and self.config["aug_params"]["mask_silence"]:
            p = random.random()
            if p <= self.config["aug_params"]["mask_silence_prob"]:
                print("mask_silence")
                cqt_spec = self._mask_silence(cqt_spec)

        # works good
        if "time_stretch" in self.config["aug_params"] and self.config["aug_params"]["time_stretch"]:
            p = random.random()
            if p <= self.config["aug_params"]["time_stretch_prob"]:
                print("time_stretch")
                cqt_spec = self._time_stretching(cqt_spec)

        # seems like works bad
        # if "time_mask" in self.config["aug_params"] and self.config["aug_params"]["time_mask"]:
        #     p = random.random()
        #     if p <= self.config["aug_params"]["time_mask_prob"]:
        #         # print("time_mask")
        # cqt_spec = self._time_mask(cqt_spec)

        # seems like works bad
        # # сомнительно. учился долго, сошелся к тому же, что модель без аугментаций..
        # if "random_time_crop" in self.config["aug_params"] and self.config["aug_params"]["random_time_crop"]:
        #     p = random.random()
        #     if p <= self.config["aug_params"]["random_timecrop_prob"]:
        #         # print("random_time_crop")
        #         cqt_spec = self._random_time_crop(cqt_spec)
        
        # # работает плохо!
        # if "random_erase" in self.config["aug_params"] and self.config["aug_params"]["random_erase"]:
        #     p = random.random()
        #     if p <= self.config["aug_params"]["random_erase_prob"]:
        #         # print("random_erase")
        #         cqt_spec = self._random_erase(cqt_spec)

        cqt_spec = self._apply_padding(cqt_spec)
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