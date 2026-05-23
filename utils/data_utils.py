import h5py
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

def create_undersampling_mask(shape, acceleration=2, center_lines=30):
    height, width = shape
    mask = np.zeros((height, width), dtype=np.float32)
    num_lines = width // acceleration
    center = width // 2
    center_start = center - center_lines // 2
    center_end = center + center_lines // 2
    mask[:, center_start:center_end] = 1
    remaining_lines = num_lines - center_lines
    if remaining_lines > 0:
        outer_lines = np.setdiff1d(np.arange(width), np.arange(center_start, center_end))
        sampled_lines = np.random.choice(outer_lines, remaining_lines, replace=False)
        mask[:, sampled_lines] = 1
    return mask


def kspace_to_image(kspace, apply_rss=True):
    kspace_shifted = np.fft.ifftshift(kspace, axes=(-2, -1))
    img = np.fft.ifft2(kspace_shifted)
    img_shifted = np.fft.fftshift(img)
    if apply_rss and len(kspace.shape) == 3:
        return np.sqrt(np.sum(np.abs(img_shifted) ** 2, axis=0))
    return np.abs(img_shifted)


class M4RawDataset(Dataset):
    def __init__(self, h5_dir, acceleration=16):
        self.h5_files = []
        all_files = [os.path.join(h5_dir, f) for f in os.listdir(h5_dir) if f.endswith('.h5')]
        for file_path in all_files:
            with h5py.File(file_path, 'r') as f:
                num_slices = f['kspace'].shape[0]
                for slice_idx in range(num_slices):
                    self.h5_files.append((file_path, slice_idx))
        self.acceleration = acceleration
        self.total_items = len(self.h5_files)

    def __len__(self):
        return self.total_items

    def __getitem__(self, idx):
        file_path, slice_idx = self.h5_files[idx]
        with h5py.File(file_path, 'r') as f:
            kspace = np.array(f['kspace'][slice_idx])
        mask = create_undersampling_mask(kspace.shape[1:], self.acceleration, center_lines=30)
        kspace_undersampled = kspace * mask[np.newaxis, :, :]
        zero_filled = kspace_to_image(kspace_undersampled, apply_rss=True)
        target = kspace_to_image(kspace, apply_rss=True)
        return zero_filled, target, f"{os.path.basename(file_path)}_slice_{slice_idx}"
