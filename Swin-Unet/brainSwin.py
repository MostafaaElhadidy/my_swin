import torch
import h5py
import numpy as np
import cv2
from torch.utils.data import Dataset
from tqdm import tqdm




class brainSS(Dataset):
    def __init__(self, file_path, transform=None, augment=False):
        self.file_path = file_path
        self.transform = transform
        self.compact = True
        self.augment = augment
        self.file_path = self.filter_black_images(self.file_path)
    
    def __len__(self):
        return len(self.file_path)
    
    def __getitem__(self, idx):
        def convert_to_three_channels(image):
            combined_channel = np.maximum(image[0, :, :], image[1, :, :])
            new_image = np.stack([image[0, :, :], image[1, :, :], combined_channel], axis=0)
            return torch.tensor(new_image, dtype=torch.float32)
        
        def convert_to_one_channel(mask):
            combined_channels = np.maximum.reduce(mask, axis=0).unsqueeze(0)
            return torch.tensor(combined_channels, dtype=torch.long)
        
        def elastic_transform(image, mask, alpha_affine):
            random_state = np.random.RandomState(None)
            shape = image.shape
            shape_size = shape[:2]

            center_square = np.float32(shape_size) // 2
            square_size = 42
            pts1 = np.float32([center_square + square_size, 
                               [center_square[0] + square_size, center_square[1] - square_size], 
                               center_square - square_size])
            pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
            M = cv2.getAffineTransform(pts1, pts2)
            image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(mask, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
            return image, mask

        def hflip_transform(image, mask):
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
            return image, mask

        def vflip_transform(image, mask):
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
            return image, mask

        def flip_transform(image, mask):
            image = cv2.flip(image, -1)
            mask = cv2.flip(mask, -1)
            return image, mask
        
        directory = self.file_path[idx]
        with h5py.File(directory, 'r') as f:
            image = f['image'][:]
            mask = f['mask'][:]
            
            image = image.transpose((2, 0, 1))
            mask = mask.transpose((2, 0, 1))

            for i in range(image.shape[0]):
                min_val = np.min(image[i])
                image[i] = image[i] - min_val
                max_val = np.max(image[i]) + 1e-4
                image[i] = image[i] / max_val

            image = torch.tensor(image, dtype=torch.float32)
            mask = torch.tensor(mask, dtype=torch.float32)
        
        if self.augment:
            alpha_affine = image.shape[1] * 0.04
            aug_functions = [
                lambda img, msk: elastic_transform(img, msk, alpha_affine),
                hflip_transform,
                vflip_transform,
                flip_transform
            ]
            aug_function = np.random.choice(aug_functions)
            image, mask = aug_function(image.numpy(), mask.numpy())
            image = torch.tensor(image, dtype=torch.float32)
            mask = torch.tensor(mask, dtype=torch.float32)
        
        sample = {'image': image, 'mask': mask}
        
        if self.compact:
            sample['image'] = convert_to_three_channels(sample['image'])
            sample['mask'] = convert_to_one_channel(sample['mask'])
            if self.transform:
                sample['image'] = self.transform['transform_3'](sample['image'])
                sample['mask'] = self.transform['transform_1'](sample['mask'])
        else:
            if self.transform:
                sample['image'] = self.transform['transform_4'](sample['image'])
                sample['mask'] = self.transform['transform_3'](sample['mask'])

        return sample
    
    def filter_black_images(self, file_paths):
        valid_file_paths = []
        for path in tqdm(file_paths, desc="Filtering black images"):
            with h5py.File(path, 'r') as f:
                image = f['image'][:]
                if np.max(image) > 0:
                    valid_file_paths.append(path)
        return valid_file_paths
