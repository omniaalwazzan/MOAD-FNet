# -*- coding: utf-8 -*-
"""
@author: omnia

"""

import os
import random
import pickle

from pathlib import Path  # (Used only for type hints or paths if needed later; safe to remove if undesired)

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login

import torch
from torch.utils.data import Dataset

# -----------------------------------------------------------------------------
# CUDA / cuDNN settings (preserved behavior)
# -----------------------------------------------------------------------------
torch.backends.cudnn.benchmark = False  # keep as in original top-level
# Important for avoiding: "RuntimeError: no valid convolution algorithms available in CuDNN"
torch.backends.cudnn.enabled = False

# Helpful for diagnosing CUDA errors
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

use_gpu = torch.cuda.is_available()
use_gpu = True  # preserved from original (forces GPU if available/configured)

# -----------------------------------------------------------------------------
# PIL image settings
# -----------------------------------------------------------------------------
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# -----------------------------------------------------------------------------
# Auth (do NOT hardcode secrets)
# -----------------------------------------------------------------------------
# Example: export HF_TOKEN="..."
# If you need to authenticate with HF Hub, either set the env var or leave this as-is if not required.
hf_token = os.getenv("HF_TOKEN", None)
if hf_token:
    login(token=hf_token, add_to_git_credential=True)
# You can also leave login() out if your environment already has credentials.

# Silence symlink warnings
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # These two lines are preserved from original function
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# -----------------------------------------------------------------------------
# Model & transforms
# -----------------------------------------------------------------------------
model = timm.create_model(
    "hf-hub:MahmoodLab/uni",
    pretrained=True,
    init_values=1e-5,
    dynamic_img_size=True
)
transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

if use_gpu:
    model = model.cuda()

# -----------------------------------------------------------------------------
# DataFrame load & basic stats
# -----------------------------------------------------------------------------
PATH_patches = "/data/DERI-MMH/DNA_meth/CSV/imgPath_MvalSentrix_20GT.csv"
df = pd.read_csv(PATH_patches, header=0)

# Column normalization (preserved)
df = df.rename(columns={'Folder': 'Patient ID'})
df.rename(columns={'lable': 'label'}, inplace=True)  # only for newly filtered data
df = df.dropna(subset=['label'])

n_classes = df['label'].nunique()
uni_cases = df['ID'].nunique()

print('Number of Cases: ', uni_cases)
print('Number of Classes: ', n_classes)

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class HistoDataset(Dataset):
    """Simple dataset that loads an image, applies transforms, and returns (tensor, label, filename)."""

    def __init__(self, dataframe: pd.DataFrame, transform):
        self.transform = transform
        self.filepaths = dataframe['ImagePath'].tolist()
        self.labels = dataframe['label'].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int):
        try:
            image_path = self.filepaths[idx]
            image = Image.open(image_path)

            # Remove alpha channel if present
            if image.mode == 'RGBA':
                image = image.convert('RGB')

            if image is None:
                raise ValueError(f"Image at index {idx} could not be loaded")

            filename = os.path.basename(image_path)
            image_tensor = self.transform(image)
            image_label = self.labels[idx]

            return image_tensor, image_label, filename

        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            return None

# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------
class Loaders:
    def slides_dataloader(
        self,
        dataframe: pd.DataFrame,
        ids,
        transform,
        slide_batch: int,
        num_workers: int,
        shuffle: bool,
        collate,
        patient_id: str = "Patient ID",
    ):
        """Create a dict of per-patient DataLoaders."""
        patient_subsets = {}
        for pid in ids:
            key = f"{pid}"
            subset_df = dataframe[dataframe[patient_id] == pid]
            patient_subset = HistoDataset(subset_df, transform)
            patient_subsets[key] = torch.utils.data.DataLoader(
                patient_subset,
                batch_size=slide_batch,
                shuffle=shuffle,
                num_workers=num_workers,
                drop_last=False,
                collate_fn=collate,
            )
        return patient_subsets

# -----------------------------------------------------------------------------
# Embedding creation
# -----------------------------------------------------------------------------
def create_embeddings(embedding_net, loader_dict, include_self: bool = False, use_gpu: bool = True):
    """
    Iterate patient-wise loaders, run the model, and collect embeddings.

    Returns:
        dict: patient_id -> [np.ndarray (N x D), last_label_tensor, list_of_filenames]
    """
    embedding_dict = {}

    embedding_net.eval()
    with torch.inference_mode():
        for patient_id, slide_loader in loader_dict.items():
            patient_embedding = []
            filenames = []
            last_label = None

            for batch in slide_loader:
                inputs, label, filename = batch
                # Expect shape: [B, 3, 224, 224], but original runs with B=1
                filename = filename[0] if isinstance(filename, (list, tuple)) else filename

                if use_gpu:
                    inputs, label = inputs.cuda(), label.cuda()

                embedding = embedding_net(inputs)  # shape: [B, ...]
                embedding = embedding.to('cpu').view(embedding.size(0), -1)

                patient_embedding.append(embedding)
                filenames.append(filename)
                last_label = label  # keep the last seen label (matches original behavior)

            if len(patient_embedding) == 0:
                print(f"Warning: no embeddings for patient {patient_id}")
                continue

            patient_embedding = torch.cat(patient_embedding, dim=0)  # [N, D]
            print(f'patient_embedding with id {patient_id} has shape {patient_embedding.shape}')

            embedding_dict[patient_id] = [patient_embedding.numpy(), last_label, filenames]

    return embedding_dict

# -----------------------------------------------------------------------------
# Params & runners
# -----------------------------------------------------------------------------
seed = 16
seed_everything(seed)

slide_batch = 1  # keep as in original; note: comment about IterableDataset left unchanged
num_workers = 0
batch_size = 1  # unused in this script; preserved as original variable

patient_id_col = 'Patient ID'
dataset_name = 'Brain_tumor'

def collate_fn_none(batch):
    """Filter out Nones from Dataset __getitem__ exceptions."""
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

train_ids = list(df[patient_id_col].unique())

train_slides = Loaders().slides_dataloader(
    df,
    train_ids,
    transform,
    slide_batch=slide_batch,
    num_workers=num_workers,
    shuffle=False,
    collate=collate_fn_none,
    patient_id=patient_id_col,
)

print('start creating Embedding...')

save_path = "/data/DERI-MMH/DNA_meth/pkl_baseline_model"
os.makedirs(save_path, exist_ok=True)

slides_dict = {
    'Brain_UNI_train_embedding_dict_': train_slides
}

for file_prefix, slides in slides_dict.items():
    embedding_dict = create_embeddings(model, slides, include_self=False, use_gpu=use_gpu)
    print(f"Started saving {file_prefix} to file")
    file_path = os.path.join(save_path, f"{file_prefix}{dataset_name}.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(embedding_dict, f)
    print(f"Done writing embedding dict into pickle file at {file_path}")
