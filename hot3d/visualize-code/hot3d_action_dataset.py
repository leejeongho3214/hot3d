import torch
from torch.utils.data import Dataset


class Hot3DActionDataset(Dataset):
    """
    Minimal dataset stub that mirrors the object which was originally pickled.

    The serialized instances only need a container holding precomputed examples,
    so the implementation keeps the expected attributes while still behaving as
    a valid ``torch.utils.data.Dataset`` whenever iteration is required.
    """

    def __init__(self, examples=None, **kwargs):
        super().__init__()
        self.examples = examples or []
        # Preserve any extra metadata that might be injected through pickle.
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

