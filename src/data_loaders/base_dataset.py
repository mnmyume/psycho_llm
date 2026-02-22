"""
Abstract base class for all dataset loaders in the Psycho LLM project.

Subclasses must implement the `load()` method to populate `self.data`
with a Hugging Face Dataset object formatted for SFTTrainer.
"""

from abc import abstractmethod


class BaseDataset:
    """Abstract base for dataset loaders.

    Attributes:
        path: Path to the dataset annotation file.
        data: The loaded Hugging Face Dataset, populated by `load()`.
    """

    def __init__(self, path: str):
        self.path = path
        self.data = None

    @abstractmethod
    def load(self):
        """Load and preprocess the dataset. Must populate self.data."""
        raise NotImplementedError

    def __len__(self):
        if self.data is None:
            return 0
        return len(self.data)

    def __getitem__(self, idx):
        if self.data is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        return self.data[idx]
