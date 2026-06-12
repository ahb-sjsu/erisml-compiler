"""Training dataset for probe calibration.

A training example is:
    - text: a passage of natural language
    - label: the target probe class (e.g., role class, commitment status,
      ethical-fact kind)
    - language: language code for the adversarial head
    - period: historical-period bucket for the adversarial head

The compiler's calibration loop expects this format. The synthetic
dataset class below supports unit testing; the real dataset class will
take a directory of corrected-IR JSONs (from the human-correction loop)
plus their source texts and yield (text, label, language, period)
tuples.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ProbeBatch:
    """One batch produced by the dataset."""

    texts: list[str]
    labels: torch.Tensor          # (B,) long
    language_idx: torch.Tensor    # (B,) long
    period_idx: torch.Tensor      # (B,) long


class ProbeTrainingDataset:
    """A simple in-memory dataset.

    Pass lists of (text, label, language, period); the dataset yields
    `ProbeBatch` objects.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        languages: list[int],
        periods: list[int],
        batch_size: int = 32,
    ):
        assert len(texts) == len(labels) == len(languages) == len(periods)
        self.texts = texts
        self.labels = labels
        self.languages = languages
        self.periods = periods
        self.batch_size = batch_size

    def __len__(self) -> int:
        return len(self.texts)

    def __iter__(self):
        for start in range(0, len(self), self.batch_size):
            end = min(start + self.batch_size, len(self))
            yield ProbeBatch(
                texts=self.texts[start:end],
                labels=torch.tensor(self.labels[start:end], dtype=torch.long),
                language_idx=torch.tensor(self.languages[start:end], dtype=torch.long),
                period_idx=torch.tensor(self.periods[start:end], dtype=torch.long),
            )


def synthetic_dataset(
    n_samples: int = 64,
    n_classes: int = 3,
    n_languages: int = 4,
    n_periods: int = 3,
    seed: int = 0,
) -> ProbeTrainingDataset:
    """Deterministic synthetic dataset for unit tests.

    Constructs short pseudo-text strings encoded with the class id so the
    probe head has a learnable signal. Language and period assignments are
    random so the adversarial losses have something to discourage.
    """
    import random
    rng = random.Random(seed)
    role_words = ["agent", "patient", "victim", "authority", "protector"]
    texts: list[str] = []
    labels: list[int] = []
    languages: list[int] = []
    periods: list[int] = []
    for i in range(n_samples):
        cls = i % n_classes
        token = role_words[cls % len(role_words)]
        # Encode the label into the text so the probe has signal.
        texts.append(f"This is a {token}-context example, sample {i}.")
        labels.append(cls)
        languages.append(rng.randrange(n_languages))
        periods.append(rng.randrange(n_periods))
    return ProbeTrainingDataset(texts, labels, languages, periods, batch_size=16)
