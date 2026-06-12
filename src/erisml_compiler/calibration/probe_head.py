"""Probe head and frozen-backbone wrapper.

Architecture (mirrors sqnd-probe v10.16.9 in spirit, simplified):

    text --[LaBSE backbone (frozen)]--> z in R^D
                                       |
                                       v
                          [VIB layer (mu, log_var)]   (optional)
                                       |
                                       v
                              [ProbeHead]  -> task logits

The VIB layer reparameterises z as N(mu, exp(log_var)) and samples z' for
the probe head input during training. At inference, z' = mu.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VIBLayer(nn.Module):
    """Variational information bottleneck.

    Maps z -> (mu, log_var), then samples z' = mu + sigma * eps in training.
    """

    def __init__(self, in_dim: int, bottleneck_dim: int):
        super().__init__()
        self.mu = nn.Linear(in_dim, bottleneck_dim)
        self.log_var = nn.Linear(in_dim, bottleneck_dim)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.mu(z)
        log_var = self.log_var(z)
        if self.training:
            sigma = (0.5 * log_var).exp()
            eps = torch.randn_like(sigma)
            sample = mu + sigma * eps
        else:
            sample = mu
        return sample, mu, log_var


class ProbeHead(nn.Module):
    """A small MLP classifier on top of the backbone (or VIB) representation."""

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        n_layers: int = 2,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(max(0, n_layers - 1)):
            layers += [nn.Linear(d, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            d = hidden_dim
        layers.append(nn.Linear(d, num_classes))
        self.net = nn.Sequential(*layers)
        self.num_classes = num_classes

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ProbeBackbone(nn.Module):
    """Frozen LaBSE + optional VIB + ProbeHead.

    LaBSE is loaded via `sentence-transformers`. We freeze its parameters
    (no gradient flows back) and use it as a deterministic feature
    extractor. The probe head and VIB are trainable.
    """

    def __init__(
        self,
        num_classes: int,
        labse_model: str = "sentence-transformers/LaBSE",
        backbone_dim: int = 768,
        use_vib: bool = True,
        vib_bottleneck: int = 256,
        head_hidden: int = 256,
        head_layers: int = 2,
        device: str = "cpu",
    ):
        super().__init__()
        from sentence_transformers import SentenceTransformer

        self._st = SentenceTransformer(labse_model, device=device)
        for p in self._st.parameters():
            p.requires_grad_(False)
        self.backbone_dim = backbone_dim
        self.use_vib = use_vib

        if use_vib:
            self.vib = VIBLayer(backbone_dim, vib_bottleneck)
            head_in = vib_bottleneck
        else:
            self.vib = None
            head_in = backbone_dim

        self.head = ProbeHead(
            in_dim=head_in,
            num_classes=num_classes,
            hidden_dim=head_hidden,
            n_layers=head_layers,
        )
        self.device_str = device

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """Encode a batch of texts to LaBSE embeddings. Returns (B, 768)
        tensor on the configured device. No gradient through the backbone."""
        with torch.no_grad():
            embs = self._st.encode(
                texts,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                device=self.device_str,
            )
        return embs

    def forward(
        self, embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        """Forward pass given pre-computed embeddings.

        Returns (logits, mu, log_var, post_vib_z). When use_vib=False,
        mu and log_var are None and post_vib_z equals the input.
        """
        if self.use_vib:
            z, mu, log_var = self.vib(embeddings)
        else:
            z, mu, log_var = embeddings, None, None
        logits = self.head(z)
        return logits, mu, log_var, z

    def state_dict_for_checkpoint(self) -> dict:
        """Save only the trainable parts (head + VIB), not the frozen backbone."""
        sd = {"head": self.head.state_dict()}
        if self.use_vib:
            sd["vib"] = self.vib.state_dict()
        sd["config"] = {
            "num_classes": self.head.num_classes,
            "use_vib": self.use_vib,
            "backbone_dim": self.backbone_dim,
        }
        return sd

    def load_state_dict_from_checkpoint(self, sd: dict) -> None:
        self.head.load_state_dict(sd["head"])
        if self.use_vib and "vib" in sd:
            self.vib.load_state_dict(sd["vib"])
