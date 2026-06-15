#!/usr/bin/env python
"""
SAAD-Net: Novel contributions injected into Ultralytics RT-DETR-L

Novel #1 - AnomalyAwareAIFI : drop-in AIFI replacement with anomaly attention bias
Novel #2 - FDA               : forward hook on backbone C3/C4/C5 features
Novel #3 - ProtoPCA          : contrastive loss patched into model.forward

Usage:
    from saad_ultralytics import build_saad_model, make_saad_trainer
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List, Dict


# ============================================================================
# Novel #1: AnomalyAwareAIFI
# Drop-in replacement for ultralytics.nn.modules.transformer.AIFI
# Adds anomaly attention bias: attn_logits += beta * score_i * score_j
# ============================================================================

class AnomalyAwareAIFI(nn.Module):
    """
    Drop-in for Ultralytics AIFI(c1, c2, hn).
    Adds per-token anomaly scores as additive attention bias.
    beta initialized to 0 -> identity at start, learns to activate.
    """

    def __init__(self, c1: int, c2: int, hn: int = 8):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.hn = hn
        d = c2

        # Standard AIFI weights (named to match Ultralytics for weight copy)
        self.self_attn = nn.MultiheadAttention(d, hn, batch_first=True)
        self.linear1   = nn.Linear(d, d * 4)
        self.linear2   = nn.Linear(d * 4, d)
        self.norm1     = nn.LayerNorm(d)
        self.norm2     = nn.LayerNorm(d)
        self.dropout   = nn.Dropout(0.0)
        self.act       = nn.GELU()

        # Input projection (only when c1 != c2)
        self.proj_in = nn.Linear(c1, c2) if c1 != c2 else nn.Identity()

        # Novel: anomaly bias
        self.anomaly_scorer = nn.Sequential(
            nn.Linear(d, d // 4),
            nn.GELU(),
            nn.Linear(d // 4, 1)
        )
        self.beta = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _sinpos2d(h: int, w: int, d: int, device) -> Tensor:
        d2  = d // 2
        div = torch.exp(torch.arange(0, d2, 2, device=device).float()
                        * -(math.log(10000.0) / d2))
        y   = torch.arange(h, device=device).float()
        x   = torch.arange(w, device=device).float()
        py  = torch.zeros(h, d2, device=device)
        py[:, 0::2] = torch.sin(y.unsqueeze(1) * div)
        py[:, 1::2] = torch.cos(y.unsqueeze(1) * div)
        px  = torch.zeros(w, d2, device=device)
        px[:, 0::2] = torch.sin(x.unsqueeze(1) * div)
        px[:, 1::2] = torch.cos(x.unsqueeze(1) * div)
        pe  = torch.cat([
            py.unsqueeze(1).expand(-1, w, -1),
            px.unsqueeze(0).expand(h, -1, -1),
        ], dim=-1).view(h * w, d)
        return pe.unsqueeze(0)  # (1, N, d)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        tokens = x.flatten(2).permute(0, 2, 1)            # (B, N, C)
        tokens = self.proj_in(tokens)                      # (B, N, d)
        pos    = self._sinpos2d(H, W, self.c2, x.device).to(tokens.dtype)
        q = k  = tokens + pos

        # Anomaly attention bias -> (B*hn, N, N)
        scores        = torch.sigmoid(self.anomaly_scorer(tokens))
        bias          = self.beta * torch.bmm(scores, scores.transpose(1, 2))
        bias_expanded = bias.unsqueeze(1).expand(-1, self.hn, -1, -1) \
                            .reshape(B * self.hn, H * W, H * W)

        attn_out, _ = self.self_attn(q, k, tokens, attn_mask=bias_expanded)
        tokens = self.norm1(tokens + self.dropout(attn_out))
        tokens = self.norm2(tokens + self.linear2(self.dropout(
            self.act(self.linear1(tokens)))))

        return tokens.permute(0, 2, 1).view(B, self.c2, H, W)


# ============================================================================
# Novel #2: Frequency Domain Augmentation (FDA)
# Injected as a forward hook on backbone feature layers.
# Enhances high-frequency texture defects (silk_spot, oil_spot).
# ============================================================================

class FrequencyDomainAugmentation(nn.Module):
    def __init__(self, channels: int, radius_ratio: float = 0.5):
        super().__init__()
        self.radius_ratio = radius_ratio
        self.hf_enhancer  = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        self.alpha = nn.Parameter(torch.zeros(1))

    def _hf_mask(self, h: int, w: int, device) -> Tensor:
        cy, cx = h // 2, w // 2
        radius = min(cy, cx) * self.radius_ratio
        Y, X   = torch.meshgrid(torch.arange(h, device=device),
                                 torch.arange(w, device=device), indexing='ij')
        dist   = ((Y - cy).float() ** 2 + (X - cx).float() ** 2).sqrt()
        return (dist > radius).float().unsqueeze(0).unsqueeze(0)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        x_fft    = torch.fft.fftshift(torch.fft.fft2(x.float(), norm='ortho'), dim=(-2, -1))
        hf_mask  = self._hf_mask(H, W, x.device)
        x_hf_fft = x_fft * hf_mask
        x_hf     = torch.fft.ifft2(torch.fft.ifftshift(x_hf_fft, dim=(-2, -1)),
                                    norm='ortho').real.to(x.dtype)
        # Cast hf_weight to float32 before FFT arithmetic:
        # inside AMP autocast hf_weight may be fp16, but x_fft is complex64 ->
        # mixing dtypes causes errors. alpha is float32; ensure consistent.
        hf_weight = self.hf_enhancer(torch.cat([x, x_hf], dim=1)).float()
        x_enh_fft = x_fft * (1 - hf_mask) + x_hf_fft * (1 + self.alpha.float() * hf_weight)
        x_rec     = torch.fft.ifft2(torch.fft.ifftshift(x_enh_fft, dim=(-2, -1)),
                                     norm='ortho').real.to(x.dtype)
        return x + self.gate(x) * (x_rec - x)


# ============================================================================
# Novel #3: Prototype-based Contrastive Alignment (ProtoPCA)
# SupCon loss on decoder query features matched to GT classes.
# ============================================================================

class PrototypePCA(nn.Module):
    def __init__(self, d_model: int = 256, proj_dim: int = 128, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.projector   = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, proj_dim)
        )

    def forward(self, query_feats: Tensor, matched_labels: Tensor) -> Tensor:
        """
        query_feats:    (B, Q, d)  -- must carry gradients (no detach upstream)
        matched_labels: (B, Q)     -- -1 = background / unmatched
        """
        device = query_feats.device
        feats_list, labels_list = [], []
        for b in range(query_feats.shape[0]):
            mask = matched_labels[b] >= 0
            if mask.sum() < 2:
                continue
            feats_list.append(query_feats[b][mask])
            labels_list.append(matched_labels[b][mask])

        if not feats_list:
            return torch.tensor(0.0, device=device, requires_grad=True)

        feats  = torch.cat(feats_list)
        labels = torch.cat(labels_list)
        N = feats.shape[0]
        if N < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        z        = F.normalize(self.projector(feats), dim=-1)
        sim      = torch.matmul(z, z.T) / self.temperature
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        pos_mask.fill_diagonal_(0)
        sim_max, _ = sim.max(dim=1, keepdim=True)
        exp_sim    = torch.exp(sim - sim_max.detach()) * (1 - torch.eye(N, device=device))
        log_prob   = (sim - sim_max.detach()) - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)
        n_pos      = pos_mask.sum(1).clamp(min=1)
        return (-(pos_mask * log_prob).sum(1) / n_pos).mean()


# ============================================================================
# Layer replacement: Ultralytics AIFI -> AnomalyAwareAIFI
# FIX (BUG3): removed unused use_anomaly_bias parameter
# ============================================================================

def replace_aifi(model: nn.Module) -> int:
    """Replace all Ultralytics AIFI with AnomalyAwareAIFI. Returns replacement count."""
    try:
        from ultralytics.nn.modules.transformer import AIFI
    except ImportError:
        print("Warning: ultralytics AIFI not found - skipping replacement")
        return 0

    replaced = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if not isinstance(child, AIFI):
                continue
            # Detect c1/c2/hn from actual Ultralytics AIFI attributes.
            # AIFI uses fc1/fc2 (not linear1/linear2) and 'ma' for attention.
            if hasattr(child, 'c1'):
                c1 = child.c1
            elif hasattr(child, 'fc1'):
                c1 = child.fc1.in_features
            else:
                c1 = 256  # RT-DETR-L default

            if hasattr(child, 'c2'):
                c2 = child.c2
            elif hasattr(child, 'fc1'):
                c2 = child.fc1.in_features
            else:
                c2 = 256

            if hasattr(child, 'ma') and hasattr(child.ma, 'num_heads'):
                hn = child.ma.num_heads
            elif hasattr(child, 'self_attn') and hasattr(child.self_attn, 'num_heads'):
                hn = child.self_attn.num_heads
            else:
                hn = 8
            new_mod = AnomalyAwareAIFI(c1, c2, hn)
            try:
                _copy_aifi_weights(child, new_mod)
            except Exception as e:
                print(f"  Weight copy skipped {name}.{child_name}: {e}")
            setattr(module, child_name, new_mod)
            replaced += 1
            print(f"  Replaced AIFI -> AnomalyAwareAIFI  {name}.{child_name} "
                  f"(c1={c1}, c2={c2}, hn={hn})")
    return replaced


def _copy_aifi_weights(src: nn.Module, dst: AnomalyAwareAIFI):
    src_sd  = src.state_dict()
    dst_sd  = dst.state_dict()
    name_map = {
        'ma.in_proj_weight': 'self_attn.in_proj_weight',
        'ma.in_proj_bias':   'self_attn.in_proj_bias',
        'ma.out_proj.weight':'self_attn.out_proj.weight',
        'ma.out_proj.bias':  'self_attn.out_proj.bias',
        'fc1.weight': 'linear1.weight', 'fc1.bias': 'linear1.bias',
        'fc2.weight': 'linear2.weight', 'fc2.bias': 'linear2.bias',
        'norm1.weight': 'norm1.weight', 'norm1.bias': 'norm1.bias',
        'norm2.weight': 'norm2.weight', 'norm2.bias': 'norm2.bias',
    }
    copied = 0
    for sk, dk in name_map.items():
        if sk in src_sd and dk in dst_sd and src_sd[sk].shape == dst_sd[dk].shape:
            dst_sd[dk] = src_sd[sk]
            copied += 1
    dst.load_state_dict(dst_sd, strict=False)
    print(f"    Copied {copied}/{len(name_map)} weight tensors")


# ============================================================================
# FDA hook manager
# FIX (BUG4): removed unused 'layers = list(model.children())' dead variable
# ============================================================================

class FDAHookManager:
    """Injects FDA modules as forward hooks on backbone feature layers."""

    def __init__(self):
        self.hooks:         list = []
        self.fda_modules:   Dict[str, FrequencyDomainAugmentation] = {}
        self._extra_params: list = []

    def inject(self, inner: nn.Module, layer_indices: List[int], channels: List[int]):
        assert len(layer_indices) == len(channels)
        # FIX (BUG4): no dead 'layers' variable here
        seq = getattr(inner, 'model', None)
        if seq is None:
            print("  Warning: cannot find model.model for FDA hooks")
            return

        seq_list = list(seq.children())
        device   = next(inner.parameters()).device

        for idx, ch in zip(layer_indices, channels):
            fda = FrequencyDomainAugmentation(ch).to(device)
            key = f'fda_{idx}'
            self.fda_modules[key] = fda
            self._extra_params.extend(list(fda.parameters()))

            def make_hook(f):
                def hook(module, input, output):
                    if isinstance(output, (list, tuple)):
                        return (f(output[0]),) + output[1:]
                    return f(output)
                return hook

            layer = seq_list[idx]
            h = layer.register_forward_hook(make_hook(fda))
            self.hooks.append(h)
            print(f"  FDA hook -> layer[{idx}] (ch={ch})")

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def parameters(self):
        return iter(self._extra_params)

    def to(self, device):
        for fda in self.fda_modules.values():
            fda.to(device)
        return self

    def train(self):
        for fda in self.fda_modules.values():
            fda.train()
        return self

    def eval(self):
        for fda in self.fda_modules.values():
            fda.eval()
        return self

    def state_dict(self) -> dict:
        return {k: v.state_dict() for k, v in self.fda_modules.items()}

    def load_state_dict(self, sd: dict):
        for k, v in sd.items():
            if k in self.fda_modules:
                self.fda_modules[k].load_state_dict(v)


# ============================================================================
# Query feature capture
# FIX (BUG2):
#   - Now hooks RTDETRDecoder.decoder[-1] OUTPUT (refined query feats, not input)
#   - Removed .detach() so gradients flow through PCA loss
# ============================================================================

class QueryFeatureCapture:
    """
    Captures refined query features from the last decoder transformer layer.
    No detach -- PCA loss must backprop through query_feats.
    """

    def __init__(self):
        self.query_feats: Optional[Tensor] = None
        self._hook = None

    def register(self, inner: nn.Module) -> bool:
        # Strategy 1: RTDETRDecoder with ModuleList .decoder
        for mod in inner.modules():
            if 'RTDETRDecoder' in type(mod).__name__:
                dec = getattr(mod, 'decoder', None)
                if dec is not None and isinstance(dec, nn.ModuleList) and len(dec) > 0:
                    last = dec[-1]
                    self._hook = last.register_forward_hook(self._hook_fn)
                    print(f"  QueryFeatureCapture -> RTDETRDecoder.decoder[-1] ({type(last).__name__})")
                    return True

        # Strategy 2: DeformableTransformerDecoder - hook the decoder object itself
        for mod in inner.modules():
            t = type(mod).__name__
            if 'DeformableTransformerDecoder' in t or ('Decoder' in t and 'RTDETR' not in t):
                self._hook = mod.register_forward_hook(self._hook_fn)
                print(f"  QueryFeatureCapture -> {t} (DeformableTransformer fallback)")
                return True

        # Strategy 3: Any module named 'decoder' inside RTDETRDecoder
        for mod in inner.modules():
            if 'RTDETRDecoder' in type(mod).__name__:
                dec = getattr(mod, 'decoder', None)
                if dec is not None:
                    self._hook = dec.register_forward_hook(self._hook_fn)
                    print(f"  QueryFeatureCapture -> RTDETRDecoder.decoder ({type(dec).__name__})")
                    return True

        # Strategy 4: last child of model.model
        seq = getattr(inner, 'model', None)
        if seq is not None:
            children = list(seq.children())
            if children:
                target = children[-1]
                self._hook = target.register_forward_hook(self._hook_fn)
                print(f"  QueryFeatureCapture last-child fallback -> {type(target).__name__}")
                return True

        print("  Warning: QueryFeatureCapture could not register hook")
        return False

    def _hook_fn(self, module, input, output):
        # Capture refined query features (B, Q, d) -- no detach, grad must flow.
        # DeformableTransformerDecoder returns (query_feats, reference_points) tuple.
        if isinstance(output, (list, tuple)):
            for item in output:
                if isinstance(item, Tensor) and item.dim() == 3:
                    self.query_feats = item   # first 3-D tensor = query feats
                    return
        elif isinstance(output, Tensor) and output.dim() == 3:
            self.query_feats = output

    def remove(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None


# ============================================================================
# Greedy label assignment for ProtoPCA
# FIX (BUG5): works from batch dict directly, no preds format dependency
# ============================================================================

def build_matched_labels(batch: dict, query_feats: Tensor) -> Tensor:
    """
    Greedy assignment: top-activated queries get GT class labels.
    Uses query feature L2-norm as activation proxy.

    batch: must contain 'cls' (N,) and 'batch_idx' (N,)
    Returns matched (B, Q): -1 = background
    """
    B, Q, _ = query_feats.shape
    device   = query_feats.device
    matched  = torch.full((B, Q), -1, dtype=torch.long, device=device)

    gt_cls = batch.get('cls',       None)
    bi     = batch.get('batch_idx', None)
    if gt_cls is None or bi is None:
        return matched

    gt_cls = gt_cls.to(device).long().view(-1)
    bi     = bi.to(device).long().view(-1)

    # Detach norms -- only used for index selection, must not affect grad graph
    query_norms = query_feats.detach().norm(dim=-1)  # (B, Q)

    for b in range(B):
        mask   = (bi == b)
        labels = gt_cls[mask]
        n_gt   = len(labels)
        if n_gt == 0:
            continue
        k      = min(n_gt, Q)
        top_q  = query_norms[b].topk(k).indices
        for i in range(k):
            matched[b, top_q[i].item()] = labels[i]

    return matched


# ============================================================================
# build_saad_model
# ============================================================================

def build_saad_model(
    weights:     str = 'rtdetr-l.pt',
    config:      str = 'v3_default',
    num_classes: int = 11,
    device           = None,
) -> dict:
    """
    Load Ultralytics RT-DETR and inject SAAD-Net novel contributions.
    Returns: model, fda_manager, proto_pca, qf_capture, config, cfg
    """
    from ultralytics import RTDETR

    CONFIGS = {
        'v3_baseline':   dict(use_aifi=False, use_fda=False, use_pca=False),
        'v3_ms_laa':     dict(use_aifi=True,  use_fda=False, use_pca=False),
        'v3_ms_laa_fda': dict(use_aifi=True,  use_fda=True,  use_pca=False),
        'v3_default':    dict(use_aifi=True,  use_fda=True,  use_pca=True),
    }
    if config not in CONFIGS:
        raise ValueError(f"Unknown config '{config}'. Options: {list(CONFIGS)}")
    cfg = CONFIGS[config]

    print(f"\nBuilding SAAD-Net [{config}]")
    print(f"  aifi={cfg['use_aifi']}  fda={cfg['use_fda']}  pca={cfg['use_pca']}")

    model = RTDETR(weights)
    inner = model.model

    if cfg['use_aifi']:
        print("Injecting AnomalyAwareAIFI...")
        n = replace_aifi(inner)
        if n == 0:
            print("  Warning: No AIFI layers found - verify architecture")

    fda_manager = FDAHookManager()
    if cfg['use_fda']:
        print("Injecting FDA hooks...")
        _inject_fda_auto(inner, fda_manager)

    proto_pca  = None
    qf_capture = None
    if cfg['use_pca']:
        print("Setting up ProtoPCA...")
        d_model    = _detect_decoder_dim(inner)
        proto_pca  = PrototypePCA(d_model=d_model).to(device or 'cpu')
        qf_capture = QueryFeatureCapture()
        qf_capture.register(inner)
        print(f"  ProtoPCA d_model={d_model}")

    return dict(model=model, fda_manager=fda_manager, proto_pca=proto_pca,
                qf_capture=qf_capture, config=config, cfg=cfg)


def _inject_fda_auto(inner: nn.Module, fda_manager: FDAHookManager):
    seq = getattr(inner, 'model', None)
    if seq is None:
        print("  Warning: cannot detect backbone layers for FDA")
        return
    candidates = []
    for i, layer in enumerate(seq.children()):
        if hasattr(layer, 'cv2') or type(layer).__name__ in ('C2f', 'Conv', 'RepC3'):
            ch = _get_out_channels(layer)
            if ch and ch >= 256:
                candidates.append((i, ch))
    if not candidates:
        print("  Warning: no suitable layers found for FDA - skipping")
        return
    chosen   = candidates[-3:] if len(candidates) >= 3 else candidates
    indices  = [c[0] for c in chosen]
    channels = [c[1] for c in chosen]
    print(f"  FDA auto-detected: {list(zip(indices, channels))}")
    fda_manager.inject(inner, indices, channels)


def _get_out_channels(module: nn.Module) -> Optional[int]:
    for attr in ('cv2', 'conv', 'm'):
        sub = getattr(module, attr, None)
        if sub is not None:
            for a2 in ('out_channels', 'weight'):
                v = getattr(sub, a2, None)
                if v is not None:
                    return v if isinstance(v, int) else v.shape[0]
    return None


def _detect_decoder_dim(inner: nn.Module) -> int:
    for mod in reversed(list(inner.modules())):
        if 'Decoder' in type(mod).__name__:
            for sub in mod.modules():
                if isinstance(sub, nn.Linear):
                    return sub.in_features
    return 256


# ============================================================================
# Custom Trainer
# FIX (BUG1): SAADTrainer.criterion() override replaced with _patch_model_loss().
#   Ultralytics training loop: self.model(batch_dict) -> model.loss() -> criterion()
#   Instance attr 'criterion' set by init_criterion() shadows any method override.
#   Fix: wrap model.forward in _setup_train so PCA loss is added BEFORE backward.
# ============================================================================

def make_saad_trainer(saad_components: dict, w_pca: float = 0.1, pca_warmup: int = 10):
    """Returns SAADTrainer (RTDETRTrainer subclass) with PCA loss injected."""
    try:
        from ultralytics.models.rtdetr.train import RTDETRTrainer
        from ultralytics.utils import LOGGER
    except ImportError:
        from ultralytics.engine.trainer import BaseTrainer as RTDETRTrainer
        from ultralytics.utils import LOGGER

    proto_pca   = saad_components.get('proto_pca')
    qf_capture  = saad_components.get('qf_capture')
    fda_manager = saad_components.get('fda_manager')

    class SAADTrainer(RTDETRTrainer):

        def _setup_train(self, world_size):
            super()._setup_train(world_size)
            device = next(self.model.parameters()).device
            if fda_manager:
                fda_manager.to(device)
            if proto_pca:
                proto_pca.to(device)
            # FIX (BUG1): patch model.forward here, BEFORE any backward call
            if proto_pca is not None and qf_capture is not None:
                self._patch_model_loss()

        def _patch_model_loss(self):
            """
            Wrap model.forward so PCA loss is added to base loss BEFORE backward.
            Ultralytics training: self.model(batch_dict) returns (loss, loss_items).
            We intercept this return value inside the same autocast context.
            """
            inner     = self.model.module if hasattr(self.model, 'module') else self.model
            orig_fwd  = inner.forward
            _pca      = proto_pca
            _qfc      = qf_capture
            _w        = w_pca
            _wu       = pca_warmup
            _trainer  = self

            def patched_forward(x):
                result = orig_fwd(x)
                # Only intercept training batch-dict calls
                if not (isinstance(x, dict) and isinstance(result, tuple)
                        and len(result) == 2):
                    return result

                base_loss, base_items = result
                epoch = getattr(_trainer, 'epoch', 0)
                w_t   = _w * min(epoch / max(_wu, 1), 1.0)

                if w_t <= 0 or _qfc.query_feats is None:
                    return result
                try:
                    qf      = _qfc.query_feats          # grad-enabled (B, Q, d)
                    matched = build_matched_labels(x, qf)
                    pca_l   = _pca(qf, matched)
                    if _trainer.epoch % 10 == 0:
                        LOGGER.info(f"[SAAD] PCA loss={pca_l.item():.4f}  w={w_t:.4f}")
                    return base_loss + w_t * pca_l, base_items
                except Exception as e:
                    LOGGER.warning(f"[SAAD] PCA loss failed: {e}")
                    return result

            inner.forward = patched_forward
            LOGGER.info("[SAAD] model.forward patched for ProtoPCA loss")

        def build_optimizer(self, model, name='AdamW', lr=0.001,
                            momentum=0.9, decay=1e-4, iterations=1e5):
            optimizer = super().build_optimizer(model, name, lr, momentum, decay, iterations)
            extra = []
            if fda_manager:
                extra.extend(list(fda_manager.parameters()))
            if proto_pca:
                extra.extend(list(proto_pca.parameters()))
            if extra:
                optimizer.add_param_group({'params': extra, 'lr': lr})
                LOGGER.info(f"[SAAD] Added {len(extra)} FDA/PCA params to optimizer")
            return optimizer

        def save_model(self):
            super().save_model()
            import os as _os
            sp = self.last
            if not _os.path.exists(sp):
                return
            ckpt = torch.load(sp, map_location='cpu', weights_only=False)
            if fda_manager and fda_manager.fda_modules:
                ckpt['fda_state'] = fda_manager.state_dict()
            if proto_pca is not None:
                ckpt['pca_state'] = proto_pca.state_dict()
            ckpt['saad_config'] = saad_components.get('config', '')
            torch.save(ckpt, sp)

    return SAADTrainer