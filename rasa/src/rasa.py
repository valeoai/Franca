# Code inspired and adapted from:
# https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/self_supervised/swav/swav_module.py
import math
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch import nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.optimizer import Optimizer

from rasa.experiments.utils import PredsmIoUKmeans, cosine_scheduler
from rasa.src.rasa_head import RASAHead


class RASA(pl.LightningModule):

    def __init__(
        self,
        gpus: int,
        num_samples: int,
        batch_size: int,
        max_epochs: int,
        lr_heads: float,
        final_lr: float,
        weight_decay_end: float,
        weight_decay: float,
        val_iters: int,
        val_iters_u_segm: int,
        num_classes: int,
        hub_repo_or_dir: str,
        model_name: str,
        num_clusters_kmeans: List[int],
        weights: Optional[str] = None,
        val_downsample_masks: bool = True,
        exclude_norm_bias: bool = True,
        optimizer: str = "adam",
        num_nodes: int = 1,
        patch_size: int = 16,
        pos_out_act_layer: str = "sigmoid",
        n_pos_layers: int = 0,
        grad_norm_clipping=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr_heads = lr_heads
        self.patch_size = patch_size
        self.gpus = gpus
        self.num_nodes = num_nodes
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.optim = optimizer
        self.exclude_norm_bias = exclude_norm_bias
        self.weight_decay = weight_decay
        self.final_lr = final_lr
        self.max_epochs = max_epochs
        self.grad_norm_clipping = grad_norm_clipping
        self.num_classes = num_classes
        self.val_downsample_masks = val_downsample_masks
        self.val_iters = val_iters
        self.val_iters_u_segm = val_iters_u_segm
        self.use_u_segm_eval = val_iters_u_segm is None or val_iters_u_segm > 0
        self.pos_out_act_layer = pos_out_act_layer
        self.n_pos_layers = n_pos_layers
        # Parameters for loading the backbone
        self.hub_repo_or_dir = hub_repo_or_dir
        self.model_name = model_name
        self.weights = weights

        # compute iters per epoch
        global_batch_size = self.num_nodes * self.gpus * self.batch_size if self.gpus > 0 else self.batch_size
        self.train_iters_per_epoch = self.num_samples // global_batch_size

        # init wd and momentum schedule
        self.wd_schedule = cosine_scheduler(
            self.weight_decay,
            weight_decay_end,
            self.max_epochs,
            self.train_iters_per_epoch,
        )

        # Load the backbone
        if self.weights is None:
            self.backbone = torch.hub.load(self.hub_repo_or_dir, self.model_name)
        else:
            self.backbone = torch.hub.load(self.hub_repo_or_dir, self.model_name, weights=self.weights)

        # Setup the RASA head
        self.head = RASAHead(
            input_dim=self.backbone.embed_dim,
            n_pos_layers=self.n_pos_layers,
            pos_out_dim=2,
            pos_out_act_layer=self.pos_out_act_layer,
        )

        # Set up the Evaluation
        if self.use_u_segm_eval:
            self.preds_miou_layer4_x = PredsmIoUKmeans(num_clusters_kmeans, num_classes)
            self.preds_miou_layer4_pos_x = PredsmIoUKmeans(num_clusters_kmeans, num_classes)
            self.preds_miou_layer4_no_pos_x = PredsmIoUKmeans(num_clusters_kmeans, num_classes)

            self.reported_u_uns_segm_x = False

    def on_train_epoch_start(self):
        self.val_loss_mse = []
        self.val_loss_mse_pos_x = []
        self.val_loss_mse_no_pos_x = []
        self.val_pos_x_to_pos_emb_sim = []
        self.val_no_pos_x_to_pos_emb_sim = []
        self.val_no_pos_x_to_x_sim = []
        self.val_pos_x_to_x_sim = []

    def configure_optimizers(self):
        # Separate head params from backbone params
        head_params_named = []
        for name, param in self.head.named_parameters():
            if name.startswith("pos_pred"):
                # Only the forefront linear layer of the head is trained in this model
                head_params_named.append((name, param))
            else:
                # The rest of the linear heads are not trained in this model
                param.requires_grad_(False)

        for name, param in self.backbone.named_parameters():
            param.requires_grad_(False)

        # Prepare param groups. Exclude norm and bias from weight decay if flag set.
        if self.exclude_norm_bias:
            head_params = self.exclude_from_wt_decay(head_params_named, weight_decay=self.weight_decay, lr=self.lr_heads)
            params = head_params
        else:
            head_params = [param for _, param in head_params_named]
            params = [{"params": head_params, "lr": self.lr_heads}]

        # Init optimizer and lr schedule
        if self.optim == "adamw":
            optimizer = torch.optim.AdamW(params, weight_decay=self.weight_decay)
        elif self.optim == "SGD":
            optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Optimizer {self.optim} not supported")
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.train_iters_per_epoch * self.max_epochs,
            eta_min=self.final_lr,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    @staticmethod
    def exclude_from_wt_decay(named_params: Iterator[Tuple[str, nn.Parameter]], weight_decay: float, lr: float):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            # do not regularize biases nor Norm parameters
            if name.endswith(".bias") or len(param.shape) == 1:
                excluded_params.append(param)
            else:
                params.append(param)
        return [
            {"params": params, "weight_decay": weight_decay, "lr": lr},
            {"params": excluded_params, "weight_decay": 0.0, "lr": lr},
        ]

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Union[Optimizer, LightningOptimizer],
        optimizer_closure: Optional[Callable[[], Any]] = None,
    ):

        if self.grad_norm_clipping is not None:
            params = []
            for name, param in self.head.named_parameters():
                if not param.requires_grad:
                    continue
                # do not regularize biases nor Norm parameters
                if name.endswith(".bias") or len(param.shape) == 1:
                    pass
                else:
                    params.append(param)

            torch.nn.utils.clip_grad_norm_(params, self.grad_norm_clipping)

        # Step weight decay schedule
        for i, param_group in enumerate(optimizer.param_groups):
            if i == 0 or i == 2:
                param_group["weight_decay"] = self.wd_schedule[self.trainer.global_step]

        if not isinstance(optimizer, LightningOptimizer):
            # wraps into LightingOptimizer only for running step
            optimizer = LightningOptimizer._to_lightning_optimizer(optimizer, self.trainer.strategy)
        optimizer.step(closure=optimizer_closure)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> float:
        out_dict = self.backbone.forward_features(batch)
        x = out_dict["x_norm_patchtokens"]
        ps = x.shape[1]
        bs = x.shape[0]
        x = self.head.forward(x, use_pos_pred=False, return_pos_info=False)
        y = self.head.forward_pos_pred(x)  # shape: (B, N, D) -> (B, N, 2)
        # Create the target positions
        ps_1d = int(math.sqrt(ps))
        r = torch.arange(ps_1d, device=x.device, dtype=torch.float) / (ps_1d - 1)
        c = torch.arange(ps_1d, device=x.device, dtype=torch.float) / (ps_1d - 1)
        r, c = torch.meshgrid(r, c, indexing="ij")
        tgt = torch.stack(
            (
                r.flatten().unsqueeze(0).repeat(bs, 1),
                c.flatten().unsqueeze(0).repeat(bs, 1),
            ),
            dim=-1,
        ).float()  # shape: (B, N, 2) ???

        # Mean Squared Error between the predicted position and the actual position
        loss = F.mse_loss(y, tgt.to(device=x.device))

        self.log(
            "lr_heads",
            self.optimizers().param_groups[1]["lr"],
            on_step=True,
            on_epoch=False,
        )  # TODO: Maybe the number of the param group is not now 2 because we removed the backbone weoghts
        self.log(
            "weight_decay",
            self.optimizers().param_groups[0]["weight_decay"],
            on_step=True,
            on_epoch=False,
        )
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def u_segm_val_step(self, embeddings: torch.Tensor, mask: torch.Tensor, preds_miou_layer4) -> None:
        # Validate for self.val_iters. Constrained to only parts of the validation set as mIoU calculation
        # would otherwise take too long.
        with torch.no_grad():
            # Process gt seg masks
            bs = mask.size(0)
            assert torch.max(mask).item() <= 1 and torch.min(mask).item() >= 0
            gt = mask * 255
            if self.val_downsample_masks:
                size_masks = 100
                gt = nn.functional.interpolate(gt, size=(size_masks, size_masks), mode="nearest")
            valid = gt != 255  # mask to remove object boundary class

            # store embeddings, valid masks and gt for clustering after validation end
            res_w = int(np.sqrt(embeddings.size(1)))
            embeddings = embeddings.permute(0, 2, 1).reshape(bs, self.backbone.embed_dim, res_w, res_w)
            preds_miou_layer4.update(valid, embeddings, gt)

    def u_segm_val_epoch_end(self, preds_miou_layer4, tag="") -> None:
        # Trigger computations for rank 0 process
        res_kmeans = preds_miou_layer4.compute(self.trainer.is_global_zero)
        preds_miou_layer4.reset()
        if res_kmeans is not None:  # res_kmeans is none for all processes with rank != 0
            if len(tag) > 0:
                tag += "/"
            for k, name, res_k in res_kmeans:
                miou_kmeans, tp, fp, fn, _, matched_bg = res_k
                self.print(miou_kmeans)
                self.logger.experiment[f"{tag}K={name}_miou_layer4"].append(round(miou_kmeans, 8))
                # Log precision and recall values for each class
                for i, (tp_class, fp_class, fn_class) in enumerate(zip(tp, fp, fn)):
                    class_name = self.trainer.datamodule.class_id_to_name(i)
                    self.logger.experiment[f"{tag}K={name}_{class_name}_precision"].append(
                        round(tp_class / max(tp_class + fp_class, 1e-8), 8)
                    )
                    self.logger.experiment[f"{tag}K={name}_{class_name}_recall"].append(
                        round(tp_class / max(tp_class + fn_class, 1e-8), 8)
                    )
                if k > self.num_classes:
                    # Log percentage of clusters assigned to background class
                    self.logger.experiment["{tag}K={name}-percentage-bg-cluster"].append(round(matched_bg, 8))

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        imgs = batch[0]
        mask = batch[1]
        if self.val_iters is None or batch_idx < self.val_iters:
            bs, c, h, w = imgs.shape
            out_dict = self.backbone.forward_features(imgs)
            x = out_dict["x_norm_patchtokens"]
            ps = x.shape[1]
            bs = x.shape[0]
            x_pre = self.head.forward(x, use_pos_pred=False, return_pos_info=False)
            y = self.head.forward_pos_pred(x_pre)  # shape: (B, N, D) -> (B, N, 2)
            # Create the target positions
            ps_1d = int(math.sqrt(ps))
            r = torch.arange(ps_1d, device=x.device, dtype=torch.float) / (ps_1d - 1)
            c = torch.arange(ps_1d, device=x.device, dtype=torch.float) / (ps_1d - 1)
            r, c = torch.meshgrid(r, c, indexing="ij")
            tgt = torch.stack(
                (
                    r.flatten().unsqueeze(0).repeat(bs, 1),
                    c.flatten().unsqueeze(0).repeat(bs, 1),
                ),
                dim=-1,
            ).float()  # shape: (B, N, 2) ???
            loss_mse = F.mse_loss(y, tgt)

            # Eval 1: Our target loss we try to minimize is the MSE loss
            # between the predicted position and the actual position of an image patch
            self.val_loss_mse.append(loss_mse)

            x_pos, x_no_pos = self.head.decompose_pos(x_pre, ll_weight=self.head.pos_pred.weight)

            # Eval 2: How good is the prediction with the pred layer when using the positional information ONLY?
            pos = self.head.forward_pos_pred(x_pos)
            loss_mse_pos_x = F.mse_loss(pos, tgt)
            self.val_loss_mse_pos_x.append(loss_mse_pos_x)

            # Eval 3: How good is the prediction with the pred layer after removing the positional information?
            pos = self.head.forward_pos_pred(x_no_pos)
            loss_mse_no_pos_x = F.mse_loss(pos, tgt)
            self.val_loss_mse_no_pos_x.append(loss_mse_no_pos_x)

            # Eval 4: How similar is the positional information to the positional embedding of the backbone?
            pos_embed = self.backbone.interpolate_pos_encoding(x, w, h)[:, 1:, :]  # shape: (1, N, D)
            pos_x_to_pos_emb_sim = torch.nn.functional.cosine_similarity(
                x_pos, pos_embed.repeat(bs, 1, 1), dim=-1, eps=1e-8
            )  # shape: (B, N)
            self.val_pos_x_to_pos_emb_sim.append(pos_x_to_pos_emb_sim.mean())

            # Eval 5: How similar is the positional information after
            # removing the positional information to the positional embedding of the backbone?
            no_pos_x_to_pos_emb_sim = torch.nn.functional.cosine_similarity(
                x_no_pos, pos_embed.repeat(bs, 1, 1), dim=-1, eps=1e-8
            )  # shape: (B, N)
            self.val_no_pos_x_to_pos_emb_sim.append(no_pos_x_to_pos_emb_sim.mean())

            # Eval 6: How similar is the positional information after
            # removing the positional information to the positional information?
            no_pos_x_to_x_sim = torch.nn.functional.cosine_similarity(x_no_pos, x, dim=-1, eps=1e-8)  # shape: (B, N)
            self.val_no_pos_x_to_x_sim.append(no_pos_x_to_x_sim.mean())

            # Eval 7: How similar is the positional information to original backbone embeddings
            # after removing the positional information?
            pos_x_to_x_sim = torch.nn.functional.cosine_similarity(x_pos, x, dim=-1, eps=1e-8)  # shape: (B, N)
            self.val_pos_x_to_x_sim.append(pos_x_to_x_sim.mean())

            if self.use_u_segm_eval and self.val_iters_u_segm is not None and batch_idx < self.val_iters_u_segm:
                if self.reported_u_uns_segm_x == False:
                    self.u_segm_val_step(x, mask, self.preds_miou_layer4_x)
                self.u_segm_val_step(x_pos, mask, self.preds_miou_layer4_pos_x)
                self.u_segm_val_step(x_no_pos, mask, self.preds_miou_layer4_no_pos_x)

    def on_validation_epoch_end(self) -> None:
        # Average the validation losses
        loss_mse = torch.stack(self.val_loss_mse).mean()
        self.logger.experiment["val/mse_loss"].append(round(loss_mse.item(), 8))
        loss_mse_pos_x = torch.stack(self.val_loss_mse_pos_x).mean()
        self.logger.experiment["val/mse_loss_pos_x"].append(round(loss_mse_pos_x.item(), 8))
        loss_mse_no_pos_x = torch.stack(self.val_loss_mse_no_pos_x).mean()
        self.logger.experiment["val/mse_loss_no_pos_x"].append(round(loss_mse_no_pos_x.item(), 8))
        pos_x_to_pos_emb_sim = torch.stack(self.val_pos_x_to_pos_emb_sim).mean()
        self.logger.experiment["val/pos_x_to_pos_emb_sim"].append(round(pos_x_to_pos_emb_sim.item(), 8))
        no_pos_x_to_pos_emb_sim = torch.stack(self.val_no_pos_x_to_pos_emb_sim).mean()
        self.logger.experiment["val/no_pos_x_to_pos_emb_sim"].append(round(no_pos_x_to_pos_emb_sim.item(), 8))
        no_pos_x_to_x_sim = torch.stack(self.val_no_pos_x_to_x_sim).mean()
        self.logger.experiment["val/no_pos_x_to_x_sim"].append(round(no_pos_x_to_x_sim.item(), 8))
        pos_x_to_x_sim = torch.stack(self.val_pos_x_to_x_sim).mean()
        self.logger.experiment["val/pos_x_to_x_sim"].append(round(pos_x_to_x_sim.item(), 8))

        # Reset the validation metrics
        self.val_loss_mse = []
        self.val_loss_mse_pos_x = []
        self.val_loss_mse_no_pos_x = []
        self.val_pos_x_to_pos_emb_sim = []
        self.val_no_pos_x_to_pos_emb_sim = []
        self.val_no_pos_x_to_x_sim = []
        self.val_pos_x_to_x_sim = []

        if self.use_u_segm_eval:
            # Evaluate for u-segm
            if self.reported_u_uns_segm_x == False:
                # We only want to report the backbone once, since it does not change afterwards
                self.reported_u_uns_segm_x = True
                self.u_segm_val_epoch_end(self.preds_miou_layer4_x, "val/x")
            self.u_segm_val_epoch_end(self.preds_miou_layer4_pos_x, "val/pos_x")
            self.u_segm_val_epoch_end(self.preds_miou_layer4_no_pos_x, "val/no_pos_x")
