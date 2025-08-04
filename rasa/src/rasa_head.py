from typing import Tuple

import torch
import torch.nn as nn


class RASAHead(nn.Module):
    def __init__(self, input_dim, n_pos_layers, pos_out_dim=2, pos_out_act_layer="sigmoid"):
        super(RASAHead, self).__init__()

        # Assertions for input validation
        assert n_pos_layers > 0, "n_pos_layers must be greater than 0"
        assert pos_out_act_layer in [
            "sigmoid",
            "tanh",
            "identity",
            None
        ], f"pos_out_act_layer must be one of ['sigmoid', 'tanh', 'identity', None], got {pos_out_act_layer}"

        if pos_out_dim not in [1, 2]:
            raise ValueError("pos_out_dim must be either 1 or 2")

        self.pos_out_dim = pos_out_dim
        # Number of positional layers that are defined except the prediction layer
        self.n_pos_layers = n_pos_layers

        # The input dimension of the head
        self.input_dim = input_dim

        pos_out_act_layer = pos_out_act_layer.lower() if isinstance(pos_out_act_layer, str) else pos_out_act_layer
        # Configure the activation layer for the positional output
        if pos_out_act_layer == "sigmoid":
            pos_out_act_layer = nn.Sigmoid
        elif pos_out_act_layer == "tanh":
            pos_out_act_layer = nn.Tanh
        elif pos_out_act_layer is None or pos_out_act_layer == "identity":
            pos_out_act_layer = nn.Identity
        self.pos_out_act_layer = pos_out_act_layer()

        # The positional prediction layer used to predict the positional information
        self.pos_pred = nn.Linear(self.input_dim, self.pos_out_dim, bias=False)
        # the layers used to decompose the positional information
        self.pre_pos_layers = torch.nn.ModuleList(
            [nn.Linear(self.input_dim, self.pos_out_dim, bias=False) for i in range(self.n_pos_layers)]
        )

    def forward(self, x, use_pos_pred=True, return_pos_info=False):
        """
        Forward method for the RASAHead.
        Args:
            x: The input tensor of shape (B, N, D) where B is the batch size,
            N is the number of patches, and D is the dimension of the patch encodings.
        Returns:
            y: The output tensor of shape (B, N, D) with the positional information
            removed after using self.pre_pos_layers linear layers iteratively.
        """
        if self.n_pos_layers > 0:
            for _, l in enumerate(self.pre_pos_layers):
                x_pos, x = self.decompose_pos(x, ll_weight=l.weight)  # the x will be the input to the next layer

        if use_pos_pred == True:
            # Use the last layer to remove the positional information
            x_pos, x = self.decompose_pos(x, ll_weight=self.pos_pred.weight)

        if return_pos_info:
            return x_pos, x
        return x

    ## Forward Head Method
    def forward_pos_pred(self, x):
        y = self.pos_pred(x)
        y = self.pos_out_act_layer(y)
        return y

    def decompose_pos(self, x, ll_weight) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pos_out_dim == 1:
            return RASAHead.decompose_pos_1D(x, ll_weight)
        elif self.pos_out_dim == 2:
            return RASAHead.decompose_pos_2D(x, ll_weight)
        else:
            raise NotImplementedError(f"Positional information for {self.pos_out_dim} dimensions is not implemented")

    @staticmethod
    def decompose_pos_1D(x, ll_weight) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the positional information from the patch encodings of the encoder's output.
        Args:
            model: The model instance
            imgs: The input images of shape (B, C, H, W)

        """
        bs = x.shape[0]  # batch size
        ps = x.shape[1]  # number of patches
        # model.pos_pred is the position prediction linear layer
        pos_v = ll_weight.squeeze(0)  # shape: (D)
        # The projection of x onto the line spanned by pos_v
        nom = torch.einsum("bnd,d->bn", x, pos_v)  # shape: (B, N)
        denom = torch.dot(pos_v, pos_v)  # shape: (1)
        x_pos = pos_v.repeat(bs, ps, 1) * (nom / denom).unsqueeze(-1)  # shape: (B, N, D)
        # Get the part of x that is orthogonal to the line spanned by pos_v
        no_pos_x = x - x_pos
        return x_pos, no_pos_x

    @staticmethod
    def decompose_pos_2D(x, ll_weight) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = x.shape[0]
        ps = x.shape[1]
        pos_vr = ll_weight[0]  # shape: (D)
        pos_vc = ll_weight[1]  # shape: (D)

        # For  3dimensional we could use the cross-product that gives us the normal vector perpendicular to the plane.
        # For n-dimensional vectors, the logic remains the same, but we need to account for the dimensionality when
        # calculating the normal vector.
        # In n-dimensions, the plane defined by two vectors A and B does not have a single normal vector but rather
        # a normal subspace. For this case, we compute the projection of v onto the subspace spanned by A and B using
        # the Gram-Schmidt process.

        # Step 1: Normalize the vectors pos_vr and pos_vc
        pos_vr = pos_vr / torch.norm(pos_vr)
        pos_vc = pos_vc / torch.norm(pos_vc)

        # Step 2: Orthogonalize pos_vc with respect to pos_vr using Gram-Schmidt
        pos_vc_orth = pos_vc - torch.dot(pos_vc, pos_vr) * pos_vr
        pos_vc_orth = pos_vc_orth / torch.norm(pos_vc_orth)  # Normalize the orthogonalized pos_vc

        # Step 3: Project x onto pos_vr and pos_vc_orth
        x_proj_pos_vr = (x * pos_vr.repeat(bs, ps, 1)).sum(dim=-1).unsqueeze(-1) * pos_vr.repeat(bs, ps, 1)
        x_proj_pos_vc_orth = (x * pos_vc_orth.repeat(bs, ps, 1)).sum(dim=-1).unsqueeze(-1) * pos_vc_orth.repeat(bs, ps, 1)

        # Step 4: Compute the projection of x onto the plane spanned by pos_vr and pos_vc
        x_pos = x_proj_pos_vr + x_proj_pos_vc_orth

        # Step 5: Remove the positional information from the vectors
        no_pos_x = x - x_pos

        return x_pos, no_pos_x
