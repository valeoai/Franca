# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch.nn as nn
from torch.nn.init import trunc_normal_
from torch.nn.utils import weight_norm

# MRL on backbone


def _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True):
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    else:
        # Initial layers up to the second-to-last layer
        layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())

        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())

        # Final layer
        layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
        if use_bn:
            layers.append(nn.BatchNorm1d(bottleneck_dim))
        layers.append(nn.GELU())

        return nn.Sequential(*layers)


class MRLDINOHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=384,
        mlp_bias=True,
        nesting_list=None,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)

        # Default nesting list is [hidden_dim//2, hidden_dim] if None is provided
        self.nesting_list = [hidden_dim // 2, hidden_dim] if nesting_list is None else nesting_list

        # Create matryoshka projection layers before MLP
        self.matryoshka_projections = nn.ModuleList([nn.Linear(dim, dim, bias=mlp_bias) for dim in self.nesting_list])

        # Build MLPs for each nesting level
        self.mlps = nn.ModuleList(
            [
                _build_mlp(
                    nlayers,
                    dim,
                    bottleneck_dim,
                    hidden_dim=hidden_dim,
                    use_bn=use_bn,
                    bias=mlp_bias,
                )
                for dim in self.nesting_list
            ]
        )

        # Output projection layers with scaling based on nesting dimension ratio
        self.last_layers = nn.ModuleList(
            [
                weight_norm(
                    nn.Linear(
                        bottleneck_dim,
                        int(out_dim * (dim / self.nesting_list[-1])),
                        bias=False,
                    )
                )
                for dim in self.nesting_list
            ]
        )

        for layer in self.last_layers:
            layer.weight_g.data.fill_(1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x shape: [batch_size, in_dim]
        outputs = []

        for i, dim in enumerate(self.nesting_list):
            # Project input to the appropriate nesting dimension
            h = self.matryoshka_projections[i](x[..., :dim])

            # Pass through MLP
            h = self.mlps[i](h)

            # Final projection
            out = self.last_layers[i](h)
            outputs.append(out)

        return tuple(outputs)
