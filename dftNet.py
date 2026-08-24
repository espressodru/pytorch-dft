import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class TorchFunctionalInputs:
    rho_a: torch.Tensor
    rho_b: torch.Tensor
    hfx_a: torch.Tensor
    hfx_b: torch.Tensor
    grid_coords: torch.Tensor
    grid_weights: torch.Tensor

class DFTNet(nn.Module):

    def __init__(
        self,
        n_input: int,
        hidden_dim: int = 256,
        n_layers: int = 6,
    ):
        super().__init__()
        layers = []
        layers.append(
            nn.Linear(n_input, hidden_dim)
        )
        for _ in range(n_layers - 1):
            layers.append(nn.SiLU())
            layers.append(
                nn.Linear(hidden_dim, hidden_dim)
            )
        layers.append(nn.SiLU())
        layers.append(
            nn.Linear(hidden_dim, 1)
        )
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DFTFunctional(nn.Module):

    def __init__(
        self
    ):
        super().__init__()
        self.model = DFTNet(
            n_input=11,
            hidden_dim=256,
            n_layers=6,
        )

    def _build_features(self, inputs):

        rho_a = inputs.rho_a
        rho_b = inputs.rho_b
        rho_only_a = rho_a[0]
        rho_only_b = rho_b[0]

        grad_a_x = rho_a[1]
        grad_a_y = rho_a[2]
        grad_a_z = rho_a[3]

        grad_b_x = rho_b[1]
        grad_b_y = rho_b[2]
        grad_b_z = rho_b[3]

        norm_grad_a = (
            grad_a_x ** 2
            + grad_a_y ** 2
            + grad_a_z ** 2
        )

        norm_grad_b = (
            grad_b_x ** 2
            + grad_b_y ** 2
            + grad_b_z ** 2
        )

        grad_x = grad_a_x + grad_b_x
        grad_y = grad_a_y + grad_b_y
        grad_z = grad_a_z + grad_b_z

        norm_grad = (
            grad_x ** 2
            + grad_y ** 2
            + grad_z ** 2
        )

        tau_a = rho_a[5]
        tau_b = rho_b[5]

        features = torch.cat(
            [
                inputs.grid_coords,

                inputs.grid_weights[:, None],

                rho_only_a[:, None],
                rho_only_b[:, None],

                tau_a[:, None],
                tau_b[:, None],

                norm_grad_a[:, None],
                norm_grad_b[:, None],
                norm_grad[:, None],

                inputs.hfx_a,
                inputs.hfx_b,
            ],
            dim=1,
        )

        return (
            features,
            rho_only_a,
            rho_only_b,
            tau_a,
            tau_b,
            norm_grad_a,
            norm_grad_b,
            norm_grad,
        )

    @staticmethod
    def _grad(
        output,
        inputs,
        create_graph=True,
    ):

        grads = torch.autograd.grad(
            outputs=output,
            inputs=inputs,
            grad_outputs=torch.ones_like(output),
            create_graph=create_graph,
            retain_graph=True,
            allow_unused=True,
        )

        result = []

        for x, g in zip(inputs, grads):

            if g is None:
                g = torch.zeros_like(x)

            result.append(g)

        return result

    def forward(self, inputs, e_lda, e_hf, e_whf):

        (
            features,
            rho_a,
            rho_b,
            tau_a,
            tau_b,
            sigma_a,
            sigma_b,
            sigma,
        ) = self._build_features(inputs)

        coeff = self.model(features)

        c_lda = coeff[:, 0]
        c_hf = coeff[:, 1]
        c_whf = coeff[:, 2]

        local_xc = (
            c_lda * e_lda
            + c_hf * e_hf
            + c_whf * e_whf
        )

        weighted_local_xc = (
            local_xc
            * inputs.grid_weights[:, None]
        )

        unweighted_xc = torch.sum(
            local_xc,
            dim=0,
        )

        xc = torch.sum(
            weighted_local_xc,
            dim=0,
        )

        total_rho = (
            rho_a
            + rho_b
        )

        vxc = (
            local_xc.squeeze(-1)
            / (total_rho + 1e-12)
        )

        vrho = self._grad(
            unweighted_xc,
            [
                rho_a,
                rho_b,
            ],
            create_graph=True,
        )

        vsigma = self._grad(
            unweighted_xc,
            [
                sigma_a,
                sigma_b,
                sigma,
            ],
            create_graph=True,
        )

        vtau = self._grad(
            unweighted_xc,
            [
                tau_a,
                tau_b,
            ],
            create_graph=True,
        )

        vhf = self._grad(
            xc,
            [
                inputs.hfx_a,
                inputs.hfx_b,
            ],
            create_graph=True,
        )

        return {
            "grid_contribution": local_xc,

            "xc": xc,

            "vxc": vxc,

            "vrho": torch.stack(vrho),

            "vsigma": torch.stack(vsigma),

            "vtau": torch.stack(vtau),

            "vhf": torch.stack(vhf),
        }