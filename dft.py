from pyscf import gto
from pyscf import dft
import torch
import dftNet
from pyscf.dft import numint

def dft_loss(
    delta_exc_pred,
    delta_exc_ref,
    delta_escf,
    lambda_s,
):
    loss_r = torch.mean(
        (delta_exc_pred - delta_exc_ref) ** 2
    )

    loss_s = torch.mean(
        delta_escf ** 2
    )

    loss = loss_r + lambda_s * loss_s

    return loss, loss_r, loss_s

mol = gto.Mole()
mol.atom = [['C', 0., 0., 0.]]
mol.spin = 2
mol.basis = 'sto-3g'
mol.build()

ni = numint.NumInt()
mf = dft.UKS(mol)
mf.small_rho_cutoff = 1.e-20
mf._numint = ni
mf.run()

dms = mf.make_rdm1()
ao = ni.eval_ao(mol, mf.grids.coords, deriv=2)
rho_a = ni.eval_rho(mol, ao, dms[0], xctype='MGGA')
rho_b = ni.eval_rho(mol, ao, dms[1], xctype='MGGA')
inputs, _ = ni.construct_functional_inputs(
    mol=mol,
    dms=dms,
    spin=1,
    coords=mf.grids.coords,
    weights=mf.grids.weights,
    rho=(rho_a, rho_b),
    ao=ao[0])

device="cuda"
model = dftNet.DFTFunctional().to(device)
optimizer=torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)
for epoch in range(10000):

    optimizer.zero_grad()
    result=model(
        inputs
    )
    E_pred=result["xc_energy"]
    loss=dft_loss(E_pred, E_ref, E_scf, 1.0)
    loss.backward()
    optimizer.step()
    if epoch%100==0:

        print(
            epoch,
            loss.item(),
            E_pred.item()
        )