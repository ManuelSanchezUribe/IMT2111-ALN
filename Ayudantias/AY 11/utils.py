import numpy as np
from meshio import write_points_cells
import pandas as pd

def TriGaussQuad(k=2, **kwargs):
    if k == 1:
        xi = np.array([1./3, 1./3])
        w = np.array([1/2])
    elif k == 2:
        xi = np.array([[2/3, 1/6], [1/6, 1/6], [1/6, 2/3]])
        w = np.array([1/6, 1/6, 1/6])
    return xi, w

def RectGaussQuad(k=2, **kwargs):
    if k == 1:
        raise NotImplementedError("Not implemented for k=1")
    if k == 2:
        c = 0.5774
        xi = np.array([[-c, -c], [-c, c], [c, -c], [c, c]])
        w = np.array([1, 1, 1, 1])
    return xi, w

def lumping_quad_P1(**kwargs):
    xi = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    w = np.array([1/6, 1/6, 1/6])
    return xi, w

def get_physical_domain_sol(u, mesh, extend_boundary=False):
    # first we remove the pml points in all arrays
    phys_mask = np.ones(mesh.xyz.shape[0], dtype=bool)
    phys_mask[mesh.pml_idxs] = 0
    xyz_phys = mesh.xyz[phys_mask]
    ID_phys = mesh.ID[phys_mask]
    if extend_boundary:
        u_ext = np.zeros(xyz_phys.shape[0])
        for idx in range(len(u_ext)):
            if ID_phys[idx] >= 0:
                u_ext[idx] = u[ID_phys[idx]]
    else:
        xyz_phys = xyz_phys[ID_phys >= 0]
        ID_u = ID_phys[ID_phys >= 0]
        u_ext = u[ID_u]
    return xyz_phys, u_ext

def plot_phys_domain_diff(t, u1, u2, mesh, inplace=False, path=None):
    u_plot = []
    for idx in range(len(t)):
        xyz, u1_ext = get_physical_domain_sol(u1[idx], mesh, 
                                                   extend_boundary=True)
        _, u2_ext = get_physical_domain_sol(u2[idx], mesh, 
                                               extend_boundary=True)
        if inplace:
            uu = np.ascontiguousarray(u1_ext-u2_ext)
            data = {"X": xyz[:, 0], "Y": xyz[:, 1], "Z": np.zeros(xyz.shape[0]), 
                    "U": uu}
            df = pd.DataFrame(data=data)
            df.to_csv(f"{path}_{idx}.csv")
        else:
            u_plot.append(u1_ext-u2_ext)
    if not inplace:
        return u_plot, t
