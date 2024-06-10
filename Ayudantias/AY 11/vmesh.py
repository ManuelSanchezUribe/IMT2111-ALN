import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import meshio
import pygmsh
import gmsh
from tqdm import tqdm
from functools import reduce
import matplotlib.patches as patches
from airfoils import Airfoil

"""
Supports reading in formats: .msh, .vtk

Boundary codes:

-1: General Dirichlet boundary (mainly for obstacles)
-2: Down boundary
-3: Down and right boundary
-4: Right boundary
-5: Up and right boundary
-6: Up boundary
-7: Up and left boundary
-8: Left boundary
-9: Left and down boundary
"""

IMPLEMENTED_TYPES = [
            "triangle",
            "triangle_left",
            "triangle_right", 
            "crisscross",
            "unionjack",
            "rectangle"
        ]

class Mesh2D:
    """
    Class for cartesian 2D mesh
    Attributes:
    - xyz [array ({ndx+1}*{ndy+1}, 2)]: nodes coordinates
    - IEN [dict of arrays]: index for xyz of nodes for each element
    - ID [array ({ndx+1}*{ndy+1},)]: dofs for each node
    - LM [dict of arrays]: dofs for nodes on each element
    - ax [float]: x-coodrinate of left boundary
    - ay [float]: x-coordinate of right boundary
    - ay [float]: y-coordinate of down boundary
    - by [float]: y-coordinate of up boundary
    - pml_idxs [array (npml, )]: array of indices of pml nodes (in xyz array)
    - h [float]: minimum element side length
    - hx [float]: side length for the construction of x-directional PML
    - hy [float]: side length for the construction of y-directional PML
    """

    def __init__(self, xyz, IEN, ID, LM, coords=None, el_type="triangle"):
        """
        Arguments:
        - xyz [ndarray (n, 2)]: array of mesh node coordinates
        - IEN [ndarray (nel, 3)]: array of elements (indexes in xyz array)
        - ID [ndarray (n, )]: array of dof of mesh nodes
        - LM [ndarray (nel, 3)]: array of dof of each node in the IEN array
        """

        self.xyz = xyz
        self.IEN = {el_type: IEN}
        self.ID = ID
        self.LM = {el_type: LM}
        if coords is None:
            self.ax, self.bx, self.ay, self.by = get_coords(xyz)
        else:
            self.ax, self.bx, self.ay, self.by = coords
        self.hx = get_h_boundary(xyz, ID, "top")
        self.hy = get_h_boundary(xyz, ID, "right")
        self.h = get_min_h(xyz, IEN)
        self.patch_order = {}
        self.triangle_el_types = [
            "triangle",
            "triangle_left", 
            "triangle_right",
            "crisscross", 
            "unionjack"
        ]

    def plot_mesh(self, figsize=(10, 10), plot_elements = True, 
                  plot_ID=False, domain="all", fp=100):
        """
        Visualization of mesh. Boundary gridpoints are shown with empty face, 
        while interior points are fully colored. Elements are shown in 
        blue.
        PML: PML points are shown in red
        """

        # plot parameters
        # dpi = 900
        XYZ, IEN, ID, LM = self.whole_domain()
        fig, ax = plt.subplots(figsize=figsize)
        triangles = None

        # triangles elements
        for tri_el in self.triangle_el_types:
            if tri_el in IEN.keys():
                if triangles is None:
                    triangles = IEN[tri_el]
                else:
                    triangles = np.concatenate((triangles, IEN[tri_el]),
                                               axis=0)
        
        if triangles is not None and plot_elements:
            triang = tri.Triangulation(XYZ[:, 0], XYZ[:, 1], 
                                       triangles)
            ax.triplot(triang, zorder=0)

        # rectangle elements
        if "rectangle" in IEN.keys() and plot_elements:
            for idx in range(IEN["rectangle"].shape[0]):
                xyz_idxs = IEN["rectangle"][idx]
                sqr1 = XYZ[xyz_idxs[0]]
                sqr2 = XYZ[xyz_idxs[1]]
                sqr3 = XYZ[xyz_idxs[2]]
                height = sqr3[1]-sqr2[1]
                width = sqr2[0]-sqr1[0]
                x, y = sqr1[0], sqr1[1]
                rect = patches.Rectangle((x, y), width, height, linewidth=2, 
                                         edgecolor="#1f77b4", facecolor="none",
                                         zorder=0)
                ax.add_patch(rect)

        # plot ID
        if plot_ID:
            for idx in range(XYZ.shape[0]):
                ax.annotate(str(ID[idx]), 
                            (XYZ[idx, 0], XYZ[idx, 1]))

        # scatter of mesh points
        phys_mask = np.ones(self.xyz.shape[0], dtype=bool)
        id_phys = self.ID[phys_mask]
        phys_nb_points = self.xyz[phys_mask][id_phys >= 0]
        left_cond = get_cond(self.ID, "left")
        bottom_cond = get_cond(self.ID, "bottom")
        right_cond = get_cond(self.ID, "right")
        top_cond = get_cond(self.ID, "top")
        b1_points = self.xyz[left_cond]
        b2_points = self.xyz[bottom_cond]
        b3_points = self.xyz[right_cond]
        b4_points = self.xyz[top_cond]
        b5_points = self.xyz[self.ID == -1]
        ax.scatter(phys_nb_points[:, 0], phys_nb_points[:, 1], color="black",
                   s=fp*self.h)
        if domain == "all":
            ax.scatter(b1_points[:, 0], b1_points[:, 1], facecolors="none", 
                       edgecolors="red", s=fp*self.h, label="D1")
            ax.scatter(b2_points[:, 0], b2_points[:, 1], facecolors="none", 
                       edgecolors="green", s=fp*self.h, label="D2")
            ax.scatter(b3_points[:, 0], b3_points[:, 1], facecolors="none", 
                       edgecolors="blue", s=fp*self.h, label="D3")
            ax.scatter(b4_points[:, 0], b4_points[:, 1], facecolors="none", 
                       edgecolors="yellow", s=fp*self.h, label="D4")
            ax.scatter(b5_points[:, 0], b5_points[:, 1], facecolors="none",
                       edgecolors="purple", s=fp*self.h, label="GD")
        return fig, ax

    def whole_domain(self):
        return self.xyz, self.IEN, self.ID, self.LM

    def plot_from_ien(self):
        triang = tri.Triangulation(self.xyz[:, 0], self.xyz[:, 1], self.IEN)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.triplot(triang, zorder=0)
        x_scatter = []
        y_scatter = []
        nel, nen = self.IEN.shape
        for e in range(nel):
            for a in range(nen):
                A = self.LM[e, a]
                if A >= 0:
                    x_scatter.append(self.xyz[self.IEN[e, a], 0])
                    y_scatter.append(self.xyz[self.IEN[e, a], 1])
        ax.scatter(x_scatter, y_scatter)
        return fig, ax

def get_cond(ID, cond):
    """
    Returns condition of all points that satisfy being in the left, right, 
    top or bottom boundary
    """
    if cond == "right":
        return (ID == -3) + (ID == -4) + (ID == -5)
    elif cond == "left":
        return (ID == -7) + (ID == -8) + (ID == -9)
    elif cond == "top":
        return (ID == -5) + (ID == -6) + (ID == -7)
    elif cond == "bottom":
        return (ID == -2) + (ID == -3) + (ID == -9)

def get_coords(xyz):
    """
    Returns coords for rectangle mesh of points xyz.

    NOTE: assumes that points are contained in a rectangle (not checked by 
    function)
    """

    xx, yy = xyz[:, 0], xyz[:, 1]
    ax, bx = np.min(xx), np.max(xx)
    ay, by = np.min(yy), np.max(yy)
    return ax, bx, ay, by

def get_boundary_h(xyz, ID):
    left_cond = get_cond(ID, "left")
    bottom_cond = get_cond(ID, "bottom")
    right_cond = get_cond(ID, "right")
    top_cond = get_cond(ID, "top")
    b1_points = np.sort(xyz[left_cond, 1])
    b2_points = np.sort(xyz[bottom_cond, 0])
    b3_points = np.sort(xyz[right_cond, 1])
    b4_points = np.sort(xyz[top_cond, 0])
    h_min = 1e10
    for idx in range(b1_points.shape[0]-1):
        h = b1_points[idx+1] - b1_points[idx]
        if h < h_min:
            h_min = h
    for idx in range(b2_points.shape[0]-1):
        h = b2_points[idx+1] - b2_points[idx]
        if h < h_min:
            h_min = h
    for idx in range(b3_points.shape[0]-1):
        h = b3_points[idx+1] - b3_points[idx]
        if h < h_min:
            h_min = h
    for idx in range(b4_points.shape[0]-1):
        h = b4_points[idx+1] - b4_points[idx]
        if h < h_min:
            h_min = h
    return h_min

def get_min_h(xyz, IEN, direction="both", boundary_only=False,
              tol=1e-10):
    """
    Given a mesh of nodes xyz and elements IEN, returns the minimal length 
    of element side among all elements.
    """

    min_h = 1e10
    for el in range(IEN.shape[0]):
        x1 = xyz[IEN[el, 0]]
        x2 = xyz[IEN[el, 1]]
        x3 = xyz[IEN[el, 2]]
        if direction == "both":
            h1 = np.linalg.norm(x1-x2)
            h2 = np.linalg.norm(x2-x3)
            h3 = np.linalg.norm(x1-x3)
        elif direction == "x":
            h1, h2, h3 = 1e10, 1e10, 1e10
            if abs(x1[1]-x2[1]) < tol:
                h1 = abs(x1[0]-x2[0])
            if abs(x2[1]-x3[1]) < tol:
                h1 = abs(x2[0]-x3[0])
            if abs(x1[1]-x3[1]) < tol:
                h1 = abs(x1[0]-x3[0])
        elif direction == "y":
            h1, h2, h3 = 1e10, 1e10, 1e10
            if abs(x1[0]-x2[0]) < tol:
                h1 = abs(x1[1]-x2[1])
            if abs(x2[0]-x3[0]) < tol:
                h1 = abs(x2[1]-x3[1])
            if abs(x1[0]-x3[0]) < tol:
                h1 = abs(x1[1]-x3[1])
        h = min(h1, h2, h3)
        if h < min_h:
            min_h = h
    return min_h

def get_max_h(xyz, IEN, direction="both", tol=1e-10):
    """
    Given a mesh of nodes xyz and elements IEN, returns the maximal length 
    of element side among all elements.
    """

    max_h = 1e-16
    for el in tqdm(range(IEN.shape[0])):
        x1 = xyz[IEN[el, 0]]
        x2 = xyz[IEN[el, 1]]
        x3 = xyz[IEN[el, 2]]
        if direction == "both":
            h1 = np.linalg.norm(x1-x2)
            h2 = np.linalg.norm(x2-x3)
            h3 = np.linalg.norm(x1-x3)
        elif direction == "x":
            h1, h2, h3 = 1e-16, 1e-16, 1e-16
            if abs(x1[1]-x2[1]) < tol:
                h1 = abs(x1[0]-x2[0])
            if abs(x2[1]-x3[1]) < tol:
                h1 = abs(x2[0]-x3[0])
            if abs(x1[1]-x3[1]) < tol:
                h1 = abs(x1[0]-x3[0])
        elif direction == "y":
            h1, h2, h3 = 1e-16, 1e-16, 1e-16
            if abs(x1[0]-x2[0]) < tol:
                h1 = abs(x1[1]-x2[1])
            if abs(x2[0]-x3[0]) < tol:
                h1 = abs(x2[1]-x3[1])
            if abs(x1[0]-x3[0]) < tol:
                h1 = abs(x1[1]-x3[1])
        h = max(h1, h2, h3)
        if h > max_h:
            max_h = h
    return max_h

def get_tag(p, coords, tol, assume_boundary=True):
    """
    Given point p and boundary coordinates coords, returns tag for boundary
    (see boundary codes above) given tolerance tol.

    If assume_boundary, then base tag is -1, else 0.
    """

    ax, bx, ay, by = coords
    left = abs(p[0]-ax) < tol
    right = abs(p[0]-bx) < tol
    up = abs(p[1]-by) < tol
    down = abs(p[1]-ay) < tol
    if down and not left and not right:
        return -2
    elif down and right:
        return -3
    elif right and not down and not up:
        return -4
    elif up and right:
        return -5
    elif up and not left and not right:
        return -6
    elif up and left:
        return -7
    elif left and not up and not down:
        return -8
    elif left and down:
        return -9
    elif assume_boundary:
        return -1
    else:
        return 0

def from_file(path, codes, neumann=False):
    """
    Supports: .vtk, .msh
    Builds mesh from file
    """

    ftype = path.split(".")[-1]
    meshio_mesh = meshio.read(path)
    xx = meshio_mesh.points[:, 0]
    yy = meshio_mesh.points[:, 1]
    ax, bx = np.min(xx), np.max(xx)
    ay, by = np.min(yy), np.max(yy)
    coords = (ax, bx, ay, by)
    xyz = np.stack((xx, yy), axis=1)
    IEN = meshio_mesh.cells_dict["triangle"]
    h = get_min_h(xyz, IEN)
    tol = h/10
    ID = -np.ones(xyz.shape[0], dtype=int)
    dof = 0
    if ftype == "msh":
        tags = meshio_mesh.point_data["gmsh:dim_tags"]
        for idx in range(xyz.shape[0]):
            codes0 = [code[0] for code in codes]
            codes1 = [code[1] for code in codes]
            neumann_cond = ((xyz[idx][0] < bx-1e-10) and 
                            (xyz[idx][0] > ax+1e-10) and 
                            (xyz[idx][1] < by-1e-10) and 
                            (xyz[idx][1] > ay+1e-10))
            if tags[idx, 0] in codes0 and tags[idx, 1] in codes1:
                ID[idx] = dof
                dof += 1
            # neumann conditions
            elif neumann and neumann_cond:
                ID[idx] = dof
                dof += 1
            else:
                tag = get_tag(xyz[idx], coords, tol)
                ID[idx] = tag
    elif ftype == "vtk":
        tags = meshio_mesh.cells_dict["vertex"].flatten()
        for idx in range(xyz.shape[0]):
            if idx not in tags:
                ID[idx] = dof
                dof += 1
            else:
                tag = get_tag(xyz[idx], coords, tol)
                ID[idx] = tag
    LM = ID[IEN]
    mesh = Mesh2D(xyz, IEN, ID, LM)
    return mesh

def from_coords(ndx, ndy, coords, random=False, random_coef=1/4, 
                el_type="triangle_right"):
    """
    Build mesh from given coordinates
    """

    # xyz
    ax, bx, ay, by = coords
    dx = np.linspace(ax, bx, ndx+1)
    dy = np.linspace(ay, by, ndy+1)
    xyz = create_grid(dx, dy, random, random_coef, el_type)

    # ID
    ID = np.zeros(xyz.shape[0], dtype = int)
    gdlCont = 0
    tol = min(dx[1]-dx[0], dy[1]-dy[0])/10
    for idx in np.arange(xyz.shape[0]):
        tag = get_tag(xyz[idx], coords, tol, assume_boundary=False)
        if tag == 0:
            ID[idx] = gdlCont
            gdlCont = gdlCont+1
        else:
            ID[idx] = tag

    # IEN and LM
    IEN = create_elements(ndx, ndy, el_type)

    # LM             
    LM = ID[IEN]
    mesh = Mesh2D(xyz, IEN, ID, LM, coords=coords, el_type=el_type)
    return mesh

def from_random(ndx, ndy, coords):
    """
    Creates mesh of random coordinates with delaunay triangulation
    """

    # grid creation
    dx = np.linspace(coords[0], coords[1], num=ndx+1)
    dy = np.linspace(coords[2], coords[3], num=ndy+1)
    npoints = (ndy+1)*(ndy+1)
    xyz = np.zeros((npoints, 2))

    idx_arr = 0
    # bottom boundary
    for idx in range(dx.shape[0]):
        xyz[idx_arr, 0] = dx[idx]
        xyz[idx_arr, 1] = dy[0]
        idx_arr += 1

    # right boundary
    for idx in range(1, dy.shape[0]):
        xyz[idx_arr, 0] = dx[-1]
        xyz[idx_arr, 1] = dy[idx]
        idx_arr += 1

    # top boundary
    for idx in range(dx.shape[0]-2, -1, -1):
        xyz[idx_arr, 0] = dx[idx]
        xyz[idx_arr, 1] = dy[-1]
        idx_arr += 1

    # left boudnary
    for idx in range(dy.shape[0]-2, 0, -1):
        xyz[idx_arr, 0] = dx[0]
        xyz[idx_arr, 1] = dy[idx]
        idx_arr += 1

    # general points
    hx = (coords[1]-coords[0])/ndx
    hy = (coords[3]-coords[2])/ndy
    for idx in range(ndx*ndy-ndx-ndy):
        p1x, p2x = dx[0]+hx, dx[-1]-hx
        p1y, p2y = dy[0]+hy, dy[-1]-hy
        px = p1x + np.random.rand()*(p2x-p1x)
        py = p1y + np.random.rand()*(p2y-p1y)
        xyz[idx_arr, 0] = px
        xyz[idx_arr, 1] = py
        idx_arr += 1

    # IEN
    triang = tri.Triangulation(xyz[:, 0], xyz[:, 1])
    IEN = triang.triangles

    # ID
    ID = np.zeros(xyz.shape[0], dtype = int)
    gdlCont = 0
    tol = min(dx[1]-dx[0], dy[1]-dy[0])/10
    for idx in np.arange(xyz.shape[0]):
        tag = get_tag(xyz[idx], coords, tol, assume_boundary=False)
        if tag == 0:
            ID[idx] = gdlCont
            gdlCont = gdlCont+1
        else:
            ID[idx] = tag

    LM = ID[IEN]
    mesh = Mesh2D(xyz, IEN, ID, LM, coords=coords, el_type="triangle")
    return mesh

def get_h_boundary(xyz, ID, cond, tol=1e-10):
    """
    Gets h of the top boundary of the physical domain
    """
    boundary_cond = get_cond(ID, cond)
    points = xyz[boundary_cond]
    if cond == "top" or cond == "bottom":
        sort_key = lambda x: x[0]
        points = sorted(points, key=sort_key)
        return points[1][0] - points[0][0]
    elif cond == "left" or cond == "right":
        sort_key = lambda x: x[1]
        points = sorted(points, key=sort_key)
        return points[1][1] - points[0][1]
    else:
        raise ValueError("Condition not supported")

def create_grid(dx, dy, random, random_coef, el_type):
    """
    Creates gridpoints from scratch, does not consider a PML region
    """

    ndx, ndy = len(dx)-1, len(dy)-1
    cont = 0
    if el_type == "crisscross":
        nnodes = (ndx+1)*(ndy+1) + ndx*ndy
        hx_random = (dx[1]-dx[0])*random_coef/2
        hy_random = (dy[1]-dy[0])*random_coef/2
    elif el_type in IMPLEMENTED_TYPES:
        nnodes = (ndx+1)*(ndy+1)
        hx_random = (dx[1]-dx[0])*random_coef
        hy_random = (dy[1]-dy[0])*random_coef
    else:
        raise NotImplementedError("Element type not implemnented")

    xyz = np.zeros((nnodes, 2))
    for idxk in range(dy.shape[0]):
        for idxj in range(dx.shape[0]):
            random_cond = random and (idxj not in [0, ndx])
            random_cond = random_cond and (idxk not in [0, ndy])
            if random_cond:
                xr = -hx_random + 2*hx_random*np.random.rand()
                yr = -hy_random + 2*hy_random*np.random.rand()
                xyz[cont, 0] = dx[idxj] + xr
                xyz[cont, 1] = dy[idxk] + yr
                cont += 1
            else:
                xyz[cont, 0] = dx[idxj]
                xyz[cont, 1] = dy[idxk]
                cont += 1
            if idxj != ndx and idxk != ndy and el_type == "crisscross":
                if random_cond:
                    xr = -hx_random + 2*hx_random*np.random.rand()
                    yr = -hy_random + 2*hy_random*np.random.rand()
                    xyz[cont, 0] = (dx[idxj]+dx[idxj+1])/2 + xr
                    xyz[cont, 1] = (dy[idxk]+dy[idxk+1])/2 + yr
                    cont += 1
                else:
                    xyz[cont, 0] = (dx[idxj]+dx[idxj+1])/2
                    xyz[cont, 1] = (dy[idxk]+dy[idxk+1])/2
                    cont += 1
    return xyz

def create_grid_right(dx, dy, start, el_type):
    """
    start is equivalent to original_nnodes
    """

    ndx, ndy = len(dx)-1, len(dy)-1
    cont = 0
    pml_map = np.zeros((dx.shape[0], dy.shape[0]), dtype=int)
    if el_type == "crisscross":
        nnodes_pml = (ndx+1)*(ndy+1) + (ndx+1)*ndy
    elif el_type in IMPLEMENTED_TYPES:
        nnodes_pml = (ndx+1)*(ndy+1)
    else:
        raise NotImplementedError("Element type not implemented")
    xyz_pml = np.zeros((nnodes_pml, 2))
    for idxk in range(dy.shape[0]):
        for idxj in range(dx.shape[0]):
            if idxj == 0 and idxk != ndy and el_type == "crisscross":
                xyz_pml[cont, 0] = dx[idxj] - (dx[1]-dx[0])/2
                xyz_pml[cont, 1] = (dy[idxk] + dy[idxk+1])/2
                cont += 1
            xyz_pml[cont, 0] = dx[idxj]
            xyz_pml[cont, 1] = dy[idxk]
            pml_map[idxj, idxk] = cont + start
            cont += 1
            if idxj != ndx and idxk != ndy and el_type == "crisscross":
                xyz_pml[cont, 0] = (dx[idxj] + dx[idxj+1])/2
                xyz_pml[cont, 1] = (dy[idxk] + dy[idxk+1])/2
                cont += 1
    idxs_pml = start + np.array(list(range(cont)))
    return xyz_pml, idxs_pml, pml_map

def create_grid_top(dx, dy, start, el_type):
    """
    start is equivalent to original_nnodes
    """

    ndx, ndy = len(dx)-1, len(dy)-1
    cont = 0
    pml_map = np.zeros((dx.shape[0], dy.shape[0]), dtype=int)
    if el_type == "crisscross":
        nnodes_pml = (ndx+1)*(ndy+1) + ndx*(ndy+1)
    elif el_type in IMPLEMENTED_TYPES:
        nnodes_pml = (ndx+1)*(ndy+1)
    else:
        raise NotImplementedError("Element type not implemented")
    xyz_pml = np.zeros((nnodes_pml, 2))
    if el_type == "crisscross":
        for idxj in range(ndx):
            xyz_pml[cont, 0] = (dx[idxj]+dx[idxj+1])/2
            xyz_pml[cont, 1] = dy[0] - (dy[1]-dy[0])/2
            cont += 1
    for idxk in range(dy.shape[0]):
        for idxj in range(dx.shape[0]):
            xyz_pml[cont, 0] = dx[idxj]
            xyz_pml[cont, 1] = dy[idxk]
            pml_map[idxj, idxk] = cont + start
            cont += 1
            if idxj != ndx and idxk != ndy and el_type == "crisscross":
                xyz_pml[cont, 0] = (dx[idxj] + dx[idxj+1])/2
                xyz_pml[cont, 1] = (dy[idxk] + dy[idxk+1])/2
                cont += 1
    idxs_pml = start + np.array(list(range(cont)))
    return xyz_pml, idxs_pml, pml_map

def create_elements(ndx, ndy, el_type):
    """
    Creates IEN array of element idxs
    """
    
    # right triangle elements
    if el_type == "triangle_right":
        nen = 3
        nel = 2*ndx*ndy
        IEN = np.zeros((nel,nen), dtype = int)
        cont = 0
        for idx in np.arange(ndy):
            b1 = np.linspace(idx*(ndx+1),idx*(ndx+1)+ndx,ndx+1)
            b2 = b1 + ndx+1
            for idx2 in range(ndx):                
                e1 = np.array([b1[idx2+0],b1[idx2+1],b2[idx2+1]])
                e2 = np.array([b1[idx2],b2[idx2+1],b2[idx2+0]])
                IEN[cont] = e1
                IEN[cont+1] = e2
                cont=cont+2

    # left triangle elements
    elif el_type == "triangle_left":
        nen = 3
        nel = 2*ndx*ndy
        IEN = np.zeros((nel,nen), dtype = int)
        cont = 0
        for idx in np.arange(ndy):
            b1 = np.linspace(idx*(ndx+1),idx*(ndx+1)+ndx,ndx+1)
            b2 = b1 + ndx+1
            for idx2 in range(ndx):                
                e1 = np.array([b1[idx2+0],b1[idx2+1],b2[idx2]])
                e2 = np.array([b1[idx2+1],b2[idx2+1],b2[idx2+0]])
                IEN[cont] = e1
                IEN[cont+1] = e2
                cont=cont+2
    
    # rectangle elements
    elif el_type == "rectangle":
        nen = 4
        nel = ndx*ndy
        IEN = np.zeros((nel, nen), dtype=int)
        cont = 0
        for idx in range(ndy):
            b1 = np.linspace(idx*(ndx+1), idx*(ndx+1)+ndx, ndx+1)
            b2 = b1 + ndx + 1
            for idx2 in range(ndx):
                e = np.array([b1[idx2], b1[idx2+1], b2[idx2+1], b2[idx2]])
                IEN[cont] = e
                cont += 1
    
    # crisscross elements
    elif el_type == "crisscross":
        nen = 3
        nel = 4*ndx*ndy
        IEN = np.zeros((nel, nen), dtype=int)
        cont = 0
        for idx in range(ndy):
            b1 = np.array(list(range(idx*(2*ndx+1), idx*(2*ndx+1) + 2*ndx+1, 2)))
            if idx != ndy-1:
                b2 = b1 + 2*ndx+1
            else:
                b2 = np.array(list(range((idx+1)*(2*ndx+1), (idx+1)*(2*ndx+1) + ndx+1)))
            for idx2 in range(ndx):
                e1 = np.array([b1[idx2], b1[idx2+1], b1[idx2]+1])
                e2 = np.array([b1[idx2+1], b2[idx2+1], b1[idx2]+1])
                e3 = np.array([b2[idx2+1], b2[idx2], b1[idx2]+1])
                e4 = np.array([b1[idx2], b1[idx2]+1, b2[idx2]])
                IEN[cont] = e1
                IEN[cont+1] = e2
                IEN[cont+2] = e3
                IEN[cont+3] = e4
                cont += 4
    
    # union jack elements
    elif el_type == "unionjack":
        nen = 3
        nel = 2*ndx*ndy
        IEN = np.zeros((nel, nen), dtype=int)
        cont = 0
        for idx in range(ndy):
            b1 = np.linspace(idx*(ndx+1), idx*(ndx+1)+ndx, ndx+1)
            b2 = b1 + ndx+1
            for idx2 in range(ndx):
                right = (idx%2+idx2+1)%2
                if right:
                    e1 = np.array([b1[idx2], b1[idx2+1], b2[idx2+1]])
                    e2 = np.array([b1[idx2], b2[idx2+1], b2[idx2]])
                else:
                    e1 = np.array([b1[idx2], b1[idx2+1], b2[idx2]])
                    e2 = np.array([b1[idx2+1], b2[idx2+1], b2[idx2]])
                IEN[cont] = e1
                IEN[cont+1] = e2
                cont += 2
    
    return IEN

def create_elements_top(ndx, ndy, start, t_idxs, el_type):
    # right triangle elements
    if el_type == "triangle_right":
        nen, nel = 3, 2*(ndx)*(ndy+1)
        IEN_pml = np.zeros((nel, nen), dtype=int)
        cont = 0
        for idx in range(ndy+1):
            if idx == 0:
                b1 = np.array(t_idxs)
                b2 = np.array(list(range(0, ndx+1))) + start
            else:
                b1 = np.array(list(range((idx-1)*(ndx+1), (idx)*(ndx+1))))
                b1 += start
                b2 = b1 + ndx+1
            for idx2 in range(ndx):
                e1 = np.array([b1[idx2+0],b1[idx2+1],b2[idx2+1]])
                e2 = np.array([b1[idx2],b2[idx2+1],b2[idx2+0]])
                IEN_pml[cont] = e1
                IEN_pml[cont+1] = e2
                cont += 2

    # left triangle elements
    elif el_type == "triangle_left":
        nen, nel = 3, 2*(ndx)*(ndy+1)
        IEN_pml = np.zeros((nel, nen), dtype=int)
        cont = 0
        for idx in range(ndy+1):
            if idx == 0:
                b1 = np.array(t_idxs)
                b2 = np.array(list(range(0, ndx+1))) + start
            else:
                b1 = np.array(list(range((idx-1)*(ndx+1), (idx)*(ndx+1))))
                b1 += start
                b2 = b1 + ndx+1
            for idx2 in range(ndx):
                e1 = np.array([b1[idx2+0],b1[idx2+1],b2[idx2]])
                e2 = np.array([b1[idx2+1],b2[idx2+1],b2[idx2+0]])
                IEN_pml[cont] = e1
                IEN_pml[cont+1] = e2
                cont += 2

    # rectangle elements
    elif el_type == "rectangle":
        nen, nel = 4, ndx*(ndy+1)
        IEN_pml = np.zeros((nel, nen), dtype=int)
        cont = 0
        for idx in range(ndy+1):
            if idx == 0:
                b1 = np.array(t_idxs)
                b2 = np.array(list(range(0, ndx+1))) + start
            else:
                b1 = np.array(list(range((idx-1)*(ndx+1), (idx)*(ndx+1))))
                b1 += start
                b2 = b1 + ndx+1
            for idx2 in range(ndx):
                e = np.array([b1[idx2], b1[idx2+1], b2[idx2+1], b2[idx2]])
                IEN_pml[cont] = e
                cont += 1
    
    # crisscross elements
    elif el_type == "crisscross":
        nen, nel = 3, 4*ndx*(ndy+1)
        IEN_pml = np.zeros((nel, nen), dtype=int)
        cont = 0
        for idx in range(ndy+1):
            if idx == 0:
                b1 = np.array(t_idxs)
                b2 = np.array(list(range(0, 2*ndx+1, 2))) + start + ndx
            else:
                b1 = np.array(list(range((idx-1)*(2*ndx+1), idx*(2*ndx+1), 2)))
                b1 += start + ndx
                if idx != ndy:
                    b2 = b1 + 2*ndx+1
                else:
                    b2 = np.array(list(range((idx)*(2*ndx+1), (idx)*(2*ndx+1) + ndx+1)))
                    b2 += start + ndx
            for idx2 in range(ndx):
                if idx == 0:
                    mid_idx = idx2 + start
                    e1 = np.array([b1[idx2], b1[idx2+1], mid_idx])
                    e2 = np.array([b1[idx2+1], b2[idx2+1], mid_idx])
                    e3 = np.array([b2[idx2+1], b2[idx2], mid_idx])
                    e4 = np.array([b1[idx2], mid_idx, b2[idx2]])
                else:
                    e1 = np.array([b1[idx2], b1[idx2+1], b1[idx2]+1])
                    e2 = np.array([b1[idx2+1], b2[idx2+1], b1[idx2]+1])
                    e3 = np.array([b2[idx2+1], b2[idx2], b1[idx2]+1])
                    e4 = np.array([b1[idx2], b1[idx2]+1, b2[idx2]])
                IEN_pml[cont] = e1
                IEN_pml[cont+1] = e2
                IEN_pml[cont+2] = e3
                IEN_pml[cont+3] = e4
                cont += 4
    
    # unionjack elements
    elif el_type == "unionjack":
        nen, nel = 3, 2*(ndx)*(ndy+1)
        IEN_pml = np.zeros((nel, nen), dtype=int)
        cont = 0
        for idx in range(ndy+1):
            if idx == 0:
                b1 = np.array(t_idxs)
                b2 = np.array(list(range(0, ndx+1))) + start
            else:
                b1 = np.array(list(range((idx-1)*(ndx+1), (idx)*(ndx+1))))
                b1 += start
                b2 = b1 + ndx+1
            for idx2 in range(ndx):
                right = (idx%2+idx2+1)%2
                if right:
                    e1 = np.array([b1[idx2], b1[idx2+1], b2[idx2+1]])
                    e2 = np.array([b1[idx2], b2[idx2+1], b2[idx2]])
                else:
                    e1 = np.array([b1[idx2], b1[idx2+1], b2[idx2]])
                    e2 = np.array([b1[idx2+1], b2[idx2+1], b2[idx2]])
                IEN_pml[cont] = e1
                IEN_pml[cont+1] = e2
                cont += 2

    else:
        raise NotImplementedError("Element type not implemented")

    return IEN_pml

def create_elements_right(ndx, ndy, start, r_idxs, el_type):
    # right triangle elements
    if el_type == "triangle_right":
        nen, nel = 3, 2*(ndx+1)*ndy
        IEN_pml = np.zeros((nel, nen), dtype=int)
        cont = 0
        for idx, idx_bdry in enumerate(r_idxs[:-1]):
            b1 = np.array(list(range(idx*(ndx+1), (idx+1)*(ndx+1))))
            b1 += start
            b2 = b1 + ndx + 1
            b1 = np.concatenate((np.array([idx_bdry]), b1))
            b2 = np.concatenate((np.array([r_idxs[idx+1]]), b2))
            for idx2 in range(ndx+1):
                e1 = np.array([b1[idx2+0],b1[idx2+1],b2[idx2+1]])
                e2 = np.array([b1[idx2],b2[idx2+1],b2[idx2+0]])
                IEN_pml[cont] = e1
                IEN_pml[cont+1] = e2
                cont += 2

    # left triangle elements
    elif el_type == "triangle_left":
        nen, nel = 3, 2*(ndx+1)*ndy
        IEN_pml = np.zeros((nel, nen), dtype=int)
        cont = 0
        for idx, idx_bdry in enumerate(r_idxs[:-1]):
            b1 = np.array(list(range(idx*(ndx+1), (idx+1)*(ndx+1))))
            b1 += start
            b2 = b1 + ndx + 1
            b1 = np.concatenate((np.array([idx_bdry]), b1))
            b2 = np.concatenate((np.array([r_idxs[idx+1]]), b2))
            for idx2 in range(ndx+1):
                e1 = np.array([b1[idx2+0],b1[idx2+1],b2[idx2]])
                e2 = np.array([b1[idx2+1],b2[idx2+1],b2[idx2+0]])
                IEN_pml[cont] = e1
                IEN_pml[cont+1] = e2
                cont += 2

    # rectangle elements
    elif el_type == "rectangle":
        nen, nel = 4, (ndx+1)*ndy
        IEN_pml = np.zeros((nel, nen), dtype=int)
        cont = 0
        for idx, idx_bdry in enumerate(r_idxs[:-1]):
            b1 = np.array(list(range(idx*(ndx+1), (idx+1)*(ndx+1))))
            b1 += start
            b2 = b1 + ndx + 1
            b1 = np.concatenate((np.array([idx_bdry]), b1))
            b2 = np.concatenate((np.array([r_idxs[idx+1]]), b2))
            for idx2 in range(ndx+1):
                e = np.array([b1[idx2], b1[idx2+1], b2[idx2+1], b2[idx2]])
                IEN_pml[cont] = e
                cont += 1

    # crisscross elements
    elif el_type == "crisscross":
        nen, nel = 3, 4*(ndx+1)*ndy
        IEN_pml = np.zeros((nel, nen), dtype=int)
        cont = 0
        for idx, idx_bdry in enumerate(r_idxs[:-1]):
            b1 = np.array(list(range(idx*(2*ndx+2), (idx+1)*(2*ndx+2), 2)))
            b1 += start+1
            if idx != ndy-1:
                b2 = b1 + 2*ndx+2
            else:
                b2 = np.array(list(range((idx+1)*(2*ndx+2), (idx+1)*(2*ndx+2) + ndx+2)))
                b2 += start
            b1 = np.concatenate((np.array([idx_bdry]), b1))
            b2 = np.concatenate((np.array([r_idxs[idx+1]]), b2))
            for idx2 in range(ndx+1):
                if idx2 == 0:
                    idx_mid = start + idx*(2*ndx+2)
                    e1 = np.array([b1[idx2], b1[idx2+1], idx_mid])
                    e2 = np.array([b1[idx2+1], b2[idx2+1], idx_mid])
                    e3 = np.array([b2[idx2+1], b2[idx2], idx_mid])
                    e4 = np.array([b1[idx2], idx_mid, b2[idx2]])
                else:
                    e1 = np.array([b1[idx2], b1[idx2+1], b1[idx2]+1])
                    e2 = np.array([b1[idx2+1], b2[idx2+1], b1[idx2]+1])
                    e3 = np.array([b2[idx2+1], b2[idx2], b1[idx2]+1])
                    e4 = np.array([b1[idx2], b1[idx2]+1, b2[idx2]])
                IEN_pml[cont] = e1
                IEN_pml[cont+1] = e2
                IEN_pml[cont+2] = e3
                IEN_pml[cont+3] = e4
                cont += 4 

    # unionjack elements
    elif el_type == "unionjack":
        nen, nel = 3, 2*(ndx+1)*ndy
        IEN_pml = np.zeros((nel, nen), dtype=int)
        cont = 0
        for idx, idx_bdry in enumerate(r_idxs[:-1]):
            b1 = np.array(list(range(idx*(ndx+1), (idx+1)*(ndx+1))))
            b1 += start
            b2 = b1 + ndx + 1
            b1 = np.concatenate((np.array([idx_bdry]), b1))
            b2 = np.concatenate((np.array([r_idxs[idx+1]]), b2))
            for idx2 in range(ndx+1):
                right = (idx%2+idx2+1)%2
                if right:
                    e1 = np.array([b1[idx2], b1[idx2+1], b2[idx2+1]])
                    e2 = np.array([b1[idx2], b2[idx2+1], b2[idx2]])
                else:
                    e1 = np.array([b1[idx2], b1[idx2+1], b2[idx2]])
                    e2 = np.array([b1[idx2+1], b2[idx2+1], b2[idx2]])
                IEN_pml[cont] = e1
                IEN_pml[cont+1] = e2
                cont += 2

    else:
        raise NotImplementedError("Element type not implemented")

    return IEN_pml

def create_gmsh(ndx, ndy, coords, save_path):
    """
    Creates "uniform" gmsh mesh and saves it to save_path as msh file
    """

    # resolution of mesh
    resx = (coords[1]-coords[0])/ndx
    resy = (coords[3]-coords[2])/ndy

    geometry = pygmsh.geo.Geometry()
    model = geometry.__enter__()
    points = [
        model.add_point((coords[0], coords[2], 0), mesh_size=resx),
        model.add_point((coords[1], coords[2], 0), mesh_size=resx),
        model.add_point((coords[1], coords[3], 0), mesh_size=resx),
        model.add_point((coords[0], coords[3], 0), mesh_size=resx)
    ]
    channel_lines = [model.add_line(points[i], points[i+1])
                     for i in range(-1, len(points)-1)]
    channel_loop = model.add_curve_loop(channel_lines)
    plane_surface = model.add_plane_surface(channel_loop)
    model.synchronize()
    # volume_marker = 6
    model.add_physical([plane_surface], "Volume")
    model.add_physical(channel_lines, "Walls")
    geometry.generate_mesh(dim=2)
    gmsh.write(save_path)
    gmsh.clear()
    geometry.__exit__()

def create_airfoil_mesh(coords, code, h, save_path, scale=1, x0=0, y0=0):
    """
    Creates airfoil mesh from airfoil code and h target for point 
    separation. x0 and y0 correspond to the start of the leading face 
    of the airfoil.
    """

    res = h

    # airfoil points
    foil = Airfoil.NACA4(code)
    num = int(4*scale/h)
    xj = np.linspace(x0, x0+scale, num=num)
    y_lower = foil.y_lower(x=(xj-x0)/scale)*scale + y0
    y_upper = foil.y_upper(x=(xj[1:-1]-x0)/scale)*scale + y0

    # add airfoil points to geometry
    geometry = pygmsh.geo.Geometry()
    model = geometry.__enter__()
    air_points = []
    for idx in range(xj.shape[0]):
        air_points.append(model.add_point((xj[idx], y_lower[idx], 0), 
                          mesh_size=res/4))
    for idx in range(y_upper.shape[0]-1, -1, -1):
        air_points.append(model.add_point((xj[1:-1][idx], y_upper[idx], 0), 
                          mesh_size=res/4))
    
    # airfoil lines
    air_lines = []
    for idx in range(-1, len(air_points)-1):
        air_lines.append(model.add_line(air_points[idx], air_points[idx+1]))

    # airfoil curve loop
    air_loop = model.add_curve_loop(air_lines)
    # air_surface = model.add_plane_surface(air_loop)

    # boundary generation
    ndx = int((coords[1]-coords[0])/h)
    ndy = int((coords[3]-coords[2])/h)
    dx = np.linspace(coords[0], coords[1], num=ndx+1)
    dy = np.linspace(coords[2], coords[3], num=ndy+1)

    points = []

    # bottom boundary
    for idx in range(dx.shape[0]):
        points.append(model.add_point((dx[idx], dy[0], 0), mesh_size=res))

    # right boundary
    for idx in range(1, dy.shape[0]):
        points.append(model.add_point((dx[-1], dy[idx], 0), mesh_size=res))

    # top boundary
    for idx in range(dx.shape[0]-2, -1, -1):
        points.append(model.add_point((dx[idx], dy[-1], 0), mesh_size=res))

    # left boundary
    for idx in range(dy.shape[0]-2, 0, -1):
        points.append(model.add_point((dx[0], dy[idx], 0), mesh_size=res))

    # Add lines between all points creating the rectangle
    channel_lines = [model.add_line(points[i], points[i+1])
                    for i in range(-1, len(points)-1)]

    # Create a line loop and plane surface for meshing
    channel_loop = model.add_curve_loop(channel_lines)
    plane_surface = model.add_plane_surface(
        channel_loop, holes=[air_loop])
    
    # Call gmsh kernel before add physical entities
    model.synchronize()
    # volume_marker = 6
    model.add_physical([plane_surface], "Volume")
    model.add_physical(channel_lines, "Walls")
    model.add_physical(air_loop.curves, "Obstacle")
    model.add_physical(points, "Boundary")
    geometry.save_geometry(save_path)
    mesh = geometry.generate_mesh(dim=2)
    # mesh.write(save_path)
    gmsh.clear()
    geometry.__exit__()

if __name__ == "__main__":
    """
    Various mesh creation and plots
    """
    
    # # create airfoil mesh (only geo file for now)
    # coords = (-4, 4, -2, 2)
    # code = "4812"
    # h = 0.3
    # x0, y0 = 0, 0
    # save_path = "../Mesh Files/airfoil_mesh.geo_unrolled"
    # scale = 3
    # create_airfoil_mesh(coords, code, h, save_path, scale=scale, x0=x0, y0=y0)

    # # load and plot airfoil mesh
    # mesh = meshio.read("../Mesh Files/airfoil_mesh.msh")
    # print(mesh.point_data["gmsh:dim_tags"])
    # mesh = from_file("../Mesh Files/airfoil_mesh.msh", code=(2,1))
    # mesh.add_domain_right(2)
    # fig, ax = mesh.plot_mesh()
    # plt.show()
