
"""
Vicente Hojas, 2022.

Finite Element solvers for the time-dependent scalar wave equation with 
discrete-analytic PML.
"""

import numpy as np
import scipy.sparse as sps
from utils import TriGaussQuad, RectGaussQuad

class LinearFEMSolver:

    def __init__(self, problem, mesh, quad="default", quad_kw=None):

        self.problem = problem
        self.mesh = mesh

        # functions for each element type
        self.element_functions = {
            "triangle": self.triangle_element,
            "triangle_right": self.triangle_element,
            "triangle_left": self.triangle_element,
            "crisscross": self.triangle_element,
            "unionjack": self.triangle_element,
            "rectangle": self.rectangle_element
        }

        self.element_err_functions = {
            "triangle": self.L2_error_triangle_element,
            "triangle_right": self.L2_error_triangle_element,
            "triangle_left": self.L2_error_triangle_element,
            "crisscross": self.L2_error_triangle_element,
            "unionjack": self.L2_error_triangle_element,
            "rectangle": self.L2_error_rectangle_element
        }

        # quadrature functions
        self.element_quads = {
            "triangle": TriGaussQuad,
            "rectangle": RectGaussQuad
        }

        self.quad_params = {
            "triangle": {"k": 2},
            "rectangle": {"k": 2}
        }

        if quad != "default":
            self.element_quads.update(quad)
        if quad_kw is not None:
            self.quad_params.update(quad_kw)

    def P1shp_rectangle(self, xeset, xi):
        """
        Arguments:
        - xeset [ndarray (2, 4)]: coordinates of element nodes [x1, x2, x3, x4]
        - xi [ndarray (1, 2)]: coordinates of quadrature evaluation 
            (standard coord)

        Returns:
        - xhat [ndarray (1, 2)]: coordinates of quadrature evaluation 
            (reference coord)
        - N [ndarray (1, 4)]: evaluation of shape functions (standard coord)
        - DN [ndarray (2, 4)]: Gradients of shape functions, with respect to 
            reference coords, evaluated in the quadrature point (standard 
            coord)
        - detJ [float]: absolute value of determinant of Jacobian matrix
        """

        xi1 = (1/4)*(1-xi[0])*(1-xi[1])
        xi2 = (1/4)*(1+xi[0])*(1-xi[1])
        xi3 = (1/4)*(1+xi[0])*(1+xi[1])
        xi4 = (1/4)*(1-xi[0])*(1+xi[1])

        N1hat = xi1
        N2hat = xi2
        N3hat = xi3
        N4hat = xi4

        Nhat = np.array([N1hat, N2hat, N3hat, N4hat])

        DN11hat = -(1/4)*(1-xi[1])
        DN12hat = -(1/4)*(1-xi[0])
        DN21hat = (1/4)*(1-xi[1])
        DN22hat = -(1/4)*(1+xi[0])
        DN31hat = (1/4)*(1+xi[1])
        DN32hat = (1/4)*(1+xi[0])
        DN41hat = -(1/4)*(1+xi[1])
        DN42hat = (1/4)*(1-xi[0])

        DNhat = np.array([[DN11hat, DN21hat, DN31hat, DN41hat],
                          [DN12hat, DN22hat, DN32hat, DN42hat]])
        
        xhat = (Nhat.dot(xeset.T))

        J = (xeset.dot(DNhat.T))
        detJ = abs(np.linalg.det(J))
        N = Nhat
        invJ = np.linalg.inv(J)
        DN = invJ.T.dot(DNhat)
        
        return xhat, N, DN, detJ

    def P1shp_triangle(self, xeset, xi):
        """
        Arguments:
        - xeset [ndarray (2, 3)]: coordinates of element nodes [x1 x2 x3]
        - xi [ndarray (1, 2)]: coordinates of quadrature evaluation 
            (standard coord)

        Returns:
        - xhat [ndarray (1, 2)]: coordinates of quadrature evaluation 
            (reference coord)
        - N [ndarray (1, 3)]: Evaluation of shape functions (standard coord)
        - DN [ndarray (2, 3)]: Gradients of shape functions, with respect to 
            reference coords, evaluated in the quadrature point 
            (standard coord)
        - detJ [float]: absolute value of determinant of jacobian matrix

        Reference triangle coordinates:
        S1 = (1, 0)
        S2 = (0, 1)
        S3 = (0, 0)
        """

        xi1=xi[0]
        xi2=xi[1]
        xi3=1-xi1-xi2
        
        N1hat=xi1
        N2hat=xi2
        N3hat=xi3

        Nhat=np.array([N1hat,N2hat,N3hat])      
        DN11hat=1
        DN12hat=0

        DN21hat=0
        DN22hat=1
        DN31hat=-1
        DN32hat=-1
        
        DNhat=np.array([[DN11hat,DN21hat,DN31hat],
                        [DN12hat,DN22hat,DN32hat]])
        
        xhat=(Nhat.dot(xeset.T))
        
        J = (xeset.dot(DNhat.T))
        detJ = abs(np.linalg.det(J))   
        N = Nhat
        invJ = np.linalg.inv(J)
        DN = invJ.T.dot(DNhat)

        return xhat, N, DN, detJ

    def triangle_element(self, xe):
        """
        Arguments:
        - t [float]: time of evaluation
        - xe [ndarray (2, 3)]: coordinates of element nodes [x1 x2 x3]

        Returns:
        - Me [ndarray (3, 3)]: Mass matrix of the triangle element
        - Se [ndarray (3, 3)]: Stiffness matrix of the triangle element
        - be [ndarray (3, )]: right-hand vector of triangle element
        """

        quad = self.element_quads["triangle"]
        quad_kw = self.quad_params["triangle"]
        xil, wl = quad(**quad_kw)
        Me = np.zeros((3, 3))
        Se = np.zeros((3, 3))
        be = np.zeros(3)

        for l in range(len(wl)):
            xi, w = xil[l], wl[l]
            xhat, N, B, detJ = self.P1shp_triangle(xe, xi)
            Me += self.problem.k*w*detJ*np.outer(N, N)
            Se += w*detJ*(-B.T.dot(B))
            fe = self.problem.f(xhat[0], xhat[1])
            be += w*N*detJ*fe

        return Me, Se, be

    def rectangle_element(self, xe):
        """
        Arguments:
        - t [float]: time of evaluation
        - xe [ndarray (2, 4)]: coordinates of element nodes [x1, x2, x3, x4]

        Returns:
        - Me [ndarray (4, 4)]: Mass matrix of the rectangle element
        - Se [ndarray (4, 4)]: Stiffness matrix of the rectangle element
        - be [ndarray (4, )]: right-hand vector of rectangle element
        """

        quad = self.element_quads["rectangle"]
        quad_kw = self.quad_params["rectangle"]
        xil, wl = quad(**quad_kw)
        Me = np.zeros((4, 4))
        Se = np.zeros((4, 4))
        be = np.zeros(4)

        for l in range(len(wl)):
            xi, w = xil[l], wl[l]
            xhat, N, B, detJ = self.P1shp_rectangle(xe, xi)
            Me += self.problem.k*w*detJ*np.outer(N, N)
            Se += w*detJ*(-B.T.dot(B))
            fe = self.problem.f(xhat[0], xhat[1])
            be += w*detJ*N*fe

        return Me, Se, be

    def model(self):
        ndofs = np.max(self.mesh.ID) + 1
        M_data = []
        M_j = []
        M_k = []
        S_data = []
        S_j = []
        S_k = []
        b_vector = np.zeros(ndofs)

        # sum of element contributions for each element type
        for el_type in self.mesh.IEN.keys():
            nel, nen = self.mesh.IEN[el_type].shape
            for e in range(nel):
                xeset = self.mesh.xyz[self.mesh.IEN[el_type][e]].T
                Me, Se, be = self.element_functions[el_type](xeset)
                for a in range(nen):
                    A = self.mesh.LM[el_type][e, a]
                    if A >= 0:
                        b_vector[A] += be[a]
                        for b in range(nen):
                            B = self.mesh.LM[el_type][e, b]
                            if B >= 0:
                                M_data.append(Me[a, b])
                                M_j.append(A)
                                M_k.append(B)
                                S_data.append(Se[a, b])
                                S_j.append(A)
                                S_k.append(B)

        M = sps.coo_matrix((M_data, (M_j, M_k)), (ndofs, ndofs))
        M = sps.csc_matrix(M)
        S = sps.coo_matrix((S_data, (S_j, S_k)), (ndofs, ndofs))
        S = sps.csc_matrix(S)
        return M, S, b_vector

    def recover_sol(self, uh):
        xyz_cut = self.mesh.xyz[self.mesh.ID >= 0]
        uh = uh[:xyz_cut.shape[0]]
        return uh

    def L2_error_triangle_element(self, ueh, u, xeset):
        xil, wl = TriGaussQuad(2)
        error = 0
        for l in range(len(wl)):
            xi, w = xil[l], wl[l]
            xhat, N, B, detJ = self.P1shp_triangle(xeset, xi)
            ueh_eval = N.dot(ueh)
            u_eval = u(xhat[0], xhat[1])
            error += w*detJ*(np.abs(ueh_eval - u_eval)**2)
        return error

    def L2_error_rectangle_element(self, ueh, u, xeset):
        xil, wl = RectGaussQuad(2)
        error = 0
        for l in range(len(wl)):
            xi, w = xil[l], wl[l]
            xhat, N, B, detJ = self.P1shp_rectangle(xeset, xi)
            ueh_eval = N.dot(ueh)
            u_eval = u(xhat[0], xhat[1])
            error += w*detJ*(np.abs(ueh_eval - u_eval)**2)
        return error

    def L2_error(self, uh, u):
        error = 0

        # Contribution of each element type set
        for el_type in self.mesh.IEN.keys():
            nel, nen = self.mesh.IEN[el_type].shape
            for e in range(nel):
                xeset = self.mesh.xyz[self.mesh.IEN[el_type][e]].T
                dof = self.mesh.LM[el_type][e]
                ueh = np.zeros(3)
                for idx in range(nen):
                    if dof[idx] >= 0:
                        ueh[idx] = uh[dof[idx]]
                error += self.element_err_functions[el_type](ueh, u, xeset)

        return np.sqrt(error)

    def extended_sol(self, u):
        u_ext = np.zeros(self.mesh.xyz.shape[0])
        for idx in range(len(u_ext)):
            if self.mesh.ID[idx] >= 0:
                u_ext[idx] = u[self.mesh.ID[idx]]
        return u_ext

    def phys_dom_sol(self, u):
        ndofs = np.max(self.mesh.ID) + 1
        u_sol = u[:ndofs]
        u_sol = self.extended_sol(u_sol)
        phys_mask = np.ones(self.mesh.xyz.shape[0], dtype=bool)
        phys_mask[self.mesh.pml_idxs] = 0
        u_sol = u_sol[phys_mask]
        return u_sol

