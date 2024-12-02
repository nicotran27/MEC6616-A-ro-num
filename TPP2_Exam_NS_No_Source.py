import numpy as np
import sympy as sp
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from mesh import Mesh
from meshGenerator import MeshGenerator
from meshConnectivity import MeshConnectivity
from meshPlotter import MeshPlotter
import pyvista as pv
import pyvistaqt as pvQt

class ConvectionDiffusionSolver:
    def __init__(self, x_min, x_max, y_min, y_max,lc):
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.lc = lc
        self.rho = 1.0
        self.Cp = 1.0
        self.R1=1
        self.R2=2
        
        # Définition des symboles pour les valeurs x et y 
        self.x, self.y = sp.symbols('x y')
        self.u = self.y / (self.x**2 + self.y**2)
        self.v = -self.x / (self.x**2 + self.y**2)
        # Define inlet temperature profile

        
        # Création des champs u et v et de la solution littérale manufacturée 

        self.T0, self.Tx, self.Txy = 400, 50, 100
        self.T_mms = self.T0 + self.Tx * sp.cos(sp.pi * self.x) + self.Txy * sp.sin(sp.pi * self.x * self.y)
        
        # Création du terme source 
        self.S_func_generator = self.compute_terme_source()
        
        # Créations des fonctions python 
        self.u_func = sp.lambdify((self.x, self.y), self.u, 'numpy')
        self.v_func = sp.lambdify((self.x, self.y), self.v, 'numpy')
        self.T_mms_func = sp.lambdify((self.x, self.y), self.T_mms, 'numpy')
        
        self.mesh = None
        self.face_areas = None
        self.cell_volumes = None
        self.normale_face = None
        self.centre_cellule = None

    def compute_terme_source(self):
        k = sp.symbols('k')
        #Calcul des termes de convection 
        convection_x = sp.diff(self.rho * self.u * self.Cp * self.T_mms, self.x)
        convection_y = sp.diff(self.rho * self.v * self.Cp * self.T_mms, self.y)
        # Calcul des termes de diffusion 
        diffusion_x = sp.diff(k * sp.diff(self.T_mms, self.x), self.x)
        diffusion_y = sp.diff(k * sp.diff(self.T_mms, self.y), self.y)
        # Calcul du terme source par méthode MMS
        S = convection_x* + convection_y - diffusion_x - diffusion_y
        
        # Création de la fonction source : fonction 
        def S_func_generator(k_val):
            return sp.lambdify((self.x, self.y), S.subs(k, k_val), 'numpy')
        
        return S_func_generator
        

    
    # Maillage
    def generate_mesh(self, mesh_type='TRI', N1=20, N2=20):
        mesh_generator = MeshGenerator()
        mesh_parameters = {
            'mesh_type': mesh_type,
            'N1': N1,
            'N2': N2
        } if mesh_type == 'QUAD' else {
            'mesh_type': mesh_type,
            'lc': self.lc
        }
        self.mesh = mesh_generator.quarter_annular(self.R1, self.R2, mesh_parameters)
        # self.mesh = mesh_generator.rectangle([self.x_min, self.x_max, self.y_min, self.y_max], mesh_parameters)
        
        mesh_connectivity = MeshConnectivity(self.mesh)
        mesh_connectivity.compute_connectivity()
        return self.mesh

    
        
    def calcul_metrique(self, maillage):
        """
        Fonction calculant les métriques du maillages.
        
        Parameters
        ----------
        maillage : Mesh object
            Mesh object containing connectivity and geometry information
            
        Returns
        -------
        metriques : dictionnaire
            Dictionnaire contenant les métriques du maillage.
        """
        # Nombre de triangles
        Ntri = maillage.get_number_of_elements()
        # Nombre d'arêtes total
        Nare = maillage.get_number_of_faces()
        # Nombre d'arêtes en frontière
        Naref = sum(1 for i in range(Nare) if maillage.get_face_to_elements(i)[1] == -1)
        
        # Vecteurs d'arêtes
        dx = np.zeros(Nare)
        dy = np.zeros(Nare)
        xmid = np.zeros(Nare)
        ymid = np.zeros(Nare)
        
        for i_face in range(Nare):
            nodes = maillage.get_face_to_nodes(i_face)
            x1, y1 = maillage.get_node_to_xycoord(nodes[0])
            x2, y2 = maillage.get_node_to_xycoord(nodes[1])
            dx[i_face] = x2 - x1
            dy[i_face] = y2 - y1
            xmid[i_face] = (x1 + x2)/2
            ymid[i_face] = (y1 + y2)/2
        
        # Vecteurs normaux aux arêtes
        dA = np.sqrt(dx**2 + dy**2)
        n = np.transpose(np.array((dy/dA, -dx/dA)))
        
        # Centroïdes des triangles
        xTri = np.zeros(Ntri)
        yTri = np.zeros(Ntri)
        
        for i_elem in range(Ntri):
            nodes = maillage.get_element_to_nodes(i_elem)
            x_coords = [maillage.get_node_to_xcoord(node) for node in nodes]
            y_coords = [maillage.get_node_to_ycoord(node) for node in nodes]
            xTri[i_elem] = np.mean(x_coords)
            yTri[i_elem] = np.mean(y_coords)
        
        # Vecteurs reliant les triangles
        dxTri = np.zeros(Nare)
        dyTri = np.zeros(Nare)
        
        for i_face in range(Nare):
            left_cell, right_cell = maillage.get_face_to_elements(i_face)
            if right_cell != -1:
                dxTri[i_face] = xTri[right_cell] - xTri[left_cell]
                dyTri[i_face] = yTri[right_cell] - yTri[left_cell]
            else:
                dxTri[i_face] = xmid[i_face] - xTri[left_cell]
                dyTri[i_face] = ymid[i_face] - yTri[left_cell]
        
        # Aires des triangles
        aireTri = np.zeros(Ntri)
        for i_elem in range(Ntri):
            nodes = maillage.get_element_to_nodes(i_elem)
            if len(nodes) == 3:  # Triangle
                x1, y1 = maillage.get_node_to_xycoord(nodes[0])
                x2, y2 = maillage.get_node_to_xycoord(nodes[1])
                x3, y3 = maillage.get_node_to_xycoord(nodes[2])
                aireTri[i_elem] = 0.5 * abs((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))
        
        # vecteur eta
        eEta = np.transpose(np.array((dx/dA, dy/dA)))
        
        # vecteur ksi
        dKsi = np.sqrt(dxTri**2 + dyTri**2)
        eKsi = np.transpose(np.array((dxTri/dKsi, dyTri/dKsi)))
        
        # produit scalaire des vecteurs n, ksi et eta
        pNKsi = np.sum(n*eKsi, axis=1)
        pKsiEta = np.sum(eKsi*eEta, axis=1)
        
        # Dictionnaire contenant les métriques
        metriques = {
            "Ntri": Ntri, "Nare": Nare, "Naref": Naref,
            "dx": dx, "dy": dy, "xmid": xmid, "ymid": ymid,
            "n": n, "dA": dA, "xTri": xTri, "yTri": yTri,
            "dxTri": dxTri, "dyTri": dyTri, "aireTri": aireTri,
            "eEta": eEta, "dKsi": dKsi, "eKsi": eKsi,
            "pNKsi": pNKsi, "pKsiEta": pKsiEta
        }
        return metriques

    def print_metrique_info(self, maillage):
        """
        Print key information about the mesh metrics
        """
        metriques = self.calcul_metrique(maillage)
    


        centre_cellule = np.zeros((self.mesh.get_number_of_elements(), 2))
        for i_elem in range(self.mesh.get_number_of_elements()):
            nodes = self.mesh.get_element_to_nodes(i_elem)
            x_coords = np.array([self.mesh.get_node_to_xcoord(node) for node in nodes])
            y_coords = np.array([self.mesh.get_node_to_ycoord(node) for node in nodes])
            centre_cellule[i_elem] = [np.mean(x_coords), np.mean(y_coords)]
        return centre_cellule
    
    def grad_least_square(self, maillage, metriques, bcdata, phi):
        """
        Fonction qui calcule le gradient de Phi dans chaque triangle par la méthode
        des moindres carrées (Least-Square).
        
        Parameters
        ----------
        maillage : Mesh object
            Maillage (coordonnées et connectivité)
        metriques : dictionnaire
            Dictionnaire contenant les métriques du maillage
        bcdata : list of tuples
            Liste des conditions limites sous forme [(type, valeur), ...]
        phi : numpy array
            Champ connu aux centres des triangles du maillage
            
        Returns
        -------
        gradPhi : numpy array
            Gradient de Phi au centre des triangles
        """
        # Initialisation
        Ntri = metriques["Ntri"]
        Naref = metriques["Naref"]
        Nare = metriques["Nare"]
        
        ATA = np.zeros((Ntri, 2, 2))
        B = np.zeros((Ntri, 2))
        dphi = np.zeros(Nare)
    
        # Internal edges
        for iare in range(Naref, Nare):
            left_cell, right_cell = maillage.get_face_to_elements(iare)
            if right_cell != -1:
                dphi[iare] = phi[right_cell] - phi[left_cell]
        
        Ap = np.zeros((2, 2))
        Bp = np.zeros(2)
        
        # Internal edges contribution
        for iare in range(Naref, Nare):
            left_cell, right_cell = maillage.get_face_to_elements(iare)
            
            Ap[0, 0] = metriques["dxTri"][iare] * metriques["dxTri"][iare]
            Ap[0, 1] = metriques["dxTri"][iare] * metriques["dyTri"][iare]
            Ap[1, 0] = Ap[0, 1]
            Ap[1, 1] = metriques["dyTri"][iare] * metriques["dyTri"][iare]
            
            Bp[0] = metriques["dxTri"][iare] * dphi[iare]
            Bp[1] = metriques["dyTri"][iare] * dphi[iare]
            
            ATA[left_cell] += Ap
            ATA[right_cell] += Ap
            B[left_cell] += Bp
            B[right_cell] += Bp
        
        # Boundary edges
        for iaref in range(Naref):
            left_cell, _ = maillage.get_face_to_elements(iaref)
            face_tag = maillage.get_boundary_face_to_tag(iaref)
            
            if bcdata is not None and face_tag < len(bcdata):
                bc_type, bc_value = bcdata[face_tag]
                
                if bc_type == 'DIRICHLET':
                    face_nodes = maillage.get_face_to_nodes(iaref)
                    _, y_face = np.mean([maillage.get_node_to_xycoord(node) for node in face_nodes], axis=0)
                    
                    # Handle inlet condition
                    bcvalue = bc_value(y_face) if callable(bc_value) else bc_value
                    dphi[iaref] = bcvalue - phi[left_cell]
                    
                    Ap[0, 0] = metriques["dxTri"][iaref] * metriques["dxTri"][iaref]
                    Ap[0, 1] = metriques["dxTri"][iaref] * metriques["dyTri"][iaref]
                    Ap[1, 0] = Ap[0, 1]
                    Ap[1, 1] = metriques["dyTri"][iaref] * metriques["dyTri"][iaref]
                    
                    Bp[0] = metriques["dxTri"][iaref] * dphi[iaref]
                    Bp[1] = metriques["dyTri"][iaref] * dphi[iaref]
                    
                    ATA[left_cell] += Ap
                    B[left_cell] += Bp
                    
                elif bc_type == 'NEUMANN':
                    # For Neumann BC
                    neumann_value = bc_value if isinstance(bc_value, (int, float)) else bc_value(y_face)
                    
                    normal = metriques["n"][iaref]
                    Bp[0] = metriques["dA"][iaref] * neumann_value * normal[0]
                    Bp[1] = metriques["dA"][iaref] * neumann_value * normal[1]
                    
                    ATA[left_cell] += Ap
                    B[left_cell] += Bp
        
        # Solve local systems
        gradPhi = np.zeros((Ntri, 2))
        for itri in range(Ntri):
            try:
                gradPhi[itri] = np.linalg.solve(ATA[itri], B[itri])
            except np.linalg.LinAlgError:
                print(f"Warning: Singular matrix for triangle {itri}")
                gradPhi[itri] = np.zeros(2)
        
        return gradPhi
        
      
    
    def calcul_champ(self, maillage, metriques, bcdata, fluide, gradPhi, Fi, source, schema):
        """
        Fonction qui calcule les valeur de Phi dans chaque triangle à partir du gradient.
        
        Parameters
        ----------
        maillage : Mesh object
            Maillage (coordonnées et connectivité)
        metriques : dictionnaire
            Dictionnaire contenant les métriques du maillage
        bcdata : list of tuples
            Conditions limites [(type, value), ...]
        fluide : dictionnaire
            Propriétés du fluide (Gamma, Rho, Cp)
        gradPhi : numpy array
            Gradient de Phi au centre des triangles
        Fi : numpy array
            Flux massiques aux centres des aretes
        source : numpy array or double
            terme source
        schema : string
            Schéma de convection ('centré' ou 'upwind')
            
        Returns
        -------
        phi : numpy array
            Champs calculé aux centres des triangles du maillage
        """
        # Initialisation de la matrice
        Ntri = metriques["Ntri"]
        Naref = metriques["Naref"]
        Nare = metriques["Nare"]
        
        A = np.zeros((Ntri, Ntri))
        # Initialisation du membre de droite par le terme source
        B = source * metriques["aireTri"]
        
        # Calculs des paramètres pour la diffusion
        gradPhi_aretes = np.zeros((Nare, 2))
        for iare in range(Nare):
            left_cell, right_cell = maillage.get_face_to_elements(iare)
            if right_cell != -1:
                gradPhi_aretes[iare] = 0.5 * (gradPhi[left_cell] + gradPhi[right_cell])
            else:
                gradPhi_aretes[iare] = gradPhi[left_cell]
        
        dPhidEta = np.sum(gradPhi_aretes * metriques["eEta"], axis=1)
        D = fluide["Gamma"] * metriques["dA"] / (metriques["pNKsi"] * metriques["dKsi"])
        S_DC = (-fluide["Gamma"] * metriques["pKsiEta"] * dPhidEta * metriques["dA"] / metriques["pNKsi"])
        
        # Aretes en frontière
        for iaref in range(Naref):
            left_cell, _ = maillage.get_face_to_elements(iaref)
            face_tag = maillage.get_boundary_face_to_tag(iaref)
            
            if bcdata is not None and face_tag < len(bcdata):
                bc_type, bc_value = bcdata[face_tag]
                
                if bc_type == 'DIRICHLET':
                    face_nodes = maillage.get_face_to_nodes(iaref)
                    _, y_face = np.mean([maillage.get_node_to_xycoord(node) for node in face_nodes], axis=0)
                    
                    if callable(bc_value):
                        bcvalue = bc_value(y_face)
                        _, y1 = maillage.get_node_to_xycoord(face_nodes[0])
                        _, y2 = maillage.get_node_to_xycoord(face_nodes[1])
                        dbcvalue = bc_value(y2) - bc_value(y1)
                    else:
                        bcvalue = bc_value
                        dbcvalue = 0
                    
                    S_DC[iaref] = (-fluide["Gamma"] * metriques["pKsiEta"][iaref] * dbcvalue / metriques["pNKsi"][iaref])
                    A[left_cell, left_cell] += D[iaref] + max(Fi[iaref], 0)
                    B[left_cell] += S_DC[iaref] + (D[iaref] + max(0, -Fi[iaref])) * bcvalue
                    
                elif bc_type == 'NEUMANN':
                    if callable(bc_value):
                        neumann_value = bc_value
                    else:
                        neumann_value = bc_value
                    
                    B[left_cell] += fluide["Gamma"] * metriques["dA"][iaref] * neumann_value
        
        # Aretes internes
        if schema.lower() == "centré":
            for iare in range(Naref, Nare):
                left_cell, right_cell = maillage.get_face_to_elements(iare)
                
                A[left_cell, left_cell] += D[iare] + Fi[iare]/2
                A[right_cell, right_cell] += D[iare] - Fi[iare]/2
                A[left_cell, right_cell] += -D[iare] + Fi[iare]/2
                A[right_cell, left_cell] += -D[iare] - Fi[iare]/2
                B[left_cell] += S_DC[iare]
                B[right_cell] += -S_DC[iare]
        else:  # schema upwind
            for iare in range(Naref, Nare):
                left_cell, right_cell = maillage.get_face_to_elements(iare)
                
                A[left_cell, left_cell] += D[iare] + max(Fi[iare], 0)
                A[right_cell, right_cell] += D[iare] + max(-Fi[iare], 0)
                A[left_cell, right_cell] += -D[iare] - max(-Fi[iare], 0)
                A[right_cell, left_cell] += -D[iare] - max(Fi[iare], 0)
                B[left_cell] += S_DC[iare]
                B[right_cell] += -S_DC[iare]
        
        # Résolution du système matriciel
        AS = sparse.csr_matrix(A)
        phi = spla.spsolve(AS, B)
        
        return phi
    def solve_convection_diffusion(self, maillage, bcdata, fluide, Fi, source, schema, Niter=15):
        """
        Fonction qui résout le problème de convection - diffusion
        
        Parameters
        ----------
        maillage : Mesh object
            Maillage (coordonnées et connectivité)
        bcdata : list of tuples
            Conditions limites [(type, value), ...]
        fluide : dictionnaire
            Propriétés du fluide (Gamma, Rho, Cp)
        Fi : numpy array
            Flux massiques aux centres des aretes
        source : numpy array or double
            terme source
        schema : string
            Schéma de convection (Centré ou Upwind)
        Niter : int, optional
            Nombre d'itération. The default is 6.
            
        Returns
        -------
        phi : numpy array
            Champ aux centres des triangles du maillage
        gradPhi : numpy array
            Gradient du champ au centre des triangles
        """
        # Initialisation
        metriques = self.calcul_metrique(maillage)
        gradPhi = np.zeros((metriques["Ntri"], 2))
        phi = None
        
        # Itérations
        for iter in range(Niter):
            # Calcul des nouvelles valeurs aux Triangles
            phi = self.calcul_champ(maillage, metriques, bcdata, fluide, gradPhi, Fi, source, schema)
            
            # Calcul du nouveau gradient
            gradPhi = self.grad_least_square(maillage, metriques, bcdata, phi)
            
            # Print iteration progress
            if iter < Niter - 1:  # Don't print for last iteration
                error = np.linalg.norm(phi) if iter == 0 else np.linalg.norm(phi - phi_old)
                print(f"Iteration {iter+1}/{Niter}, Error: {error:.2e}")
            
            phi_old = phi.copy()
        
        return phi, gradPhi
    
    def compute_cross_diffusion_vectors(self):
        cross_diff_vectors = np.zeros((self.mesh.get_number_of_faces(), 2))
        for i_face in range(self.mesh.get_number_of_faces()):
            left_cell, right_cell = self.mesh.get_face_to_elements(i_face)
            face_center = np.mean([self.mesh.get_node_to_xycoord(node) for node in self.mesh.get_face_to_nodes(i_face)], axis=0)
            if right_cell != -1:  # Internal face
                cell_vector = self.centre_cellule[right_cell] - self.centre_cellule[left_cell]
                cross_diff_vectors[i_face] = face_center - (self.centre_cellule[left_cell] + 0.5 * cell_vector)
            else:  # Boundary face
                cross_diff_vectors[i_face] = face_center - self.centre_cellule[left_cell]
        return cross_diff_vectors

    # Création des paramètres propres au maillage pour les utiliser plus loin            
    def compute_mesh_properties(self):
        self.face_areas = self.compute_longueur_face()
        self.cell_volumes = self.compute_cell_volumes()
        self.normale_face = self.compute_normale_face()
        self.centre_cellule = self.compute_centre_cellule()
        self.cross_diff_vectors = self.compute_cross_diffusion_vectors() 
        
    # Définition de la surface des cellules 
    def compute_longueur_face(self):
        surface_face= np.zeros(self.mesh.get_number_of_faces())
        for i_face in range(self.mesh.get_number_of_faces()):
            nodes = self.mesh.get_face_to_nodes(i_face)
            x1, y1 = self.mesh.get_node_to_xycoord(nodes[0])
            x2, y2 = self.mesh.get_node_to_xycoord(nodes[1])
            surface_face[i_face] = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return surface_face
    
    # Nouvelle fonction de création des volumes 
    def compute_cell_volumes(self):
        cell_volumes = np.zeros(self.mesh.get_number_of_elements())
        for i_elem in range(self.mesh.get_number_of_elements()):
            nodes = self.mesh.get_element_to_nodes(i_elem)
            if len(nodes) == 3:  # Triangles
                x1, y1 = self.mesh.get_node_to_xycoord(nodes[0])
                x2, y2 = self.mesh.get_node_to_xycoord(nodes[1])
                x3, y3 = self.mesh.get_node_to_xycoord(nodes[2])
                cell_volumes[i_elem] = 0.5 * abs((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))
            elif len(nodes) == 4:  # Quadrilatères
                x1, y1 = self.mesh.get_node_to_xycoord(nodes[0])
                x2, y2 = self.mesh.get_node_to_xycoord(nodes[1])
                x3, y3 = self.mesh.get_node_to_xycoord(nodes[2])
                x4, y4 = self.mesh.get_node_to_xycoord(nodes[3])
                cell_volumes[i_elem] = 0.5 * abs((x3 - x1)*(y4 - y2) - (x4 - x2)*(y3 - y1)) # COM : Correction TPP1
        return cell_volumes

    def compute_normale_face(self):
        normale_face = np.zeros((self.mesh.get_number_of_faces(), 2))
        for i_face in range(self.mesh.get_number_of_faces()):
            nodes = self.mesh.get_face_to_nodes(i_face)
            x1, y1 = self.mesh.get_node_to_xycoord(nodes[0])
            x2, y2 = self.mesh.get_node_to_xycoord(nodes[1])
            nx = y2 - y1
            ny = x1 - x2
            norm = np.sqrt(nx**2 + ny**2)
            normale_face[i_face] = [nx/norm, ny/norm]
        return normale_face

    def compute_centre_cellule(self):
        centre_cellule = np.zeros((self.mesh.get_number_of_elements(), 2))
        for i_elem in range(self.mesh.get_number_of_elements()):
            nodes = self.mesh.get_element_to_nodes(i_elem)
            x_coords = np.array([self.mesh.get_node_to_xcoord(node) for node in nodes])
            y_coords = np.array([self.mesh.get_node_to_ycoord(node) for node in nodes])
            centre_cellule[i_elem] = [np.mean(x_coords), np.mean(y_coords)]
        return centre_cellule
    
    def compute_error(self, T_numerical):
        error = np.zeros_like(T_numerical)
        for i_cell in range(self.mesh.get_number_of_elements()):
            x_cell, y_cell = self.centre_cellule[i_cell]
            error[i_cell] = abs(T_numerical[i_cell] - self.T_mms_func(x_cell, y_cell))
         
        L1_error = np.sum(np.abs(error) * self.cell_volumes)/ (self.mesh.get_number_of_elements())
        L2_error = np.sqrt(np.sum(np.abs(error)**2 * self.cell_volumes))/self.mesh.get_number_of_elements()
        Linf_error = np.max(np.abs(error))
         
        return L1_error, L2_error, Linf_error
    
    def plot_resultats(self, T_numerical, k, scheme):
        nodes = np.array([self.mesh.get_node_to_xycoord(i) for i in range(self.mesh.get_number_of_nodes())])
        elements = np.array([self.mesh.get_element_to_nodes(i) for i in range(self.mesh.get_number_of_elements())])
        

        triangles = []
        triangle_to_element = []
        for i, element in enumerate(elements):
            if len(element) == 3:
                triangles.append(element)
                triangle_to_element.append(i)
            elif len(element) == 4:
                triangles.append([element[0], element[1], element[2]])
                triangles.append([element[0], element[2], element[3]])
                triangle_to_element.extend([i, i])
        
        triangles = np.array(triangles)
        triangle_to_element = np.array(triangle_to_element)
        

        T_triangles = T_numerical[triangle_to_element]
        
        tri = Triangulation(nodes[:, 0], nodes[:, 1], triangles)

        plt.figure(figsize=(10, 8))
        plt.tripcolor(tri, T_triangles)
        plt.colorbar(label='Température')
        plt.title(f'Distribution de température (Pe={1/k}, {scheme} scheme)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()



class ConvectionDiffusionAnalyzer:
    def __init__(self, x_min, x_max, y_min, y_max, lc):
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.solver = ConvectionDiffusionSolver(x_min, x_max, y_min, y_max, lc)
        self.lc = lc
        self.mesh = None
        self.centre_cellule = None
        self.T_mms_values = None

    def generate_mesh(self, lc):
        self.mesh = self.solver.generate_mesh()
        self.solver.compute_mesh_properties()
        return self.mesh

    def plot_results(self, T_numerical, k, scheme):
        grid = self.create_pyvista_grid(T_numerical)
        pl = pvQt.BackgroundPlotter()
        pl.add_mesh(grid, show_edges=True, scalars="Température", cmap="RdBu")
        contours = grid.contour(scalars='Température', isosurfaces=10)
        pl.add_mesh(contours, color="white", line_width=2)
        pl.add_text(f'Distribution de température (Pe={1/k}, {scheme} scheme)', font_size=12)
        pl.show()

    def get_temperature_profile(self, grid, start_point, end_point):
        line = grid.sample_over_line(start_point, end_point, resolution=100)
        return line['Distance'], line['Température']

    def create_pyvista_grid(self, T_values):
        plotter = MeshPlotter()  
        nodes, elements = plotter.prepare_data_for_pyvista(self.mesh)
        points = np.array(nodes)
        cells = elements
        
        cell_types = []
        i = 0
        while i < len(cells):
            num_points = cells[i]
            if num_points == 3:
                cell_types.append(pv.CellType.TRIANGLE)
            elif num_points == 4:
                cell_types.append(pv.CellType.QUAD)
            i += num_points + 1
        
        cell_types = np.array(cell_types)
        
        grid = pv.UnstructuredGrid(cells, cell_types, points)
        if len(T_values) == grid.n_cells:
            grid.cell_data['Température'] = T_values
            grid = grid.cell_data_to_point_data()
        elif len(T_values) == grid.n_points:
            grid.point_data['Température'] = T_values
        
        return grid

   

    def plot_T_mms(self):
        plotter = MeshPlotter()  
        nodes, elements = plotter.prepare_data_for_pyvista(self.mesh)
        points = np.array(nodes)

        x, y = sp.symbols('x y')
        T0, Tx, Txy = 400, 50, 100
        T_mms = T0 + Tx * sp.cos(sp.pi * x) + Txy * sp.sin(sp.pi * x * y)
        
        T_mms_func = sp.lambdify((x, y), T_mms, 'numpy')
        
        self.T_mms_values = T_mms_func(points[:, 0], points[:, 1])
        
        grid = self.create_pyvista_grid(self.T_mms_values)
        
        # pl = pvQt.BackgroundPlotter()
        # pl.add_mesh(grid, scalars='Température', show_edges=True, cmap="RdBu", 
        #             clim=[self.T_mms_values.min(), self.T_mms_values.max()])
        # contours = grid.contour(scalars='Température', isosurfaces=10)
        # pl.add_mesh(contours, color="white", line_width=2)
        # pl.add_text('T_mms Distribution', font_size=12)
        # pl.show()
        
    def plot_cross_section_profiles(self, all_profiles, analytical_profiles, constant_type='x'):
        plt.figure(figsize=(15, 8))
        
        if constant_type == 'x':
            positions = [0, 0.5]  # x positions
            for i, x_pos in enumerate(positions):
                plt.subplot(1, 2, i+1)
                for k, scheme, (distance, temperature) in all_profiles[x_pos]:
                    plt.plot(distance, temperature, label=f'Pe={1/k}, {scheme}')
                
                # Plot analytical solution
                plt.plot(analytical_profiles[x_pos][0], analytical_profiles[x_pos][1], 
                        'k--', label='Analytique')
                
                plt.xlabel('Distance y')
                plt.ylabel('Température')
                plt.title(f'Distribution de température - Coupe à x={x_pos}')
                plt.legend()
                plt.grid(True)
        else:  # constant_type == 'y'
            positions = [0, 0.5]  # y positions
            for i, y_pos in enumerate(positions):
                plt.subplot(1, 2, i+1)
                for k, scheme, (distance, temperature) in all_profiles[y_pos]:
                    plt.plot(distance, temperature, label=f'Pe={1/k}, {scheme}')
                
                # Plot analytical solution
                plt.plot(analytical_profiles[y_pos][0], analytical_profiles[y_pos][1], 
                        'k--', label='Analytique')
                
                plt.xlabel('Distance x')
                plt.ylabel('Température')
                plt.title(f'Distribution de température - Coupe à y={y_pos}')
                plt.legend()
                plt.grid(True)
        
        plt.tight_layout()
        plt.show()

    def run_analysis(self, k_values, schemes, lc, mesh_sizes, bcdata):  # Added bcdata parameter
        self.generate_mesh(lc)
        self.plot_T_mms()
        
        x_min, x_max = 0, 2
        y_min, y_max = 0, 2
        z_min, z_max = 0, 0
        L = x_max - x_min
    
        # Group cross sections by type
        x_sections = {
            0: ("Coupe Verticale à x=0", [0, y_min, z_min], [0, y_max, z_max]),
            0.5: ("Coupe Verticale à x=0.5", [0.5, y_min, z_min], [0.5, y_max, z_max])
        }
        
        y_sections = {
            0: ("Coupe Horizontale à y=0", [x_min, 0, z_min], [x_max, 0, z_max]),
            0.5: ("Coupe Horizontale à y=0.5", [x_min, 0.5, z_min], [x_max, 0.5, z_max])
        }
    
        # Setup fluid properties
        fluide = {
            "Gamma": 1.0,
            "Rho": 1.0,
            "Cp": 1.0
        }
    
        # Calculate face fluxes and source terms
        n_faces = self.mesh.get_number_of_faces()
        Fi = np.zeros(n_faces)
        for i_face in range(n_faces):
            face_nodes = self.mesh.get_face_to_nodes(i_face)
            x_face, y_face = np.mean([self.mesh.get_node_to_xycoord(node) for node in face_nodes], axis=0)
            u = self.solver.u_func(x_face, y_face)
            v = self.solver.v_func(x_face, y_face)
            normale = self.solver.normale_face[i_face]
            Fi[i_face] = fluide["Rho"] * (u * normale[0] + v * normale[1]) * self.solver.face_areas[i_face]
    
        source = np.zeros(self.mesh.get_number_of_elements())
        source_func = self.solver.S_func_generator(fluide["Gamma"])
        for i in range(self.mesh.get_number_of_elements()):
            x, y = self.solver.centre_cellule[i]
            source[i] = source_func(x, y)*0
    
        # Collect profiles for constant x and constant y
        x_profiles = {pos: [] for pos in x_sections}
        y_profiles = {pos: [] for pos in y_sections}
        x_analytical = {}
        y_analytical = {}
    
        # Run simulations and collect profiles
        for k in k_values:
            fluide["Gamma"] = k
            for scheme in schemes:
                phi, gradPhi = self.solver.solve_convection_diffusion(
                    self.mesh, bcdata, fluide, Fi, source, scheme
                )
                self.plot_results(phi, k, scheme)
    
                grid = self.create_pyvista_grid(phi)
                grid_analytical = self.create_pyvista_grid(self.T_mms_values)
    
        #         # Collect constant x profiles
        #         for pos, (name, start, end) in x_sections.items():
        #             distance, temperature = self.get_temperature_profile(grid, start, end)
        #             x_profiles[pos].append((k, scheme, (distance, temperature)))
        #             if pos not in x_analytical:
        #                 x_analytical[pos] = self.get_temperature_profile(grid_analytical, start, end)
    
        #         # Collect constant y profiles
        #         for pos, (name, start, end) in y_sections.items():
        #             distance, temperature = self.get_temperature_profile(grid, start, end)
        #             y_profiles[pos].append((k, scheme, (distance, temperature)))
        #             if pos not in y_analytical:
        #                 y_analytical[pos] = self.get_temperature_profile(grid_analytical, start, end)
    
        # # Plot combined cross sections
        # self.plot_cross_section_profiles(x_profiles, x_analytical, 'x')
        # self.plot_cross_section_profiles(y_profiles, y_analytical, 'y')
    
        # Run convergence study
        errors = self.run_convergence_study(k_values, schemes, mesh_sizes, bcdata)
        max_slopes = self.calculate_max_slope(mesh_sizes, errors)
        self.plot_convergence(mesh_sizes, errors, L)
        self.print_max_slopes(max_slopes)
        

    def run_convergence_study(self, k_values, schemes, mesh_sizes, bcdata):  # Added bcdata parameter
        errors = {scheme: {error_type: {k: [] for k in k_values} for error_type in ['L1', 'L2', 'Linf']} for scheme in schemes}
        
        for lc in mesh_sizes:
            self.solver = ConvectionDiffusionSolver(self.x_min, self.x_max, self.y_min, self.y_max, lc)
            self.generate_mesh(lc)
            
            # Define fluid properties
            fluide = {
                "Gamma": 1.0,
                "Rho": 1.0,
                "Cp": 1.0
            }
            
            # Calculate face fluxes
            n_faces = self.mesh.get_number_of_faces()
            Fi = np.zeros(n_faces)
            for i_face in range(n_faces):
                face_nodes = self.mesh.get_face_to_nodes(i_face)
                x_face, y_face = np.mean([self.mesh.get_node_to_xycoord(node) for node in face_nodes], axis=0)
                u = self.solver.u_func(x_face, y_face)
                v = self.solver.v_func(x_face, y_face)
                normale = self.solver.normale_face[i_face]
                Fi[i_face] = fluide["Rho"] * (u * normale[0] + v * normale[1]) * self.solver.face_areas[i_face]
            
            # Calculate source term
            source = np.zeros(self.mesh.get_number_of_elements())
            source_func = self.solver.S_func_generator(fluide["Gamma"])
            for i in range(self.mesh.get_number_of_elements()):
                x, y = self.solver.centre_cellule[i]
                source[i] = source_func(x, y)*0
            
            for scheme in schemes:
                for k in k_values:
                    fluide["Gamma"] = k  # Adjust diffusion coefficient
                    phi, gradPhi = self.solver.solve_convection_diffusion(
                        self.mesh, bcdata, fluide, Fi, source, scheme
                    )
                    L1_error, L2_error, Linf_error = self.solver.compute_error(phi)
                    
                    errors[scheme]['L1'][k].append(L1_error)
                    errors[scheme]['L2'][k].append(L2_error)
                    errors[scheme]['Linf'][k].append(Linf_error)
        
        return errors

    def calculate_max_slope(self, mesh_sizes, errors):
        max_slopes = {scheme: {error_type: {} for error_type in ['L1', 'L2', 'Linf']} for scheme in errors}
        
        for scheme in errors:
            for error_type in errors[scheme]:
                for k, error_values in errors[scheme][error_type].items():
                    slopes = []
                    for i in range(1, len(mesh_sizes)):
                        dx = np.log(mesh_sizes[i]) - np.log(mesh_sizes[i-1])
                        dy = np.log(error_values[i]) - np.log(error_values[i-1])
                        slope = abs(dy / dx)
                        slopes.append(slope)
                    max_slopes[scheme][error_type][k] = max(slopes)
        
        return max_slopes
    
    def print_max_slopes(self, max_slopes):
        print("\nOrdre de convergence:")
        for scheme in max_slopes:
            print(f"\nSchéma: {scheme}")
            for error_type in max_slopes[scheme]:
                print(f" Erreur {error_type} :")
                for k, slope in max_slopes[scheme][error_type].items():
                    print(f"   Pe={1/k:.0e}: {slope:.2f}")

    def plot_convergence(self, mesh_sizes, errors, L):
        plt.figure(figsize=(15, 5))
        error_types = ['L1', 'L2', 'Linf']
        for i, error_type in enumerate(error_types):
            plt.subplot(1, 3, i+1)
            for scheme in errors:
                for k, error_values in errors[scheme][error_type].items():
                    plt.loglog(mesh_sizes, error_values, '-o', label=f'{scheme}, Pe={1/k}')
            
            plt.xlabel('Taille de cellule')
            plt.ylabel(f'{error_type} Erreur')
            plt.title(f"{error_type} Convergeance de l'erreur")
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()

    def get_analytical_bcdata(self):
        """Set boundary conditions according to the problem"""
        y = sp.symbols('y')
        T_inlet = sp.tanh(-20*(y-1.6)) - sp.tanh(-20*(y-1.4))
        T_inlet_func = sp.lambdify((y,), T_inlet, 'numpy')  # Note the tuple (s,)
        
        bcdata = [
            ('DIRICHLET',  T_inlet_func),  # Inlet
            ('NEUMANN', 0.0),           # Upper curved boundary
            ('NEUMANN', 0.0),           # Lower curved boundary  
            ('NEUMANN', 0.0)            # Outlet
        ]
        
        return bcdata
    def plot_outlet_temperature(self, phi):
        """Plot temperature profile at outlet (R=2)"""
        outlet_faces = []
        x_coords = []
        temps = []
        
        for i_face in range(self.mesh.get_number_of_faces()):
            nodes = self.mesh.get_face_to_nodes(i_face)
            x1, y1 = self.mesh.get_node_to_xycoord(nodes[0])
            x2, y2 = self.mesh.get_node_to_xycoord(nodes[1])
            r1 = np.sqrt(x1**2 + y1**2)
            r2 = np.sqrt(x2**2 + y2**2)
            
            if abs(r1 - self.solver.R2) < 1e-6 and abs(r2 - self.solver.R2) < 1e-6:
                outlet_faces.append(i_face)
                x_mid = (x1 + x2)/2
                y_mid = (y1 + y2)/2
                x_coords.append(y_mid)  # Using y coordinate for x-axis
                
                left_cell, _ = self.mesh.get_face_to_elements(i_face)
                temps.append(phi[left_cell])
        
        sort_idx = np.argsort(x_coords)
        x_coords = np.array(x_coords)[sort_idx]
        temps = np.array(temps)[sort_idx]
        
        plt.figure(figsize=(8, 6))
        plt.plot(x_coords, temps, '-o')
        plt.xlabel('y')
        plt.ylabel('Temperature')
        plt.title('Temperature Profile at Outlet (R=2)')
        plt.grid(True)
        plt.show()
        
      
def main():
    lc_values = [0.2,0.1]
    k_values = [0.1,0.0001]
    schemes = ['centré', 'upwind']
    
    for lc in lc_values:
        analyzer = ConvectionDiffusionAnalyzer(0, 2, 0, 2, lc)
        analyzer.generate_mesh(lc)
        
        for k in k_values:
            fluide = {"Gamma": k, "Rho": 1.0, "Cp": 1.0}
            bcdata = analyzer.get_analytical_bcdata()
            
            for scheme in schemes:
                print(f"\nRunning simulation with lc={lc}, Γ={k}, scheme={scheme}")
                
                mesh = analyzer.mesh
                n_faces = mesh.get_number_of_faces()
                Fi = np.zeros(n_faces)
                
                # Calculate face fluxes
                for i_face in range(n_faces):
                    face_nodes = mesh.get_face_to_nodes(i_face)
                    x_face, y_face = np.mean([mesh.get_node_to_xycoord(node) for node in face_nodes], axis=0)
                    u = analyzer.solver.u_func(x_face, y_face)
                    v = analyzer.solver.v_func(x_face, y_face)
                    normale = analyzer.solver.normale_face[i_face]
                    Fi[i_face] = fluide["Rho"] * (u * normale[0] + v * normale[1]) * analyzer.solver.face_areas[i_face]
                
                source = np.zeros(mesh.get_number_of_elements())
                
                phi, _ = analyzer.solver.solve_convection_diffusion(mesh, bcdata, fluide, Fi, source, scheme)
                analyzer.plot_outlet_temperature(phi)  # Add outlet temperature plot
    analyzer.run_analysis(k_values, schemes, lc, lc_values, analyzer.get_analytical_bcdata())

        

if __name__ == "__main__":
    main()
