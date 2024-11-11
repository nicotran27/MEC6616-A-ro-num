import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import pyvista as pv
import pyvistaqt as pvQt
from mesh import Mesh
from meshGenerator import MeshGenerator
from meshConnectivity import MeshConnectivity
from meshPlotter import MeshPlotter
from typing import Tuple, Dict, List
import matplotlib.collections

class CouetteFlow:
    """
    Classe implémentant le problème de l'écoulement de Couette avec l'interpolation de vitesse de Rhie-Chow.
    """
    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float):
        """
        Initialisation du problème de l'écoulement de Couette.
        
        Paramètres:
        -----------
        x_min, x_max : float
            Limites du domaine en direction x
        y_min, y_max : float
            Limites du domaine en direction y
        """
        # Limites du domaine
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        
        # Propriétés physiques
        self.rho = 1.0  # Masse volumique [kg/m^3]
        self.mu = 0.1  # Viscosité dynamique [N*s/m^2]
        self.U = 1.0    # Vitesse de la paroi supérieure [m/s]
        
        # Propriétés du maillage
        self.mesh = None
        self.face_areas = None
        self.cell_volumes = None
        self.normal_face = None
        self.cell_centers = None

    def generate_mesh(self, mesh_type: str = 'TRI', Nx: float = 3, Ny: float = 3) -> None:
        """
        Génération du maillage de calcul.
        
        Paramètres:
        -----------
        mesh_type : str
            'TRI' pour maillage triangulaire ou 'QUAD' pour maillage quadrilatéral
        lc : float
            Longueur caractéristique du maillage
        """
        mesh_generator = MeshGenerator()
        mesh_parameters = {'mesh_type': mesh_type, 'Nx': Nx,'Ny': Ny}
        self.mesh = mesh_generator.rectangle(
            [self.x_min, self.x_max, self.y_min, self.y_max], 
            mesh_parameters
        )
        mesh_connectivity = MeshConnectivity(self.mesh)
        mesh_connectivity.compute_connectivity()
        self._compute_mesh_properties()
    
    def _compute_mesh_properties(self) -> None:
        """Calcul des propriétés géométriques du maillage."""
        self.face_areas = self._compute_face_areas()
        self.cell_volumes = self._compute_cell_volumes()
        self.normal_face = self._compute_face_normals()
        self.cell_centers = self._compute_cell_centers()
    
    def _compute_face_areas(self) -> np.ndarray:
        """Calcul des aires de toutes les faces."""
        face_areas = np.zeros(self.mesh.get_number_of_faces())
        for i_face in range(self.mesh.get_number_of_faces()):
            nodes = self.mesh.get_face_to_nodes(i_face)
            x1, y1 = self.mesh.get_node_to_xycoord(nodes[0])
            x2, y2 = self.mesh.get_node_to_xycoord(nodes[1])
            face_areas[i_face] = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return face_areas
    
    def _compute_cell_volumes(self) -> np.ndarray:
        """Calcul des volumes (aires en 2D) de toutes les cellules."""
        cell_volumes = np.zeros(self.mesh.get_number_of_elements())
        for i_elem in range(self.mesh.get_number_of_elements()):
            nodes = self.mesh.get_element_to_nodes(i_elem)
            coords = np.array([self.mesh.get_node_to_xycoord(node) for node in nodes])
            if len(nodes) == 3:  # Triangle
                cell_volumes[i_elem] = 0.5 * np.abs(
                    np.cross(coords[1] - coords[0], coords[2] - coords[0])
                )
            elif len(nodes) == 4:  # Quadrilatère
                cell_volumes[i_elem] = 0.5 * np.abs(
                    np.cross(coords[2] - coords[0], coords[3] - coords[1])
                )
        return cell_volumes
    
    def _compute_face_normals(self) -> np.ndarray:
        """Calcul des vecteurs normaux unitaires pour toutes les faces."""
        normal_face = np.zeros((self.mesh.get_number_of_faces(), 2))
        for i_face in range(self.mesh.get_number_of_faces()):
            nodes = self.mesh.get_face_to_nodes(i_face)
            x1, y1 = self.mesh.get_node_to_xycoord(nodes[0])
            x2, y2 = self.mesh.get_node_to_xycoord(nodes[1])
            nx, ny = y2 - y1, x1 - x2  # Vecteur normal
            norm = np.sqrt(nx**2 + ny**2)
            normal_face[i_face] = [nx/norm, ny/norm]
        return normal_face
    
    def _compute_cell_centers(self) -> np.ndarray:
        """Calcul des centres géométriques de toutes les cellules."""
        cell_centers = np.zeros((self.mesh.get_number_of_elements(), 2))
        for i_elem in range(self.mesh.get_number_of_elements()):
            nodes = self.mesh.get_element_to_nodes(i_elem)
            coords = np.array([self.mesh.get_node_to_xycoord(node) for node in nodes])
            cell_centers[i_elem] = np.mean(coords, axis=0)
        return cell_centers
    
    def analytical_solution(self, x: float, y: float, P: float) -> Tuple[float, float]:
        """
        Solution analytique pour l'écoulement de Couette.
        
        Paramètres:
        -----------
        x, y : float
            Coordonnées du point
        P : float
            Paramètre de gradient de pression
            
        Retourne:
        --------
        u, v : float
            Composantes de la vitesse
        """
        u = 10-x  # Solution analytique corrigée
        v = 0.0
        return u, v
    
    def champ_P(self, x: float, y: float, P: float = 0) -> float:
        """
        Calcul du champ de pression basé sur le paramètre P.
        
        Pour P=0: pression nulle partout
        Pour P≠0: pression linéaire en x avec coefficient P
        """
        if P == 0:
            return 0.0
        else:
            # Pression linéaire en direction x, mise à l'échelle par P
            return -P * x

    def assemble_momentum_system(self, P: float, u: np.ndarray, v: np.ndarray) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """
        Assemblage du système linéaire pour les équations de quantité de mouvement.
        
        Paramètres:
        -----------
        P : float
            Paramètre de gradient de pression
        u, v : np.ndarray
            Champ de vitesse actuel
            
        Retourne:
        --------
        A : scipy.sparse.csr_matrix
            Matrice du système
        b : np.ndarray
            Vecteur second membre
        """
        n_cells = self.mesh.get_number_of_elements()
        A = sparse.lil_matrix((2*n_cells, 2*n_cells))
        b = np.zeros(2*n_cells)
        
        # Ajout des termes diffusifs et convectifs
        for i_face in range(self.mesh.get_number_of_faces()):
            left_cell, right_cell = self.mesh.get_face_to_elements(i_face)
            
            if right_cell != -1:  # Face interne
                self._add_internal_face_contribution(A, u, v, left_cell, right_cell, i_face)
            else:  # Face frontière
                self._add_boundary_face_contribution(A, b, left_cell, i_face, P)
        
        # Ajout des termes sources
        b[:n_cells] += -2 * P * self.cell_volumes
        
        return A.tocsr(), b
    
    def _add_internal_face_contribution(self, A: sparse.lil_matrix, u: np.ndarray, v: np.ndarray, 
                                      left_cell: int, right_cell: int, i_face: int) -> None:
        """Ajout des contributions des faces internes à la matrice du système."""
        n_cells = self.mesh.get_number_of_elements()
        
        # Calcul de la vitesse à la face
        u_face = 0.5 * (u[left_cell] + u[right_cell])
        v_face = 0.5 * (v[left_cell] + v[right_cell])
        
        # Vitesse normale à la face
        nx,ny = self.normal_face[i_face]
        vel_normal = u_face * nx + v_face * ny
       
        
        # Terme convectif
        conv_coeff = 0.5 * self.rho * vel_normal * self.face_areas[i_face]
        
        # Terme diffusif
        dx = self.cell_centers[right_cell] - self.cell_centers[left_cell]
        distance = np.linalg.norm(dx)
        diff_coeff = self.mu * self.face_areas[i_face] / distance
        
        # Ajout des contributions à la matrice
        for i in range(2):
            base_idx = i * n_cells
            # Termes diffusifs
            A[base_idx + left_cell, base_idx + right_cell] -= diff_coeff
            A[base_idx + right_cell, base_idx + left_cell] -= diff_coeff
            A[base_idx + left_cell, base_idx + left_cell] += diff_coeff
            A[base_idx + right_cell, base_idx + right_cell] += diff_coeff
            
            # Termes convectifs
            A[base_idx + left_cell, base_idx + right_cell] -= conv_coeff
            A[base_idx + right_cell, base_idx + left_cell] += conv_coeff
            A[base_idx + left_cell, base_idx + left_cell] += conv_coeff
            A[base_idx + right_cell, base_idx + right_cell] -= conv_coeff
    
    def _add_boundary_face_contribution(self, A: sparse.lil_matrix, b: np.ndarray, 
                                      left_cell: int, i_face: int, P: float) -> None:
        """Ajout des contributions des faces de frontieres à la matrice du système."""
        n_cells = self.mesh.get_number_of_elements()
        
     
        x_face, y_face = np.mean([self.mesh.get_node_to_xycoord(node) 
                                 for node in self.mesh.get_face_to_nodes(i_face)], axis=0)
        left_cell, right_cell = self.mesh.get_face_to_elements(i_face)
        nodes = self.mesh.get_face_to_nodes(i_face)
        
        xa,ya =self.mesh.get_node_to_xycoord(nodes[0]) 
        xb,yb =self.mesh.get_node_to_xycoord(nodes[1]) 
        
        delta_yi = yb -ya
        delta_xi = xb- xa
        xtg,ytg = self.cell_centers[left_cell]
        xtd,ytd =self.cell_centers[right_cell]
        
        PNKSI = 1
                  
         
        
        
        if np.isclose(y_face, self.y_min) or np.isclose(y_face, self.y_max) or np.isclose(x_face, self.x_min) :
            # Dirichlet
            u_boundary, v_boundary = self.analytical_solution(x_face, y_face, P)
            diff_coeff = self.mu * self.face_areas[i_face] / (0.5 * self.face_areas[i_face])
            A[left_cell, left_cell] += diff_coeff
            A[n_cells + left_cell, n_cells + left_cell] += diff_coeff
            b[left_cell] += diff_coeff * u_boundary
            b[n_cells + left_cell] += diff_coeff * v_boundary
    
    def _least_squares_gradient(self, field_values: np.ndarray) -> np.ndarray:
        """
        Calcule les gradients en utilisant la méthode des moindres carrés pondérés.
        
        Paramètres:
        -----------
        field_values : np.ndarray
            Valeurs du champ aux centres des cellules
            
        Retourne:
        --------
        np.ndarray
            Gradients aux centres des cellules (forme: n_elements x 2)
        """
        n_elements = self.mesh.get_number_of_elements()
        ATA = np.zeros((n_elements, 2, 2))
        B = np.zeros((n_elements, 2))
        
        # Boucle sur les faces
        for i_face in range(self.mesh.get_number_of_faces()):
            left_cell, right_cell = self.mesh.get_face_to_elements(i_face)
            
            if right_cell != -1:  # Face interne
                # Vecteur entre les centres des cellules
                dx = self.cell_centers[right_cell] - self.cell_centers[left_cell]
                distance = np.linalg.norm(dx)
                
                # Poids basé sur l'inverse de la distance
                weight = 1.0 / distance
                
                # Matrice de contribution locale
                A_local = weight * np.outer(dx, dx)
                
                # Ajouter les contributions aux deux cellules
                ATA[left_cell] += A_local
                ATA[right_cell] += A_local
                
                # Différence des valeurs du champ
                df = field_values[right_cell] - field_values[left_cell]
                
                # Ajouter les contributions pondérées à B
                B[left_cell] += weight * df * dx
                B[right_cell] += weight * df * dx
                
            else:  # Face frontière
                # Obtenir les coordonnées du centre de la face
                face_nodes = self.mesh.get_face_to_nodes(i_face)
                x_face = 0.5 * (self.mesh.get_node_to_xcoord(face_nodes[0]) + 
                               self.mesh.get_node_to_xcoord(face_nodes[1]))
                y_face = 0.5 * (self.mesh.get_node_to_ycoord(face_nodes[0]) + 
                               self.mesh.get_node_to_ycoord(face_nodes[1]))
                
                # Vecteur du centre de la cellule au centre de la face
                dx = np.array([x_face, y_face]) - self.cell_centers[left_cell]
                distance = np.linalg.norm(dx)
                
                weight = 1.0 / distance
                A_local = weight * np.outer(dx, dx)
                
                # Ajouter la contribution à la cellule
                ATA[left_cell] += A_local
                
                # Pour le gradient de pression, on utilise un gradient nul aux frontières
                face_value = field_values[left_cell]
                df = face_value - field_values[left_cell]
                B[left_cell] += weight * df * dx
        
        # Calculer les gradients
        gradients = np.zeros((n_elements, 2))
        for i in range(n_elements):
            try:
                # Ajouter une petite régularisation pour éviter les matrices singulières
                ATA[i] += 1e-10 * np.eye(2)
                gradients[i] = np.linalg.solve(ATA[i], B[i])
            except np.linalg.LinAlgError:
                print(f"Attention: Matrice singulière à l'élément {i}")
                gradients[i] = B[i] / (np.trace(ATA[i]) + 1e-10)
        
        return gradients
    
    def relaxation(self, u_new: np.ndarray, v_new: np.ndarray, 
              u_prec: np.ndarray, v_prec: np.ndarray, 
              alpha: float = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applique une sous-relaxation sur les champs de vitesse.
        
        Paramètres:
        -----------
        u_new, v_new : np.ndarray
            Composantes de vitesse nouvellement calculées
        u_prec, v_prec : np.ndarray
            Composantes de vitesse de l'itération précédente
        alpha : float, optional (default=0.7)
            Facteur de sous-relaxation (entre 0 et 1)
            
        Retourne:
        --------
        Tuple[np.ndarray, np.ndarray]
            Composantes de vitesse sous-relaxées (u_final, v_final)
        """
        u_final = alpha * u_new + (1 - alpha) * u_prec
        v_final = alpha * v_new + (1 - alpha) * v_prec
        return u_final, v_final

    def Rhie_Chow(self, P: float = 0, max_iterations: int = 1000, 
              tolerance: float = 1e-6, alpha: float = 1,P_sortie = 0) -> np.ndarray:
        """
        Interpolation de Rhie-Chow modifiée avec sous-relaxation.
        
        Paramètres:
        -----------
        P : float, optional (default=0)
            Paramètre de pression
        max_iterations : int, optional (default=1000)
            Nombre maximum d'itérations
        tolerance : float, optional (default=1e-6)
            Critère de convergence
        alpha : float, optional (default=0.7)
            Facteur de sous-relaxation
                
        Retourne:
        --------
        np.ndarray
            Vitesses normales aux faces
        """
        # Initialisation des champs de vitesse
        n_elements = self.mesh.get_number_of_elements()
        #u = self.u_lap_6()
        u = np.zeros(n_elements)
        v = np.zeros(n_elements)
        
        # Boucle principale d'itération
        for iteration in range(max_iterations):
            # Obtenir les coefficients de l'équation de quantité de mouvement
            A, b = self.assemble_momentum_system(P, u, v)
            aP = A.diagonal()[:n_elements]
            
            # Calculer la nouvelle solution
            solution = spla.spsolve(A, b)
            u_new = solution[:n_elements]
            v_new = solution[n_elements:]
            
            # Appliquer la sous-relaxation
            u, v = self.relaxation(u_new, v_new, u, v, alpha)
            
            # Vérifier la convergence
            if (np.max(np.abs(u_new - u)) < tolerance and 
                np.max(np.abs(v_new - v)) < tolerance):
                print(f"Convergence atteinte après {iteration + 1} itérations")
                break
                
            if iteration == max_iterations - 1:
                print("Attention: Nombre maximum d'itérations atteint sans convergence")
    
       
        # print("u : " ,u)
        #print("v : " ,v)
        # Calculer le champ de pression
        P_field = np.zeros(n_elements)
        for i_elem in range(n_elements):
            x, y = self.cell_centers[i_elem]
            P_field[i_elem] = -P * x  # Champ de pression linéaire
        
        # Calculer les gradients de pression en utilisant les moindres carrés
        grad_P = self._least_squares_gradient(P_field)
        
        
        # Initialiser le tableau des vitesses aux faces
        U_face = np.zeros(self.mesh.get_number_of_faces())
        
        # Calculer les vitesses aux faces
        for i_face in range(self.mesh.get_number_of_faces()):
            left_cell, right_cell = self.mesh.get_face_to_elements(i_face)
            nx, ny = self.normal_face[i_face]
           
            if right_cell != -1:  # Face interne
                # Interpolation simple de la vitesse
                u_avg = 0.5 * (u[left_cell] + u[right_cell])
                v_avg = 0.5 * (v[left_cell] + v[right_cell])
                U_avg = u_avg * nx + v_avg * ny
                #print("u_avg", u_avg)
                #print("nx",nx)
                
                if P != 0:  # Appliquer la correction de pression uniquement si le gradient existe
                    # Calculer la distance entre les centres des cellules
                    dx = self.cell_centers[right_cell] - self.cell_centers[left_cell]
                    distance = np.linalg.norm(dx)
                    #print("distance : ", distance)
                    
                    # Coefficient pondéré par le volume
                    vol_aP_avg = 0.5 * (self.cell_volumes[left_cell]/aP[left_cell] + 
                                       self.cell_volumes[right_cell]/aP[right_cell])
                    
                    # Termes de gradient de pression utilisant les gradients par moindres carrés
                    dP = (P_field[right_cell] - P_field[left_cell]) / distance
                    grad_P_avg = 0.5 * (
                        (grad_P[left_cell, 0] * nx + grad_P[left_cell, 1] * ny) +
                        (grad_P[right_cell, 0] * nx + grad_P[right_cell, 1] * ny)
                    )
                    
                    # Ajouter la correction de pression
                    U_face[i_face] = U_avg - vol_aP_avg * (dP - grad_P_avg)
                else:
                    U_face[i_face] = U_avg
                    
            else:  # Face frontière
                # Obtenir les coordonnées du centre de la face
                x_face, y_face = np.mean([self.mesh.get_node_to_xycoord(node) 
                                        for node in self.mesh.get_face_to_nodes(i_face)], axis=0)
                
                if np.isclose(y_face, self.y_min) or np.isclose(y_face, self.y_max) or np.isclose(x_face,self.x_min) :
                    # Condition limite de Dirichlet - utiliser la vitesse exacte 
                    # FAce ouest, nord ,sud 
                    u_bound, v_bound = self.analytical_solution(x_face, y_face, P)
                   
                    U_face[i_face] = u_bound * nx + v_bound * ny
                    #print("U face ",U_face[i_face])
                    #print("u_limit : ",u_bound)
                    #print(nx)
                    #print("Ubound", U_face[i_face])
                elif np.isclose(x_face,self.x_max) : 
                    # Face est 
                    dx = np.array([x_face, y_face]) - self.cell_centers[left_cell]
                    xtg,ytg = self.cell_centers[left_cell]
                    distance = np.linalg.norm(dx)
                    # Condition limite de Neumann - utiliser les valeurs au centre de la cellule
                    
                    U_face[i_face] = u[left_cell] * nx + v[left_cell] * ny  + self.cell_volumes[left_cell]/aP[left_cell]*((P_field[left_cell]-P_sortie)/distance) + self.cell_volumes[left_cell]/aP[left_cell]*(grad_P[left_cell][0]*(x_face-xtg)+grad_P[left_cell][1]*(y_face-ytg)) 
                    
                    
        #print("U_face", U_face)
        
        return U_face, aP
    
    def Correction_pression(self, U_face, aP, P: float = 0, max_iterations: int = 1000, 
          tolerance: float = 1e-6, alpha: float = 1) -> np.ndarray:
        """
        Interpolation de Rhie-Chow modifiée avec sous-relaxation.
        
        Paramètres:
        -----------
        P : float, optional (default=0)
            Paramètre de pression
        max_iterations : int, optional (default=1000)
            Nombre maximum d'itérations
        tolerance : float, optional (default=1e-6)
            Critère de convergence
        alpha : float, optional (default=0.7)
            Facteur de sous-relaxation
                
        Retourne:
        --------
        np.ndarray
            Vitesses normales aux faces
        """
        # Initialisation
        n_elements = self.mesh.get_number_of_elements()
        n_faces = self.mesh.get_number_of_faces()
        
        # Création des matrices B et NP 
        NP = sparse.lil_matrix((n_elements, n_elements))
        B = np.zeros(n_elements)
        
        # Assemblage
        for i_face in range(n_faces): 
            left_cell, right_cell = self.mesh.get_face_to_elements(i_face)
            
            ap = aP[left_cell]
            
            nodes = self.mesh.get_face_to_nodes(i_face)
            xa, ya = self.mesh.get_node_to_xycoord(nodes[0])
            xb, yb = self.mesh.get_node_to_xycoord(nodes[1])
            delta_Ai = np.sqrt((xb-xa)**2 + (yb-ya)**2)
            
            if right_cell != -1:  # Face interne
                aa = aP[right_cell]
                dx = self.cell_centers[right_cell] - self.cell_centers[left_cell]
                distance = np.linalg.norm(dx)
                
                # Moyenne harmonique pour dfi
                dfi = (self.cell_volumes[left_cell] * self.cell_volumes[right_cell]) / \
                      (distance * (self.cell_volumes[left_cell] * aa + 
                                 self.cell_volumes[right_cell] * ap))
                
                NP[left_cell, left_cell] += self.rho * dfi * delta_Ai
                NP[left_cell, right_cell] -= self.rho * dfi * delta_Ai
                NP[right_cell, right_cell] += self.rho * dfi * delta_Ai
                NP[right_cell, left_cell] -= self.rho * dfi * delta_Ai
                
                B[left_cell] -= self.rho * U_face[i_face] * delta_Ai
                B[right_cell] += self.rho * U_face[i_face] * delta_Ai
                
            else:  # Face limite
                x_face, y_face = np.mean([self.mesh.get_node_to_xycoord(node) 
                                        for node in self.mesh.get_face_to_nodes(i_face)], axis=0)
                
                if np.isclose(x_face, self.x_min) or np.isclose(y_face, self.y_max) or np.isclose(y_face, self.y_min):
                    B[left_cell] -= self.rho * U_face[i_face] * delta_Ai
                    
                elif np.isclose(x_face, self.x_max):
                    dx = np.array([x_face, y_face]) - self.cell_centers[left_cell]
                    distance = np.linalg.norm(dx)
                    dfi = self.cell_volumes[left_cell]/(ap*distance)
                    NP[left_cell, left_cell] += self.rho * dfi * delta_Ai
                    B[left_cell] -= self.rho * U_face[i_face] * delta_Ai
        
        # Format de matrice
        NP = NP.tocsr()
      
        # Resolution
        pression = spla.spsolve(NP, B)
        
        # Correction des vitesses
        U_final = np.zeros(n_faces)
        for i_face in range(n_faces):
            left_cell, right_cell = self.mesh.get_face_to_elements(i_face)
            ap = aP[left_cell]
            
            nodes = self.mesh.get_face_to_nodes(i_face)
            xa, ya = self.mesh.get_node_to_xycoord(nodes[0])
            xb, yb = self.mesh.get_node_to_xycoord(nodes[1])
            
            if right_cell != -1:  # Face interne
                aa = aP[right_cell]
                dx = self.cell_centers[right_cell] - self.cell_centers[left_cell]
                distance = np.linalg.norm(dx)
                
                # Use same interpolation as in pressure equation
                # Moyenne harmonique et non linéaire 
                dfi = (self.cell_volumes[left_cell] * self.cell_volumes[right_cell]) / \
                      (distance * (self.cell_volumes[left_cell] * aa + 
                                 self.cell_volumes[right_cell] * ap))
                
                U_final[i_face] = U_face[i_face] + dfi * (pression[left_cell] - pression[right_cell])
                
            else:  # Face limite
                x_face, y_face = np.mean([self.mesh.get_node_to_xycoord(node) 
                                        for node in self.mesh.get_face_to_nodes(i_face)], axis=0)
                
                if np.isclose(x_face, self.x_max):
                    dx = np.array([x_face, y_face]) - self.cell_centers[left_cell]
                    distance = np.linalg.norm(dx)
                    dfi = self.cell_volumes[left_cell]/(ap*distance)
                    U_final[i_face] = U_face[i_face] + dfi * pression[left_cell]
                    
                else:
                    U_final[i_face] = U_face[i_face]
        
        return pression, U_final
                
                    
        
        def Calcul_divergence (self,U_face) :
            n_elements = self.mesh.get_number_of_elements()
            n_faces = self.mesh.get_number_of_faces()
            
            divergence = np.zeros((n_elements,1))
            for i_elem in range(n_elements):
                
                divergence_face_i = 0
                faces = self.mesh.get_face_to_elements(i_elem)
                
                for i_face in faces:
                    
                    nodes = self.mesh.get_face_to_nodes(i_face)
                    xa,ya = self.mesh.get_node_to_xycoord(nodes[0])
                    xb,yb = self.mesh.get_node_to_xycoord(nodes[1])
                    delta_Ai = ((xb-xa)**2+(yb-ya)**2)**(1/2)
                    
                    divergence_face_i += self.rho*U_face[i_face]*delta_Ai
                    
                divergence[i_elem] = divergence_face_i
                
                
            return divergence
                    
    def calculate_simple_interpolation(self, P: float) -> np.ndarray:
        """
        Calcul des vitesse par moyenne lineaire
        
        Parameters:
        -----------
        P : float
            Pression
            
        Returns:
        --------
        np.ndarray
            Vitesse aux face utilisant l'interpolation
        """
        # Initialisation
        U_simple = np.zeros(self.mesh.get_number_of_faces())
        
        # Vitesse au centre des cellules
        A, b = self.assemble_momentum_system(P, np.zeros(self.mesh.get_number_of_elements()), 
                                           np.zeros(self.mesh.get_number_of_elements()))
        solution = spla.spsolve(A, b)
        u = solution[:self.mesh.get_number_of_elements()]
        v = solution[self.mesh.get_number_of_elements():]
        
        # Calcul aux faces
        for i_face in range(self.mesh.get_number_of_faces()):
            left_cell, right_cell = self.mesh.get_face_to_elements(i_face)
            nx, ny = self.normal_face[i_face]
            
            if right_cell != -1:  # Face interne
                u_avg = 0.5 * (u[left_cell] + u[right_cell])
                v_avg = 0.5 * (v[left_cell] + v[right_cell])
                U_simple[i_face] = u_avg * nx + v_avg * ny
            else:  # Face limite
                x_face, y_face = np.mean([self.mesh.get_node_to_xycoord(node) 
                                        for node in self.mesh.get_face_to_nodes(i_face)], axis=0)
                if np.isclose(y_face, self.y_min) or np.isclose(y_face, self.y_max):
                    u_bound, v_bound = self.analytical_solution(x_face, y_face, P)
                    U_simple[i_face] = u_bound * nx + v_bound * ny
        
        return U_simple
    
    def Calcul_divergence (self,U_face) :
        """
        Calcul de la divergence
        
        Parameters:
        -----------
        U_face : 
        np.ndarray
            Vitesses normales aux facesn
            
        Returns:
        --------
        np.ndarray
            Divergence de chaque cellule
        """
        n_elements = self.mesh.get_number_of_elements()
        n_faces = self.mesh.get_number_of_faces()
        
        divergence = np.zeros(n_elements)
        for i_face in range(n_faces):
            el1,el2  = self.mesh.get_face_to_elements(i_face)
            
            nodes = self.mesh.get_face_to_nodes(i_face)
            xa,ya = self.mesh.get_node_to_xycoord(nodes[0])
            xb,yb = self.mesh.get_node_to_xycoord(nodes[1])
            delta_Ai = ((xb-xa)**2+(yb-ya)**2)**(1/2)
            
            flux = self.rho*U_face[i_face]*delta_Ai
            flux = flux.item()
            divergence[el1] =  divergence[el1]+ flux
            if el2 != -1 : 
                divergence[el2] = divergence[el2] - flux
            
        return divergence
    
class VelocityFieldPlotter:
    def __init__(self, solver):
        self.solver = solver
        
    def plot_all_fields(self, U_face_initial: np.ndarray, U_face_corrected: np.ndarray, 
                       P_prime: np.ndarray, title: str = "Champs d'écoulement"):
        """
        Affiche les champs de vitesse et la correction de pression.
        
        Paramètres:
        -----------
        U_face_initial : np.ndarray
            Vitesses initiales aux faces (Rhie-Chow)
        U_face_corrected : np.ndarray
            Vitesses corrigées aux faces après correction de pression
        P_prime : np.ndarray
            Champ de correction de pression
        title : str
            Titre du graphique
        """
        fig = plt.figure(figsize=(18, 6))
        gs = plt.GridSpec(1, 3, figure=fig)
        
        # Affichage du champ de vitesse initial
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_velocity_field(ax1, U_face_initial.flatten(), "Champ de vitesse initial")
        
        # Affichage du champ de vitesse corrigé
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_velocity_field(ax2, U_face_corrected.flatten(), "Champ de vitesse corrigé")
        
        # Affichage de la correction de pression
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_pressure_field(ax3, P_prime, "Correction de pression")
        
        # Ajout du titre général
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        return fig
        
    def _plot_velocity_field(self, ax, U_face: np.ndarray, subtitle: str):
        """Affichage d'un champ de vitesse"""
        # Tracé des éléments du maillage
        for i_elem in range(self.solver.mesh.get_number_of_elements()):
            nodes = self.solver.mesh.get_element_to_nodes(i_elem)
            xy_coords = []
            for node in nodes:
                x, y = self.solver.mesh.get_node_to_xycoord(node)
                xy_coords.append([x, y])
            xy_coords = np.array(xy_coords)
            xy_coords = np.vstack((xy_coords, xy_coords[0]))
            ax.plot(xy_coords[:, 0], xy_coords[:, 1], 'k-', linewidth=0.5)
        
        # Tracé des vecteurs vitesse
        for i_face in range(self.solver.mesh.get_number_of_faces()):
            nodes = self.solver.mesh.get_face_to_nodes(i_face)
            xy_coords = []
            for node in nodes:
                x, y = self.solver.mesh.get_node_to_xycoord(node)
                xy_coords.append([x, y])
            face_center = np.mean(xy_coords, axis=0)
            
            nx, ny = self.solver.normal_face[i_face]
            velocity = U_face[i_face]
            
            scale = 0.2
            dx = velocity * nx * scale
            dy = velocity * ny * scale
            
            # Utilisation de quiver pour une meilleure visualisation des vecteurs
            ax.quiver(face_center[0], face_center[1], dx, dy,
                     angles='xy', scale_units='xy', scale=1,
                     color='blue', width=0.005)
        
        self._set_plot_properties(ax, subtitle)
    
    def _plot_pressure_field(self, ax, P_prime: np.ndarray, subtitle: str):
        """Affichage du champ de correction de pression"""
        # Création des patches pour la visualisation de la pression
        patches = []
        p_values = []
        for i_elem in range(self.solver.mesh.get_number_of_elements()):
            nodes = self.solver.mesh.get_element_to_nodes(i_elem)
            xy_coords = []
            for node in nodes:
                x, y = self.solver.mesh.get_node_to_xycoord(node)
                xy_coords.append([x, y])
            patches.append(plt.Polygon(xy_coords))
            p_values.append(abs(P_prime[i_elem]))
        
        # Création de la collection et ajout au graphique
        p_collection = matplotlib.collections.PatchCollection(patches)
        p_collection.set_array(np.array(p_values))
        ax.add_collection(p_collection)
        plt.colorbar(p_collection, ax=ax, label='Correction de pression')
        
        self._set_plot_properties(ax, subtitle)
    
    def _set_plot_properties(self, ax, subtitle: str):
        """Configuration des propriétés communes des graphiques"""
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(subtitle)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        x_min, x_max = self.solver.x_min, self.solver.x_max
        y_min, y_max = self.solver.y_min, self.solver.y_max
        padding = 0.1 * max(x_max - x_min, y_max - y_min)
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)

def main():
    """Fonction principale pour tester et visualiser les champs avec différentes tailles de maillage"""
    # Création de l'instance du solveur
    flow = CouetteFlow(0, 10, 0, 10)
    
    # Test de différentes configurations de maillage
    mesh_parameters = [["QUAD",3],["TRI",3]]
    for i in range(2): 
        if 0==0 : 
            Nx = mesh_parameters[i][1]
            mesh_type = mesh_parameters[i][0]
            Ny = Nx
            
            # Génération du maillage avec la taille spécifiée
            flow.generate_mesh(mesh_type=mesh_type, Nx=Nx, Ny=Ny)
            
            # Calcul du champ de vitesse initial
            U_face_initial,aP = flow.Rhie_Chow(0)
            #print("Uface",U_face_initial)
            # Initialisation du champ de vitesse de base pour la correction
            n_elements = flow.mesh.get_number_of_elements()
            u = np.zeros(n_elements)
            v = np.zeros(n_elements)
            
            # Calcul de la correction de pression et des vitesses corrigées
            P_prime, U_face_corrected = flow.Correction_pression(U_face_initial, aP)
            #print("Uface",U_face_corrected)
            # Visualisation des résultats
            plotter = VelocityFieldPlotter(flow)
            fig = plotter.plot_all_fields(
                U_face_initial, 
                U_face_corrected,
                P_prime,
                f"Champs d'écoulement - Maillage {mesh_type} (Nx={Nx}, Ny={Ny})"
            )
            
            # Affichage des statistiques de divergence
            div_initial = flow.Calcul_divergence(U_face_initial)
            print("Divergence initiale:", div_initial)
            div_corrected = flow.Calcul_divergence(U_face_corrected)
            print("Divergence corrigée:", div_corrected)
            print(f"Divergence maximale initiale: {np.max(np.abs(div_initial)):.2e}")
            print(f"Divergence maximale corrigée: {np.max(np.abs(div_corrected)):.2e}")
            print(f"Facteur d'amélioration: {np.max(np.abs(div_initial))/np.max(np.abs(div_corrected)):.2f}x")

if __name__ == "__main__":
    main()