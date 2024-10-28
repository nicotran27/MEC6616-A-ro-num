
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
        self.mu = 1.0   # Viscosité dynamique [N*s/m^2]
        self.U = 1.0    # Vitesse de la paroi supérieure [m/s]
        
        # Propriétés du maillage
        self.mesh = None
        self.face_areas = None
        self.cell_volumes = None
        self.normal_face = None
        self.cell_centers = None

    def generate_mesh(self, mesh_type: str = 'TRI', lc: float = 0.1) -> None:
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
        mesh_parameters = {'mesh_type': mesh_type, 'lc': lc}
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
        u = y - 0.5 * P * y * (1.0 - y)  # Solution analytique corrigée
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
        vel_normal = (u_face * self.normal_face[i_face, 0] + 
                     v_face * self.normal_face[i_face, 1])
        
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
        
        if np.isclose(y_face, self.y_min) or np.isclose(y_face, self.y_max):
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
        # Initialisation des champs de vitesse
        n_elements = self.mesh.get_number_of_elements()
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
                
                if P != 0:  # Appliquer la correction de pression uniquement si le gradient existe
                    # Calculer la distance entre les centres des cellules
                    dx = self.cell_centers[right_cell] - self.cell_centers[left_cell]
                    distance = np.linalg.norm(dx)
                    
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
                
                if np.isclose(y_face, self.y_min) or np.isclose(y_face, self.y_max):
                    # Condition limite de Dirichlet - utiliser la vitesse exacte
                    u_bound, v_bound = self.analytical_solution(x_face, y_face, P)
                    U_face[i_face] = u_bound * nx + v_bound * ny
                else:
                    # Condition limite de Neumann - utiliser les valeurs au centre de la cellule
                    U_face[i_face] = u[left_cell] * nx + v[left_cell] * ny
        
        return U_face

    
    def calculate_simple_interpolation(self, P: float) -> np.ndarray:
        """
        Calcul des vitesse par moyenne lin/aire
        
        Parameters:
        -----------
        P : float
            Pression
            
        Returns:
        --------
        np.ndarray
            Vitesse aux face utilisant l'interpolation
        """
        # Initialize
        U_simple = np.zeros(self.mesh.get_number_of_faces())
        
        # Get cell-centered velocities
        A, b = self.assemble_momentum_system(P, np.zeros(self.mesh.get_number_of_elements()), 
                                           np.zeros(self.mesh.get_number_of_elements()))
        solution = spla.spsolve(A, b)
        u = solution[:self.mesh.get_number_of_elements()]
        v = solution[self.mesh.get_number_of_elements():]
        
        # Calculate at each face
        for i_face in range(self.mesh.get_number_of_faces()):
            left_cell, right_cell = self.mesh.get_face_to_elements(i_face)
            nx, ny = self.normal_face[i_face]
            
            if right_cell != -1:  # Internal face
                u_avg = 0.5 * (u[left_cell] + u[right_cell])
                v_avg = 0.5 * (v[left_cell] + v[right_cell])
                U_simple[i_face] = u_avg * nx + v_avg * ny
            else:  # Boundary face
                x_face, y_face = np.mean([self.mesh.get_node_to_xycoord(node) 
                                        for node in self.mesh.get_face_to_nodes(i_face)], axis=0)
                if np.isclose(y_face, self.y_min) or np.isclose(y_face, self.y_max):
                    u_bound, v_bound = self.analytical_solution(x_face, y_face, P)
                    U_simple[i_face] = u_bound * nx + v_bound * ny
        
        return U_simple
    
    def test_rhie_chow_method(self) -> Dict:
       """
       Test de l'implémentation de Rhie-Chow avec des tolérances appropriées.
       
       Retourne:
       --------
       Dict
           Résultats des tests pour chaque cas
       """
       test_cases = {
           'mesh_types': ['TRI', 'QUAD'],
           'P_values': [0, 3],
           'nx': 8,
           'ny': 8
       }
       
       tolerances = {
           0: 1e-10,  # Tolérance  pour P=0
           3: 1e-10    # Tolérance  pour P=3
       }
       
       results = {}
       
       for mesh_type in test_cases['mesh_types']:
           for P in test_cases['P_values']:
               print(f"\nTest avec maillage {mesh_type} et P={P}")
               
               # Génération du maillage et calcul des vitesses
               self.generate_mesh(mesh_type=mesh_type, lc=1/test_cases['nx'])
               U_RC = self.Rhie_Chow(P)
               U_simple = self.calculate_simple_interpolation(P)
               
               # Comparaison uniquement pour les faces internes
               internal_faces = [i for i in range(self.mesh.get_number_of_faces()) 
                               if self.mesh.get_face_to_elements(i)[1] != -1]
               
               diff = np.abs(U_RC[internal_faces] - U_simple[internal_faces])
               max_diff = np.max(diff)
               avg_diff = np.mean(diff)
               
               # Stockage des résultats
               results[f'{mesh_type}_P{P}'] = {
                   'max_difference': max_diff,
                   'average_difference': avg_diff,
                   'passed': max_diff < tolerances[P],
                   'tolerance_used': tolerances[P]
               }


       
       return results
   
    # def create_pyvista_grid(self, u_values: np.ndarray, v_values: np.ndarray) -> pv.UnstructuredGrid:
    #    """
    #    Création d'une grille PyVista pour la visualisation.
       
    #    Paramètres:
    #    -----------
    #    u_values, v_values : np.ndarray
    #        Composantes de la vitesse
           
    #    Retourne:
    #    --------
    #    pv.UnstructuredGrid
    #        Grille pour la visualisation
    #    """
    #    # Préparation des données pour PyVista
    #    plotter = MeshPlotter()
    #    nodes, elements = plotter.prepare_data_for_pyvista(self.mesh)
    #    points = np.array(nodes)
    #    cells = elements
       
    #    # Définition des types de cellules
    #    cell_types = []
    #    i = 0
    #    while i < len(cells):
    #        num_points = cells[i]
    #        cell_types.append(pv.CellType.TRIANGLE if num_points == 3 else pv.CellType.QUAD)
    #        i += num_points + 1
       
    #    # Création de la grille
    #    grid = pv.UnstructuredGrid(cells, np.array(cell_types), points)
       
    #    # Ajout des données de vitesse
    #    grid.cell_data['u'] = u_values
    #    grid.cell_data['v'] = v_values
    #    grid.cell_data['velocity_magnitude'] = np.sqrt(u_values**2 + v_values**2)
       
    #    return grid

    # def create_analytical_grid(self, P: float) -> pv.UnstructuredGrid:
    #    """
    #    Création d'une grille PyVista pour la solution analytique.
       
    #    Paramètres:
    #    -----------
    #    P : float
    #        Paramètre de gradient de pression
           
    #    Retourne:
    #    --------
    #    pv.UnstructuredGrid
    #        Grille avec la solution analytique
    #    """
    #    grid = self.create_pyvista_grid(
    #        np.zeros(self.mesh.get_number_of_elements()),
    #        np.zeros(self.mesh.get_number_of_elements())
    #    )
       
    #    # Calcul de la solution analytique
    #    u_analytical = np.zeros(self.mesh.get_number_of_elements())
    #    v_analytical = np.zeros(self.mesh.get_number_of_elements())
       
    #    for i in range(self.mesh.get_number_of_elements()):
    #        x, y = self.cell_centers[i]
    #        u_analytical[i], v_analytical[i] = self.analytical_solution(x, y, P)
       
    #    grid.cell_data['u'] = u_analytical
    #    grid.cell_data['v'] = v_analytical
    #    grid.cell_data['velocity_magnitude'] = np.sqrt(u_analytical**2 + v_analytical**2)
       
    #    return grid

    # def plot_results(self, results: Dict) -> None:
    #    """
    #    Visualisation des résultats avec PyVista.
       
    #    Paramètres:
    #    -----------
    #    results : Dict
    #        Dictionnaire contenant les résultats
    #    """
    #    for result in results:
    #        lc, P, scheme = result['lc'], result['P'], result['scheme']
    #        u, v = result['u'], result['v']

    #        # Génération des grilles
    #        self.generate_mesh(lc=lc)
    #        numerical_grid = self.create_pyvista_grid(u, v)
    #        analytical_grid = self.create_analytical_grid(P)
           
    #        # Conversion des données de cellules en données de points
    #        numerical_grid = numerical_grid.cell_data_to_point_data()
    #        analytical_grid = analytical_grid.cell_data_to_point_data()
           
    #        # Création du plotter
    #        pl = pvQt.BackgroundPlotter(shape=(2, 2))
           
    #        # Solution numérique avec contours
    #        pl.subplot(0, 0)
    #        pl.add_mesh(numerical_grid, scalars="velocity_magnitude", 
    #                   cmap="viridis", show_edges=True)
    #        contours_num = numerical_grid.contour(scalars="velocity_magnitude", 
    #                                            isosurfaces=10)
    #        pl.add_mesh(contours_num, color="white", line_width=2)
    #        pl.add_text("Solution Numérique", font_size=10)
           
    #        # Solution analytique avec contours
    #        pl.subplot(0, 1)
    #        pl.add_mesh(analytical_grid, scalars="velocity_magnitude", 
    #                   cmap="viridis", show_edges=True)
    #        contours_ana = analytical_grid.contour(scalars="velocity_magnitude", 
    #                                             isosurfaces=10)
    #        pl.add_mesh(contours_ana, color="white", line_width=2)
    #        pl.add_text("Solution Analytique", font_size=10)
           
    #        # Titre général
    #        pl.add_text(f"Champ de vitesse (P={P}, schéma={scheme}, lc={lc})", 
    #                   font_size=6, position='lower_edge')
       
    #        # Erreur
    #        pl.subplot(1, 0)
    #        error_grid = numerical_grid.copy()
    #        error_grid.point_data['error'] = np.abs(
    #            numerical_grid.point_data['velocity_magnitude'] - 
    #            analytical_grid.point_data['velocity_magnitude']
    #        )
    #        pl.add_mesh(error_grid, scalars="error", cmap="RdBu", show_edges=True)
    #        contours_error = error_grid.contour(scalars="error", isosurfaces=10)
    #        pl.add_mesh(contours_error, color="black", line_width=1)
    #        pl.add_text(f"Erreur (P={P}, schéma={scheme}, lc={lc})", font_size=6)
    #        pl.link_views()
           
    #        # Profils de vitesse
    #        pl.subplot(1, 1)
    #        x_values = [0.25, 0.5, 0.75]
    #        for x in x_values:
    #            # Échantillonnage des lignes
    #            num_line = numerical_grid.sample_over_line((x, 0, 0), (x, 1, 0))
    #            ana_line = analytical_grid.sample_over_line((x, 0, 0), (x, 1, 0))
               
    #            # Extraction des données
    #            num_y = num_line.points[:, 1]
    #            ana_y = ana_line.points[:, 1]
    #            num_vel = num_line['velocity_magnitude']
    #            ana_vel = ana_line['velocity_magnitude']
               
    #            # Tracé des profils
    #            plt.plot(num_y, num_vel, label=f'Numérique x={x}')
    #            plt.plot(ana_y, ana_vel, '--', label=f'Analytique x={x}')
           
    #        plt.xlabel('y')
    #        plt.ylabel('Vitesse (Magnitude)')
    #        plt.title(f'Coupes à X constant (P={P}, schéma={scheme})')
    #        plt.legend()
    #        plt.grid(True)
    #        plt.tight_layout()
           
    #        # Sauvegarde et affichage de la figure
    #        plt.savefig(f'profil_vitesse_P{P}_{scheme}_lc{lc}.png', 
    #                   dpi=300, bbox_inches='tight')
    #        plt.close()
           
    #        # Ajout de la figure sauvegardée au plot
    #        plane = pv.Plane(center=(0.5, 0.5, 0.01), i_size=1, j_size=1, 
    #                        direction=(0, 0, 1))
    #        texture = pv.read_texture(f'profil_vitesse_P{P}_{scheme}_lc{lc}.png')
    #        plane.texture_map_to_plane(inplace=True)
    #        pl.add_mesh(plane, texture=texture)
           
    #        # Configuration de la caméra
    #        pl.camera_position = [(0.5, 0.5, 2), (0.5, 0.5, 0), (0, 1, 0)]
    #        pl.camera.zoom(0.75)
       
    #        pl.show()


        
def main():
    """Fonction principale pour exécuter les tests et la visualisation."""
    # Création du solveur
    flow = CouetteFlow(0, 1, 0, 1)
    
    # Tests avec différents paramètres
    P_values = [0, 3]
    schemes = ['centered']
    lc_values = [1/8]  # Maillage 8x8
    
    results = []
    for lc in lc_values:
        flow.generate_mesh(lc=lc)
        for P in P_values:
            for scheme in schemes:
                # Calcul de la solution
                A, b = flow.assemble_momentum_system(P, np.zeros(flow.mesh.get_number_of_elements()), 
                                                   np.zeros(flow.mesh.get_number_of_elements()))
                solution = spla.spsolve(A, b)
                u = solution[:flow.mesh.get_number_of_elements()]
                v = solution[flow.mesh.get_number_of_elements():]
                
                # Stockage des résultats
                results.append({
                    'lc': lc,
                    'P': P,
                    'scheme': scheme,
                    'u': u,
                    'v': v
                })
    
    # Visualisation des résultats
    # flow.plot_results(results)
    
    # Exécution des tests Rhie-Chow
    test_results = flow.test_rhie_chow_method()
    
    # Affichage du résumé
    print("\nRésumé des tests:")
    for case, result in test_results.items():
        print(f"\n{case}:")
        print(f"  Différence maximale: {result['max_difference']:.2e}")
        print(f"  Différence moyenne: {result['average_difference']:.2e}")


if __name__ == "__main__":
    main()