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
        self.mu = 0.02  # Viscosité dynamique [N*s/m^2]
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
    def analytical_neumman (self,x: float, y: float, P: float):
        """
        Solution analytique pour le gradient imposé par neumann
        
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
        valeur = 0
        return valeur
        
    
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
    
        u = 1 # Solution analytique corrigée
        v = 0
        if np.isclose(y, self.y_min) or np.isclose(y, self.y_max):
            u = 0
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
        
    # Création de l'étape 2 pour bien initilaisater les flux ( à priori qui sont nuls)   
    def initialisation_flux(self, u : np.ndarray,v :np.ndarray) :
        n_faces = self.mesh.get_number_of_faces()
        
        Fi = np.zeros((n_faces,1)) # Matrice avec Fxi et Fyi 
        
        for i_face in range (n_faces):
            # Récupération des voisins 
            left_cell, right_cell = self.mesh.get_face_to_elements(i_face)
            nodes = self.mesh.get_face_to_nodes(i_face)
            xa, ya = self.mesh.get_node_to_xycoord(nodes[0])
            xb, yb = self.mesh.get_node_to_xycoord(nodes[1])
            delta_Ai = np.sqrt((xb-xa)**2 + (yb-ya)**2)
            
            if right_cell !=-1 : 
                nx,ny = self.normal_face[i_face]
                Fxp = self.rho*u[left_cell]
                Fxa = self.rho*u[right_cell]
                
                Fxi = (Fxp + Fxa)/2
                
                Fyp = self.rho*v[left_cell]
                Fya = self.rho*v[right_cell]
                
                Fyi = (Fyp+Fya)/2
                
                Fi[i_face] = (Fxi*nx + Fyi*ny)*delta_Ai
               
                
        return Fi
        
    
    def assemble_momentum_system(self, P: float, u: np.ndarray, v: np.ndarray,Fi,grad_P, bcdata = [["LIBRE",0],["NEUMANN",1],["DIRICHLET",2],["NEUMANN",3]]) -> Tuple[sparse.csr_matrix, np.ndarray]:
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
        n_elements = n_cells
        A = sparse.lil_matrix((2*n_cells, 2*n_cells))
        b = np.zeros(2*n_cells)
        # Ajout des termes diffusifs et convectifs
        for i_face in range(self.mesh.get_number_of_faces()):
            left_cell, right_cell = self.mesh.get_face_to_elements(i_face)
            
            if right_cell != -1:  # Face interne
                self._add_internal_face_contribution(A,b, u, v,grad_P, left_cell, right_cell, i_face,Fi)
            else:  # Face frontière
                self._add_boundary_face_contribution(A, b, left_cell, i_face, P,bcdata,u,v,grad_P,Fi)
                
        for i_elem in range (n_elements):   
            volume =self.cell_volumes[i_elem]
            # Ajout des termes sources avec le gradient 
            b[i_elem] -= grad_P[i_elem][0]*volume
            b[n_cells +i_elem] -= grad_P[i_elem][1]*volume
            
        return A.tocsr(), b
    
    def _add_internal_face_contribution(self, A: sparse.lil_matrix, b: np.ndarray, u: np.ndarray, v: np.ndarray,grad : np.ndarray, 
                                      left_cell: int, right_cell: int, i_face: int,Fi,resol = "CENTRE", ) -> None:
        """Ajout des contributions des faces internes à la matrice du système."""
        n_cells = self.mesh.get_number_of_elements()
        nodes = self.mesh.get_face_to_nodes(i_face)
        xa, ya = self.mesh.get_node_to_xycoord(nodes[0])
        xb, yb = self.mesh.get_node_to_xycoord(nodes[1])
        delta_Ai = np.sqrt((xb-xa)**2 + (yb-ya)**2)
        xA,yA = self.cell_centers[right_cell]
        xP,yP = self.cell_centers[left_cell]
        
        # Calcul de la vitesse à la face
        u_face = 0.5 * (u[left_cell] + u[right_cell])
        v_face = 0.5 * (v[left_cell] + v[right_cell])
        
        # Vitesse normale à la face
        nx,ny = self.normal_face[i_face]
        vel_normal = u_face * nx + v_face * ny
       
        
        # Terme convectif REVU par méthode 
        conv_coeff =  Fi[i_face] # Récupération de l'itération précedente 
        
        # Terme diffusif
        
        dx = self.cell_centers[right_cell] - self.cell_centers[left_cell]
        distance = np.linalg.norm(dx)
        delta_yi = yb-ya
        delta_xi = xb-xa
        
        PNKSI = (delta_yi*(xA-xP)-delta_xi*(yA-yP))/delta_Ai/distance
        delta_n = delta_Ai
        
        PKSIETA = ((xA-xP)*(xa-xb)+(yA-yP)*(ya-yb))/distance/delta_n
        
        e_eta = [delta_xi/delta_Ai,delta_yi/delta_Ai]
        # Récupération des gradients 
        grad_Ax,grad_Ay = grad[right_cell]
        grad_Px, grad_Py = grad[left_cell]
        
        
        grad_moy = [(grad_Ax+grad_Px)/2,(grad_Ay+grad_Py)/2]
        #print("grad_moy", grad_moy)
        
        gradient_terme = np.dot(grad_moy,e_eta)
    
        #print("PNKSI", PNKSI)
        diff_coeff = self.mu * self.face_areas[i_face] / distance/ PNKSI
        
        SD_i_cross = - self.mu * (PKSIETA/PNKSI)*delta_Ai*gradient_terme
        
        # Ajout des contributions à la matrice
        if resol == "CENTRE" : 
            for i in range(2):
                base_idx = i * n_cells
                # Termes diffusifs
                #print("diff_coeff",diff_coeff)
                A[base_idx + left_cell, base_idx + right_cell] -= diff_coeff
                A[base_idx + right_cell, base_idx + left_cell] -= diff_coeff
                A[base_idx + left_cell, base_idx + left_cell] += diff_coeff
                A[base_idx + right_cell, base_idx + right_cell] += diff_coeff
                #print("Sd",SD_i_cross)
                b[base_idx + left_cell] += SD_i_cross
                b[base_idx + right_cell]-= SD_i_cross
                
                # Termes convectifs
                
                A[base_idx + left_cell, base_idx + right_cell] += conv_coeff[0]/2
                A[base_idx + right_cell, base_idx + left_cell] -= conv_coeff[0]/2
                A[base_idx + left_cell, base_idx + left_cell] += conv_coeff[0]/2
                A[base_idx + right_cell, base_idx + right_cell] -= conv_coeff[0]/2
                
        if resol == "UPWIND" : 
            for i in range(2):
                base_idx = i * n_cells
                # Termes diffusifs
                #print("diff_coeff",diff_coeff)
                A[base_idx + left_cell, base_idx + right_cell] -= diff_coeff
                A[base_idx + right_cell, base_idx + left_cell] -= diff_coeff
                A[base_idx + left_cell, base_idx + left_cell] += diff_coeff
                A[base_idx + right_cell, base_idx + right_cell] += diff_coeff
                #print("Sd",SD_i_cross)
                b[base_idx + left_cell] += SD_i_cross
                b[base_idx + right_cell]-= SD_i_cross
                
                # Termes convectifs
                #print("conv_coeff",conv_coeff)
                A[base_idx + left_cell, base_idx + right_cell] -= max(0,-conv_coeff)
                A[base_idx + right_cell, base_idx + left_cell] -= max(conv_coeff,0)
                A[base_idx + left_cell, base_idx + left_cell] += max(conv_coeff,0)
                A[base_idx + right_cell, base_idx + right_cell] -= max(0,-conv_coeff)
        
        
    def _add_boundary_face_contribution(self, A: sparse.lil_matrix, b: np.ndarray, 
                                      left_cell: int, i_face: int, P: float,bcdata : np.ndarray,u : np.ndarray, v :np.ndarray, grad_P : np.ndarray,Fi : np.ndarray) -> None:
        """Ajout des contributions des faces de frontieres à la matrice du système."""
        n_cells = self.mesh.get_number_of_elements()
        tag = self.mesh.get_boundary_face_to_tag(i_face)
        #print("tag",tag)
        for i_tag in range (len(bcdata)): # Récupération de la ocndition limite par le Tag
            tag_bc = bcdata[i_tag][1]
            if tag_bc == tag :
                condition = bcdata[i_tag][0]
                #print("Tag",tag,condition)
        
        left_cell, right_cell = self.mesh.get_face_to_elements(i_face) # Récupération des éléments de gauche et de droite 
        
        if condition == "DIRICHLET": 
            x_face, y_face = np.mean([self.mesh.get_node_to_xycoord(node) 
                                     for node in self.mesh.get_face_to_nodes(i_face)], axis=0)
            
            nodes = self.mesh.get_face_to_nodes(i_face) # Récupération des noeuds de l'arrête 
            xa,ya =self.mesh.get_node_to_xycoord(nodes[0]) # Coordonnées des limites de l'arête 
            xb,yb =self.mesh.get_node_to_xycoord(nodes[1]) # Coordonnées des limites de l'arrête
            delta_Ai = np.sqrt((xb-xa)**2 + (yb-ya)**2) # TAille de l'arrête
             
            xP,yP = self.cell_centers[left_cell]
            dx = np.array([x_face,y_face])- self.cell_centers[left_cell]
            distance = np.linalg.norm(dx)
            
            e_ksi = [(x_face-xP)/distance,(y_face-yP)/distance]
            nx,ny = self.normal_face[i_face] # Récupératon des normales
            PNKSI = np.dot([nx,ny],e_ksi)
            
            e_eta = [(xb-xa)/delta_Ai,(yb-ya)/delta_Ai]
            PKSIETA = np.dot(e_ksi,e_eta)
            
            diff_coeff = self.mu * self.face_areas[i_face] / distance/ PNKSI
            
            #Calcul par solution analytique à la face 
            #PHI1_u,PHI1_v = self.analytical_solution(xb, yb, P)
            #PHI2_u,PHI2_v = self.analytical_solution(xa, ya, P)
            grad_moy = (grad_P[left_cell]+grad_P[right_cell])/2
            
            gradient_terme = np.dot(grad_moy,e_eta)
            
            # Approximation des gradients par méthode du cours 
            #gradient_terme_u  = (PHI1_u -PHI2_u)/delta_Ai
            #gradient_terme_v  = (PHI1_u -PHI2_u)/delta_Ai
            
            # Calcul des termes de crossdiffusion
            SD_i_cross_u = - self.mu * (PKSIETA/PNKSI)*delta_Ai*gradient_terme
            SD_i_cross_v = - self.mu * (PKSIETA/PNKSI)*delta_Ai*gradient_terme
            
            
            u_boundary, v_boundary = self.analytical_solution(x_face, y_face, P)
            
            # Terme convectif REVU par méthode 
            conv_coeff =  Fi[i_face]
            # Terme A                                                 
            A[left_cell, left_cell] += diff_coeff + max(conv_coeff[0],0)
            A[n_cells + left_cell, n_cells + left_cell] += diff_coeff + max(conv_coeff[0],0)
            #Terme B 
            b[left_cell] += SD_i_cross_u + diff_coeff*u_boundary + max(0,-conv_coeff)*u_boundary
            b[n_cells + left_cell] += SD_i_cross_v + diff_coeff*v_boundary + max(0,-conv_coeff)*v_boundary
            
        if  condition == "NEUMANN" : 
            x_face, y_face = np.mean([self.mesh.get_node_to_xycoord(node) 
                                     for node in self.mesh.get_face_to_nodes(i_face)], axis=0)
            
            nodes = self.mesh.get_face_to_nodes(i_face) # Récupération des noeuds de l'arrête 
            xa,ya =self.mesh.get_node_to_xycoord(nodes[0]) # Coordonnées des limites de l'arête 
            xb,yb =self.mesh.get_node_to_xycoord(nodes[1]) # Coordonnées des limites de l'arrête
            delta_Ai = np.sqrt((xb-xa)**2 + (yb-ya)**2) # TAille de l'arrête
             
            xP,yP = self.cell_centers[left_cell]
            dx = np.array([x_face,y_face])- self.cell_centers[left_cell]
            distance = np.linalg.norm(dx)
            
            e_ksi = [(x_face-xP)/distance,(y_face-yP)/distance]
            nx,ny = self.normal_face[i_face] # Récupératon des normales
            PNKSI = np.dot([nx,ny],e_ksi)
            
            e_eta = [(xb-xa)/delta_Ai,(yb-ya)/delta_Ai]
            PKSIETA = np.dot(e_ksi,e_eta)
            
            diff_coeff = self.mu * self.face_areas[i_face] / distance/ PNKSI
            
            grad_moy = (grad_P[left_cell]+grad_P[right_cell])/2
           
            gradient_terme = np.dot(grad_moy,e_eta)
            
            # Calcul des termes de crossdiffusion
            SD_i_cross_u = - self.mu * (PKSIETA/PNKSI)*delta_Ai*gradient_terme
            SD_i_cross_v = - self.mu * (PKSIETA/PNKSI)*delta_Ai*gradient_terme
            
            
            
            # Terme convectif REVU par méthode 
            conv_coeff =  Fi[i_face]
            
            
            Valeur = self.analytical_neumman(0,0,0)
            # Terme  en A
            A[left_cell, left_cell] += conv_coeff[0]
            A[n_cells + left_cell, n_cells + left_cell] += conv_coeff[0]
            b[left_cell] += self.mu*Valeur*delta_Ai - conv_coeff[0]*Valeur*PNKSI*distance
            b[n_cells + left_cell] += self.mu*Valeur*delta_Ai - conv_coeff[0]*Valeur*PNKSI*distance
            
           
            
    
    def least_squares_gradient(self, field_values: np.ndarray, bcdata = [["Libre",0],["NEUMANN",1],["DIRICHLET",2],["NEUMANN",3]] ) -> np.ndarray:
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
                xA,yA = self.cell_centers[right_cell]
                xP,yP = self.cell_centers[left_cell]
                di = self.cell_centers[right_cell] - self.cell_centers[left_cell]
                distance = np.linalg.norm(di)
                #print("di",di)
                dx = xA - xP 
                dy = yA - yP
                
                ALS = np.zeros((2,2))
                
                ALS [0,0] = dx*dx
                ALS [1,0] = dx*dy
                ALS [0,1] = dy*dx
                ALS [1,1] = dy*dy
                
                
                # Ajouter les contributions aux deux cellules
                ATA[left_cell] += ALS
                ATA[right_cell] += ALS
                
                # Différence des valeurs du champ
                df = field_values[right_cell] - field_values[left_cell]
                
                # Ajouter les contributions pondérées à B
                B[left_cell][0] += dx*df
                B[left_cell][0] += dy*df
                B[right_cell][0] += dx*df
                B[right_cell][1] += dy*df
                
            else :   
                tag = self.mesh.get_boundary_face_to_tag(i_face)
                #print("tag",tag)
                for i_tag in range (len(bcdata)): # Récupération de la ocndition limite par le Tag
                    tag_bc = bcdata[i_tag][1]
                    if tag_bc == tag :
                        condition = bcdata[i_tag][0]
                        #print("Tag",tag,condition)
                 # Obtenir les coordonnées du centre de la face
                face_nodes = self.mesh.get_face_to_nodes(i_face)
                x_face = 0.5 * (self.mesh.get_node_to_xcoord(face_nodes[0]) + 
                                self.mesh.get_node_to_xcoord(face_nodes[1]))
                y_face = 0.5 * (self.mesh.get_node_to_ycoord(face_nodes[0]) + 
                                self.mesh.get_node_to_ycoord(face_nodes[1]))
                
                xtg,ytg = self.cell_centers[left_cell]
                
                nx, ny = self.normal_face[i_face]
                
                # Face frontière dirichlet
                if condition == "DIRICHLET" : 
            
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
                  
                elif condition == "NEUMANN" :
                    dx = x_face - xtg
                    dy =y_face = ytg
                    valeur = self.analytical_neumman(x_face,y_face,0)
                    valeur = 0
                    
                    B[left_cell][0] += (dx*nx +dy*ny)*(dx*nx +dy*ny)*valeur
                    B[left_cell][1] += (dx*ny +dy*nx)*(dx*nx +dy*ny)*valeur
                    
                elif condition =="LIBRE" : 
                    i_m = 0
                    
                    
                    
            
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
    
    def relaxation_Fn( self,Fn :np.ndarray,F_prec :np.ndarray, alpha = 0.7):
        Fn = alpha * Fn + (1 - alpha) * F_prec
        return Fn
    
    def relaxation_pression(self, P_new : np.ndarray, P_prec : np.ndarray, 
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
        P_final = alpha * P_new + (1 - alpha) * P_prec
        
        return P_final

       
    def assemblage_lap_4(self,Fi,u,v,grad_P,bcdata,P =0,alpha = 0.7):
        n_elements = self.mesh.get_number_of_elements()
        for i in range (1) : 
            A, b = self.assemble_momentum_system(P, u, v,Fi,grad_P,bcdata)
            aP = A.diagonal()[:n_elements]
        
            # Calculer la nouvelle solution
            solution = spla.spsolve(A, b)
            u_new = solution[:n_elements]
            v_new = solution[n_elements:]
            
            # Appliquer la sous-relaxation
            u, v = self.relaxation(u_new, v_new, u, v, 1)
            #print("A",A)
            #print("b",b)
            #print("u", u)
        
        return u,v,aP
        

    def Rhie_Chow(self, F_initial,u,v,grad_P,aP,P_field,bcdata ,P: float = 0, max_iterations: int = 1000, 
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
        
        
        
        # Initialiser le tableau des vitesses aux faces
        U_face = np.zeros(self.mesh.get_number_of_faces())
        V_face = np.zeros(self.mesh.get_number_of_faces())
        
        # Calculer les vitesses aux faces
        for i_face in range(self.mesh.get_number_of_faces()):
            n_cells = self.mesh.get_number_of_elements()
            
            left_cell, right_cell = self.mesh.get_face_to_elements(i_face)
            nx, ny = self.normal_face[i_face]
           
            if right_cell != -1:  # Face interne
                # Interpolation simple de la vitesse
                u_avg = 0.5 * (u[left_cell] + u[right_cell])
                v_avg = 0.5 * (v[left_cell] + v[right_cell])
                #print("uleft",u[left_cell])
                #print("uright",u[right_cell])
                #print("nx",nx)
                #print("u-",u_avg)
                U_avg = u_avg * nx + v_avg * ny
                #print("V_avg", V_avg)
                #print("u_avg", u_avg)
                #print("nx",nx)
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
                #print("U_avg", U_avg)
                #print("U_face",U_face[i_face])
                V_face[i_face] = U_avg - vol_aP_avg * (dP - grad_P_avg)
                
            else:  # Face frontière
                tag = self.mesh.get_boundary_face_to_tag(i_face)
                #print("tag",tag)
                for i_tag in range (len(bcdata)): # Récupération de la ocndition limite par le Tag
                    tag_bc = bcdata[i_tag][1]
                    if tag_bc == tag :
                        condition = bcdata[i_tag][0]
                        #print("Tag",tag,condition)
                        
                        
                # Obtenir les coordonnées du centre de la face
                x_face, y_face = np.mean([self.mesh.get_node_to_xycoord(node) 
                                        for node in self.mesh.get_face_to_nodes(i_face)], axis=0)
                
                if condition == "DIRICHLET" : 
                    # FAce ouest, nord ,sud 
                    u_bound, v_bound = self.analytical_solution(x_face, y_face, P)
                   
                    U_face[i_face] = u_bound * nx 
                    V_face[i_face] = v_bound*ny
                    #print("U face ",U_face[i_face])
                    #print("u_limit : ",u_bound)
                    #print(nx)
                    #print("Ubound", U_face[i_face])
                elif condition == "NEUMANN" : 
                    # Face est 
                    dx = np.array([x_face, y_face]) - self.cell_centers[left_cell]
                    xtg,ytg = self.cell_centers[left_cell]
                    distance = np.linalg.norm(dx)
                    # Condition limite de Neumann - utiliser les valeurs au centre de la cellule
                    
                    U_face[i_face] = u[left_cell] * nx  + self.cell_volumes[left_cell]/aP[left_cell]*((P_field[left_cell]-P_sortie)/distance) + self.cell_volumes[left_cell]/aP[left_cell]*(grad_P[left_cell][0]*(x_face-xtg)+grad_P[left_cell][1]*(y_face-ytg)) 
        #print("U_face", U_face)
        
        return U_face,V_face, aP
    
    def Correction_pression(self, U_face, aP,bcdata = [["ENTREE",0],["PAROI",1],["SORTIE",2],["PAROI",3]], P: float = 0, max_iterations: int = 1000, 
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
                tag = self.mesh.get_boundary_face_to_tag(i_face)
                #print("tag",tag)
                for i_tag in range (len(bcdata)): # Récupération de la ocndition limite par le Tag
                    tag_bc = bcdata[i_tag][1]
                    if tag_bc == tag :
                        condition = bcdata[i_tag][0]
                        #print("Tag",tag,condition)
                        
                x_face, y_face = np.mean([self.mesh.get_node_to_xycoord(node) 
                                        for node in self.mesh.get_face_to_nodes(i_face)], axis=0)
                
                if condition == "ENTREE" or condition == "PAROI" : 
                    B[left_cell] -= self.rho * U_face[i_face] * delta_Ai
                    
                elif condition == "SORTIE" : 
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
                tag = self.mesh.get_boundary_face_to_tag(i_face)
                #print("tag",tag)
                for i_tag in range (len(bcdata)): # Récupération de la ocndition limite par le Tag
                    tag_bc = bcdata[i_tag][1]
                    if tag_bc == tag :
                        condition = bcdata[i_tag][0]
                        #print("Tag",tag,condition)
                
                if condition == "SORTIE" : 
                    dx = np.array([x_face, y_face]) - self.cell_centers[left_cell]
                    distance = np.linalg.norm(dx)
                    dfi = self.cell_volumes[left_cell]/(ap*distance)
                    U_final[i_face] = U_face[i_face] + dfi * pression[left_cell]
                    
                else:
                    U_final[i_face] = U_face[i_face]
        
        return pression, U_final
                
    
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
            #print(el1,el2)
            
            nodes = self.mesh.get_face_to_nodes(i_face)
            xa,ya = self.mesh.get_node_to_xycoord(nodes[0])
            xb,yb = self.mesh.get_node_to_xycoord(nodes[1])
            delta_Ai = ((xb-xa)**2+(yb-ya)**2)**(1/2)
            
            flux = self.rho*U_face[i_face]*delta_Ai
            flux = flux.item()
            divergence[el1] =  divergence[el1]+ flux
            #print("div",divergence[el1])
            #print("flux",flux)
            if el2 != -1 : 
                divergence[el2] = divergence[el2] - flux
            
        return divergence
    
    def relaxation_RC (self,UnRC,Un,alphaRC):
        n_faces = self.mesh.get_number_of_faces()
        for i_face in range(n_faces):
            UnRC [i_face] = Un[i_face] + alphaRC*(UnRC[i_face]-Un[i_face])
        return UnRC
    
    def Fi_nouveau(self,up,vp):
        n_face =self.mesh.get_number_of_faces()
        Fi = np.zeros((n_face,1))
        for i_face in range(n_face):
            left_cell, right_cell = self.mesh.get_face_to_elements(i_face)
            nodes = self.mesh.get_face_to_nodes(i_face)
            xa,ya = self.mesh.get_node_to_xycoord(nodes[0])
            xb,yb = self.mesh.get_node_to_xycoord(nodes[1])
            delta_Ai = ((xb-xa)**2+(yb-ya)**2)**(1/2)
            nx, ny = self.normal_face[i_face]
            
            Fxi = self.rho *delta_Ai *(up[left_cell]+up[right_cell])/2
            Fyi = self.rho *delta_Ai *(vp[left_cell]+vp[right_cell])/2
            
            
            Fi_face = (Fxi*nx+Fyi*ny)*delta_Ai
            Fi[i_face] = Fi_face
    
        return Fi
        
    




class VelocityFieldPlotter:
    def __init__(self, solver):
        self.solver = solver

    def plot_all_fields(self, U_face_initial: np.ndarray, U_face_corrected: np.ndarray, 
                        P_prime: np.ndarray, u: np.ndarray, v: np.ndarray, 
                        title: str = "Champs d'écoulement"):
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
        u, v : np.ndarray
            Composantes de vitesse au centre des cellules
        title : str
            Titre du graphique
        """
        # Création de la première figure
        fig1 = plt.figure(figsize=(18, 6))
        gs = plt.GridSpec(1, 3, figure=fig1)
        
        # Affichage du champ de vitesse initial
        ax1 = fig1.add_subplot(gs[0, 0])
        self._plot_velocity_field(ax1, U_face_initial.flatten(), "Champ de vitesse initial")
        
        # Affichage du champ de vitesse corrigé
        ax2 = fig1.add_subplot(gs[0, 1])
        self._plot_velocity_field(ax2, U_face_corrected.flatten(), "Champ de vitesse corrigé")
        
        # Affichage de la correction de pression
        ax3 = fig1.add_subplot(gs[0, 2])
        self._plot_pressure_field(ax3, P_prime, "Correction de pression")
        
        # Ajout du titre général
        fig1.suptitle(title, fontsize=14)
        plt.tight_layout()

        # Création d'une deuxième figure pour le champ de vitesse au centre des cellules
        fig2 = plt.figure(figsize=(6, 6))
        ax4 = fig2.add_subplot(111)  # Une seule grille pour la 4ème plot
        self._plot_velocity_field1(ax4, u, v, "Champ de vitesse au centre des cellules")
        fig2.suptitle("Champ au centre des cellules", fontsize=14)
        plt.tight_layout()

        return fig1, fig2  # Retour des deux figures
    


        
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
            
            scale = 0.1
            dx = velocity * nx * scale
            dy = velocity * ny * scale
            
            # Utilisation de quiver pour une meilleure visualisation des vecteurs
            ax.quiver(face_center[0], face_center[1], dx, dy,
                     angles='xy', scale_units='xy', scale=1,
                     color='blue', width=0.005)
        
        self._set_plot_properties(ax, subtitle)
        
    def _plot_velocity_field1(self, ax, ux: np.ndarray, uy : np.ndarray, subtitle: str):
        """Affichage d'un champ de vitesse au centre des cellules"""
        # Tracé des éléments du maillage
        for i_elem in range(self.solver.mesh.get_number_of_elements()):
            nodes = self.solver.mesh.get_element_to_nodes(i_elem)
            xy_coords = []
            for node in nodes:
                x, y = self.solver.mesh.get_node_to_xycoord(node)
                xy_coords.append([x, y])
            xy_coords = np.array(xy_coords)
            xy_coords = np.vstack((xy_coords, xy_coords[0]))  # Boucle le contour
            ax.plot(xy_coords[:, 0], xy_coords[:, 1], 'k-', linewidth=0.5)
    
            norms = np.sqrt(ux**2 + uy**2)  # Norme des vitesses
        max_velocity = np.max(norms)  # Norme max
        scale = 0.1/ max_velocity if max_velocity > 0 else 0.1  # Évite div. par 0
    
        for i_elem in range(self.solver.mesh.get_number_of_elements()):
            # Récupération des nœuds de l'élément
            nodes = self.solver.mesh.get_element_to_nodes(i_elem)
            xy_coords = []
            for node in nodes:
                x, y = self.solver.mesh.get_node_to_xycoord(node)
                xy_coords.append([x, y])
            xy_coords = np.array(xy_coords)
            
            # Calcul du centre de l'élément
            cell_center = np.mean(xy_coords, axis=0)
            
            # Composantes de vitesse au centre de la cellule
            velocity_x = ux[i_elem] 
            velocity_y = uy[i_elem]
            
            # Mise à l'échelle des vecteurs
            dx = velocity_x * scale
            dy = velocity_y * scale
            
            # Tracé des vecteurs avec `quiver`
            ax.quiver(cell_center[0], cell_center[1], dx, dy,
                      angles='xy', scale_units='xy', scale=1,
                      color='blue', width=0.005)
    
        # Configuration du titre
        ax.set_title(subtitle)

    
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

def track_residuals(flow, mesh_parameters, bcdata):
    """Track and plot residuals during the SIMPLE algorithm iteration"""
    # Initialize storage for residuals
    max_iterations = 2000
    u_residuals = []
    v_residuals = []
    p_residuals = []
    div_residuals = []
    
    # Setup initial conditions
    Nx = mesh_parameters[0][1]
    mesh_type = mesh_parameters[0][0]
    Ny = mesh_parameters[0][2]
    flow.generate_mesh(mesh_type=mesh_type, Nx=Nx, Ny=Ny)
    
    # Initial fields
    n_elements = flow.mesh.get_number_of_elements()
    u = np.zeros(n_elements)
    v = np.zeros(n_elements)
    P_field = np.zeros(n_elements)
    grad_P = flow.least_squares_gradient(P_field)
    F_initial = flow.initialisation_flux(u,v)
    
    # Reference values for residuals
    u_ref = None
    v_ref = None
    p_ref = None
    
    condition = 1
    iteration = 0
    
    while condition > 10**-10 and iteration < max_iterations:
        iteration += 1
        
        # Store previous iteration values for residual calculation
        u_prev = u.copy() if iteration > 1 else None
        v_prev = v.copy() if iteration > 1 else None
        p_prev = P_field.copy() if iteration > 1 else None
        
        # Main SIMPLE algorithm steps
        up, vp, aP = flow.assemblage_lap_4(F_initial, u, v, grad_P, bcdata)
        UnRC, VnRC, aP = flow.Rhie_Chow(F_initial, up, vp, grad_P, aP, P_field, bcdata)
        
        if iteration == 1:
            Un = UnRC
            u_ref = np.linalg.norm(up)
            v_ref = np.linalg.norm(vp)
            p_ref = np.linalg.norm(P_field)
            
        # Rhie-Chow relaxation
        alphaRC = 0.1
        UnRC = flow.relaxation_RC(UnRC, Un, alphaRC)
        
        # Pressure correction
        bcdpc = [["ENTREE",0],["PAROI",1],["SORTIE",2],["PAROI",3]]
        P_prime, UF = flow.Correction_pression(UnRC, aP, bcdpc)
        
        # Calculate divergence
        div_initial = flow.Calcul_divergence(UnRC)
        div_max = np.max(np.abs(div_initial))
        
        # Pressure correction with relaxation
        Alpha_P = 0.1
        P_field += Alpha_P * P_prime
        
        # Calculate residuals
        if iteration > 1:
            u_res = np.linalg.norm(up - u_prev) / u_ref
            v_res = np.linalg.norm(vp - v_prev) / v_ref
            p_res = np.linalg.norm(P_field - p_prev) / p_ref
            
            u_residuals.append(u_res)
            v_residuals.append(v_res)
            p_residuals.append(p_res)
            div_residuals.append(div_max)
        
        # Update for next iteration
        bcdp = [["Libre",0],["NEUMANN",1],["DIRICHLET",2],["NEUMANN",3]]
        grad_P = flow.least_squares_gradient(P_field, bcdp)
        Un = UF
        condition = div_max
        F_initial = flow.Fi_nouveau(up, vp)
        
        # Update velocity fields
        u = up
        v = vp
    
    # Plot residuals
    iterations = range(1, len(u_residuals) + 1)
    
    plt.figure(figsize=(12, 8))
    plt.semilogy(iterations, u_residuals, 'b-', label='u velocity')
    plt.semilogy(iterations, v_residuals, 'r-', label='v velocity')
    plt.semilogy(iterations, p_residuals, 'g-', label='pressure')
    plt.semilogy(iterations, div_residuals, 'k-', label='max divergence')
    
    plt.grid(True)
    plt.xlabel('Iteration')
    plt.ylabel('Residual (log scale)')
    plt.title(f'Convergence History - {mesh_type} Mesh ({Nx}x{Ny})')
    plt.legend()
    
    return plt.gcf()

def main():
    """Fonction principale pour tester et visualiser les champs avec différentes tailles de maillage"""
    # Création de l'instance du solveur
    flow = CouetteFlow(0,1,0,1)
    bcdata = [["DIRICHLET",0],["DIRICHLET",1],["DIRICHLET",2],["DIRICHLET",3]]
    # Test de différentes configurations de maillage
    mesh_parameters = [["QUAD",5,5],["QUAD",20,10],["QUAD",40,20]]
    for i in range(1): 
        if 0==0 : 
            Nx = mesh_parameters[i][1]
            mesh_type = mesh_parameters[i][0]
            Ny = mesh_parameters[i][2]
            # Génération du maillage avec la taille spécifiée
            flow.generate_mesh(mesh_type=mesh_type, Nx=Nx, Ny=Ny)
            
            #Etape 1 de l'algorithme Simple 
            n_elements = flow.mesh.get_number_of_elements()
            u = np.zeros(n_elements)
            v = np.zeros(n_elements)
            
            P_field = np.zeros(n_elements)
            # Calculer les gradients de pression en utilisant les moindres carrés
            grad_P = flow.least_squares_gradient(P_field)
            #print("grad_P",grad_P)
        
            # Etape 2 de l'algorithme Simple 
            # Calcul du champ de vitesse initial
            F_initial = flow.initialisation_flux(u,v)
            #print("F_initial", F_initial)
            #print(np.size(F_initial))

            condition = 1
            iteration = 0
            while condition > 10**-10 and iteration <50: 
                iteration+=1
                # Etape 3
                up,vp,aP = flow.assemblage_lap_4(F_initial,u,v,grad_P,bcdata)
                print("up",up)
                print("vp",vp)
                
                # Etape 4 : Moyenne des vitesses de Rhie Chow 
                UnRC,VnRC,aP = flow.Rhie_Chow(F_initial,up,vp,grad_P,aP,P_field,bcdata)
                #print(UnRC)
                
                if iteration ==1 : 
                    Un = UnRC
                # SOus relaxation de Rhie CHOW  
                alphaRC =.9
                UnRC = flow.relaxation_RC(UnRC,Un,alphaRC)
                
                # Etape 5 
                bcdpc = [["ENTREE",0],["PAROI",1],["SORTIE",2],["PAROI",3]]
                P_prime, UF = flow.Correction_pression(UnRC,aP,bcdpc)
                
                # Visualisation des résultats
                plotter = VelocityFieldPlotter(flow)
                fig = plotter.plot_all_fields(
                    UnRC, 
                    UF,
                    P_prime,
                    up,
                    vp,
                    f"Champs d'écoulement - Maillage {mesh_type} (Nx={Nx}, Ny={Ny}) - Iteration = {iteration}"
                )
                # # Affichage des statistiques de divergence
                div_initial = flow.Calcul_divergence(UnRC)
                print("Divergence initiale:", div_initial)
                div_corrected = flow.Calcul_divergence(UF)
                print("Divergence corrigée:", div_corrected)
                
                #Correction du champ de pression 
                Alpha_P =0.1
                P_field += Alpha_P*P_prime
                
                # Gradient de pression pour l'itération suivante 
                bcdp = [["Libre",0],["NEUMANN",1],["DIRICHLET",2],["NEUMANN",3]]
                grad_P = flow.least_squares_gradient(P_field,bcdp)
                Un = UF
                condition = np.max(np.abs(div_initial))
                
                # NOuveau Fi pour
                Fi_initial = flow.Fi_nouveau(up,vp)
                
                
            fig = track_residuals(flow, mesh_parameters, bcdata)
            plt.show()      
                #print(f"Divergence maximale initiale: {np.max(np.abs(div_initial)):.2e}")
                #print(f"Divergence maximale corrigée: {np.max(np.abs(div_corrected)):.2e}")
                #print(f"Facteur d'amélioration: {np.max(np.abs(div_initial))/np.max(np.abs(div_corrected)):.2f}x")

if __name__ == "__main__":
    main()