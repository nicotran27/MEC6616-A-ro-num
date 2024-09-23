"""
Exemple Solveur Simple Moyenne - Semaine 3
MEC6616 - Aérodynamique numérique
Date de création: 2022 - 01 - 21
Auteurs: Nicolas Tran
"""

# import sys
# sys.path.append('mesh_path')

import numpy as np
import sympy as sp
from meshGenerator import MeshGenerator

def pause():
    input('Press [return] to continue ...')
#%%Programme 1-A
mesher = MeshGenerator()
def Euler(mesh_obj):
    f=mesh_obj.get_number_of_elements()
    a=mesh_obj.get_number_of_faces()
    s=mesh_obj.get_number_of_nodes()
    h=0
    if mesh_obj.get_number_of_boundaries ==5:
        h=1  
    Euler=f-a+s+h
    if Euler==0:
        print("Le maillage ne respecte pas la relation d'Euler")
    else:
        print("Le maillage respecte la relation d'Euler")
    return 
print('Partie 1-A')
#Rectangle : maillage non structuré avec des triangles
mesh_parameters = {'mesh_type': 'TRI',
                    'lc': 0.5
                    }
print('Rectangle : maillage non structuré avec des triangles')
mesh_obj = mesher.rectangle([0.0, 1.0, 0.0, 1.0], mesh_parameters)
Euler(mesh_obj)

#Rectangle : maillage non structuré avec des quadrilatères
print('Rectangle : maillage non structuré avec des quadrilatères')
mesh_parameters = {'mesh_type': 'QUAD',
                    'lc': 0.2
                    }
mesh_obj = mesher.rectangle([0.0, 2.0, 0.0, 1.0], mesh_parameters)
Euler(mesh_obj)

#Rectangle : maillage non structuré mix
print('Rectangle : maillage non structuré mix')
mesh_parameters = {'mesh_type': 'MIX',
                    'lc': 0.2
                    }
mesh_obj = mesher.rectangle([0.0, 2.0, 0.0, 1.0], mesh_parameters)
Euler(mesh_obj)

#Rectangle : maillage transfinis avec des triangles
print('Rectangle : maillage transfinis avec des triangles')
mesh_parameters = {'mesh_type': 'TRI',
                    'Nx': 5,
                    'Ny': 4
                    }
mesh_obj = mesher.rectangle([0.0, 1.0, 0.0, 1.0], mesh_parameters)
Euler(mesh_obj)

#Rectangle : maillage transfinis avec des quadrilatères
print('Rectangle : maillage transfinis avec des quadrilatères')

mesh_parameters = {'mesh_type': 'QUAD',
                    'Nx': 10,
                    'Ny': 15
                    }
mesh_obj = mesher.rectangle([0.0, 1.0, 1.0, 2.5], mesh_parameters)
Euler(mesh_obj)

#Rectangle : maillage transfinis mix
print('Rectangle : maillage transfinis mix')
mesh_parameters = {'mesh_type': 'MIX',
                    'Nx': 16,
                    'Ny': 10
                    }
mesh_obj = mesher.rectangle([0.0, 2.0, 0.0, 1.0], mesh_parameters)
Euler(mesh_obj)

#Back Step : maillage non structuré avec des triangles
print('Back Step : maillage non structuré avec des triangles')
mesh_parameters = {'mesh_type': 'TRI',
                    'lc': 0.3
                    }
mesh_obj = mesher.back_step(1.0, 1.5, 1.0, 3.0, mesh_parameters)
Euler(mesh_obj)

#Back Step : maillage non structuré avec des quadrilatères
print('Back Step : maillage non structuré avec des quadrilatères')
mesh_parameters = {'mesh_type': 'QUAD',
                    'lc': 0.2
                    }
mesh_obj = mesher.back_step(1.0, 1.5, 1.0, 3.0, mesh_parameters)
Euler(mesh_obj)

#Back Step : maillage non structuré mix
print('Back Step : maillage non structuré mix')
mesh_parameters = {'mesh_type': 'MIX',
                    'lc': 0.2
                    }
mesh_obj = mesher.back_step(1.0, 1.5, 1.0, 3.0, mesh_parameters)
Euler(mesh_obj)

#Cercle : maillage non structuré avec des triangles
print('Cercle : maillage non structuré avec des triangles')
mesh_parameters = {'mesh_type': 'TRI',
                    'lc_rectangle': 0.3,
                    'lc_circle': 0.1
                    }
rayon = 0.25
mesh_obj = mesher.circle([0.0, 2.0, 0.0, 1.0], rayon, mesh_parameters)
Euler(mesh_obj)

#Cercle : maillage non structuré avec des quadrilatères
print('Cercle : maillage non structuré avec des quadrilatères')
mesh_parameters = {'mesh_type': 'QUAD',
                    'lc_rectangle': 0.2,
                    'lc_circle': 0.05
                    }
rayon = 0.25
mesh_obj = mesher.circle([0.0, 2.0, 0.0, 1.0], rayon, mesh_parameters)
Euler(mesh_obj)

#Cercle : maillage transfini avec des triangles
print('Cercle : maillage transfini avec des triangles')
mesh_parameters = {'mesh_type': 'TRI',
                    'Nx': 25,
                    'Ny': 5,
                    'Nc': 20
                    }
rayon = 0.25
mesh_obj = mesher.circle([0.0, 2.0, 0.0, 1.0], rayon, mesh_parameters)

Euler(mesh_obj)

#Cercle : maillage transfini avec des quadrilatères
print('Cercle : maillage transfini avec des quadrilatères')
mesh_parameters = {'mesh_type': 'QUAD',
                    'Nx': 60,
                    'Ny': 20,
                    'Nc': 60
                    }
rayon = 0.25
mesh_obj = mesher.circle([0.0, 5.0, 0.0, 3.0], rayon, mesh_parameters)
Euler(mesh_obj)

#Quart Anneau : maillage non structuré avec des triangles
print('Quart Anneau : maillage non structuré avec des triangles')
mesh_parameters = {'mesh_type': 'TRI', 'lc': 0.2}
mesh_obj = mesher.quarter_annular(1.0, 2.0, mesh_parameters)
Euler(mesh_obj)

#Quart Anneau : maillage non structuré avec des quadrilatères
print('Quart Anneau : maillage non structuré avec des quadrilatères')
mesh_parameters = {'mesh_type': 'QUAD', 'lc': 0.1}
mesh_obj = mesher.quarter_annular(1.0, 2.0, mesh_parameters)
Euler(mesh_obj)

#Quart Anneau : maillage transfini avec des triangles
print('Quart Anneau : maillage transfini avec des triangles')
mesh_parameters = {'mesh_type': 'TRI',
                    'N1': 50,
                    'N2': 10
                    }
mesh_obj = mesher.quarter_annular(2.0, 4.0, mesh_parameters)
Euler(mesh_obj)

#Quart Anneau : maillage transfini avec des quadrilatères
print('Quart Anneau : maillage transfini avec des quadrilatères')
mesh_parameters = {'mesh_type': 'QUAD',
                    'N1': 50,
                    'N2': 10
                    }
mesh_obj = mesher.quarter_annular(2.0, 3.0, mesh_parameters)
Euler(mesh_obj)
print('')
print('')
pause()
#%%Programme 1-B

x, y = sp.symbols('x y') #Variables symboliques
#Création du champ
a=1
b=1
V = a*x+b*y
fV = sp.lambdify([x, y], V, 'numpy')


def normal_of_line(mesh_obj):
    """
    Fonction qui renvoit les normales des faces

    Parameters
    ----------
    mesh_obj : TYPE
        Maillage choisi.

    Returns
    -------
    normals : array
        Matrice qui retournes les normales de chaque face.

    """
    #Création de la matrice des normales
    number_of_faces = mesh_obj.get_number_of_faces()
    normals = []
    
    for i_face in range(number_of_faces):
        # Trouver les coordonnées des noeuds reliés aux faces
        
        face_nodes = mesh_obj.face_to_nodes[i_face]
        x1 = mesh_obj.get_node_to_xcoord(face_nodes[0])
        y1 = mesh_obj.get_node_to_ycoord(face_nodes[0])
        x2 = mesh_obj.get_node_to_xcoord(face_nodes[1])
        y2 = mesh_obj.get_node_to_ycoord(face_nodes[1])
        
        # Vecteurs de direction pour la normale
        dx = x2 - x1
        dy = y2 - y1
        
        # Vecteur normal
        normal = (-dy, dx)
        
        # Ajouts dans la matrice des normales
        normals.append(normal)
    return normals

def face_lengths(mesh_obj):
    """
    Fonction qui renvoit les longeurs des faces

    Parameters
    ----------
    mesh_obj : TYPE
        Maillage choisi.

    Returns
    -------
    lenghts : array
        Matrice qui retournes les longeurs de chaque face.

    """
    #Création de la matrice des longueurs
    number_of_faces = mesh_obj.get_number_of_faces()
    lengths = []
    
    for i_face in range(number_of_faces):
        # Trouver les coordonnées des noeuds reliés aux faces
        face_nodes = mesh_obj.face_to_nodes[i_face]
        x1 = mesh_obj.get_node_to_xcoord(face_nodes[0])
        y1 = mesh_obj.get_node_to_ycoord(face_nodes[0])
        x2 = mesh_obj.get_node_to_xcoord(face_nodes[1])
        y2 = mesh_obj.get_node_to_ycoord(face_nodes[1])
        
        # Calcul de la distance
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Ajouts dans la matrice des longueurs
        lengths.append(length)
    
    return lengths

def dot_product_V_and_normals(mesh_obj, a, b):
    """
    Fonction qui renvoit le produit scalaire du champ et des normales

    Parameters
    ----------
    mesh_obj : TYPE
        Maillage choisi.
    a : INT
        Constante du champ en x
    b : INT
        Constante du champ en y
    Returns
    -------
    dot_products : array
        Matrice qui retournes les produits scalaire du champ et des normales.

    """
    #Création de la matrice des produits scalaires
    number_of_faces = mesh_obj.get_number_of_faces()
    dot_products = []

    #Fonction qui calcul le gradient du champ
    def grad_V(x, y):
        return np.array([a, b])  

    # Calculs les vecteur normals des faces
    normals = normal_of_line(mesh_obj)

    for i_face in range(number_of_faces):
        # Trouver les coordonnées des noeuds reliés aux faces
        face_nodes = mesh_obj.face_to_nodes[i_face]

        x1 = mesh_obj.get_node_to_xcoord(face_nodes[0])
        y1 = mesh_obj.get_node_to_ycoord(face_nodes[0])
        x2 = mesh_obj.get_node_to_xcoord(face_nodes[1])
        y2 = mesh_obj.get_node_to_ycoord(face_nodes[1])

        # Centres des faces
        face_center_x = (x1 + x2) / 2
        face_center_y = (y1 + y2) / 2

        # Vecteurs normaux des faces
        normal = normals[i_face]

        # Calculs des gradients au centre
        gradient = grad_V(face_center_x, face_center_y)

        # Calcul le produits scalaire et ajout à la matrice
        dot_product = np.dot(gradient, normal)
        dot_products.append(dot_product)

    return dot_products


def compute_total_flux(mesh_obj, a, b):
    """
    Fonction qui calcule les flux incident et sortant des éléments

    Parameters
    ----------
    mesh_obj : TYPE
        Maillage choisi.
    a : INT
        Constante du champ en x
    b : INT
        Constante du champ en y
    Returns
    -------
    flux_per_cell : array
        Matrice qui retournes les flux totaux dans les éléments.

    """
    number_of_elements = mesh_obj.get_number_of_elements()
    number_of_faces = mesh_obj.get_number_of_faces()
    
    # Matrice pour les fluxs
    flux_per_cell = np.zeros(number_of_elements)
    
    # Calculs des produits scalire du champ et des normales
    dot_products = dot_product_V_and_normals(mesh_obj, a, b)
    
    # Itération sur chaque face
    for i_face in range(number_of_faces):
        # Elements adjacents
        elements_adjacent_to_face = mesh_obj.get_face_to_elements(i_face)
        
        # Calcul du flux au travers de la face
        flux = dot_products[i_face] * face_lengths(mesh_obj)[i_face]  
        
        # Équilibrage des flux
        if len(elements_adjacent_to_face) == 2:
            # Pour un face interne
            element_1, element_2 = elements_adjacent_to_face
            flux_per_cell[element_1] -= flux 
            flux_per_cell[element_2] += flux
        elif len(elements_adjacent_to_face) == 1:
            # Pour des faces en frontières
            element = elements_adjacent_to_face[0]
            flux_per_cell[element] += flux  
    return flux_per_cell



def test_divergence(mesh_obj, a, b):
    """
    Fontion qui calcule et test si la divergence est égale à 0 dans chaque élément d'un maillage

    Parameters:
    - mesh_obj : Maillage choisi
    - a et b : Constantes d'un champ

    Returns:
    -Indique si la divergence est nulle partout dans le maillage
    """

    # Calcul des fluxs
    dot_products_list = dot_product_V_and_normals(mesh_obj, a, b)
    total_flux_per_cell = compute_total_flux(mesh_obj, a, b)
    face_lengths_list = face_lengths(mesh_obj)
    number_of_elements = mesh_obj.get_number_of_elements()
    
    #Initialise les divergences
    Div = np.zeros(number_of_elements)
    

    
    #Calcul de la divergence dans chaque éléments
    all_divergence_zero = True  
    
    for i in range(number_of_elements):
        Div[i] = Div[i] + total_flux_per_cell[i] 
        
        if Div[i] != 0:
            print(f"La divergence dans l'élément {i} est non-nulle.")
            all_divergence_zero = False
    if all_divergence_zero:
        print("La divergence est nulle dans tous les éléments.")
    else:
        print("Il y a au moins un élément avec une divergence non-nulle.")


print('')
print('Partie 1-B')
print('')
mesh_parameters = {'mesh_type': 'TRI',
                    'lc': 0.5
                    }
print('Rectangle : maillage non structuré avec des triangles')
mesh_obj = mesher.rectangle([0.0, 1.0, 0.0, 1.0], mesh_parameters)
test_divergence(mesh_obj, a, b)

#Rectangle : maillage non structuré avec des quadrilatères
print('Rectangle : maillage non structuré avec des quadrilatères')
mesh_parameters = {'mesh_type': 'QUAD',
                    'lc': 0.2
                    }
mesh_obj = mesher.rectangle([0.0, 2.0, 0.0, 1.0], mesh_parameters)
test_divergence(mesh_obj, a, b)

#Rectangle : maillage non structuré mix
print('Rectangle : maillage non structuré mix')
mesh_parameters = {'mesh_type': 'MIX',
                    'lc': 0.2
                    }
mesh_obj = mesher.rectangle([0.0, 2.0, 0.0, 1.0], mesh_parameters)
test_divergence(mesh_obj, a, b)

#Rectangle : maillage transfinis avec des triangles
print('Rectangle : maillage transfinis avec des triangles')
mesh_parameters = {'mesh_type': 'TRI',
                    'Nx': 5,
                    'Ny': 4
                    }
mesh_obj = mesher.rectangle([0.0, 1.0, 0.0, 1.0], mesh_parameters)
test_divergence(mesh_obj, a, b)

#Rectangle : maillage transfinis avec des quadrilatères
print('Rectangle : maillage transfinis avec des quadrilatères')

mesh_parameters = {'mesh_type': 'QUAD',
                    'Nx': 10,
                    'Ny': 15
                    }
mesh_obj = mesher.rectangle([0.0, 1.0, 1.0, 2.5], mesh_parameters)
test_divergence(mesh_obj, a, b)

#Rectangle : maillage transfinis mix
print('Rectangle : maillage transfinis mix')
mesh_parameters = {'mesh_type': 'MIX',
                    'Nx': 16,
                    'Ny': 10
                    }
mesh_obj = mesher.rectangle([0.0, 2.0, 0.0, 1.0], mesh_parameters)
test_divergence(mesh_obj, a, b)

#Back Step : maillage non structuré avec des triangles
print('Back Step : maillage non structuré avec des triangles')
mesh_parameters = {'mesh_type': 'TRI',
                    'lc': 0.3
                    }
mesh_obj = mesher.back_step(1.0, 1.5, 1.0, 3.0, mesh_parameters)
test_divergence(mesh_obj, a, b)

#Back Step : maillage non structuré avec des quadrilatères
print('Back Step : maillage non structuré avec des quadrilatères')
mesh_parameters = {'mesh_type': 'QUAD',
                    'lc': 0.2
                    }
mesh_obj = mesher.back_step(1.0, 1.5, 1.0, 3.0, mesh_parameters)
test_divergence(mesh_obj, a, b)

#Back Step : maillage non structuré mix
print('Back Step : maillage non structuré mix')
mesh_parameters = {'mesh_type': 'MIX',
                    'lc': 0.2
                    }
mesh_obj = mesher.back_step(1.0, 1.5, 1.0, 3.0, mesh_parameters)
test_divergence(mesh_obj, a, b)

#Cercle : maillage non structuré avec des triangles
print('Cercle : maillage non structuré avec des triangles')
mesh_parameters = {'mesh_type': 'TRI',
                    'lc_rectangle': 0.3,
                    'lc_circle': 0.1
                    }
rayon = 0.25
mesh_obj = mesher.circle([0.0, 2.0, 0.0, 1.0], rayon, mesh_parameters)
test_divergence(mesh_obj, a, b)

#Cercle : maillage non structuré avec des quadrilatères
print('Cercle : maillage non structuré avec des quadrilatères')
mesh_parameters = {'mesh_type': 'QUAD',
                    'lc_rectangle': 0.2,
                    'lc_circle': 0.05
                    }
rayon = 0.25
mesh_obj = mesher.circle([0.0, 2.0, 0.0, 1.0], rayon, mesh_parameters)
test_divergence(mesh_obj, a, b)

#Cercle : maillage transfini avec des triangles
print('Cercle : maillage transfini avec des triangles')
mesh_parameters = {'mesh_type': 'TRI',
                    'Nx': 25,
                    'Ny': 5,
                    'Nc': 20
                    }
rayon = 0.25
mesh_obj = mesher.circle([0.0, 2.0, 0.0, 1.0], rayon, mesh_parameters)
test_divergence(mesh_obj, a, b)

#Cercle : maillage transfini avec des quadrilatères
print('Cercle : maillage transfini avec des quadrilatères')
mesh_parameters = {'mesh_type': 'QUAD',
                    'Nx': 60,
                    'Ny': 20,
                    'Nc': 60
                    }
rayon = 0.25
mesh_obj = mesher.circle([0.0, 5.0, 0.0, 3.0], rayon, mesh_parameters)
test_divergence(mesh_obj, a, b)

#Quart Anneau : maillage non structuré avec des triangles
print('Quart Anneau : maillage non structuré avec des triangles')
mesh_parameters = {'mesh_type': 'TRI', 'lc': 0.2}
mesh_obj = mesher.quarter_annular(1.0, 2.0, mesh_parameters)
test_divergence(mesh_obj, a, b)

#Quart Anneau : maillage non structuré avec des quadrilatères
print('Quart Anneau : maillage non structuré avec des quadrilatères')
mesh_parameters = {'mesh_type': 'QUAD', 'lc': 0.1}
mesh_obj = mesher.quarter_annular(1.0, 2.0, mesh_parameters)
test_divergence(mesh_obj, a, b)

#Quart Anneau : maillage transfini avec des triangles
print('Quart Anneau : maillage transfini avec des triangles')
mesh_parameters = {'mesh_type': 'TRI',
                    'N1': 50,
                    'N2': 10
                    }
mesh_obj = mesher.quarter_annular(2.0, 4.0, mesh_parameters)
test_divergence(mesh_obj, a, b)

#Quart Anneau : maillage transfini avec des quadrilatères
print('Quart Anneau : maillage transfini avec des quadrilatères')
mesh_parameters = {'mesh_type': 'QUAD',
                    'N1': 50,
                    'N2': 10
                    }
mesh_obj = mesher.quarter_annular(2.0, 3.0, mesh_parameters)
test_divergence(mesh_obj, a, b)












































