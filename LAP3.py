
"""
LAP3 Structure de donnes -QUAD et TRIANGLES
MEC6616 - Aérodynamique numérique
Date de création: 2024 - 09 - 23
Auteurs: Nicolas Tran et Mathis Turquet
"""

#%% Classes des differentes questions

class Q1A():
    def __init__(self):
        from meshGenerator import MeshGenerator

        mesher = MeshGenerator()

        def Euler(mesh_obj):
            f = mesh_obj.get_number_of_elements()
            a = mesh_obj.get_number_of_faces()
            s = mesh_obj.get_number_of_nodes()
            h = 0
            if mesh_obj.get_number_of_boundaries == 5:
                h = 1
            Euler = f - a + s + h
            if Euler == 0:
                print("Le maillage ne respecte pas la relation d'Euler")
            else:
                print("Le maillage respecte la relation d'Euler")
            return

        print('Partie 1-A')
        # Rectangle : maillage non structuré avec des triangles
        mesh_parameters = {'mesh_type': 'TRI',
                           'lc': 0.5
                           }
        print('Rectangle : maillage non structuré avec des triangles')
        mesh_obj = mesher.rectangle([0.0, 1.0, 0.0, 1.0], mesh_parameters)
        Euler(mesh_obj)

        # Rectangle : maillage non structuré avec des quadrilatères
        print('Rectangle : maillage non structuré avec des quadrilatères')
        mesh_parameters = {'mesh_type': 'QUAD',
                           'lc': 0.2
                           }
        mesh_obj = mesher.rectangle([0.0, 2.0, 0.0, 1.0], mesh_parameters)
        Euler(mesh_obj)

        # Rectangle : maillage non structuré mix
        print('Rectangle : maillage non structuré mix')
        mesh_parameters = {'mesh_type': 'MIX',
                           'lc': 0.2
                           }
        mesh_obj = mesher.rectangle([0.0, 2.0, 0.0, 1.0], mesh_parameters)
        Euler(mesh_obj)

        # Rectangle : maillage transfinis avec des triangles
        print('Rectangle : maillage transfinis avec des triangles')
        mesh_parameters = {'mesh_type': 'TRI',
                           'Nx': 5,
                           'Ny': 4
                           }
        mesh_obj = mesher.rectangle([0.0, 1.0, 0.0, 1.0], mesh_parameters)
        Euler(mesh_obj)

        # Rectangle : maillage transfinis avec des quadrilatères
        print('Rectangle : maillage transfinis avec des quadrilatères')

        mesh_parameters = {'mesh_type': 'QUAD',
                           'Nx': 10,
                           'Ny': 15
                           }
        mesh_obj = mesher.rectangle([0.0, 1.0, 1.0, 2.5], mesh_parameters)
        Euler(mesh_obj)

        # Rectangle : maillage transfinis mix
        print('Rectangle : maillage transfinis mix')
        mesh_parameters = {'mesh_type': 'MIX',
                           'Nx': 16,
                           'Ny': 10
                           }
        mesh_obj = mesher.rectangle([0.0, 2.0, 0.0, 1.0], mesh_parameters)
        Euler(mesh_obj)

        # Back Step : maillage non structuré avec des triangles
        print('Back Step : maillage non structuré avec des triangles')
        mesh_parameters = {'mesh_type': 'TRI',
                           'lc': 0.3
                           }
        mesh_obj = mesher.back_step(1.0, 1.5, 1.0, 3.0, mesh_parameters)
        Euler(mesh_obj)

        # Back Step : maillage non structuré avec des quadrilatères
        print('Back Step : maillage non structuré avec des quadrilatères')
        mesh_parameters = {'mesh_type': 'QUAD',
                           'lc': 0.2
                           }
        mesh_obj = mesher.back_step(1.0, 1.5, 1.0, 3.0, mesh_parameters)
        Euler(mesh_obj)

        # Back Step : maillage non structuré mix
        print('Back Step : maillage non structuré mix')
        mesh_parameters = {'mesh_type': 'MIX',
                           'lc': 0.2
                           }
        mesh_obj = mesher.back_step(1.0, 1.5, 1.0, 3.0, mesh_parameters)
        Euler(mesh_obj)

        # Cercle : maillage non structuré avec des triangles
        print('Cercle : maillage non structuré avec des triangles')
        mesh_parameters = {'mesh_type': 'TRI',
                           'lc_rectangle': 0.3,
                           'lc_circle': 0.1
                           }
        rayon = 0.25
        mesh_obj = mesher.circle([0.0, 2.0, 0.0, 1.0], rayon, mesh_parameters)
        Euler(mesh_obj)

        # Cercle : maillage non structuré avec des quadrilatères
        print('Cercle : maillage non structuré avec des quadrilatères')
        mesh_parameters = {'mesh_type': 'QUAD',
                           'lc_rectangle': 0.2,
                           'lc_circle': 0.05
                           }
        rayon = 0.25
        mesh_obj = mesher.circle([0.0, 2.0, 0.0, 1.0], rayon, mesh_parameters)
        Euler(mesh_obj)

        # Cercle : maillage transfini avec des triangles
        print('Cercle : maillage transfini avec des triangles')
        mesh_parameters = {'mesh_type': 'TRI',
                           'Nx': 25,
                           'Ny': 5,
                           'Nc': 20
                           }
        rayon = 0.25
        mesh_obj = mesher.circle([0.0, 2.0, 0.0, 1.0], rayon, mesh_parameters)

        Euler(mesh_obj)

        # Cercle : maillage transfini avec des quadrilatères
        print('Cercle : maillage transfini avec des quadrilatères')
        mesh_parameters = {'mesh_type': 'QUAD',
                           'Nx': 60,
                           'Ny': 20,
                           'Nc': 60
                           }
        rayon = 0.25
        mesh_obj = mesher.circle([0.0, 5.0, 0.0, 3.0], rayon, mesh_parameters)
        Euler(mesh_obj)

        # Quart Anneau : maillage non structuré avec des triangles
        print('Quart Anneau : maillage non structuré avec des triangles')
        mesh_parameters = {'mesh_type': 'TRI', 'lc': 0.2}
        mesh_obj = mesher.quarter_annular(1.0, 2.0, mesh_parameters)
        Euler(mesh_obj)

        # Quart Anneau : maillage non structuré avec des quadrilatères
        print('Quart Anneau : maillage non structuré avec des quadrilatères')
        mesh_parameters = {'mesh_type': 'QUAD', 'lc': 0.1}
        mesh_obj = mesher.quarter_annular(1.0, 2.0, mesh_parameters)
        Euler(mesh_obj)

        # Quart Anneau : maillage transfini avec des triangles
        print('Quart Anneau : maillage transfini avec des triangles')
        mesh_parameters = {'mesh_type': 'TRI',
                           'N1': 50,
                           'N2': 10
                           }
        mesh_obj = mesher.quarter_annular(2.0, 4.0, mesh_parameters)
        Euler(mesh_obj)

        # Quart Anneau : maillage transfini avec des quadrilatères
        print('Quart Anneau : maillage transfini avec des quadrilatères')
        mesh_parameters = {'mesh_type': 'QUAD',
                           'N1': 50,
                           'N2': 10
                           }
        mesh_obj = mesher.quarter_annular(2.0, 3.0, mesh_parameters)
        Euler(mesh_obj)
        return

    

class Q1B():
    def __init__(self):
        import sympy as sp
        import numpy as np
        from meshGenerator import MeshGenerator

        mesher = MeshGenerator()
        x, y = sp.symbols('x y')  # Variables symboliques
        # Création du champ
        a = 1
        b = 1

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
            # Création de la matrice des normales
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
            # Création de la matrice des longueurs
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
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

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
            # Création de la matrice des produits scalaires
            number_of_faces = mesh_obj.get_number_of_faces()
            dot_products = []

            # Fonction qui calcul le gradient du champ
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
            total_flux_per_cell = compute_total_flux(mesh_obj, a, b)
            number_of_elements = mesh_obj.get_number_of_elements()

            # Initialise les divergences
            Div = np.zeros(number_of_elements)

            # Calcul de la divergence dans chaque éléments
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

        # Rectangle : maillage non structuré avec des quadrilatères
        print('Rectangle : maillage non structuré avec des quadrilatères')
        mesh_parameters = {'mesh_type': 'QUAD',
                           'lc': 0.2
                           }
        mesh_obj = mesher.rectangle([0.0, 2.0, 0.0, 1.0], mesh_parameters)
        test_divergence(mesh_obj, a, b)

        # Rectangle : maillage non structuré mix
        print('Rectangle : maillage non structuré mix')
        mesh_parameters = {'mesh_type': 'MIX',
                           'lc': 0.2
                           }
        mesh_obj = mesher.rectangle([0.0, 2.0, 0.0, 1.0], mesh_parameters)
        test_divergence(mesh_obj, a, b)

        # Rectangle : maillage transfinis avec des triangles
        print('Rectangle : maillage transfinis avec des triangles')
        mesh_parameters = {'mesh_type': 'TRI',
                           'Nx': 5,
                           'Ny': 4
                           }
        mesh_obj = mesher.rectangle([0.0, 1.0, 0.0, 1.0], mesh_parameters)
        test_divergence(mesh_obj, a, b)

        # Rectangle : maillage transfinis avec des quadrilatères
        print('Rectangle : maillage transfinis avec des quadrilatères')

        mesh_parameters = {'mesh_type': 'QUAD',
                           'Nx': 10,
                           'Ny': 15
                           }
        mesh_obj = mesher.rectangle([0.0, 1.0, 1.0, 2.5], mesh_parameters)
        test_divergence(mesh_obj, a, b)

        # Rectangle : maillage transfinis mix
        print('Rectangle : maillage transfinis mix')
        mesh_parameters = {'mesh_type': 'MIX',
                           'Nx': 16,
                           'Ny': 10
                           }
        mesh_obj = mesher.rectangle([0.0, 2.0, 0.0, 1.0], mesh_parameters)
        test_divergence(mesh_obj, a, b)

        # Back Step : maillage non structuré avec des triangles
        print('Back Step : maillage non structuré avec des triangles')
        mesh_parameters = {'mesh_type': 'TRI',
                           'lc': 0.3
                           }
        mesh_obj = mesher.back_step(1.0, 1.5, 1.0, 3.0, mesh_parameters)
        test_divergence(mesh_obj, a, b)

        # Back Step : maillage non structuré avec des quadrilatères
        print('Back Step : maillage non structuré avec des quadrilatères')
        mesh_parameters = {'mesh_type': 'QUAD',
                           'lc': 0.2
                           }
        mesh_obj = mesher.back_step(1.0, 1.5, 1.0, 3.0, mesh_parameters)
        test_divergence(mesh_obj, a, b)

        # Back Step : maillage non structuré mix
        print('Back Step : maillage non structuré mix')
        mesh_parameters = {'mesh_type': 'MIX',
                           'lc': 0.2
                           }
        mesh_obj = mesher.back_step(1.0, 1.5, 1.0, 3.0, mesh_parameters)
        test_divergence(mesh_obj, a, b)

        # Cercle : maillage non structuré avec des triangles
        print('Cercle : maillage non structuré avec des triangles')
        mesh_parameters = {'mesh_type': 'TRI',
                           'lc_rectangle': 0.3,
                           'lc_circle': 0.1
                           }
        rayon = 0.25
        mesh_obj = mesher.circle([0.0, 2.0, 0.0, 1.0], rayon, mesh_parameters)
        test_divergence(mesh_obj, a, b)

        # Cercle : maillage non structuré avec des quadrilatères
        print('Cercle : maillage non structuré avec des quadrilatères')
        mesh_parameters = {'mesh_type': 'QUAD',
                           'lc_rectangle': 0.2,
                           'lc_circle': 0.05
                           }
        rayon = 0.25
        mesh_obj = mesher.circle([0.0, 2.0, 0.0, 1.0], rayon, mesh_parameters)
        test_divergence(mesh_obj, a, b)

        # Cercle : maillage transfini avec des triangles
        print('Cercle : maillage transfini avec des triangles')
        mesh_parameters = {'mesh_type': 'TRI',
                           'Nx': 25,
                           'Ny': 5,
                           'Nc': 20
                           }
        rayon = 0.25
        mesh_obj = mesher.circle([0.0, 2.0, 0.0, 1.0], rayon, mesh_parameters)
        test_divergence(mesh_obj, a, b)

        # Cercle : maillage transfini avec des quadrilatères
        print('Cercle : maillage transfini avec des quadrilatères')
        mesh_parameters = {'mesh_type': 'QUAD',
                           'Nx': 60,
                           'Ny': 20,
                           'Nc': 60
                           }
        rayon = 0.25
        mesh_obj = mesher.circle([0.0, 5.0, 0.0, 3.0], rayon, mesh_parameters)
        test_divergence(mesh_obj, a, b)

        # Quart Anneau : maillage non structuré avec des triangles
        print('Quart Anneau : maillage non structuré avec des triangles')
        mesh_parameters = {'mesh_type': 'TRI', 'lc': 0.2}
        mesh_obj = mesher.quarter_annular(1.0, 2.0, mesh_parameters)
        test_divergence(mesh_obj, a, b)

        # Quart Anneau : maillage non structuré avec des quadrilatères
        print('Quart Anneau : maillage non structuré avec des quadrilatères')
        mesh_parameters = {'mesh_type': 'QUAD', 'lc': 0.1}
        mesh_obj = mesher.quarter_annular(1.0, 2.0, mesh_parameters)
        test_divergence(mesh_obj, a, b)

        # Quart Anneau : maillage transfini avec des triangles
        print('Quart Anneau : maillage transfini avec des triangles')
        mesh_parameters = {'mesh_type': 'TRI',
                           'N1': 50,
                           'N2': 10
                           }
        mesh_obj = mesher.quarter_annular(2.0, 4.0, mesh_parameters)
        test_divergence(mesh_obj, a, b)

        # Quart Anneau : maillage transfini avec des quadrilatères
        print('Quart Anneau : maillage transfini avec des quadrilatères')
        mesh_parameters = {'mesh_type': 'QUAD',
                           'N1': 50,
                           'N2': 10
                           }
        mesh_obj = mesher.quarter_annular(2.0, 3.0, mesh_parameters)
        test_divergence(mesh_obj, a, b)
        return

    
        
class Q2():
    def __init__(self):
        import sympy as sp
        import numpy as np
        import pyvista as pv
        import pyvistaqt as pvQt
        from meshGenerator import MeshGenerator
        from meshConnectivity import MeshConnectivity
        from meshPlotter import MeshPlotter
        from mesh import Mesh
        import matplotlib.pyplot as plt 

        ## test récupération nombre de faces  
        mesher = MeshGenerator()
        plotter = MeshPlotter()



        def normal(xa, ya, xb, yb):
            """
            Fonction pour renvoyer a et b du vecteur normal unitaire à une arête 

            Parameters
            ----------
            xa, ya : Coordonnées du point A
            xb, yb : Coordonnées du point B

            Returns
            -------
            a, b : Composantes du vecteur normal unitaire
            """
            dx = xb - xa
            dy = yb - ya
            
            # Calcul de la longueur de l'arête (norme du vecteur)
            Delta_A = np.sqrt(dx**2 + dy**2)
            
           
            a = dy / Delta_A
            b = - dx / Delta_A
            
            return a, b


        def centre_element_2D(mesh_obj,i_element):
            """
            Fonction pour renvoyer  xmoy et ymoy centre des faces et lieu des éléments géométriques en 2D 

            Parameters
            ----------
            Maillage et numéro de l'élément

            Returns
            -------
            xmoy et y moy 
            """
            # Récupération des données des sommets 
            start = mesh_obj.get_element_to_nodes_start(i_element)
            fin = mesh_obj.get_element_to_nodes_start(i_element+1)
            noeuds_i_elements = mesh_obj.element_to_nodes[start:fin] # Création des listes pour les noeuds autour d'un élément
            # Calcul du centre géométrique du triangle
            xmoy = 0 # Initilaisation des coordonnées virtuelles de l'élément 
            ymoy = 0
            for i in range(len(noeuds_i_elements)):
                x,y = mesh_obj.get_node_to_xycoord(noeuds_i_elements[i])
                xmoy += x
                ymoy += y 
            xmoy = xmoy/len(noeuds_i_elements)
            ymoy = ymoy/len(noeuds_i_elements)
            return xmoy,ymoy

        def phi(x,y):
            """
            Fonction pour rcrééer le champ 

            Parameters
            ----------
            coordonnées x et y 

            Returns
            -------
            Phi, le champ 
            """
            #np.sin(x)+ np.cos(y)
            return  x*2 +y*2


        mesh_parameters = {'mesh_type': 'TRI',
                           'lc': 10.0
                           }
        mesh_obj = mesher.rectangle([0.0, 10.0, 0.0, 10.0], mesh_parameters)
        conec = MeshConnectivity(mesh_obj)
        conec.compute_connectivity()
        #plotter.plot_mesh(mesh_obj)


        def test_euler(mesh_object):
            f = mesh_obj.number_of_elements
            a =mesh_obj.number_of_faces
            s = mesh_obj.number_of_nodes
            h = 0
            if f-a+s+h ==1:
                return True 
            

           
        def least_square(mesh_obj,bcdata,phi):
            
            bc_types = [item[0] for item in bcdata]  # Extrait la première valeur de chaque tuple
            bc_numbers = [item[1] for item in bcdata]
            
            #Création des données
            number_of_faces = mesh_obj.number_of_faces
            number_of_elements = mesh_obj.number_of_elements
            
            # Création des coordonnées des éléments 
            
            coordonnees_elements_x = []
            coordonnees_elements_y = []
            
            voisins = []
            
            f = mesh_obj.number_of_elements
            a =mesh_obj.number_of_faces
            s = mesh_obj.number_of_nodes
            bcdata =mesh_obj.get_boundary_faces_to_tag()
            
            # Création de la matrice de calcul ATA 
            
            ATA = np.zeros((f,2,2))
            
            # Création de la matrice B 
            
            B = np.zeros((f,2))
            
            for i in range(number_of_elements):
                
                # Récupération des différents noeuds autour d'un éléments
                
                start = mesh_obj.get_element_to_nodes_start(i)
                fin = mesh_obj.get_element_to_nodes_start(i+1)
                noeuds_i_elements = mesh_obj.element_to_nodes[start:fin] 
                
                
                # Création du centre gémoétrique des différentes formes 
                
                x_e, y_e = centre_element_2D(mesh_obj, i)
                
                # Récupération dans deux listes distinctes 
                
                coordonnees_elements_x.append(x_e)
                coordonnees_elements_y.append(y_e)  
                
            # Parcours de l'ensemble des arrêtes pour la création des différentes matrices 
            
            for i in range(number_of_faces):
                neighbours_elements = mesh_obj.get_face_to_elements(i)
                # Récupération des éléments voisins
                voisins.append(neighbours_elements)
                
                Tg,Td = neighbours_elements
                
                if neighbours_elements[1] == -1 : # pour les faces frontières 
                    
                    tag = mesh_obj.get_boundary_face_to_tag(i)
                    bc_type = bc_types[tag]
                    bc_number = bc_numbers [tag]
                    
                    
                    
                    if ( bc_type == 'DIRICHLET' ) :
                        # DIRICHLET
                        # Création du centre de l'arrête 
                        noeuds_faces = mesh_obj.get_face_to_nodes(i)
                        
                        # Milieu
                        
                        xa = mesh_obj.get_node_to_xcoord(noeuds_faces[0])
                        ya = mesh_obj.get_node_to_ycoord(noeuds_faces[0])
                        xb = mesh_obj.get_node_to_xcoord(noeuds_faces[1])
                        yb = mesh_obj.get_node_to_xcoord(noeuds_faces[1])
                        
                        
                        x_milieu = (xa+xb)/2
                        y_milieu = (yb+ya)/2
                        
                        # Dx et Dy
                        xtg = coordonnees_elements_x[neighbours_elements[0]]
                        ytg = coordonnees_elements_y[neighbours_elements[0]]
                        
                        dx = (xtg - x_milieu)
                        dy = (ytg - y_milieu)
                        
                        # Pour les arêtes internes 
                        
                        ALS = np.zeros((2,2))
                        
                        # Création des différents paramèetres de la matrice 2x2
                        
                        ALS[0,0]= dx*dx
                        ALS[1,0]= dx*dy
                        ALS[0,1]= dy*dx
                        ALS[1,1]= dy*dy
                        # Remplissage
                        
                        ATA[Tg] += ALS
                        
                        Phi_A =  phi(x_milieu,y_milieu)
                        Phi_tg = phi(xtg,ytg)
                        
                        B[Tg,0] = B[Tg,0] + (x_milieu - xtg) * (Phi_A - Phi_tg)
                        B[Tg,1] = B[Tg,1] + (y_milieu - ytg) * (Phi_A - Phi_tg)
                        
                        
                    if ( bc_type == 'NEUMANN' ) :
                        # Neumann
                        noeuds_faces = mesh_obj.get_face_to_nodes(i)
                        
                        xa = mesh_obj.get_node_to_xcoord(noeuds_faces[0])
                        ya = mesh_obj.get_node_to_ycoord(noeuds_faces[0])
                        xb = mesh_obj.get_node_to_xcoord(noeuds_faces[1])
                        yb = mesh_obj.get_node_to_xcoord(noeuds_faces[1])
                        
                        nx,ny = normal(xa,ya,xb,yb)
                        
                        xtg = coordonnees_elements_x[neighbours_elements[0]]
                        ytg = coordonnees_elements_y[neighbours_elements[0]] 
                        
                        x_milieu = ( mesh_obj.get_node_to_xcoord(noeuds_faces[1]) + mesh_obj.get_node_to_xcoord(noeuds_faces[0]))/2
                        y_milieu = ( mesh_obj.get_node_to_ycoord(noeuds_faces[1]) + mesh_obj.get_node_to_ycoord(noeuds_faces[0]))/2
                        
                        dx1 =((x_milieu-xtg))
                        dy1 = ((y_milieu-ytg))
                        
                        dx = (dx1*nx+dy1*ny)*nx
                        dy = (dx1*nx + dy1*ny)*ny
                        
                        # Matric intermédiaire
                        ALS = np.zeros((2,2))
                        
                        # Création des différents paramètres de la matrice 2x2
                        
                        ALS[0,0]= dx*dx
                        ALS[1,0]= ALS[0,1]= dy*dx
                        ALS[1,1]= dy*dy
                        
                        
                        
                        # Remplissage 
                        ATA[Tg] += ALS
                        
                        xtg = coordonnees_elements_x[neighbours_elements[0]]
                        ytg = coordonnees_elements_y[neighbours_elements[0]]
                        
                        dx = ((x_milieu-xtg)*nx + (y_milieu-ytg)*ny)*nx
                        dy = ((x_milieu-xtg)*nx + (y_milieu-ytg)*ny)*ny
                        
                        Phi_A = bc_number # phi(x_milieu,y_milieu) # Phi milieu 
                        Phi_N = bc_number
                        
                        delta_phi = ((x_milieu-xtg)*nx + (y_milieu-ytg)*ny)*Phi_N
                        
                        B[Tg,0] = B[Tg,0] + dx * delta_phi
                        B[Tg,1] = B[Tg,1] + dy * delta_phi
                        
                        
                        
                    if (bc_type == 'LIBRE'):
                       B[Tg,0] = B[Tg,0] 
                       B[Tg,1] = B[Tg,1] 

                # Pour les arrêtes internes
                
                if neighbours_elements[1] != -1 :
                    
                    # Coordonnées X triangles 
                    
                    xtg = coordonnees_elements_x[neighbours_elements[0]]
                    xtd = coordonnees_elements_x[neighbours_elements[1]]
                    
                    # Coordonnées Y triangles
                    
                    ytg = coordonnees_elements_y[neighbours_elements[0]]
                    ytd = coordonnees_elements_y[neighbours_elements[1]]
                    
                    # Récupération des dx et dy pour une arrête
                    
                    dx = xtg - xtd
                    dy = ytg - ytd
                    
                    # Pour les arêtes internes
                    
                    ALS = np.zeros((2,2))
                    
                    # Création des différents paramèetres de la matrice 2x2
                    
                    ALS[0,0]= dx*dx
                    ALS[1,0]= dx*dy
                    ALS[0,1]= dy*dx
                    ALS[1,1]= dy*dy
                    
                    ATA[Tg]+= ALS
                    ATA[Td]+= ALS
                    
                    # Remplissage de B 
                    
                    Phi_td = phi(xtd,ytd)
                    Phi_tg = phi(xtg,ytg)
                    
                    B[Tg,0] = B[Tg,0] + (xtd-xtg) * (Phi_td -Phi_tg)
                    B[Tg,1] = B[Tg,1] + (ytd-ytg) * (Phi_td -Phi_tg)
                    B[Td,0] = B[Td,0] + (xtd-xtg) * (Phi_td -Phi_tg)
                    B[Td,1] = B[Td,1] + (ytd-ytg) * (Phi_td -Phi_tg)
                    
            # Création de la matrice ATAI    
            
            ATAI = np.zeros((f,2,2))
            
            # BOucle sur les triangles 
            
            for i in range(number_of_elements):
                
                AL = ATA[i] # Sélection de chaque élément
                ALI = np.linalg.inv(AL) # Inversion de chaque élément
                ATAI[i]= ALI # Ajout à la nouvelle matrice
            
            # Création du gradient 
            
            Grad =np.zeros((f,2))
            
            # Résolution numérique 
            
            for i in range(number_of_elements):
                
                Grad[i] = np.dot(ATAI[i],B[i]) # Multiplication des deux matrices
                
            return Grad



        bcdata = (['DIRICHLET', 2 ], ['DIRICHLET', 2 ],
                  ['DIRICHLET', 2], ['DIRICHLET', 2])



        f = mesh_obj.number_of_elements
        #%% Création de la fonction analytique 

        def analytique(f,mesh_obj):
            
            Grad = np.zeros((f,2))
            coordonnees_elements_x = []
            coordonnees_elements_y = []
            for i in range(f):
                x_e, y_e = centre_element_2D(mesh_obj, i)
                coordonnees_elements_x.append(x_e)
                coordonnees_elements_y.append(y_e)  
                Grad[i][0] =  2
                Grad[i][1] = 3
            return Grad


        #%%

        def surface_cellule(noeuds, mesh_obj):
            n = len(noeuds)
            surface = 0
            points = []
            
            # Récupérer les coordonnées (x, y) des noeuds associés à la cellule
            for i in range(n):
                x, y = mesh_obj.get_node_to_xycoord(noeuds[i])  # Utiliser noeuds[i] pour les bons indices
                points.append((x, y))
                
            # Calcul de la surface avec la formule du déterminant (shoelace)
            for i in range(n):
                x1, y1 = points[i]
                x2, y2 = points[(i + 1) % n]  # Le sommet suivant, en bouclant sur le premier point
                surface += x1 * y2 - x2 * y1
            
            return abs(surface) / 2

        def surface_moyenne(sol_num,mesh_obj):
            moyenne = 0
            surface_total = 0
            
            for i in range(len(sol_num)):
                # Récupération des nœuds associés à l'élément i
                start = mesh_obj.get_element_to_nodes_start(i)
                fin = mesh_obj.get_element_to_nodes_start(i + 1)
                noeuds_i_elements = mesh_obj.element_to_nodes[start:fin]
                
                # Calcul de la surface de la cellule
                taille_cellule = surface_cellule(noeuds_i_elements, mesh_obj)
                moyenne += taille_cellule**2
            moyenne = np.sqrt(moyenne/len(sol_num))
            
            return moyenne

        def erreur_L2(sol_num, sol_anal, mesh_obj):
            erreur = 0
            surface_total = 0
            
            for i in range(len(sol_num)):
                # Récupération des nœuds associés à l'élément i
                start = mesh_obj.get_element_to_nodes_start(i)
                fin = mesh_obj.get_element_to_nodes_start(i + 1)
                noeuds_i_elements = mesh_obj.element_to_nodes[start:fin]
                
                # Calcul de la surface de la cellule
                taille_cellule = surface_cellule(noeuds_i_elements, mesh_obj)
                
                # Calcul des écarts pondérés par la taille de la cellule
                x = ((sol_num[i][0] - sol_anal[i][0])**2) * taille_cellule
                y= ((sol_num[i][1] - sol_anal[i][1])**2) * taille_cellule
                
                surface_total += taille_cellule
                
                erreur += (x+y)
                
            
            
            erreur =np.sqrt(erreur/len(sol_anal))
            # print(erreur)
            
            # Optionnel : affichage pour vérifier les calculs
            # print(f"Surface totale: {surface_total}")
            
            return erreur


        def erreur_infinie(sol_num, sol_anal, mesh_obj):
            max_erreur_x = 0
            max_erreur_y = 0
            
            for i in range(len(sol_num)):
                # Calcul des écarts absolus pour chaque composante (x et y)
                ecart_x = abs(sol_num[i][0] - sol_anal[i][0])
                ecart_y = abs(sol_num[i][1] - sol_anal[i][1])
                
                # Mise à jour des erreurs maximales
                max_erreur_x = max(max_erreur_x, ecart_x)
                max_erreur_y = max(max_erreur_y, ecart_y)
            
            # Calcul de la norme infinie comme le maximum entre les deux composantes
            norme_infinie = max(max_erreur_x, max_erreur_y)
            
            return max_erreur_x, max_erreur_y, norme_infinie



        def OrdreConvergeance(x, y):
            dx = np.diff(x)
            dy = np.diff(y)
            slope = dy / dx
            return slope

        #%% Création du code pour les erreurs
        def phi1(x,y):
            return x**2 + y**2


        def analytique1(f,mesh_obj):
            
            Grad = np.zeros((f,2))
            coordonnees_elements_x = []
            coordonnees_elements_y = []
            for i in range(f):
                x_e, y_e = centre_element_2D(mesh_obj, i)
                coordonnees_elements_x.append(x_e)
                coordonnees_elements_y.append(y_e)  
                Grad[i][0] =  2*x_e
                Grad[i][1] =  2*y_e
            return Grad


        # Création des points de difference  
        lc= .5
        n=6
        a1= erreur_infinie(least_square(mesh_obj,bcdata,phi),analytique(f,mesh_obj),mesh_obj)
        h1 =1/f 
        H= []
        T = []
        bcdata1 = (['DIRICHLET', 2 ], ['DIRICHLET', 2 ],
                  ['DIRICHLET', 2], ['DIRICHLET', 2])
        lc_list=np.zeros(n)
        for i in range(n):
            if i==0:
                lc_list[i]=lc
            else:
                lc_list[i]=lc*(i+1)/(2**(i+1))
            
        for i in range(n):
            mesh_parameters2 = {'mesh_type': 'QUAD',
                               'lc': lc_list[i]
                               }
            mesh_obj1= mesher.rectangle([0.0, 1.0, 0.0, 1.0], mesh_parameters2)
            conec = MeshConnectivity(mesh_obj1)
            conec.compute_connectivity()
            
            
            f2 = mesh_obj1.number_of_elements
            calcul = least_square(mesh_obj1,bcdata1,phi1)
            normalien= analytique1(f2,mesh_obj1)

            a2 = erreur_L2(calcul,normalien,mesh_obj1)

            f2 = mesh_obj1.number_of_elements
            h2 = surface_moyenne(calcul,mesh_obj1)
            t2 = a2
            
            H.append(h2)
            T.append(t2)

           
           
        plt.figure(2)
        plt.loglog(H, T,'-o', label="Erreur T")
        plt.legend()
        plt.xlabel("h")
        plt.ylabel("Erreur(h)")
        plt.legend()
        # plt.axis([10**-4, 10**-2, 10**-3, 10**-1])
        plt.grid(True)
        plt.title("Convergence de l'erreur ")
        plt.show()

        #Ordre de convergeance
        D=str(max(abs(OrdreConvergeance(np.log(H),np.log(T)))))
        print("La convergeance de l'erreur pour les fonctions complexes est de :"+ D)
        return
    