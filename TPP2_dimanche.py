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
    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.rho = 1.0
        self.Cp = 1.0
        
        # Define symbolic variables
        self.x, self.y = sp.symbols('x y')
        
        # Define velocity field and temperature field
        self.u = (2*self.x**2 - self.x**4 - 1) * (self.y - self.y**3)
        self.v = -(2*self.y**2 - self.y**4 - 1) * (self.x - self.x**3)
        self.T0, self.Tx, self.Txy = 400, 50, 100
        self.T_mms = self.T0 + self.Tx * sp.cos(sp.pi * self.x) + self.Txy * sp.sin(sp.pi * self.x * self.y)
        
        # Compute source term
        self.S_func_generator = self.compute_source_term()
        
        # Convert symbolic expressions to Python functions
        self.u_func = sp.lambdify((self.x, self.y), self.u, 'numpy')
        self.v_func = sp.lambdify((self.x, self.y), self.v, 'numpy')
        self.T_mms_func = sp.lambdify((self.x, self.y), self.T_mms, 'numpy')
        
        self.mesh = None
        self.face_areas = None
        self.cell_volumes = None
        self.face_normals = None
        self.cell_centroids = None

    def compute_source_term(self):
        k = sp.symbols('k')
        convection_x = sp.diff(self.rho * self.u * self.Cp * self.T_mms, self.x)
        convection_y = sp.diff(self.rho * self.v * self.Cp * self.T_mms, self.y)
        diffusion_x = sp.diff(k * sp.diff(self.T_mms, self.x), self.x)
        diffusion_y = sp.diff(k * sp.diff(self.T_mms, self.y), self.y)
        S = convection_x + convection_y - diffusion_x - diffusion_y
        
        def S_func_generator(k_val):
            return sp.lambdify((self.x, self.y), S.subs(k, k_val), 'numpy')
        
        return S_func_generator

    def generate_mesh(self, mesh_type='QUAD', Nx=20, Ny=20):
        mesh_generator = MeshGenerator()
        mesh_parameters = {
            'mesh_type': mesh_type,
            'Nx': Nx,
            'Ny': Ny
        }
        self.mesh = mesh_generator.rectangle([self.x_min, self.x_max, self.y_min, self.y_max], mesh_parameters)
        mesh_connectivity = MeshConnectivity(self.mesh)
        mesh_connectivity.compute_connectivity()
        self.compute_mesh_properties()  # Ensure this is called after generating the mesh
        return self.mesh

    def plot_mesh(self):
        plotter = MeshPlotter()
        plotter.plot_mesh(self.mesh, label_elements=True, label_faces=True)
        self.compute_cross_diffusion_vectors()
        
    def compute_cross_diffusion_vectors(self):
        cross_diff_vectors = np.zeros((self.mesh.get_number_of_faces(), 2))
        for i_face in range(self.mesh.get_number_of_faces()):
            left_cell, right_cell = self.mesh.get_face_to_elements(i_face)
            face_center = np.mean([self.mesh.get_node_to_xycoord(node) for node in self.mesh.get_face_to_nodes(i_face)], axis=0)
            if right_cell != -1:  # Internal face
                cell_vector = self.cell_centroids[right_cell] - self.cell_centroids[left_cell]
                cross_diff_vectors[i_face] = face_center - (self.cell_centroids[left_cell] + 0.5 * cell_vector)
            else:  # Boundary face
                cross_diff_vectors[i_face] = face_center - self.cell_centroids[left_cell]
        return cross_diff_vectors

                
    def compute_mesh_properties(self):
        self.face_areas = self.compute_face_areas()
        self.cell_volumes = self.compute_cell_volumes()
        self.face_normals = self.compute_face_normals()
        self.cell_centroids = self.compute_cell_centroids()
        self.cross_diff_vectors = self.compute_cross_diffusion_vectors()  # Add this line

    def compute_face_areas(self):
        face_areas = np.zeros(self.mesh.get_number_of_faces())
        for i_face in range(self.mesh.get_number_of_faces()):
            nodes = self.mesh.get_face_to_nodes(i_face)
            x1, y1 = self.mesh.get_node_to_xycoord(nodes[0])
            x2, y2 = self.mesh.get_node_to_xycoord(nodes[1])
            face_areas[i_face] = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return face_areas

    def compute_cell_volumes(self):
        cell_volumes = np.zeros(self.mesh.get_number_of_elements())
        for i_elem in range(self.mesh.get_number_of_elements()):
            nodes = self.mesh.get_element_to_nodes(i_elem)
            if len(nodes) == 3:  # Triangle
                x1, y1 = self.mesh.get_node_to_xycoord(nodes[0])
                x2, y2 = self.mesh.get_node_to_xycoord(nodes[1])
                x3, y3 = self.mesh.get_node_to_xycoord(nodes[2])
                cell_volumes[i_elem] = 0.5 * abs((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))
            elif len(nodes) == 4:  # Quadrilateral
                x1, y1 = self.mesh.get_node_to_xycoord(nodes[0])
                x2, y2 = self.mesh.get_node_to_xycoord(nodes[1])
                x3, y3 = self.mesh.get_node_to_xycoord(nodes[2])
                x4, y4 = self.mesh.get_node_to_xycoord(nodes[3])
                cell_volumes[i_elem] = 0.5 * abs((x3 - x1)*(y4 - y2) - (x4 - x2)*(y3 - y1))
        return cell_volumes

    def compute_face_normals(self):
        face_normals = np.zeros((self.mesh.get_number_of_faces(), 2))
        for i_face in range(self.mesh.get_number_of_faces()):
            nodes = self.mesh.get_face_to_nodes(i_face)
            x1, y1 = self.mesh.get_node_to_xycoord(nodes[0])
            x2, y2 = self.mesh.get_node_to_xycoord(nodes[1])
            nx = y2 - y1
            ny = x1 - x2
            norm = np.sqrt(nx**2 + ny**2)
            face_normals[i_face] = [nx/norm, ny/norm]
        return face_normals

    def compute_cell_centroids(self):
        cell_centroids = np.zeros((self.mesh.get_number_of_elements(), 2))
        for i_elem in range(self.mesh.get_number_of_elements()):
            nodes = self.mesh.get_element_to_nodes(i_elem)
            x_coords = np.array([self.mesh.get_node_to_xcoord(node) for node in nodes])
            y_coords = np.array([self.mesh.get_node_to_ycoord(node) for node in nodes])
            cell_centroids[i_elem] = [np.mean(x_coords), np.mean(y_coords)]
        return cell_centroids

    def assemble_system(self, k, scheme='centered'):
        n_cells = self.mesh.get_number_of_elements()
        A = sparse.lil_matrix((n_cells, n_cells))
        b = np.zeros(n_cells)
        
        S_func = self.S_func_generator(k)
        
        for i_face in range(self.mesh.get_number_of_faces()):
            left_cell, right_cell = self.mesh.get_face_to_elements(i_face)
            
            if right_cell != -1:  # Internal face
                x_face, y_face = np.mean([self.mesh.get_node_to_xycoord(node) for node in self.mesh.get_face_to_nodes(i_face)], axis=0)
                u_face = self.u_func(x_face, y_face)
                v_face = self.v_func(x_face, y_face)
                velocity_dot_normal = u_face * self.face_normals[i_face, 0] + v_face * self.face_normals[i_face, 1]
                
                # Convection term
                if scheme == 'centered':
                    coeff = 0.5 * self.rho * self.Cp * velocity_dot_normal * self.face_areas[i_face]
                    A[left_cell, right_cell] -= coeff
                    A[right_cell, left_cell] += coeff
                    A[left_cell, left_cell] += coeff
                    A[right_cell, right_cell] -= coeff
                elif scheme == 'upwind':
                    coeff = self.rho * self.Cp * velocity_dot_normal * self.face_areas[i_face]
                    if velocity_dot_normal > 0:
                        A[right_cell, left_cell] += coeff
                        A[left_cell, left_cell] -= coeff
                    else:
                        A[left_cell, right_cell] -= coeff
                        A[right_cell, right_cell] += coeff
                
                # Diffusion term with cross-diffusion correction
                dx = self.cell_centroids[right_cell] - self.cell_centroids[left_cell]
                distance = np.linalg.norm(dx)
                diff_coeff = k * self.face_areas[i_face] / distance
                A[left_cell, right_cell] -= diff_coeff
                A[right_cell, left_cell] -= diff_coeff
                A[left_cell, left_cell] += diff_coeff
                A[right_cell, right_cell] += diff_coeff
                
                # Cross-diffusion correction
                cross_diff_vector = self.cross_diff_vectors[i_face]
                cross_diff_coeff = k * np.dot(cross_diff_vector, self.face_normals[i_face]) * self.face_areas[i_face] / distance**2
                A[left_cell, left_cell] += cross_diff_coeff
                A[right_cell, right_cell] += cross_diff_coeff
                A[left_cell, right_cell] -= cross_diff_coeff
                A[right_cell, left_cell] -= cross_diff_coeff
            
            else:  # Boundary face
                x_face, y_face = np.mean([self.mesh.get_node_to_xycoord(node) for node in self.mesh.get_face_to_nodes(i_face)], axis=0)
                T_boundary = self.T_mms_func(x_face, y_face)
                
                # Diffusion term with cross-diffusion correction (Dirichlet boundary condition)
                dx = np.array([x_face, y_face]) - self.cell_centroids[left_cell]
                distance = np.linalg.norm(dx)
                diff_coeff = k * self.face_areas[i_face] / distance
                A[left_cell, left_cell] += diff_coeff
                b[left_cell] += diff_coeff * T_boundary
                
                # Cross-diffusion correction for boundary
                cross_diff_vector = self.cross_diff_vectors[i_face]
                cross_diff_coeff = k * np.dot(cross_diff_vector, self.face_normals[i_face]) * self.face_areas[i_face] / distance**2
                A[left_cell, left_cell] += cross_diff_coeff
                b[left_cell] += cross_diff_coeff * T_boundary
                
                # Convection term
                u_face = self.u_func(x_face, y_face)
                v_face = self.v_func(x_face, y_face)
                velocity_dot_normal = u_face * self.face_normals[i_face, 0] + v_face * self.face_normals[i_face, 1]
                if velocity_dot_normal < 0:  # Inflow
                    conv_coeff = self.rho * self.Cp * velocity_dot_normal * self.face_areas[i_face]
                    b[left_cell] -= conv_coeff * T_boundary
        
        # Source term
        for i_cell in range(n_cells):
            x_cell, y_cell = self.cell_centroids[i_cell]
            b[i_cell] += S_func(x_cell, y_cell) * self.cell_volumes[i_cell]
        
        return A.tocsr(), b
    
    def get_boundary_type(self, x, y):
        # Define boundary types based on coordinates
        # This is an example, modify according to your specific problem
        if np.isclose(x, -1) or np.isclose(x, 1):
            return 'Dirichlet'
        elif np.isclose(y, -1) or np.isclose(y, 1):
            return 'Neumann'
        else:
            raise ValueError(f"Point ({x}, {y}) is not on the boundary")

    def get_neumann_value(self, x, y):
        # Compute the Neumann boundary condition (dT/dn) based on the manufactured solution
        dTdx = self.Tx * (-np.pi * np.sin(np.pi * x)) + self.Txy * y * np.pi * np.cos(np.pi * x * y)
        dTdy = self.Txy * x * np.pi * np.cos(np.pi * x * y)
        
        if np.isclose(y, -1):
            return -dTdy  # normal is (0, -1)
        elif np.isclose(y, 1):
            return dTdy   # normal is (0, 1)
        else:
            raise ValueError(f"Point ({x}, {y}) is not on a Neumann boundary")

    def approximate_boundary_temperature(self, cell_index, dTdn, normal):
        # Approximate boundary temperature for Neumann condition at inflow
        cell_center = self.cell_centroids[cell_index]
        distance_to_boundary = self.face_areas[cell_index] / 2  # Approximate
        T_cell = self.T_mms_func(cell_center[0], cell_center[1])
        T_boundary = T_cell - distance_to_boundary * dTdn
        return T_boundary
    def solve_system(self, A, b):
        return spla.spsolve(A, b)

    def compute_error(self, T_numerical):
        error = np.zeros_like(T_numerical)
        for i_cell in range(self.mesh.get_number_of_elements()):
            x_cell, y_cell = self.cell_centroids[i_cell]
            error[i_cell] = T_numerical[i_cell] - self.T_mms_func(x_cell, y_cell)
        
        L1_error = np.sum(np.abs(error) * self.cell_volumes) / np.sum(self.cell_volumes)
        L2_error = np.sqrt(np.sum(error**2 * self.cell_volumes) / np.sum(self.cell_volumes))
        Linf_error = np.max(np.abs(error))
        
        return L1_error, L2_error, Linf_error

    def plot_results(self, T_numerical, k, scheme):
        nodes = np.array([self.mesh.get_node_to_xycoord(i) for i in range(self.mesh.get_number_of_nodes())])
        elements = np.array([self.mesh.get_element_to_nodes(i) for i in range(self.mesh.get_number_of_elements())])
        
        # Split quadrilaterals into triangles and create a mapping from original elements to new triangles
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
        
        # Create temperature array for triangles
        T_triangles = T_numerical[triangle_to_element]
        
        tri = Triangulation(nodes[:, 0], nodes[:, 1], triangles)
        
        plt.figure(figsize=(10, 8))
        plt.tripcolor(tri, T_triangles)
        plt.colorbar(label='Temperature')
        plt.title(f'Temperature distribution (k={k:.2e}, {scheme} scheme)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
        
   

    def run_convergence_study(self, k_values, schemes, mesh_sizes):
        errors = {scheme: {k: [] for k in k_values} for scheme in schemes}
        
        for nx in mesh_sizes:
            self.generate_mesh(Nx=nx, Ny=nx)
            self.compute_mesh_properties()
            
            for scheme in schemes:
                for k in k_values:
                    A, b = self.assemble_system(k, scheme=scheme)
                    T_numerical = self.solve_system(A, b)
                    _, L2_error, _ = self.compute_error(T_numerical)
                    errors[scheme][k].append((nx, L2_error))
        
        return errors
    def calculate_convergence_order(self, errors):
       convergence_orders = {scheme: {k: [] for k in errors[scheme]} for scheme in errors}
       
       for scheme in errors:
           for k in errors[scheme]:
               error_data = errors[scheme][k]
               for i in range(0, len(error_data)):
                   h1, e1 = error_data[i-1]
                   h2, e2 = error_data[i]
                   order = np.log(e1/e2) / np.log(h2/h1)
                   convergence_orders[scheme][k].append((h2, order))
       
       return convergence_orders

def plot_convergence(mesh_sizes, errors,L):
    plt.figure(figsize=(10, 6))
    mesh_sizes1=L/mesh_sizes
    for scheme in errors:
        for k, error_values in errors[scheme].items():
            plt.loglog(mesh_sizes1, error_values, '-o', 
                       label=f'{scheme}, Pe={1/k}')
    
    plt.xlabel('Mesh size')
    plt.ylabel('L2 Error')
    plt.title('Convergence study')
    plt.legend()
    plt.grid(True)
    plt.show()
    
# def main():
#     # Create an instance of the solver
#     solver = ConvectionDiffusionSolver(-1, 1, -1, 1)

#     # Define parameters for the simulation
#     k_values = [1, 1/100, 1/10000000]  # Corresponding to Peclet numbers [1, 100, 10000]
#     schemes = ['centered', 'upwind']
#     mesh_sizes = np.array([10,50,100])
#     L=2

#     # Run convergence study
#     print("Running convergence study...")
#     errors = solver.run_convergence_study(k_values, schemes, mesh_sizes)
    
#     # Plot convergence results
#     print("Plotting convergence results...")
#     plot_convergence(mesh_sizes, errors,L)
#     print("\nCalculating convergence orders...")
#     convergence_orders = solver.calculate_convergence_order(errors)
    
#     for scheme in convergence_orders:
#         for k in convergence_orders[scheme]:
#             print(f"\nScheme: {scheme}, Pe = {1/k:.2e}")
#             print("Mesh Size | Convergence Order")
#             print("--------------------------")
#             for mesh_size, order in convergence_orders[scheme][k]:
#                 print(f"{mesh_size:9d} | {order:.4f}")
#     # Example of solution visualization and comparison for one case
#     nx = ny = 50
#     solver.generate_mesh(Nx=nx, Ny=ny)
#     solver.compute_mesh_properties()
    
#     for k in k_values:
#         for scheme in schemes:
            
#             A, b = solver.assemble_system(k, scheme=scheme)
#             T_numerical = solver.solve_system(A, b)
#             L1_error, L2_error, Linf_error = solver.compute_error(T_numerical)
            
#             solver.plot_results(T_numerical, k, scheme)
            
    

# if __name__ == "__main__":
#     main()
mesher = MeshGenerator()
plotter = MeshPlotter()
# Create an instance of the solver
solver = ConvectionDiffusionSolver(-1, 1, -1, 1)

# Define parameters for the simulation
k_values = [1, 1/100, 1/10000000]  # Corresponding to Peclet numbers [1, 100, 10000]
schemes = ['centered', 'upwind']
mesh_sizes = np.array([10,50,100])
L=2

# Run convergence study
print("Running convergence study...")
errors = solver.run_convergence_study(k_values, schemes, mesh_sizes)

# Plot convergence results
print("Plotting convergence results...")
plot_convergence(mesh_sizes, errors,L)
print("\nCalculating convergence orders...")
convergence_orders = solver.calculate_convergence_order(errors)

for scheme in convergence_orders:
    for k in convergence_orders[scheme]:
        print(f"\nScheme: {scheme}, Pe = {1/k:.2e}")
        print("Mesh Size | Convergence Order")
        print("--------------------------")
        for mesh_size, order in convergence_orders[scheme][k]:
            print(f"{mesh_size:9d} | {order:.4f}")
# Example of solution visualization and comparison for one case
nx = ny = 50
solver.generate_mesh(Nx=nx, Ny=ny)
M1=solver.generate_mesh(Nx=nx, Ny=ny)
solver.compute_mesh_properties()

for k in k_values:
    for scheme in schemes:
        
        A, b = solver.assemble_system(k, scheme=scheme)
        T_numerical = solver.solve_system(A, b)
        L1_error, L2_error, Linf_error = solver.compute_error(T_numerical)
        
        solver.plot_results(T_numerical, k, scheme)
        nodes, elements = plotter.prepare_data_for_pyvista(M1)
        pv_mesh = pv.PolyData(nodes, elements)
        pv_mesh['Température'] = T_numerical
        #contours =pv_mesh.contour()
        pl = pvQt.BackgroundPlotter()

        # Tracé du champ
        print("\nVoir le champ moyenne dans la fenêtre de PyVista \n")
        pl.add_mesh(pv_mesh, show_edges=True, scalars="Température", cmap="RdBu")
        #pl.add_mesh(contours, color="white")
