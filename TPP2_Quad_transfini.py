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

    def assemble_system(self, k, scheme='centré'):
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
                if scheme == 'centré':
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
        
        L1_error = np.sum(np.abs(error) * self.cell_volumes) / (self.mesh.get_number_of_elements())
        L2_error = np.sqrt(np.sum(error**2 * self.cell_volumes) / (self.mesh.get_number_of_elements()))
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
        plt.title(f'Distribution Temperature  (Pe={1/k}, {scheme} scheme)')
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



#%%
class ConvectionDiffusionAnalyzer:
    def __init__(self, x_min, x_max, y_min, y_max,mesh_sizes):
        self.solver = ConvectionDiffusionSolver(x_min, x_max, y_min, y_max)
        self.mesh = None
        self.T_mms_values = None
        self.mesh_sizes = mesh_sizes

    def generate_mesh(self, nx, ny):
        self.mesh = self.solver.generate_mesh(Nx=nx, Ny=ny)
        return self.mesh

    def plot_convergence(self, mesh_sizes, errors,L):
        plt.figure(figsize=(15, 5))
        error_types = ['L1', 'L2', 'Linf']
        mesh_sizes1=L/mesh_sizes
        for i, error_type in enumerate(error_types):
            plt.subplot(1, 3, i+1)
            for scheme in errors:
                for k, error_values in errors[scheme][error_type].items():
                    plt.loglog(mesh_sizes1, error_values, '-o', label=f'{scheme}, Pe={1/k:.0e}')
            
            plt.xlabel('Mesh size')
            plt.ylabel(f'{error_type} Erreur')
            plt.title(f'{error_type} Convergence')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    
    def run_convergence_study(self, k_values, schemes, mesh_sizes):
        errors = {scheme: {error_type: {k: [] for k in k_values} for error_type in ['L1', 'L2', 'Linf']} for scheme in schemes}
        
        for nx in mesh_sizes:
            self.generate_mesh(nx, nx)
            self.solver.compute_mesh_properties()
            
            for scheme in schemes:
                for k in k_values:
                    A, b = self.solver.assemble_system(k, scheme=scheme)
                    T_numerical = self.solver.solve_system(A, b)
                    L1_error, L2_error, Linf_error = self.solver.compute_error(T_numerical)
                    
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
                
    def plot_results(self, T_numerical, k, scheme):
        grid = self.create_pyvista_grid(T_numerical)
        
        pl = pvQt.BackgroundPlotter()
        pl.add_mesh(grid, show_edges=True, scalars="Temperature", cmap="RdBu")
        contours = grid.contour(scalars='Temperature', isosurfaces=10)
        pl.add_mesh(contours, color="white", line_width=2)
        pl.add_text(f"Distribution Temperature  (Pe={1/k}, scheme={scheme})", font_size=12)
        pl.show()

    def get_temperature_profile(self, grid, start_point, end_point):
        line = grid.sample_over_line(start_point, end_point, resolution=100)
        return line['Distance'], line['Temperature']

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
            grid.cell_data['Temperature'] = T_values
            grid = grid.cell_data_to_point_data()
        elif len(T_values) == grid.n_points:
            grid.point_data['Temperature'] = T_values
        
        return grid

    def plot_cross_section_profiles(self, profiles, analytical_profile, title, xlabel):
        plt.figure(figsize=(12, 8))
        
        for k, scheme, (distance, temperature) in profiles:
            plt.plot(distance, temperature, label=f'Pe={1/k}, {scheme}')
        
        plt.plot(analytical_profile[0], analytical_profile[1], 'k--', label='Analytical')
        
        plt.xlabel(xlabel)
        plt.ylabel('Temperature')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

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
        
        pl = pvQt.BackgroundPlotter()
        pl.add_mesh(grid, scalars='Temperature', show_edges=True, cmap="RdBu", clim=[self.T_mms_values.min(), self.T_mms_values.max()])
        contours = grid.contour(scalars='Temperature', isosurfaces=10)
        pl.add_mesh(contours, color="white", line_width=2)
        pl.add_text('T_mms Distribution', font_size=12)
        pl.show()

    def run_analysis(self, k_values, schemes, nx, ny,mesh_sizes):
        self.generate_mesh(nx, ny)
        self.plot_T_mms()
        x_min, x_max = -1, 1
        y_min, y_max = -1, 1
        z_min, z_max = 0, 0
        L=x_max-x_min
        cross_sections = [
            ("Coupe Verticale à x=0", [0, y_min, z_min], [0, y_max, z_max], "Distance  y"),
            ("Coupe Verticale à x=0.5", [0.5, y_min, z_min], [0.5, y_max, z_max], "Distance  y"),
            ("Coupe Horizontale à y=0", [x_min, 0, z_min], [x_max, 0, z_max], "Distance  x"),
            ("Coupe Horizontale à y=0.5", [x_min, 0.5, z_min], [x_max, 0.5, z_max], "Distance  x")
        ]

        for k in k_values:
            for scheme in schemes:
                A, b = self.solver.assemble_system(k, scheme=scheme)
                T_numerical = self.solver.solve_system(A, b)
                self.plot_results(T_numerical, k, scheme)

        for section_name, start_point, end_point, xlabel in cross_sections:
            profiles = []
            
            for k in k_values:
                for scheme in schemes:
                    A, b = self.solver.assemble_system(k, scheme=scheme)
                    T_numerical = self.solver.solve_system(A, b)
                    grid = self.create_pyvista_grid(T_numerical)
                    distance, temperature = self.get_temperature_profile(grid, start_point, end_point)
                    profiles.append((k, scheme, (distance, temperature)))
            
            grid_analytical = self.create_pyvista_grid(self.T_mms_values)
            analytical_profile = self.get_temperature_profile(grid_analytical, start_point, end_point)
            
            self.plot_cross_section_profiles(profiles, analytical_profile, f"Distribution Temperature  - {section_name}", xlabel)
        errors = self.run_convergence_study(k_values, schemes, mesh_sizes)
        max_slopes = self.calculate_max_slope(mesh_sizes, errors)
        self.plot_convergence(mesh_sizes, errors, L)
        self.print_max_slopes(max_slopes)

                    
#%%
def main():
    mesh_sizes = np.array([10, 20, 40, 80]) 
    analyzer = ConvectionDiffusionAnalyzer(-1, 1, -1, 1,mesh_sizes)
    k_values = [1, 1/100, 1/10000000]
    schemes = ['centré', 'upwind']
    nx = ny = 50
    
    
    analyzer.run_analysis(k_values, schemes, nx, ny, mesh_sizes)

if __name__ == "__main__":
    main()