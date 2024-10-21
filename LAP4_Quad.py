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

class CouetteFlow:
    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.rho = 1.0  # kg/m^3
        self.mu = 1.0  # N*s/m^2
        self.U = 1.0  # m/s
        self.b = 1.0  # m
        
        self.mesh = None
        self.face_areas = None
        self.cell_volumes = None
        self.normal_face = None
        self.cell_centers = None
        
    def generate_mesh(self, mesh_type='QUAD', Nx=20, Ny=20):
        mesh_generator = MeshGenerator()
        mesh_parameters = {'mesh_type': mesh_type, 'Nx': Nx, 'Ny': Ny}
        self.mesh = mesh_generator.rectangle([self.x_min, self.x_max, self.y_min, self.y_max], mesh_parameters)
        mesh_connectivity = MeshConnectivity(self.mesh)
        mesh_connectivity.compute_connectivity()
        self.compute_mesh_properties()
        return self.mesh
    
    def compute_mesh_properties(self):
        self.face_areas = self.compute_face_areas()
        self.cell_volumes = self.compute_cell_volumes()
        self.normal_face = self.compute_normal_face()
        self.cell_centers = self.compute_cell_centers()
    
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
            coords = np.array([self.mesh.get_node_to_xycoord(node) for node in nodes])
            if len(nodes) == 3:  # Triangles
                cell_volumes[i_elem] = 0.5 * np.abs(np.cross(coords[1] - coords[0], coords[2] - coords[0]))
            elif len(nodes) == 4:  # Quadrilaterals
                cell_volumes[i_elem] = 0.5 * np.abs(np.cross(coords[2] - coords[0], coords[3] - coords[1]))
        return cell_volumes
    
    def compute_normal_face(self):
        normal_face = np.zeros((self.mesh.get_number_of_faces(), 2))
        for i_face in range(self.mesh.get_number_of_faces()):
            nodes = self.mesh.get_face_to_nodes(i_face)
            x1, y1 = self.mesh.get_node_to_xycoord(nodes[0])
            x2, y2 = self.mesh.get_node_to_xycoord(nodes[1])
            nx, ny = y2 - y1, x1 - x2
            norm = np.sqrt(nx**2 + ny**2)
            normal_face[i_face] = [nx/norm, ny/norm]
        return normal_face
    
    def compute_cell_centers(self):
        cell_centers = np.zeros((self.mesh.get_number_of_elements(), 2))
        for i_elem in range(self.mesh.get_number_of_elements()):
            nodes = self.mesh.get_element_to_nodes(i_elem)
            coords = np.array([self.mesh.get_node_to_xycoord(node) for node in nodes])
            cell_centers[i_elem] = np.mean(coords, axis=0)
        return cell_centers
    
    def analytical_solution(self, x, y, P):
        return y * (1 - P * (1 - y)), 0
    
    def assemble_system(self, P, u, v, scheme='centered'):
        n_cells = self.mesh.get_number_of_elements()
        A = sparse.lil_matrix((2*n_cells, 2*n_cells))
        b = np.zeros(2*n_cells)
        
        for i_face in range(self.mesh.get_number_of_faces()):
            left_cell, right_cell = self.mesh.get_face_to_elements(i_face)
            
            if right_cell != -1:  # Internal face
                self.add_internal_face_contribution(A, u, v, left_cell, right_cell, i_face, scheme)
            else:  # Boundary face
                self.add_boundary_face_contribution(A, b, left_cell, i_face, P)
        
        # Source terms
        b[:n_cells] += -2 * P * self.cell_volumes  # Sx
        # Sy is zero for this problem
        
        return A.tocsr(), b
    
    def add_internal_face_contribution(self, A, u, v, left_cell, right_cell, i_face, scheme):
        n_cells = self.mesh.get_number_of_elements()
        u_face = 0.5 * (u[left_cell] + u[right_cell])
        v_face = 0.5 * (v[left_cell] + v[right_cell])
        velocity_dot_normal = u_face * self.normal_face[i_face, 0] + v_face * self.normal_face[i_face, 1]
        
        # Convective term
        if scheme == 'centered':
            conv_coeff = 0.5 * self.rho * velocity_dot_normal * self.face_areas[i_face]
            A[left_cell, right_cell] -= conv_coeff
            A[right_cell, left_cell] += conv_coeff
            A[left_cell, left_cell] += conv_coeff
            A[right_cell, right_cell] -= conv_coeff
            
            A[n_cells + left_cell, n_cells + right_cell] -= conv_coeff
            A[n_cells + right_cell, n_cells + left_cell] += conv_coeff
            A[n_cells + left_cell, n_cells + left_cell] += conv_coeff
            A[n_cells + right_cell, n_cells + right_cell] -= conv_coeff
        
        elif scheme == 'upwind':
            conv_coeff = self.rho * abs(velocity_dot_normal) * self.face_areas[i_face]
            if velocity_dot_normal > 0:
                A[right_cell, left_cell] += conv_coeff
                A[left_cell, left_cell] -= conv_coeff
                A[n_cells + right_cell, n_cells + left_cell] += conv_coeff
                A[n_cells + left_cell, n_cells + left_cell] -= conv_coeff
            else:
                A[left_cell, right_cell] -= conv_coeff
                A[right_cell, right_cell] += conv_coeff
                A[n_cells + left_cell, n_cells + right_cell] -= conv_coeff
                A[n_cells + right_cell, n_cells + right_cell] += conv_coeff
        
        # Diffusive term
        dx = self.cell_centers[right_cell] - self.cell_centers[left_cell]
        distance = np.linalg.norm(dx)
        diff_coeff = self.mu * self.face_areas[i_face] / distance
        A[left_cell, right_cell] -= diff_coeff
        A[right_cell, left_cell] -= diff_coeff
        A[left_cell, left_cell] += diff_coeff
        A[right_cell, right_cell] += diff_coeff
        
        A[n_cells + left_cell, n_cells + right_cell] -= diff_coeff
        A[n_cells + right_cell, n_cells + left_cell] -= diff_coeff
        A[n_cells + left_cell, n_cells + left_cell] += diff_coeff
        A[n_cells + right_cell, n_cells + right_cell] += diff_coeff
    
    def add_boundary_face_contribution(self, A, b, left_cell, i_face, P):
        n_cells = self.mesh.get_number_of_elements()
        x_face, y_face = np.mean([self.mesh.get_node_to_xycoord(node) for node in self.mesh.get_face_to_nodes(i_face)], axis=0)
        
        if np.isclose(y_face, self.y_min) or np.isclose(y_face, self.y_max):
            # Dirichlet boundary condition
            u_boundary, v_boundary = self.analytical_solution(x_face, y_face, P)
            
            diff_coeff = self.mu * self.face_areas[i_face] / (0.5 * self.face_areas[i_face])
            A[left_cell, left_cell] += diff_coeff
            A[n_cells + left_cell, n_cells + left_cell] += diff_coeff
            b[left_cell] += diff_coeff * u_boundary
            b[n_cells + left_cell] += diff_coeff * v_boundary
        
        elif np.isclose(x_face, self.x_min) or np.isclose(x_face, self.x_max):
            # Neumann boundary condition (zero gradient)
            pass
    
    def solve_system(self, A, b):
        return spla.spsolve(A, b)
    
    def compute_error(self, u_numerical, v_numerical, P):
        error_u = np.zeros_like(u_numerical)
        error_v = np.zeros_like(v_numerical)
        for i_cell in range(len(u_numerical)):
            x, y = self.cell_centers[i_cell]
            u_analytical, v_analytical = self.analytical_solution(x, y, P)
            error_u[i_cell] = abs(u_numerical[i_cell] - u_analytical)
            error_v[i_cell] = abs(v_numerical[i_cell] - v_analytical)
        
        # Calculate L1, L2, and Linf errors
        L1_error_u = np.mean(error_u)
        L2_error_u = np.sqrt(np.mean(error_u**2))
        Linf_error_u = np.max(error_u)
        
        L1_error_v = np.mean(error_v)
        L2_error_v = np.sqrt(np.mean(error_v**2))
        Linf_error_v = np.max(error_v)
        
        return {
            'L1_u': L1_error_u, 'L2_u': L2_error_u, 'Linf_u': Linf_error_u,
            'L1_v': L1_error_v, 'L2_v': L2_error_v, 'Linf_v': Linf_error_v
        }
    
    def run_analysis(self, P_values, schemes, mesh_sizes, max_iterations=1000, tolerance=1e-6):
        results = []
        for mesh_size in mesh_sizes:
            self.generate_mesh(Nx=mesh_size, Ny=mesh_size)
            n_elements = self.mesh.get_number_of_elements()
            print(f"Solving for mesh size {mesh_size} with {n_elements} elements")
            
            for P in P_values:
                for scheme in schemes:
                    u, v = np.zeros(n_elements), np.zeros(n_elements)
                    for iteration in range(max_iterations):
                        A, b = self.assemble_system(P, u, v, scheme)
                        solution = self.solve_system(A, b)
                        u_new, v_new = solution[:n_elements], solution[n_elements:]
                        if np.max(np.abs(u_new - u)) < tolerance and np.max(np.abs(v_new - v)) < tolerance:
                            break
                        u, v = u_new, v_new
                    
                    errors = self.compute_error(u, v, P)
                    results.append({
                        'mesh_size': mesh_size, 'P': P, 'scheme': scheme,
                        'iterations': iteration + 1, 'errors': errors,
                        'u': u, 'v': v
                    })
        return results

    def create_pyvista_grid(self, u_values, v_values):
        plotter = MeshPlotter()
        nodes, elements = plotter.prepare_data_for_pyvista(self.mesh)
        points = np.array(nodes)
        cells = elements
        
        cell_types = []
        i = 0
        while i < len(cells):
            num_points = cells[i]
            cell_types.append(pv.CellType.TRIANGLE if num_points == 3 else pv.CellType.QUAD)
            i += num_points + 1
        
        grid = pv.UnstructuredGrid(cells, np.array(cell_types), points)
        
        grid.cell_data['u'] = u_values
        grid.cell_data['v'] = v_values
        grid.cell_data['velocity_magnitude'] = np.sqrt(u_values**2 + v_values**2)
        
        grid = grid.cell_data_to_point_data()
        grid.point_data['velocity'] = np.column_stack((grid.point_data['u'], grid.point_data['v'], np.zeros_like(grid.point_data['u'])))
        
        return grid
    
    def create_analytical_grid(self, P):
        plotter = MeshPlotter()
        nodes, elements = plotter.prepare_data_for_pyvista(self.mesh)
        points = np.array(nodes)
        cells = elements
        
        cell_types = []
        i = 0
        while i < len(cells):
            num_points = cells[i]
            cell_types.append(pv.CellType.TRIANGLE if num_points == 3 else pv.CellType.QUAD)
            i += num_points + 1
        
        grid = pv.UnstructuredGrid(cells, np.array(cell_types), points)
        
        u_analytical = np.zeros(grid.n_cells)
        v_analytical = np.zeros(grid.n_cells)
        
        for i in range(grid.n_cells):
            x, y = self.cell_centers[i]
            u_analytical[i], v_analytical[i] = self.analytical_solution(x, y, P)
        
        grid.cell_data['u'] = u_analytical
        grid.cell_data['v'] = v_analytical
        grid.cell_data['velocity_magnitude'] = np.sqrt(u_analytical**2 + v_analytical**2)
        
        grid = grid.cell_data_to_point_data()
        grid.point_data['velocity'] = np.column_stack((grid.point_data['u'], grid.point_data['v'], np.zeros_like(grid.point_data['u'])))
        
        return grid

    def plot_results(self, results):
        for result in results:
            mesh_size, P, scheme = result['mesh_size'], result['P'], result['scheme']
            u, v = result['u'], result['v']

            self.generate_mesh(Nx=mesh_size, Ny=mesh_size)
            numerical_grid = self.create_pyvista_grid(u, v)
            analytical_grid = self.create_analytical_grid(P)
            
            pl = pvQt.BackgroundPlotter(shape=(2, 2))
            
            # Plot numerical solution
            pl.subplot(0, 0)
            pl.add_mesh(numerical_grid, scalars="velocity_magnitude", cmap="viridis", show_edges=True)
            pl.add_text("Solution Numérique", font_size=10)
            
            # Plot analytical solution
            pl.subplot(0, 1)
            pl.add_mesh(analytical_grid, scalars="velocity_magnitude", cmap="viridis", show_edges=True)
            pl.add_text("Solution Analytique", font_size=10)
            
            # Add contours to both plots
            try:
                for i in range(2):
                    pl.subplot(0, i)
                    grid = numerical_grid if i == 0 else analytical_grid
                    contours = grid.contour(scalars='velocity_magnitude', isosurfaces=10)
                    if contours.n_points > 0:
                        pl.add_mesh(contours, color="white", line_width=2)
                  
            except Exception as e:
                print(f"Warning: Could not generate contours. Error: {e}")
            
            pl.add_text(f"Champ de Vitesse (P={P}, schéma={scheme}, mesh={mesh_size})", font_size=6, position='lower_edge')

           

            # Plot error
            pl.subplot(1, 0)
            error_grid = numerical_grid.copy()
            error_grid.point_data['error'] = np.abs(numerical_grid.point_data['velocity_magnitude'] - analytical_grid.point_data['velocity_magnitude'])
 
            pl.add_mesh(error_grid, scalars="error", cmap="RdBu", show_edges=True)
            pl.add_text(f"Erreur  (P={P}, schéma={scheme}, mesh={mesh_size})", font_size=6)
            pl.link_views()
            
            
            pl.subplot(1, 1)
            x_values = [0.25, 0.5, 0.75]
            for x in x_values:
                num_line = numerical_grid.sample_over_line((x, 0, 0), (x, 1, 0))
                ana_line = analytical_grid.sample_over_line((x, 0, 0), (x, 1, 0))
                
                # Extract y-coordinates and velocity magnitudes
                num_y = num_line.points[:, 1]  # y-coordinate is the second column
                ana_y = ana_line.points[:, 1]
                num_vel = num_line['velocity_magnitude']
                ana_vel = ana_line['velocity_magnitude']
                
                plt.plot(num_y, num_vel, label=f'Numerical x={x}')
                plt.plot(ana_y, ana_vel, '--', label=f'Analytical x={x}')
            
            plt.xlabel('y')
            plt.ylabel('Vitesse (Magnitude)')
            plt.title(f'Coupe à X constant (P={P}, scheme={scheme})')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'velocity_profile_P{P}_{scheme}_mesh{mesh_size}.png', dpi=300, bbox_inches='tight')
            plt.close()
            

            plane = pv.Plane(center=(0.5, 0.5, 0.01), i_size=1, j_size=1, direction=(0, 0, 1))
            texture = pv.read_texture(f'velocity_profile_P{P}_{scheme}_mesh{mesh_size}.png')
            plane.texture_map_to_plane(inplace=True)
            pl.add_mesh(plane, texture=texture)
            
            # Adjust camera position to view the new plane
            pl.camera_position = [(0.5, 0.5, 2), (0.5, 0.5, 0), (0, 1, 0)]
            pl.camera.zoom(0.75)

            pl.show()
        



def run_analysis_and_plot(flow_solver, P_values, schemes, mesh_sizes):
    results = flow_solver.run_analysis(P_values, schemes, mesh_sizes)
    flow_solver.plot_results(results)
    
    error_types = ['L1_u', 'L2_u', 'Linf_u']
    
    plt.figure(figsize=(10, 6))
    for P in P_values:
        for scheme in schemes:
            for error_type in error_types:
                errors = [result['errors'][error_type] for result in results if result['P'] == P and result['scheme'] == scheme]
                h = [1/result['mesh_size'] for result in results if result['P'] == P and result['scheme'] == scheme]
                plt.loglog(h, errors, 'o-', label=f'{error_type}, P={P}, {scheme}')
                if len(errors) > 1:
                    order = np.log(errors[-2] / errors[-1]) / np.log(h[-2] / h[-1])
                    print(f"Ordre de Convergence {error_type} pour P={P}, schéma={scheme}: {order:.2f}")
    
    plt.xlabel('Taille des éléments (h)')
    plt.ylabel('Erreur')
    plt.title('Comparaison des erreurs L1, L2, et Linf')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    P_values = [0, 1, -3]
    schemes = ['centered']
    mesh_sizes = [10,20,40]


    classic_couette_flow = CouetteFlow(0, 1, 0, 1)
    run_analysis_and_plot(classic_couette_flow, P_values, schemes, mesh_sizes)


if __name__ == "__main__":
    main()