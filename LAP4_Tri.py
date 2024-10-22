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
from typing import Tuple

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
        
    def generate_mesh(self, mesh_type='TRI', lc=0.1):
        mesh_generator = MeshGenerator()
        mesh_parameters = {'mesh_type': mesh_type, 'lc': lc}
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
            'L1_v': L1_error_v, 'L2_v': L2_error_v, 'Linf_v': Linf_error_v}
    
    def run_analysis(self, P_values, schemes, lc_values, max_iterations=1000, tolerance=1e-6):
        results = []
        for lc in lc_values:
            self.generate_mesh(lc=lc)
            n_elements = self.mesh.get_number_of_elements()

            
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
                        'lc': lc, 'P': P, 'scheme': scheme,
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
        
        return grid

    def create_analytical_grid(self, P):
        grid = self.create_pyvista_grid(np.zeros(self.mesh.get_number_of_elements()), np.zeros(self.mesh.get_number_of_elements()))
        
        u_analytical = np.zeros(self.mesh.get_number_of_elements())
        v_analytical = np.zeros(self.mesh.get_number_of_elements())
        
        for i in range(self.mesh.get_number_of_elements()):
            x, y = self.cell_centers[i]
            u_analytical[i], v_analytical[i] = self.analytical_solution(x, y, P)
        
        grid.cell_data['u'] = u_analytical
        grid.cell_data['v'] = v_analytical
        grid.cell_data['velocity_magnitude'] = np.sqrt(u_analytical**2 + v_analytical**2)
        
        return grid

    def plot_results(self, results):
        for result in results:
            lc, P, scheme = result['lc'], result['P'], result['scheme']
            u, v = result['u'], result['v']

            self.generate_mesh(lc=lc)
            numerical_grid = self.create_pyvista_grid(u, v)
            analytical_grid = self.create_analytical_grid(P)
            
            # Convert cell data to point data
            numerical_grid = numerical_grid.cell_data_to_point_data()
            analytical_grid = analytical_grid.cell_data_to_point_data()
            
            pl = pvQt.BackgroundPlotter(shape=(2, 2))
            
            # Plot numerical solution with contours
            pl.subplot(0, 0)
            pl.add_mesh(numerical_grid, scalars="velocity_magnitude", cmap="viridis", show_edges=True)
            contours_num = numerical_grid.contour(scalars="velocity_magnitude", isosurfaces=10)
            pl.add_mesh(contours_num, color="white", line_width=2)
            pl.add_text("Solution Numérique", font_size=10)
            
            # Plot analytical solution with contours
            pl.subplot(0, 1)
            pl.add_mesh(analytical_grid, scalars="velocity_magnitude", cmap="viridis", show_edges=True)
            contours_ana = analytical_grid.contour(scalars="velocity_magnitude", isosurfaces=10)
            pl.add_mesh(contours_ana, color="white", line_width=2)
            pl.add_text("Solution Analytique", font_size=10)
            
            pl.add_text(f"Champ de vitesse (P={P}, schéma={scheme}, lc={lc})", font_size=6, position='lower_edge')
    
            # Plot error
            pl.subplot(1, 0)
            error_grid = numerical_grid.copy()
            error_grid.point_data['error'] = np.abs(numerical_grid.point_data['velocity_magnitude'] - analytical_grid.point_data['velocity_magnitude'])
            pl.add_mesh(error_grid, scalars="error", cmap="RdBu", show_edges=True)
            contours_error = error_grid.contour(scalars="error", isosurfaces=10)
            pl.add_mesh(contours_error, color="black", line_width=1)
            pl.add_text(f"Erreur (P={P}, schema={scheme}, lc={lc})", font_size=6)
            pl.link_views()
            
            # Plot velocity profiles
            pl.subplot(1, 1)
            x_values = [0.25, 0.5, 0.75]
            for x in x_values:
                num_line = numerical_grid.sample_over_line((x, 0, 0), (x, 1, 0))
                ana_line = analytical_grid.sample_over_line((x, 0, 0), (x, 1, 0))
                
                num_y = num_line.points[:, 1]
                ana_y = ana_line.points[:, 1]
                num_vel = num_line['velocity_magnitude']
                ana_vel = ana_line['velocity_magnitude']
                
                plt.plot(num_y, num_vel, label=f'Numérique x={x}')
                plt.plot(ana_y, ana_vel, '--', label=f'Analytique x={x}')
            
            plt.xlabel('y')
            plt.ylabel('Vitesse (Magnitude)')
            plt.title(f'Coupes à X constant (P={P}, scheme={scheme})')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'velocity_profile_P{P}_{scheme}_lc{lc}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Add the saved figure to the plot
            plane = pv.Plane(center=(0.5, 0.5, 0.01), i_size=1, j_size=1, direction=(0, 0, 1))
            texture = pv.read_texture(f'velocity_profile_P{P}_{scheme}_lc{lc}.png')
            plane.texture_map_to_plane(inplace=True)
            pl.add_mesh(plane, texture=texture)
            
            pl.camera_position = [(0.5, 0.5, 2), (0.5, 0.5, 0), (0, 1, 0)]
            pl.camera.zoom(0.75)
    
            pl.show()
        

class RotatedCouetteFlow(CouetteFlow):
    def __init__(self, x_min, x_max, y_min, y_max, theta_degrees):
        super().__init__(x_min, x_max, y_min, y_max)
        self.theta = np.radians(theta_degrees)
        self.rotation_matrix = np.array([
            [np.cos(self.theta), -np.sin(self.theta)],
            [np.sin(self.theta), np.cos(self.theta)]
        ])
        self.rotation_matrix_inverse = self.rotation_matrix.T
        
    def rotate_coordinates(self, x: float, y: float) -> Tuple[float, float]:
        """Transform coordinates from rotated to original frame"""
        coords = np.array([x, y])
        original_coords = self.rotation_matrix_inverse @ coords
        return original_coords[0], original_coords[1]
    
    def rotate_velocities(self, u: float, v: float) -> Tuple[float, float]:
        """Transform velocities from original to rotated frame"""
        velocities = np.array([u, v])
        rotated_velocities = self.rotation_matrix @ velocities
        return rotated_velocities[0], rotated_velocities[1]
    
    def rotate_gradients(self, dx: float, dy: float) -> Tuple[float, float]:
        """Transform pressure gradients from original to rotated frame"""
        gradients = np.array([dx, dy])
        rotated_gradients = self.rotation_matrix @ gradients
        return rotated_gradients[0], rotated_gradients[1]
    
    def analytical_solution(self, x: float, y: float, P: float) -> Tuple[float, float]:
        """Compute analytical solution in rotated frame"""
        # Transform coordinates to original frame
        x_orig, y_orig = self.rotate_coordinates(x, y)
        
        # Compute solution in original frame
        u_orig, v_orig = super().analytical_solution(x_orig, y_orig, P)
        
        # Transform velocities to rotated frame
        return self.rotate_velocities(u_orig, v_orig)
    
    def assemble_system(self, P: float, u: np.ndarray, v: np.ndarray, scheme='centered'):
        n_cells = self.mesh.get_number_of_elements()
        A = sparse.lil_matrix((2*n_cells, 2*n_cells))
        b = np.zeros(2*n_cells)
        
        for i_face in range(self.mesh.get_number_of_faces()):
            left_cell, right_cell = self.mesh.get_face_to_elements(i_face)
            
            if right_cell != -1:  # Internal face
                self.add_internal_face_contribution(A, u, v, left_cell, right_cell, i_face, scheme)
            else:  # Boundary face
                self.add_boundary_face_contribution(A, b, left_cell, i_face, P)
        
        # Transform pressure gradient source terms to rotated frame
        Sx_rot, Sy_rot = self.rotate_gradients(-2*P, 0)
        b[:n_cells] += Sx_rot * self.cell_volumes  # Rotated Sx
        b[n_cells:] += Sy_rot * self.cell_volumes  # Rotated Sy
        
        return A.tocsr(), b

def run_rotated_analysis(P_values, schemes, mesh_sizes,theta_degrees):
    
    # Create rotated Couette flow solver
    rotated_couette_flow = RotatedCouetteFlow(0, 1, 0, 1, theta_degrees)
    
    # Run analysis and plot results
    results = rotated_couette_flow.run_analysis(P_values, schemes, mesh_sizes)
    rotated_couette_flow.plot_results(results)
    
    # Calculate convergence order
    error_types = ['L1_u', 'L2_u', 'Linf_u']
    plt.figure(figsize=(10, 6))
   
    for P in P_values:
        for scheme in schemes:
            for error_type in error_types:
                errors = [result['errors'][error_type] for result in results 
                          if result['P'] == P and result['scheme'] == scheme]
                h = [result['lc'] for result in results 
                      if result['P'] == P and result['scheme'] == scheme]
                
                plt.loglog(h, errors, 'o-', 
                          label=f'{error_type}, P={P}, {scheme}, θ={theta_degrees}°')
                
                if len(errors) > 1:
                    order = np.log(errors[-2] / errors[-1]) / np.log(h[-2] / h[-1])
                    print(f"Ordre de Convergence {error_type} pour P={P}, "
                          f"schéma={scheme}, θ={theta_degrees}°: {order:.2f}")
    
    plt.xlabel('Taille des éléments (h)')
    plt.ylabel('Erreur')
    plt.title('Comparaison des erreurs L1, L2, et Linf (Cas Tourné)')
    plt.legend()
    plt.grid(True)
    plt.show()

def run_analysis_and_plot(flow_solver, P_values, schemes, mesh_sizes):
    results = flow_solver.run_analysis(P_values, schemes, mesh_sizes)
    flow_solver.plot_results(results)
    
    error_types = ['L1_u', 'L2_u', 'Linf_u']
    
    plt.figure(figsize=(10, 6))
    for P in P_values:
        for scheme in schemes:
            for error_type in error_types:
                errors = [result['errors'][error_type] for result in results if result['P'] == P and result['scheme'] == scheme]
                h = [result['lc'] for result in results if result['P'] == P and result['scheme'] == scheme]
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
    lc_values = [0.1, 0.05, 0.025]
    theta=45
    
    classic_couette_flow = CouetteFlow(0, 1, 0, 1)
    run_analysis_and_plot(classic_couette_flow, P_values, schemes, lc_values)
    run_rotated_analysis( P_values, schemes, lc_values,theta)


if __name__ == "__main__":
    main()