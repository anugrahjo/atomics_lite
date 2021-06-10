import dolfin as df
import numpy as np

# import openmdao.api as om
import om_lite.api as om
# from kirchoff_elastic_model import KirchoffElasticModel
from lsdo_optimizer.api import SQP_OSQP
from hyperelastic_model import HyperElasticModel

import scipy.sparse as sp

from atomics_lite.api import PDEProblem, AtomicsGroup
from atomics_lite.pdes.st_kirchhoff import get_residual_form
from atomics_lite.general_filter_comp import GeneralFilterComp


np.random.seed(0)

'''
1. Define the mesh
'''
NUM_ELEMENTS_X = 40
# NUM_ELEMENTS_X = 80
NUM_ELEMENTS_Y = 20
# NUM_ELEMENTS_Y = 40
LENGTH_X = 160.
LENGTH_Y = 80.

mesh = df.RectangleMesh.create(
    [df.Point(0.0, 0.0), df.Point(LENGTH_X, LENGTH_Y)],
    [NUM_ELEMENTS_X, NUM_ELEMENTS_Y],
    df.CellType.Type.quadrilateral,
)

'''
2. Define the traction boundary conditions
'''
# here traction force is applied on the middle of the right edge
class TractionBoundary(df.SubDomain):
    def inside(self, x, on_boundary):
        return ((abs(x[1] - LENGTH_Y/2) < LENGTH_Y/NUM_ELEMENTS_Y + df.DOLFIN_EPS) and (abs(x[0] - LENGTH_X ) < df.DOLFIN_EPS*1.5e15))

# Define the traction boundary
sub_domains = df.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
upper_edge = TractionBoundary()
upper_edge.mark(sub_domains, 6)
dss = df.Measure('ds')(subdomain_data=sub_domains)
f = df.Constant((0, -1. / 4 ))

'''
3. Setup the PDE problem
'''
# PDE problem
pde_problem = PDEProblem(mesh)

# Add input to the PDE problem:
# name = 'density', function = density_function (function is the solution vector here)
density_function_space = df.FunctionSpace(mesh, 'DG', 0)
density_function = df.Function(density_function_space)
pde_problem.add_input('density', density_function)

# Add states to the PDE problem (line 58):
# name = 'displacements', function = displacements_function (function is the solution vector here)
# residual_form = get_residual_form(u, v, rho_e) from atomics_lite.pdes.thermo_mechanical_uniform_temp
# *inputs = density (can be multiple, here 'density' is the only input)

displacements_function_space = df.VectorFunctionSpace(mesh, 'Lagrange', 1)
displacements_function = df.Function(displacements_function_space)
v = df.TestFunction(displacements_function_space)
method='SIMP'
residual_form = get_residual_form(
    displacements_function, 
    v, 
    density_function,
    method=method
)

residual_form -= df.dot(f, v) * dss(6)
pde_problem.add_state('displacements', displacements_function, residual_form, 'density')

# Add output-avg_density to the PDE problem:
volume = df.assemble(df.Constant(1.) * df.dx(domain=mesh))
avg_density_form = density_function / (df.Constant(1. * volume)) * df.dx(domain=mesh)
pde_problem.add_scalar_output('avg_density', avg_density_form, 'density')

# Add output-compliance to the PDE problem:
compliance_form = df.dot(f, displacements_function) * dss(6)
pde_problem.add_scalar_output('compliance', compliance_form, 'displacements')

# Add Dirichlet boundary conditions to the PDE problem:
pde_problem.add_bc(df.DirichletBC(displacements_function_space, df.Constant((0.0, 0.0)), '(abs(x[0]-0.) < DOLFIN_EPS)'))

'''
4. Setup the optimization problem
'''
# Define the OpenMDAO problem and model

# prob = om.Problem()

num_dof_density = pde_problem.inputs_dict['density']['function'].function_space().dim()

# Define the omlite model
x_val = np.full((num_dof_density,), .1)
y_val = np.full((3,), 1.)
psi = np.full((3,), 1.)

# tol = 1.11 * 1e-3
tol = 1.e-15
method = 'surf'

model = HyperElasticModel()
model.setup(num_dof_density, density_function_space, pde_problem)

# y_val, f, res, c = model.evaluate_functions(x_val, y_val, method = method, tol=tol)

# pf_px, pf_py, pc_px, pc_py, psi, pR_px, pR_py = model.evaluate_derivatives(x_val, y_val, psi, method = method, tol=tol)

class KirchoffElasticProblem(om.Problem):
    # def __init__(self, model):
    #     super().__init__(model)

    def initialize(self):
        self.nc = 1 + 2 * self.nx
        
        self.x_and_y_vector_dict['x'] = dict(shape=(self.nx,))
        self.x_and_y_vector_dict['y'] = dict(shape=(self.ny,))
        
        self.cons_and_res_vector_dict['average_density'] = dict(shape=(1,))
        self.cons_and_res_vector_dict['density_lower_bound'] = dict(shape=(self.nx,))
        self.cons_and_res_vector_dict['density_upper_bound'] = dict(shape=(self.nx,))
        self.cons_and_res_vector_dict['residuals+'] = dict(shape=(self.ny,))
        self.cons_and_res_vector_dict['residuals-'] = dict(shape=(self.ny,))

    def evaluate_constraints(self, x):
        # if self.hot_x_for_fn_evals != x:
        nx = self.nx
        if not(np.array_equal(self.hot_x_for_fn_evals, x)):
            self.y_val, self.f, self.res, self.c = model.evaluate_functions(x[:nx], x[nx:], method = method, tol=tol)
            self.hot_x_for_fn_evals[:] = 1. * x

        # bound_constraints = x[:nx]
        bound_constraints_upp = model.bound_constraints['upper'] - x[:nx]
        bound_constraints_low = x[:nx] - model.bound_constraints['lower']
        
        # return np.concatenate(([self.c - 0.5], bound_constraints))
        return np.concatenate(([self.c - 0.5], bound_constraints_low, bound_constraints_upp))
        # return np.concatenate(([0.5 - self.c], bound_constraints_low, bound_constraints_upp))

    def compute_constraint_jacobian(self, x):
        # if self.hot_x_for_deriv_evals != x:
        nx = self.nx
        ny = self.ny
        if not(np.array_equal(self.hot_x_for_deriv_evals, x)):
            self.pf_px, self.pf_py, self.pc_px, self.pc_py, self.psi, self.pR_px, self.pR_py = model.evaluate_derivatives(x[:nx], x[nx:], self.psi, method = method, tol=tol)
            self.hot_x_for_deriv_evals[:] = 1. * x

        # pc_pv = -np.append(self.pc_px, self.pc_py).reshape((1, nx+ny))
        pc_pv = np.append(self.pc_px, self.pc_py).reshape((1, nx+ny))
        pLB_pv = np.append(np.identity(nx), np.zeros((nx, ny)), axis=1)

        return sp.csc_matrix(np.concatenate((pc_pv, pLB_pv, -pLB_pv)))



prob = KirchoffElasticProblem(model, formulation=method, res_tol=tol)

optimizer = SQP_OSQP(prob, opt_tol=1e-6, feas_tol=1e-6)
optimizer.setup()
optimizer.run()
# optimizer.print(table = True)


# comp = om.IndepVarComp()
# comp.add_output(
#     'density_unfiltered', 
#     shape=num_dof_density, 
#     val=np.ones(num_dof_density),
# )
# prob.model.add_subsystem('indep_var_comp', comp, promotes=['*'])

# comp = GeneralFilterComp(density_function_space=density_function_space)
# prob.model.add_subsystem('general_filter_comp', comp, promotes=['*'])


# group = AtomicsGroup(pde_problem=pde_problem)
# prob.model.add_subsystem('atomics_lite_group', group, promotes=['*'])

# prob.model.add_design_var('density_unfiltered',upper=1, lower=1e-3)
# prob.model.add_objective('compliance')
# prob.model.add_constraint('avg_density',upper=0.50)

# set up the optimizer
# prob.driver = driver = om.pyOptSparseDriver()
# driver.options['optimizer'] = 'SNOPT'
# driver.opt_settings['Verify level'] = 0

# driver.opt_settings['Major iterations limit'] = 100000
# driver.opt_settings['Minor iterations limit'] = 100000
# driver.opt_settings['Iterations limit'] = 100000000
# driver.opt_settings['Major step limit'] = 2.0

# driver.opt_settings['Major feasibility tolerance'] = 1.0e-6
# driver.opt_settings['Major optimality tolerance'] =1.e-10

# prob.setup()
# prob.run_model()
# print(prob['compliance']); exit()
# prob.run_driver()


#save the solution vector
# if method =='SIMP':
#     penalized_density  = df.project(density_function**3, density_function_space) 
# else:
#     penalized_density  = df.project(density_function/(1 + 8. * (1. - density_function)), density_function_space) 

# df.File('solutions/case_1/cantilever_beam_kirchhoff/displacement.pvd') << displacements_function
# df.File('solutions/case_1/cantilever_beam_kirchhoff/penalized_density.pvd') << penalized_density