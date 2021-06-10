import dolfin as df

import numpy as np

# import openmdao.api as om
import om_lite.api as om
from hyperelastic_model import HyperElasticModel
# from lsdo_optimizer.api import Problem
from lsdo_optimizer.api import SQP_OSQP
from array_manager.api import VectorComponentsDict

import scipy.sparse as sp

from atomics_lite.api import PDEProblem, AtomicsGroup
from atomics_lite.pdes.neo_hookean_addtive import get_residual_form
from atomics_lite.general_filter_comp import GeneralFilterComp

np.random.seed(0)

# Define the mesh and create the PDE problem
# NUM_ELEMENTS_X = 120 
NUM_ELEMENTS_X = 60
# NUM_ELEMENTS_Y = 30 
NUM_ELEMENTS_Y = 15

# nx = NUM_ELEMENTS_X * NUM_ELEMENTS_Y

LENGTH_X = 4.8 # 0.12
LENGTH_Y = 1.6 # 0.03

LENGTH_X = 0.12
LENGTH_Y = 0.03

mesh = df.RectangleMesh.create(
    [df.Point(0.0, 0.0), df.Point(LENGTH_X, LENGTH_Y)],
    [NUM_ELEMENTS_X, NUM_ELEMENTS_Y],
    df.CellType.Type.quadrilateral,
)

# Define the traction condition:
# here traction force is applied on the middle of the right edge
class TractionBoundary(df.SubDomain):
    def inside(self, x, on_boundary):
        return ((abs(x[1] - LENGTH_Y/2) < LENGTH_Y/NUM_ELEMENTS_Y + df.DOLFIN_EPS) and (abs(x[0] - LENGTH_X ) < df.DOLFIN_EPS*1.5e15))

# Define the traction boundary
sub_domains = df.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
upper_edge = TractionBoundary()
upper_edge.mark(sub_domains, 6)
dss = df.Measure('ds')(subdomain_data=sub_domains)
tractionBC = dss(6)
f = df.Constant((0.0, -9.e-1 ))

# f = df.Constant((0.0, -9.e-1))
k = 10
# k = 3e9

# f = df.Constant((0.0, -120/ (8.*LENGTH_Y/NUM_ELEMENTS_Y ) ))

# PDE problem
pde_problem = PDEProblem(mesh)

# Add input to the PDE problem:
# name = 'density', function = density_function (function is the solution vector here)
density_function_space = df.FunctionSpace(mesh, 'DG', 0)
density_function = df.Function(density_function_space)
density_function.vector().set_local(np.ones(density_function_space.dim()))
pde_problem.add_input('density', density_function)

# Add states to the PDE problem (line 58):
# name = 'displacements', function = displacements_function (function is the solution vector here)
# residual_form = get_residual_form(u, v, rho_e) from atomics_lite.pdes.thermo_mechanical_uniform_temp
# *inputs = density (can be multiple, here 'density' is the only input)
displacements_function_space = df.VectorFunctionSpace(mesh, 'Lagrange', 1)
displacements_function = df.Function(displacements_function_space)
v = df.TestFunction(displacements_function_space)
residual_form = get_residual_form(
    displacements_function, 
    v, 
    density_function,
    density_function_space,
    tractionBC,
    f,
    1
)

pde_problem.add_state('displacements', displacements_function, residual_form, 'density')

# Add output-avg_density to the PDE problem:
volume = df.assemble(df.Constant(1.) * df.dx(domain=mesh))
avg_density_form = density_function / (df.Constant(1. * volume)) * df.dx(domain=mesh)
pde_problem.add_scalar_output('avg_density', avg_density_form, 'density')

# Add output-compliance to the PDE problem:
compliance_form = df.dot(f, displacements_function) * dss(6)
pde_problem.add_scalar_output('compliance', compliance_form, 'displacements')

# Add boundary conditions to the PDE problem:
pde_problem.add_bc(df.DirichletBC(displacements_function_space, df.Constant((0.0, 0.0)), '(abs(x[0]-0.) < DOLFIN_EPS)'))


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

class HyperElasticProblem(om.Problem):
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



prob = HyperElasticProblem(model, formulation=method, res_tol=tol)

x = np.full((prob.n,), 1.)
x[450:] = .5
f = prob.evaluate_objective(x)
c = prob.evaluate_constraints(x)

print(f)
print(c)
# y = prob.evaluate_states(x)

# y = prob.evaluate_states(x)
# y = prob.evaluate_states(x)

# optimizer = SQP_OSQP(prob, opt_tol=1e-6, feas_tol=1e-6)
# optimizer.setup()
# optimizer.run()



# optimizer.print(table = True)
# prob = HyperElasticProblem(nx=, ny=, nc=)



# comp = om.IndepVarComp()
# comp.add_output(
#     'density_unfiltered', 
#     shape=num_dof_density, 
#     val=np.ones(num_dof_density),
#     # val=np.random.random(num_dof_density) * 0.86,
# )
# prob.model.add_subsystem('indep_var_comp', comp, promotes=['*'])

# comp = GeneralFilterComp(density_function_space=density_function_space)
# prob.model.add_subsystem('general_filter_comp', comp, promotes=['*'])

# group = AtomicsGroup(pde_problem=pde_problem)
# prob.model.add_subsystem('atomics_group', group, promotes=['*'])

# prob.model.add_design_var('density_unfiltered',upper=1, lower=5e-3 )
# prob.model.add_objective('compliance')
# prob.model.add_constraint('avg_density',upper=0.50)

# prob.driver = driver = om.pyOptSparseDriver()
# driver.options['optimizer'] = 'SNOPT'
# driver.opt_settings['Verify level'] = 0

# driver.opt_settings['Major iterations limit'] = 100000
# driver.opt_settings['Minor iterations limit'] = 100000
# driver.opt_settings['Iterations limit'] = 100000000
# driver.opt_settings['Major step limit'] = 2.0

# driver.opt_settings['Major feasibility tolerance'] = 1.0e-5
# driver.opt_settings['Major optimality tolerance'] =1.3e-9

# prob.setup()
# prob.run_model()
# prob.check_partials(compact_print=True)
# prob.run_driver()













# eps = df.sym(df.grad(displacements_function))
# eps_dev = eps - 1/3 * df.tr(eps) * df.Identity(2)
# eps_eq = df.sqrt(2.0 / 3.0 * df.inner(eps_dev, eps_dev))
# eps_eq_proj = df.project(eps_eq, density_function_space)   
# ratio = eps / eps_eq

# fFile = df.HDF5File(df.MPI.comm_world,"eps_eq_proj_1000.h5","w")
# fFile.write(eps_eq_proj,"/f")
# fFile.close()

# F_m = df.grad(displacements_function) + df.Identity(2)
# det_F_m = df.det(F_m)
# det_F_m_proj = df.project(det_F_m, density_function_space)

# fFile = df.HDF5File(df.MPI.comm_world,"det_F_m_proj_1000.h5","w")
# fFile.write(det_F_m_proj,"/f")
# fFile.close()
# f2 = df.Function(density_function_space)

# #save the solution vector
# df.File('solutions/case_1/hyperelastic_cantilever_beam/displacement.pvd') << displacements_function
# stiffness  = df.project(density_function/(1 + 8. * (1. - density_function)), density_function_space) 
# df.File('solutions/case_1/hyperelastic_cantilever_beam/stiffness.pvd') << stiffness
# df.File('solutions/case_1/hyperelastic_cantilever_beam/eps_eq_proj_1000.pvd') << eps_eq_proj
# df.File('solutions/case_1/hyperelastic_cantilever_beam/detF_m_1000.pvd') << det_F_m_proj