import numpy as np
import scipy.sparse as sp
import om_lite.api as om

# from atomics_lite.api import PDEProblem, AtomicsGroup

from atomics_lite.api import PDEProblem

# from atomics_lite.pdes.neo_hookean_addtive import get_residual_form

from atomics_lite.general_filter_comp import GeneralFilterComp
from atomics_lite.states_comp import StatesComp
from atomics_lite.scalar_output_comp import ScalarOutputsComp
from atomics_lite.field_output_comp import FieldOutputsComp

class KirchhoffElasticModel(object):

    def setup(self, num_dof_density, density_function_space, pde_problem):
        self.inputs = {}
        self.outputs = {}
        self.partials = {}
        self.residuals = {}
        self.ex_components_list = []  # does not include IndepVarComp

        self.x = om.IndepVarComp()
        self.x.outputs = self.outputs
        self.x.add_output(
            'density_unfiltered',
            shape=num_dof_density,
            val=np.ones(num_dof_density),
            # val=x_val,
            # val=np.random.random(num_dof_density) * 0.86,
        )

        self.density_filter = GeneralFilterComp(
            density_function_space=density_function_space)
        self.ex_components_list.append(self.density_filter)
        # density_filter.setup()

        # group = AtomicsGroup(pde_problem=pde_problem)
        # components_list.append(group)

        self.c = ScalarOutputsComp(
                pde_problem=pde_problem,
                scalar_output_name='avg_density', 
            )
        self.ex_components_list.append(self.c)
        # c.setup()

        linear_solver_ = 'petsc_cg_ilu'
        problem_type = 'nonlinear_problem'
        visualization = 'False'

        self.y = StatesComp(
                pde_problem = pde_problem,
                state_name='displacements',
                linear_solver_=linear_solver_,
                problem_type=problem_type,
                visualization=visualization
            )
        # self.ex_components_list.append(y)
        # y.setup()

        self.f = ScalarOutputsComp(
                pde_problem=pde_problem,
                scalar_output_name='compliance', 
            )
        self.ex_components_list.append(self.f)
        # f.setup()

        # Setup (Note: Promotes all)
        for comp in self.ex_components_list:
            comp.inputs = self.inputs
            comp.outputs = self.outputs
            comp.partials = self.partials
            comp.setup()

        self.y.inputs = self.inputs
        self.y.outputs = self.outputs
        self.y.partials = self.partials
        self.y.setup()


    # tol=10 means tol=None runs the equivalent of full-space method (just one nonlinear iteration is done)
    def evaluate_functions(self, x_val, y_val, psi=None, tol=1.11*1e-3, method='surf'):
        self.x.set_val(x_val)
        self.y.set_val(y_val)

        inputs = self.inputs
        outputs = self.outputs
        partials = self.partials

        self.density_filter.compute(inputs, outputs)

        self.c.compute(inputs, outputs)

        # TODO: Add apply_nonlinear in states_comp
        if method == 'fs':
            self.y.apply_nonlinear(inputs, outputs, tol)
        elif method == 'cfs':
            # solve_nonlinear is different from OM
            self.y.solve_nonlinear(inputs, outputs, tol)
        elif method == 'surf':
            self.y.solve_nonlinear(inputs, outputs, tol)
            # self.y.set_value(y_val)
            self.y.apply_nonlinear(inputs, outputs, tol)

        self.f.compute(inputs, outputs)

        return y_val, outputs['compliance'], outputs['avg_density']
        

    def evaluate_derivatives(self, x_val=None, y_val=None, psi=None, tol=1.11*1e-3, method='surf'):
        
        if x_val is not None:
            self.x.set_value('density_unfiltered', x_val)
        if y_val is not None:
            self.y.set_value('displacements', y_val)
        if psi is not None:
            self.y.set_value('displacements', y_val)

        inputs = self.inputs
        outputs = self.outputs
        partials = self.partials
        
        # self.density_filter.compute_partials(inputs, partials)
        self.c.compute_partials(inputs, partials)
        self.y.linearize(inputs, outputs, partials)
        self.f.compute_partials(inputs, partials)

        # Adjoint vector
        # psi = np.linalg.solve(partials['R', 'y'].T, partials['f', 'y'].T)
        # psi = np.linalg.solve(partials['displacements', 'displacements'].T, partials['compliance', 'displacements'])

        # psi = self.y.solve_linear(d_outputs, d_residuals, mode='rev')

        d_outputs = {'displacements' : partials['compliance', 'displacements']}
        d_residuals = {'displacements' : np.zeros((outputs['displacements'].size, 1), dtype=float)}
        self.y.solve_linear(d_outputs, d_residuals, mode='rev')
        
        if method != 'fs':
            self.psi = d_residuals['displacements']

        # Making partials scipy coo matrix
        py_px1 = partials['displacements', 'density']
        coo = partials['displacements', 'density', 'coo']
        partials['displacements', 'density'] = sp.coo_matrix((py_px1, coo), shape=(outputs['displacements'].size, inputs['density'].size))

        dx1_dx = partials['density', 'density_unfiltered']
        coo = partials['density', 'density_unfiltered', 'coo']
        partials['density', 'density_unfiltered'] = sp.coo_matrix((dx1_dx, coo), shape=(outputs['density'].size, inputs['density_unfiltered'].size))
        
        # df_dx1 = psi @ partials['R', 'x1']
        if method != 'fs':
            df_dx1 = psi @ partials['displacements', 'density']

            # Chain rule
            # dc_dx = partials['c', 'x1'] @ partials['x1', 'x']
            dc_dx = partials['avg_density', 'density'] @ partials['density', 'density_unfiltered']
            # df_dx = partials['f', 'x1'] @ partials['x1', 'x']
            df_dx = partials['compliance', 'density'] @ partials['density', 'density_unfiltered']
            # pR_px = partials['R', 'x1'] @ partials['x1', 'x']
            self.pR_px = partials['displacements', 'density'] @ partials['density', 'density_unfiltered']

        else:
            df_dx1 = psi @ partials['displacements', 'density']

            # Chain rule
            # dc_dx = partials['c', 'x1'] @ partials['x1', 'x']
            dc_dx = partials['avg_density', 'density'] @ partials['density', 'density_unfiltered']
            # df_dx = partials['f', 'x1'] @ partials['x1', 'x']
            df_dx = partials['compliance', 'density'] @ partials['density', 'density_unfiltered']
            # pR_px = partials['R', 'x1'] @ partials['x1', 'x']
            self.pR_px = partials['displacements', 'density'] @ partials['density', 'density_unfiltered']

        return df_dx, dc_dx, self.psi, self.pR_px
