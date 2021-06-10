import numpy as np
import scipy.sparse as sp
import om_lite.api as om

# from atomics_lite.api import PDEProblem, AtomicsGroup

from atomics_lite.api import PDEProblem

# from atomics_lite.pdes.neo_hookean_addtive import get_residual_form

from atomics_lite.general_filter_comp import GeneralFilterComp
from atomics_lite.states_comp import StatesComp
from atomics_lite.scalar_output_comp import ScalarOutputsComp
# from atomics_lite.field_output_comp import FieldOutputsComp

class HyperElasticModel(om.Model):

    def setup(self, num_dof_density, density_function_space, pde_problem):
        
        x = om.IndepVarComp()
        self.add_subsystem('x', x) # inputs and outputs can be added only after adding subsystems to the model

        x.add_output(
            'density_unfiltered',
            val=np.full((num_dof_density,), .4),
        ) # this can be moved to setup() of indepvar comp also later

        # self.nx = nx = num_dof_density
        self.nx = nx = self.outputs['density_unfiltered'].size

        density_filter = GeneralFilterComp(
            density_function_space=density_function_space)
        self.add_subsystem('density_filter', density_filter)

        dx1_dx_data = self.partials['density', 'density_unfiltered']
        coo = self.partials['density', 'density_unfiltered', 'coo']
        self.dx1_dx = sp.coo_matrix((dx1_dx_data, coo), shape=(self.outputs['density'].size, self.inputs['density_unfiltered'].size))

        # group = AtomicsGroup(pde_problem=pde_problem)
        # self.add_subsystem('group', group)

        c = ScalarOutputsComp(
                pde_problem=pde_problem,
                scalar_output_name='avg_density', 
            )
        self.add_subsystem('c', c)

        linear_solver_ = 'petsc_cg_ilu'
        problem_type = 'nonlinear_problem'
        visualization = 'False'

        y = StatesComp(
                pde_problem=pde_problem,
                state_name='displacements',
                linear_solver_=linear_solver_,
                problem_type=problem_type,
                visualization=visualization
            )
        self.add_subsystem('y', y)

        self.ny = ny = self.outputs['displacements'].size

        f = ScalarOutputsComp(
                pde_problem=pde_problem,
                scalar_output_name='compliance', 
            )
        self.add_subsystem('f', f)

        self.pf_px = np.zeros((self.inputs['density_unfiltered'].size,),)
        self.pc_py = sp.coo_matrix(np.zeros((self.outputs['displacements'].size, 1),))


        # print(self.inputs)
        # print(self.outputs)
        # print(self.partials)
        # print(self.residuals)

        # super().setup()

        self.linear_eq_constraints = {}
        self.linear_eq_constraints['A'] = None
        self.linear_eq_constraints['b'] = None

        self.linear_ineq_constraints = {}
        self.linear_ineq_constraints['A'] = None
        self.linear_ineq_constraints['b'] = None


        self.bound_constraints = {}
        self.bound_constraints['dv_indices'] = np.arange(nx)
        # self.bound_constraints['upper'] = np.full((nx,), .99)
        self.bound_constraints['upper'] = np.full((nx,), 1.)
        self.bound_constraints['lower'] = np.full((nx,), 5e-3)

    def evaluate_functions(self, x_val, y_val, tol=1.11*1e-3, method='surf'):
        # if x_val is not None:
        #     self.x.set_val('density_unfiltered', x_val)
        # if y_val is not None:
        #     self.y.set_val('displacements', y_val)

        inputs = self.inputs
        outputs = self.outputs
        partials = self.partials
        residuals = self.residuals

        self.x.set_val('density_unfiltered', x_val)

        self.density_filter._compute(inputs, outputs)
        self.c._compute(inputs, outputs)

        # TODO: Add apply_nonlinear in states_comp
        if method == 'fs':
            self.y.apply_nonlinear(inputs, outputs, residuals)
        elif method == 'cfs':
            # solve_nonlinear is different from OM
            self.y._solve_nonlinear(inputs, outputs, tol)
        elif method == 'surf':
            self.y._solve_nonlinear(inputs, outputs, tol)
            # self.y.set_value(y_val)
            self.y.apply_nonlinear(inputs, outputs, residuals)

        self.f._compute(inputs, outputs)

        # print(self.inputs)
        # print(self.outputs)
        # print(self.partials)
        # print(self.residuals)

        print(outputs['avg_density'])

        return outputs['displacements'], outputs['compliance'], residuals['displacements'], outputs['avg_density']
        

    def evaluate_derivatives(self, x_val=None, y_val=None, psi=None, tol=1.11*1e-3, method='surf'):
        
        # if x_val is not None:
        #     self.x.set_val('density_unfiltered', x_val)
        # if y_val is not None:
        #     self.y.set_val('displacements', y_val)
        # if psi is not None:
        #     self.y.set_val('displacements', y_val)

        inputs = self.inputs
        outputs = self.outputs
        partials = self.partials

        self.x.set_val('density_unfiltered', x_val)
        
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
        
        # if method != 'fs':
        self.psi = -d_residuals['displacements']
        print(self.psi)

        # Making partials scipy coo matrix
        pR_px1_data = partials['displacements', 'density']
        coo = partials['displacements', 'density', 'coo']
        pR_px1 = sp.coo_matrix((pR_px1_data, coo), shape=(outputs['displacements'].size, inputs['density'].size))

        pR_py_data = partials['displacements', 'displacements']
        coo = partials['displacements', 'displacements', 'coo']
        pR_py = sp.coo_matrix((pR_py_data, coo), shape=(outputs['displacements'].size, inputs['displacements'].size))

        dx1_dx = self.dx1_dx
  
        if method != 'fs':
            pf_px = self.pf_px
            pf_py = partials['compliance', 'displacements']

            if method == 'rs':
                df_dx1 = self.psi @ pR_px1
                df_dx = df_dx1 @ dx1_dx
            
            # Cheating
            # if method :
            #     df_dx1 = self.psi @ pR_px1
            #     df_dx = df_dx1 @ dx1_dx

            #     dR_dx = sp.csr_matrix(np.zeros((outputs['displacements'].size, inputs['density'].size)))
            
            # Chain rule
            pc_px = dc_dx = partials['avg_density', 'density'] @ dx1_dx
            pc_py = self.pc_py

            self.pR_px = pR_px1 @ dx1_dx
            self.pR_py = pR_py

            print("reached END")

        # print(self.inputs)
        # print(self.outputs)
        # print(self.partials)
        # print(self.residuals)


        # print(self.pR_px.data)
        # print(len(self.pR_px.data))
        # print(type(self.pR_px))

        # print(self.pR_py.data)
        # print(len(self.pR_py.data))
        # print(type(self.pR_py))

        return pf_px, pf_py, pc_px, pc_py, self.psi, self.pR_px, self.pR_py
        # return df_dx, pf_py, pc_px, pc_py, self.psi, dR_dx, self.pR_py

    def MAUD(self):
        
        # if x_val is not None:
        #     self.x.set_val('density_unfiltered', x_val)
        # if y_val is not None:
        #     self.y.set_val('displacements', y_val)
        # if psi is not None:
        #     self.y.set_val('displacements', y_val)

        inputs = self.inputs
        outputs = self.outputs
        partials = self.partials

        self.c.compute_partials(inputs, partials)
        self.y.linearize(inputs, outputs, partials)
        self.f.compute_partials(inputs, partials)

        # Assemble_jacobians
        for of in outputs:
            for wrt in inputs:
                rows = None
                cols = None
                if (of, wrt, 'coo') in partials:
                    rows=partials[of, wrt, 'coo'][0]
                    cols=partials[of, wrt, 'coo'][1]

                self.full_jac_dict[of, wrt] = dict(
                vals = partials[of, wrt],
                rows=rows,
                cols=cols,
                ind_ptr=None,
                vals_shape = None,
            )

            full_jac = Matrix(full_jac_dict)

        # solve MAUD




        # psi = self.y.solve_linear(d_outputs, d_residuals, mode='rev')

        d_outputs = {'displacements' : partials['compliance', 'displacements']}
        d_residuals = {'displacements' : np.zeros((outputs['displacements'].size, 1), dtype=float)}
        self.y.solve_linear(d_outputs, d_residuals, mode='rev')
        
        # if method != 'fs':
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
            df_dx1 = self.psi @ partials['displacements', 'density']

            # Chain rule
            # dc_dx = partials['c', 'x1'] @ partials['x1', 'x']
            dc_dx = partials['avg_density', 'density'] @ partials['density', 'density_unfiltered']
            # df_dx = partials['f', 'x1'] @ partials['x1', 'x']
            df_dx = partials['compliance', 'density'] @ partials['density', 'density_unfiltered']
            # pR_px = partials['R', 'x1'] @ partials['x1', 'x']
            self.pR_px = partials['displacements', 'density'] @ partials['density', 'density_unfiltered']


        return df_dx, dc_dx, self.psi, self.pR_px
