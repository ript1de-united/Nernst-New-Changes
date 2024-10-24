import openmdao.api as om
import numpy as np

class nthFuelCellStackComp(om.ExplicitComponent):
    """
    Fuel cell stack model calculates following parameters for an nth fuel cell system
    module operating at current and delivering power:
    ratio of electrical power produced by the fuel cell stack in the module to the voltage
    of the cell in the fuel cell stack, maximum electrical power produced by the
    fuel cell system module, and the electrical efficiency of the fuel cell stack.

    Inputs
    ------
    current_nthfcstack : float
        Current of the nth fuel cell stack (vector, A).
    pwr_el_nthfcmodule_bop : float
        Electrical power consumed by the balance of plant (BOP) of the fuel cell system module (vector, W).

    Outputs
    -------
    ratio_nthpowerfcstackbycellvoltage : float
        Ratio of electrical power produced by the fuel cell stack in the module
        to the voltage of the cell in the fuel cell stack (vector, W/V).
    pwr_el_del_per_nthfcsysmodule : float
        Electrical power delivered per fuel cell system module (vector, W).
    eff_el_nthfcstack : float
        Electrical efficiency of the fuel cell stack in the nth fuel cell system module (vector, dimensionless).

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless).
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        # Global Design Variable
        self.add_input('current_nthfcstack', val=1.0*np.ones(nn))#, units='A'
        # Coupling parameter
        self.add_input('pwr_el_nthfcmodule_bop', val=1.0*np.ones(nn))#, units='J/s'

        # Outputs
        # Coupling output
        self.add_output('ratio_nthpowerfcstackbycellvoltage', val=1.0*np.ones(nn))#, units='W/V'
        self.add_output('pwr_el_del_per_nthfcsysmodule', val=1.0*np.ones(nn))#, units='J/s'
        # Non-Coupling outputs
        self.add_output('eff_el_nthfcstack', val=1.0*np.ones(nn))#, units=None

        # Partials
        ar=np.arange(nn)
        self.declare_partials(of='ratio_nthpowerfcstackbycellvoltage', wrt='current_nthfcstack', rows=ar, cols=ar)
        self.declare_partials(of='ratio_nthpowerfcstackbycellvoltage', wrt='pwr_el_nthfcmodule_bop', rows=ar, cols=ar)
        self.declare_partials(of='pwr_el_del_per_nthfcsysmodule', wrt='current_nthfcstack', rows=ar, cols=ar)
        self.declare_partials(of='pwr_el_del_per_nthfcsysmodule', wrt='pwr_el_nthfcmodule_bop', rows=ar, cols=ar)
        self.declare_partials(of='eff_el_nthfcstack', wrt='current_nthfcstack', rows=ar, cols=ar)
        self.declare_partials(of='eff_el_nthfcstack', wrt='pwr_el_nthfcmodule_bop', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        current_nthfcstack = inputs['current_nthfcstack']
        pwr_el_nthfcmodule_bop = inputs['pwr_el_nthfcmodule_bop']
        num_cells_in_fcstack = 309
        # stack_voltage_reversible = num_cells_in_fcstack * 1.23
        stack_voltage_thermoneutral = num_cells_in_fcstack * 1.48


        stack_voltage = (-2e-7 * current_nthfcstack**3) + (0.0003 * current_nthfcstack**2) - (0.2041 * current_nthfcstack) + 274.36
        pwr_el_nthfcstack = stack_voltage * current_nthfcstack
        outputs['ratio_nthpowerfcstackbycellvoltage'] = pwr_el_nthfcstack / (stack_voltage/num_cells_in_fcstack)
        # outputs['pwr_ht_nthfcstack'] = (stack_voltage_thermoneutral - stack_voltage) * current_nthfcstack

        outputs['eff_el_nthfcstack'] = stack_voltage / stack_voltage_thermoneutral

        outputs['pwr_el_del_per_nthfcsysmodule'] = pwr_el_nthfcstack - pwr_el_nthfcmodule_bop

    def compute_partials(self, inputs, partials):
        current_nthfcstack = inputs['current_nthfcstack']
        num_cells_in_fcstack = 309
        # stack_voltage_reversible = num_cells_in_fcstack * 1.23
        stack_voltage_thermoneutral = num_cells_in_fcstack * 1.48

        partials['ratio_nthpowerfcstackbycellvoltage','current_nthfcstack'] = num_cells_in_fcstack * 1
        partials['ratio_nthpowerfcstackbycellvoltage','pwr_el_nthfcmodule_bop'] = 0
        partials['pwr_el_del_per_nthfcsysmodule','current_nthfcstack'] = (4 * -2e-7 * current_nthfcstack**3) + (3 * 0.0003 * current_nthfcstack**2) - (2 * 0.2041 * current_nthfcstack) + 274.36
        partials['pwr_el_del_per_nthfcsysmodule','pwr_el_nthfcmodule_bop'] = -1
        partials['eff_el_nthfcstack','current_nthfcstack'] = ((3 * -2e-7 * current_nthfcstack**2) + (2 * 0.0003 * current_nthfcstack**1) - (1 * 0.2041 * current_nthfcstack**0))/stack_voltage_thermoneutral
        partials['eff_el_nthfcstack','pwr_el_nthfcmodule_bop'] = 0


class nthFuelCellBoPComp(om.ExplicitComponent):
    """
    Fuel cell system balance of plant model, of the fuel cell stack in the
    nth fuel cell system module operating at current, calculates various
    properties of the balance of plant components (BoP)
    including power required by the air compressor, hydrogen usage rate,
    air usage rate, and overall efficiency of the fuel cell system module.

    Inputs
    ------
    ratio_nthpowerfcstackbycellvoltage : float
        Ratio of electrical power produced by the fuel cell stack in the module
        to the voltage of the cell in the fuel cell stack (vector, W/V).
    pwr_el_del_per_nthfcsysmodule : float
        Electrical power delivered per fuel cell system module (vector, W).

    Outputs
    -------
    pwr_el_nthfcmodule_bop : float
        Electrical power consumed by the balance of plant (BOP) of the fuel cell system module (vector, W).
    pwr_aircmprsr_nthfcstack : float
        Power required by the air compressor for the fuel cell stack (vector, W).
    nthstack_hydrogenusage_rate : float
        Hydrogen usage rate of the fuel cell stack (vector, kg/s).
    nthstack_airusage_rate : float
        Air usage rate of the fuel cell stack (vector, kg/s).
    eff_nthfcsysmodule : float
        Efficiency of the fuel cell system module (vector, dimensionless).

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless).
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)
    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        # Coupling parameters
        self.add_input('ratio_nthpowerfcstackbycellvoltage', val=1.0*np.ones(nn))#, units='W/V'
        self.add_input('pwr_el_del_per_nthfcsysmodule', val=1.0*np.ones(nn))#, units='J/s'

        # Outputs
        # Coupling output
        self.add_output('pwr_el_nthfcmodule_bop', val=1.0*np.ones(nn))#, units='J/s'
        # Non-Coupling output
        self.add_output('eff_nthfcsysmodule', val=1.0*np.ones(nn))#, units=None
        self.add_output('nthstack_airusage_rate', val=1.0*np.ones(nn))#, units='kg/s'
        self.add_output('nthstack_hydrogenusage_rate', val=1.0*np.ones(nn))#, units='kg/s'
        self.add_output('nthstack_waterprodn_rate', val=1.0*np.ones(nn))#, units='kg/s'

        # Partials
        ar=np.arange(nn)
        self.declare_partials(of='pwr_el_nthfcmodule_bop', wrt='ratio_nthpowerfcstackbycellvoltage', rows=ar, cols=ar)
        self.declare_partials(of='pwr_el_nthfcmodule_bop', wrt='pwr_el_del_per_nthfcsysmodule', rows=ar, cols=ar)
        self.declare_partials(of='eff_nthfcsysmodule', wrt='ratio_nthpowerfcstackbycellvoltage', rows=ar, cols=ar)
        self.declare_partials(of='eff_nthfcsysmodule', wrt='pwr_el_del_per_nthfcsysmodule', rows=ar, cols=ar)
        self.declare_partials(of='nthstack_airusage_rate', wrt='ratio_nthpowerfcstackbycellvoltage', rows=ar, cols=ar)
        self.declare_partials(of='nthstack_airusage_rate', wrt='pwr_el_del_per_nthfcsysmodule', rows=ar, cols=ar)
        self.declare_partials(of='nthstack_hydrogenusage_rate', wrt='ratio_nthpowerfcstackbycellvoltage', rows=ar, cols=ar)
        self.declare_partials(of='nthstack_hydrogenusage_rate', wrt='pwr_el_del_per_nthfcsysmodule', rows=ar, cols=ar)
        self.declare_partials(of='nthstack_waterprodn_rate', wrt='ratio_nthpowerfcstackbycellvoltage', rows=ar, cols=ar)
        self.declare_partials(of='nthstack_waterprodn_rate', wrt='pwr_el_del_per_nthfcsysmodule', rows=ar, cols=ar)


    def compute(self, inputs, outputs):
        ratio_nthpowerfcstackbycellvoltage = inputs['ratio_nthpowerfcstackbycellvoltage']
        pwr_el_del_per_nthfcsysmodule = inputs['pwr_el_del_per_nthfcsysmodule']
        lambda_air_nthfcstack = 2.2
        lambda_hydrogen_nthfcstack = 1.2
        hhv_h2 = 1.417e8

        outputs['nthstack_airusage_rate'] = nthstack_airusage_rate = 3.58e-7 * ratio_nthpowerfcstackbycellvoltage * lambda_air_nthfcstack
        outputs['nthstack_hydrogenusage_rate'] = nthstack_hydrogenusage_rate = 1.05e-8 * ratio_nthpowerfcstackbycellvoltage * lambda_hydrogen_nthfcstack
        outputs['nthstack_waterprodn_rate'] = 9.34e-8 * ratio_nthpowerfcstackbycellvoltage

        pwr_aircmprsr_nthfcstack = 1004 * (298/0.7) * (((3*101325)/101325)**(0.286) - 1) * nthstack_airusage_rate
        outputs['pwr_el_nthfcmodule_bop'] = pwr_aircmprsr_nthfcstack

        # Following efficiency calculation does not account for stoichiometric ratio of hydrogen at inlet.
        # Therefore, it is divided to get the real efficiency of fc system module.
        outputs['eff_nthfcsysmodule'] = pwr_el_del_per_nthfcsysmodule / ((nthstack_hydrogenusage_rate / lambda_hydrogen_nthfcstack) * hhv_h2)

    def compute_partials(self,inputs, partials):
        ratio_nthpowerfcstackbycellvoltage = inputs['ratio_nthpowerfcstackbycellvoltage']
        pwr_el_del_per_nthfcsysmodule = inputs['pwr_el_del_per_nthfcsysmodule']
        lambda_air_nthfcstack = 2.2
        lambda_hydrogen_nthfcstack = 1.2
        hhv_h2 = 1.417e8

        partials['pwr_el_nthfcmodule_bop', 'ratio_nthpowerfcstackbycellvoltage'] = 1004 * (298/0.7) * (((3*101325)/101325)**(0.286) - 1) *  3.58e-7 * 1 * lambda_air_nthfcstack
        partials['pwr_el_nthfcmodule_bop', 'pwr_el_del_per_nthfcsysmodule'] = 0
        partials['eff_nthfcsysmodule', 'ratio_nthpowerfcstackbycellvoltage'] = (-1 * pwr_el_del_per_nthfcsysmodule * lambda_hydrogen_nthfcstack) / (hhv_h2 * 1.05e-8 * lambda_hydrogen_nthfcstack * ratio_nthpowerfcstackbycellvoltage**2)
        partials['eff_nthfcsysmodule', 'pwr_el_del_per_nthfcsysmodule'] = (1 * lambda_hydrogen_nthfcstack) / (hhv_h2 * 1.05e-8 * lambda_hydrogen_nthfcstack * ratio_nthpowerfcstackbycellvoltage)
        partials['nthstack_airusage_rate', 'ratio_nthpowerfcstackbycellvoltage'] = 3.58e-7 * 1 * lambda_air_nthfcstack
        partials['nthstack_airusage_rate', 'pwr_el_del_per_nthfcsysmodule'] = 0
        partials['nthstack_hydrogenusage_rate', 'ratio_nthpowerfcstackbycellvoltage'] = 1.05e-8 * 1 * lambda_hydrogen_nthfcstack
        partials['nthstack_hydrogenusage_rate', 'pwr_el_del_per_nthfcsysmodule'] = 0
        partials['nthstack_waterprodn_rate', 'ratio_nthpowerfcstackbycellvoltage'] = 9.34e-8 * 1
        partials['nthstack_waterprodn_rate', 'pwr_el_del_per_nthfcsysmodule'] = 0


class nthFCModuleGroup(om.Group):
    """
    This group models the optimisation cycle of the nth fuel cell module operating at the
    a current, and includes modeling of the fuel cell stack, and its balance of plant.

    Subsystems
    ----------
    nthcyclefcstackandbop : Group
        A group containing subsystems for modeling the fuel cell stack and BoP.

    Components
    ----------
    d1 : nthFuelCellStackComp
        Component modeling the fuel cell stack.
    d2 : nthFuelCellBoPComp
        Component modeling the balance of plant (BoP) for the fuel cell stack.
    obj_cmp : ExecComp
        Component calculating the objective function based on the efficiency of the fuel cell stack or system.

    Nonlinear Solver (not necessary and can be avoided)
    ----------------
    nlsolver : NonlinearBlockGS
        Nonlinear solver used for solving the nonlinear system within the group.

    Linear Solver
    -------------
    lsolver : DirectSolver
        Linear solver used for solving the linear system within the group.

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless).
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        nthcyclefcstackandbop = self.add_subsystem('nthcyclefcstackandbop', om.Group(), promotes=['*'])

        nthcyclefcstackandbop.add_subsystem('d1', nthFuelCellStackComp(num_nodes=nn), promotes_inputs=['current_nthfcstack', 'pwr_el_nthfcmodule_bop'],
                            promotes_outputs=['ratio_nthpowerfcstackbycellvoltage', 'eff_el_nthfcstack', 'pwr_el_del_per_nthfcsysmodule'])

        nthcyclefcstackandbop.add_subsystem('d2', nthFuelCellBoPComp(num_nodes=nn), promotes_inputs=['ratio_nthpowerfcstackbycellvoltage', 'pwr_el_del_per_nthfcsysmodule'],
                            promotes_outputs=['pwr_el_nthfcmodule_bop', 'nthstack_airusage_rate','nthstack_hydrogenusage_rate','eff_nthfcsysmodule', 'nthstack_waterprodn_rate'])

        nlsolver = nthcyclefcstackandbop.nonlinear_solver = om.NonlinearBlockGS()
        # nlsolver = nthcyclefcstackandbop.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        # nlsolver = nthcyclefcstackandbop.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)

        # lsolver = nthcyclefcstackandbop.linear_solver = om.LinearBlockGS()
        # lsolver = nthcyclefcstackandbop.linear_solver = om.ScipyKrylov()
        lsolver = nthcyclefcstackandbop.linear_solver = om.DirectSolver()

        iterations = 100
        # lsolver.options['maxiter'] = iterations
        lsolver.options['iprint'] = -1

        nlsolver.options['maxiter'] = iterations
        nlsolver.options['iprint'] = -1
        # # nlsolver.options['rtol'] = 1e-16
        # # nlsolver.options['atol'] = 1e-16
        # nlsolver.options['err_on_non_converge'] = True
        # nlsolver.options['debug_print'] = True
        # if nlsolver == om.NonlinearBlockGS():
        #     nlsolver.options['use_aitken'] = False

        # recorder = om.SqliteRecorder("casesnthfcsysmodule.sql")
        # nlsolver.add_recorder(recorder)
        # nlsolver.recording_options['record_abs_error'] = True
        # nlsolver.recording_options['record_rel_error'] = True
        # nlsolver.recording_options['record_inputs'] = True
        # nlsolver.recording_options['record_outputs'] = True
        # if nlsolver == om.NonlinearBlockGS():
        #     nlsolver.options['use_apply_nonlinear'] = True

        # obj_cmp = om.ExecComp('obj = eff_el_nthfcstack', shape=nn, obj={'units': None}, eff_el_nthfcstack={'units': None})
        # self.add_subsystem('obj_cmp', obj_cmp, promotes_inputs=['eff_el_nthfcstack'], promotes_outputs=['obj'])
        # obj_cmp.declare_partials('*', '*', method='cs')
        self.add_subsystem('obj_cmp', om.ExecComp('obj = eff_el_nthfcstack', shape=nn, obj={'units': None}, eff_el_nthfcstack={'units': None}), promotes_inputs=['eff_el_nthfcstack'], promotes_outputs=['obj'])
        # self.add_constraint('obj', upper=0.6)
        # self.add_objective('obj', scaler=-1)

        # self.add_subsystem('con_cmp1', om.ExecComp('maxcon1 = pwr_el_req_per_nthfcsysmodule - pwr_el_del_per_nthfcsysmodule'), promotes_inputs=['pwr_el_del_per_nthfcsysmodule', 'pwr_el_req_per_nthfcsysmodule'], promotes_outputs=['maxcon1'])
        # self.add_constraint('maxcon1', upper=0.0)


# prob = om.Problem()

# prob.model.add_subsystem('fcsysmodule', nthFCModuleGroup(num_nodes=1))

# prob.driver = om.ScipyOptimizeDriver()
# prob.driver.options['optimizer'] = 'SLSQP'
# prob.driver.options['maxiter'] = 100
# prob.driver.options['tol'] = 1e-16

# prob.model.add_design_var('fcsysmodule.current_nthfcstack', lower=5, upper=630)
# prob.model.add_objective('fcsysmodule.obj', scaler=1)

# # Ask OpenMDAO to finite-difference across the model to compute the gradients for the optimizer
# prob.model.approx_totals()

# prob.setup()
# # prob.model.list_inputs(units=True, shape=True)
# # prob.model.list_outputs(units=True, shape=True)
# # prob.check_partials(method='cs', compact_print=False, show_only_incorrect=True)
# # prob.check_partials(method='cs', compact_print=True, show_only_incorrect=True)

# # prob.set_solver_print(level=2)

# prob.run_driver()

# print('minimum found at')
# print(prob.get_val('fcsysmodule.current_nthfcstack'))

# print('Power delivered by FC System Module [W]')
# print(prob.get_val('fcsysmodule.pwr_el_del_per_nthfcsysmodule')[0])

# print('Power consumed by the air compressor [W]')
# print(prob.get_val('fcsysmodule.pwr_el_nthfcmodule_bop')[0])

# # print('Power produced by FC Stack [W]')
# # print(prob.get_val('fcsysmodule.pwr_el_nthfcstack')[0])

# print('Stack Air Flow Rate [kg/s]')
# print(prob.get_val('fcsysmodule.nthstack_airusage_rate')[0])

# print('Stack Elec Efficiency [%]')
# print(prob.get_val('fcsysmodule.eff_el_nthfcstack')[0])

# print('Fuel Cell System Module Electrical Efficiency [%]')
# print(prob.get_val('fcsysmodule.eff_nthfcsysmodule')[0])

# print('Stack Hydrogen Usage rate [kg/s]')
# print(prob.get_val('fcsysmodule.nthstack_hydrogenusage_rate')[0])