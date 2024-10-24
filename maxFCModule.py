import openmdao.api as om
import numpy as np

class maxFuelCellStackComp(om.ExplicitComponent):
    """
    Fuel cell stack model calculates following parameters for a fuel cell system
    module operating at maximum current delivering maximum power:
    ratio of electrical power produced by the fuel cell stack in the module to the voltage
    of the cell in the fuel cell stack, maximum electrical power produced by the
    fuel cell system module, and the electrical efficiency of the fuel cell stack.

    Inputs
    ------
    current_maxfcstack : float
        Current of the maximum fuel cell stack (vector, A).
    pwr_el_maxfcmodule_bop : float
        Electrical power of the balance of plant (BOP) for the maximum fuel cell module (vector, W).

    Outputs
    -------
    ratio_maxpowerfcstackbycellvoltage : float
        Ratio of maximum electrical power produced by the fuel cell stack in the module
        to the voltage of the cell in the fuel cell stack (vector, W/V).
    pwr_el_del_per_maxfcsysmodule : float
        Electrical power delivered per fuel cell system module (vector, W).
    eff_el_maxfcstack : float
        Electrical efficiency of the fuel cell stack (vector, dimensionless).

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
        self.add_input('current_maxfcstack', val=1*np.ones(nn))#, units='A', tags=['dymos.static_target'])
        # Coupling parameter
        self.add_input('pwr_el_maxfcmodule_bop', val=1*np.ones(nn))#, units='J/s'

        # Outputs
        # Coupling outputs
        self.add_output('ratio_maxpowerfcstackbycellvoltage', val=1.0*np.ones(nn))#, units='W/V'
        self.add_output('pwr_el_del_per_maxfcsysmodule', val=1.0*np.ones(nn))#, units='J/s'
        self.add_output('pwr_el_maxfcstack', val=1.0*np.ones(nn))   #added 
        # Non-Coupling outputs
        self.add_output('eff_el_maxfcstack', val=1.0*np.ones(nn))#, units=None

        # Partials
        ar=np.arange(nn)
        self.declare_partials(of='ratio_maxpowerfcstackbycellvoltage', wrt='current_maxfcstack', rows=ar, cols=ar)
        self.declare_partials(of='ratio_maxpowerfcstackbycellvoltage', wrt='pwr_el_maxfcmodule_bop', rows=ar, cols=ar)
        self.declare_partials(of='pwr_el_del_per_maxfcsysmodule', wrt='current_maxfcstack', rows=ar, cols=ar)
        self.declare_partials(of='pwr_el_del_per_maxfcsysmodule', wrt='pwr_el_maxfcmodule_bop', rows=ar, cols=ar)
        self.declare_partials(of='eff_el_maxfcstack', wrt='current_maxfcstack', rows=ar, cols=ar)
        self.declare_partials(of='eff_el_maxfcstack', wrt='pwr_el_maxfcmodule_bop', rows=ar, cols=ar)
        self.declare_partials(of='pwr_el_maxfcstack', wrt='current_maxfcstack', rows=ar, cols=ar) #added
        self.declare_partials(of='pwr_el_maxfcstack', wrt='pwr_el_maxfcmodule_bop', rows=ar, cols=ar) #added

    def compute(self, inputs, outputs):
        current_maxfcstack = inputs['current_maxfcstack']
        pwr_el_maxfcmodule_bop = inputs['pwr_el_maxfcmodule_bop']
        num_cells_in_fcstack = 309
        # stack_voltage_reversible = num_cells_in_fcstack * 1.23
        stack_voltage_thermoneutral = num_cells_in_fcstack * 1.48

        stack_voltage = (-2e-7 * current_maxfcstack**3) + (0.0003 * current_maxfcstack**2) - (0.2041 * current_maxfcstack) + 274.36
        pwr_el_maxfcstack = stack_voltage * current_maxfcstack
        #pwr_el_maxfcstack = (-2e-7 * current_maxfcstack**4) + (0.0003 * current_maxfcstack**3) - (0.2041 * current_maxfcstack**2) + 274.36 * current_maxfcstack
        outputs['ratio_maxpowerfcstackbycellvoltage'] = pwr_el_maxfcstack / (stack_voltage/num_cells_in_fcstack)
       
        outputs['eff_el_maxfcstack'] = stack_voltage / stack_voltage_thermoneutral

        outputs['pwr_el_del_per_maxfcsysmodule'] = pwr_el_maxfcstack - pwr_el_maxfcmodule_bop

    def compute_partials(self, inputs, partials):
        current_maxfcstack = inputs['current_maxfcstack']
        num_cells_in_fcstack = 309
        stack_voltage_thermoneutral = num_cells_in_fcstack * 1.48

        partials['ratio_maxpowerfcstackbycellvoltage','current_maxfcstack'] = num_cells_in_fcstack * 1
        partials['ratio_maxpowerfcstackbycellvoltage','pwr_el_maxfcmodule_bop'] = 0
        partials['pwr_el_del_per_maxfcsysmodule','current_maxfcstack'] = (4 * -2e-7 * current_maxfcstack**3) + (3 * 0.0003 * current_maxfcstack**2) - (2 * 0.2041 * current_maxfcstack) + 274.36
        partials['pwr_el_del_per_maxfcsysmodule','pwr_el_maxfcmodule_bop'] = -1
        partials['eff_el_maxfcstack','current_maxfcstack'] = ((3 * -2e-7 * current_maxfcstack**2) + (2 * 0.0003 * current_maxfcstack**1) - (1 * 0.2041 * current_maxfcstack**0))/stack_voltage_thermoneutral
        partials['eff_el_maxfcstack','pwr_el_maxfcmodule_bop'] = 0
        #changed partials 
        partials['pwr_el_maxfcstack','current_maxfcstack'] = (4*-2e-7 * current_maxfcstack**3)+ (3*0.0003 * current_maxfcstack**2) - (2*0.2041 * current_maxfcstack)+ 274.36
        partials['pwr_el_maxfcstack','pwr_el_maxfcmodule_bop'] = 0
        


class maxFuelCellBoPComp(om.ExplicitComponent):
    """
    Fuel cell system balance of plant model, of the fuel cell stack operating at maximum current,
    calculates various properties of the balance of plant components (BoP)
    including power required by the air compressor, hydrogen usage rate,
    airusage rate, and overall efficiency of the fuel cell system module.

    Inputs
    ------
    ratio_maxpowerfcstackbycellvoltage : float
        Ratio of maximum electrical power produced by the fuel cell stack in the module
        to the voltage of the cell in the fuel cell stack (vector, W/V).
    pwr_el_del_per_maxfcsysmodule : float
        Electrical power delivered per fuel cell system module (vector, W).

    Outputs
    -------
    pwr_el_maxfcmodule_bop : float
        Electrical power consumed by the balance of plant (BOP) of the fuel cell system module (vector, W).
    pwr_aircmprsr_maxfcstack : float
        Power required by the air compressor for the fuel cell stack (vector, W).
    maxstack_hydrogenusage_rate : float
        Hydrogen usage rate of the fuel cell stack (vector, kg/s).
    maxstack_airusage_rate : float
        Airusage rate of the fuel cell stack (vector, kg/s).
    eff_maxfcsysmodule : float
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
        # Coupling inputs
        self.add_input('ratio_maxpowerfcstackbycellvoltage', val=1.0*np.ones(nn))#, units='W/V'
        self.add_input('pwr_el_del_per_maxfcsysmodule', val=1.0*np.ones(nn))#, units='J/s'

        # Outputs
        # Coupling output
        self.add_output('pwr_el_maxfcmodule_bop', val=1.0*np.ones(nn))#, units='J/s'
        # Non-Coupling outputs
        self.add_output('pwr_aircmprsr_maxfcstack', val=1.0*np.ones(nn))#, units='J/s'
        self.add_output('maxstack_hydrogenusage_rate', val=1.0*np.ones(nn))#, units='kg/s'
        self.add_output('maxstack_airusage_rate', val=1.0*np.ones(nn))#, units='kg/s'
        self.add_output('eff_maxfcsysmodule', val=1.0*np.ones(nn))#, units=None

        # Partials
        ar=np.arange(nn)
        self.declare_partials(of='pwr_el_maxfcmodule_bop', wrt='ratio_maxpowerfcstackbycellvoltage', rows=ar, cols=ar)
        self.declare_partials(of='pwr_el_maxfcmodule_bop', wrt='pwr_el_del_per_maxfcsysmodule', rows=ar, cols=ar)
        self.declare_partials(of='pwr_aircmprsr_maxfcstack', wrt='ratio_maxpowerfcstackbycellvoltage', rows=ar, cols=ar)
        self.declare_partials(of='pwr_aircmprsr_maxfcstack', wrt='pwr_el_del_per_maxfcsysmodule', rows=ar, cols=ar)
        self.declare_partials(of='maxstack_airusage_rate', wrt='ratio_maxpowerfcstackbycellvoltage', rows=ar, cols=ar)
        self.declare_partials(of='maxstack_airusage_rate', wrt='pwr_el_del_per_maxfcsysmodule', rows=ar, cols=ar)
        self.declare_partials(of='maxstack_hydrogenusage_rate', wrt='ratio_maxpowerfcstackbycellvoltage', rows=ar, cols=ar)
        self.declare_partials(of='maxstack_hydrogenusage_rate', wrt='pwr_el_del_per_maxfcsysmodule', rows=ar, cols=ar)
        self.declare_partials(of='eff_maxfcsysmodule', wrt='ratio_maxpowerfcstackbycellvoltage', rows=ar, cols=ar)
        self.declare_partials(of='eff_maxfcsysmodule', wrt='pwr_el_del_per_maxfcsysmodule', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        ratio_maxpowerfcstackbycellvoltage = inputs['ratio_maxpowerfcstackbycellvoltage']
        pwr_el_del_per_maxfcsysmodule = inputs['pwr_el_del_per_maxfcsysmodule']
        lambda_air = 2.2
        lambda_hydrogen = 1.2
        hhv_h2 = 1.417e8

        outputs['maxstack_airusage_rate'] = maxstack_airusage_rate = 3.58e-7 * ratio_maxpowerfcstackbycellvoltage * lambda_air
        outputs['pwr_aircmprsr_maxfcstack'] = pwr_aircmprsr_maxfcstack = 1004 * (298/0.7) * (((3*101325)/101325)**(0.286) - 1) * maxstack_airusage_rate
        outputs['pwr_el_maxfcmodule_bop'] = pwr_aircmprsr_maxfcstack

        outputs['maxstack_hydrogenusage_rate'] = maxstack_hydrogenusage_rate = 1.05e-8 * ratio_maxpowerfcstackbycellvoltage * lambda_hydrogen

        #outputs['stack_water_production_rate'] = 9.34e-8 * ratio_maxpowerfcstackbycellvoltage

        # Following efficiency calculation does not account for stoichiometric ratio of hydrogen at inlet.
        # Therefore, it is divided to get the real efficiency of fc system module.
        outputs['eff_maxfcsysmodule'] = pwr_el_del_per_maxfcsysmodule / ((maxstack_hydrogenusage_rate/lambda_hydrogen) * hhv_h2)

    def compute_partials(self,inputs, partials):
        ratio_maxpowerfcstackbycellvoltage = inputs['ratio_maxpowerfcstackbycellvoltage']
        pwr_el_del_per_maxfcsysmodule = inputs['pwr_el_del_per_maxfcsysmodule']
        lambda_air = 2.2
        lambda_hydrogen = 1.2
        hhv_h2 = 1.417e8

        partials['pwr_el_maxfcmodule_bop', 'ratio_maxpowerfcstackbycellvoltage'] = 1004 * (298/0.7) * (((3*101325)/101325)**(0.286) - 1) *  3.58e-7 * 1 * lambda_air
        partials['pwr_el_maxfcmodule_bop', 'pwr_el_del_per_maxfcsysmodule'] = 0
        partials['maxstack_airusage_rate', 'ratio_maxpowerfcstackbycellvoltage'] = 3.58e-7 * 1 * lambda_air
        partials['maxstack_airusage_rate', 'pwr_el_del_per_maxfcsysmodule'] = 0
        partials['pwr_aircmprsr_maxfcstack', 'ratio_maxpowerfcstackbycellvoltage'] = 1004 * (298/0.7) * (((3*101325)/101325)**(0.286) - 1) * 3.58e-7 * 1 * lambda_air
        partials['pwr_aircmprsr_maxfcstack', 'pwr_el_del_per_maxfcsysmodule'] = 0
        partials['maxstack_hydrogenusage_rate', 'ratio_maxpowerfcstackbycellvoltage'] = 1.05e-8 * 1 * lambda_hydrogen
        partials['maxstack_hydrogenusage_rate', 'pwr_el_del_per_maxfcsysmodule'] = 0
        partials['eff_maxfcsysmodule', 'ratio_maxpowerfcstackbycellvoltage'] = -1 * pwr_el_del_per_maxfcsysmodule / (((1.05e-8 * ratio_maxpowerfcstackbycellvoltage**2 * lambda_hydrogen)/lambda_hydrogen) * hhv_h2)
        partials['eff_maxfcsysmodule', 'pwr_el_del_per_maxfcsysmodule'] = 1 / (((1.05e-8 * ratio_maxpowerfcstackbycellvoltage * lambda_hydrogen)/lambda_hydrogen) * hhv_h2)

class maxFCModuleGroup(om.Group):
    """
    This group models the optimisation cycle of the fuel cell module operating at the
    maximum current, and includes modeling of the fuel cell stack, and its balance of plant.

    Subsystems
    ----------
    maxcyclefcstackandbop : Group
        A group containing subsystems for modeling the fuel cell stack and BoP.

    Components
    ----------
    d1 : maxFuelCellStackComp
        Component modeling the fuel cell stack.
    d2 : maxFuelCellBoPComp
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

        maxcyclefcstackandbop = self.add_subsystem('maxcyclefcstackandbop', om.Group(), promotes=['*'])

        maxcyclefcstackandbop.add_subsystem('d1', maxFuelCellStackComp(num_nodes=nn),
                                            promotes_inputs=['current_maxfcstack', 'pwr_el_maxfcmodule_bop'],
                                            promotes_outputs=['ratio_maxpowerfcstackbycellvoltage', 'pwr_el_del_per_maxfcsysmodule','eff_el_maxfcstack','pwr_el_maxfcstack'])

        maxcyclefcstackandbop.add_subsystem('d2', maxFuelCellBoPComp(num_nodes=nn),
                                            promotes_inputs=['ratio_maxpowerfcstackbycellvoltage', 'pwr_el_del_per_maxfcsysmodule'],
                                            promotes_outputs=['pwr_el_maxfcmodule_bop', 'pwr_aircmprsr_maxfcstack',
                                                              'maxstack_airusage_rate','maxstack_hydrogenusage_rate','eff_maxfcsysmodule'])

        nlsolver = maxcyclefcstackandbop.nonlinear_solver = om.NonlinearBlockGS()
        # nlsolver = maxcyclefcstackandbop.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        # nlsolver = maxcyclefcstackandbop.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)

        # lsolver = maxcyclefcstackandbop.linear_solver = om.LinearBlockGS()
        # lsolver = maxcyclefcstackandbop.linear_solver = om.ScipyKrylov()
        lsolver = maxcyclefcstackandbop.linear_solver = om.DirectSolver()

        iterations = 100
        # lsolver.options['maxiter'] = iterations
        lsolver.options['iprint'] = -1

        nlsolver.options['maxiter'] = iterations
        nlsolver.options['iprint'] = -1
        # nlsolver.options['rtol'] = 1e-16
        # nlsolver.options['atol'] = 1e-16
        # nlsolver.options['err_on_non_converge'] = True
        # nlsolver.options['debug_print'] = True
        # if nlsolver == om.NonlinearBlockGS():
        #     nlsolver.options['use_aitken'] = False

        # if nlsolver == om.NonlinearBlockGS():
        #     nlsolver.options['use_apply_nonlinear'] = True

        self.add_subsystem('obj_cmp', om.ExecComp('obj = eff_el_maxfcstack', shape=nn, obj={'units': None}, eff_el_maxfcstack={'units': None}), promotes_inputs=['eff_el_maxfcstack'], promotes_outputs=['obj'])
        # self.add_constraint('obj', upper=0.5)
        # self.add_objective('obj', scaler=-1)

        #self.add_subsystem('con_cmp1', om.ExecComp('maxcon1 = 112500 - pwr_el_del_per_maxfcsysmodule'), promotes_inputs=['pwr_el_del_per_maxfcsysmodule'], promotes_outputs=['maxcon1'])
        #self.add_subsystem('con_cmp1', om.ExecComp('maxcon1 = pwr_el_req_per_max_fcsysmodule - pwr_el_del_per_maxfcsysmodule'), promotes_inputs=['pwr_el_del_per_maxfcsysmodule', 'pwr_el_req_per_max_fcsysmodule'], promotes_outputs=['maxcon1'])
        #self.add_constraint('maxcon1', upper=0.0)



# prob = om.Problem()

# prob.model.add_subsystem('fcsysmodule', maxFCModuleGroup(num_nodes=1))

# prob.driver = om.ScipyOptimizeDriver()
# prob.driver.options['optimizer'] = 'SLSQP'
# prob.driver.options['maxiter'] = 100
# prob.driver.options['tol'] = 1e-16

# prob.model.add_design_var('fcsysmodule.current_maxfcstack', lower=5, upper=630)
# prob.model.add_objective('fcsysmodule.obj', scaler=1)

# # Ask OpenMDAO to finite-difference across the model to compute the gradients for the optimizer
# prob.model.approx_totals()

# prob.setup()
# # prob.model.list_inputs(units=True, shape=True)
# # prob.model.list_outputs(units=True, shape=True)
# # prob.check_partials(method='cs', compact_print=False, show_only_incorrect=True)
# prob.check_partials(method='cs', compact_print=True, show_only_incorrect=True)

# # prob.set_solver_print(level=2)

# prob.run_driver()
# #//XXX: Only required if current is fixed and an analysis must be done
# # prob.run_model()

# print('minimum found at')
# print(prob.get_val('fcsysmodule.current_maxfcstack'))

# print('Power delivered by FC System Module [W]')
# print(prob.get_val('fcsysmodule.pwr_el_del_per_maxfcsysmodule')[0])

# print('Power consumed by the air compressor [W]')
# print(prob.get_val('fcsysmodule.pwr_el_maxfcmodule_bop')[0])

# print('Power produced by FC Stack [W]')
# print(prob.get_val('fcsysmodule.pwr_el_maxfcstack')[0])

# print('Stack Air Flow Rate [kg/s]')
# print(prob.get_val('fcsysmodule.maxstack_airusage_rate')[0])

# print('Stack Elec Efficiency [%]')
# print(prob.get_val('fcsysmodule.eff_el_maxfcstack')[0])

# print('Fuel Cell System Module Electrical Efficiency [%]')
# print(prob.get_val('fcsysmodule.eff_maxfcsysmodule')[0])

# print('Stack Hydrogen Usage rate [kg/s]')
# print(prob.get_val('fcsysmodule.maxstack_hydrogenusage_rate')[0])