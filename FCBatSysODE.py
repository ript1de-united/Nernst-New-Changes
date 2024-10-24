import openmdao.api as om

# Battery
from BatSysState import BatSysStateComp
from BatSysProperties import BatSysPropertiesComp

# Imports and Power Splitting
from Power import PowerComp
from Motor import MotorComp
from BatFCOperation import BatFCOperationComp
from TotalPowerConstraint import TotalPowerConstraintComp

# Fuel Cells
from FCSys import FCSysGroup
from maxFCModule import maxFCModuleGroup
from FCSysSizing import FCSysSizingComp
from FCSysModuleProperties import FCSysModulePropertiesComp
from FCSysProperties import FCSysPropertiesComp

from TakeOffMass import TakeOffMassComp


class FCBatSysODEGroup(om.Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem(name='gearbox_power', subsys=PowerComp(num_nodes=nn),
                           promotes_inputs=['time_phase'], promotes_outputs=['pwr_in_gearbox', 'pwr_max_gearbox'])

        self.add_subsystem(name='motor_power', subsys=MotorComp(num_nodes=nn),
                           promotes_inputs=['pwr_in_gearbox', 'pwr_max_gearbox'],
                           promotes_outputs=['pwr_el_in_motors', 'pwr_el_max_in_motors'])

        self.add_subsystem(name='battery_fuel_cell_system_operation_and_limit', subsys=BatFCOperationComp(num_nodes=nn),
                           promotes_inputs=['pwr_el_max_in_motors', 'split_ratio', 'egy_batsys', 'C_rate'],
                           promotes_outputs=['pwr_el_inout_batsys', 'pwr_el_max_fcsys'])

        self.add_subsystem(name='fuel_cell_system', subsys=FCSysGroup(num_nodes=nn),
                           promotes_inputs=['current_nthfcstack', 'current_nminus1fcstack', 'nminus1_active_fcsysmodules'],
                           promotes_outputs=['pwr_el_del_per_nthfcsysmodule',
                                             'nthstack_airusage_rate', 'nthstack_hydrogenusage_rate', 'nthstack_waterprodn_rate',
                                             'pwr_el_del_per_nminus1fcsysmodule',
                                             'nminus1stack_airusage_rate', 'nminus1stack_hydrogenusage_rate', 'nminus1stack_waterprodn_rate',
                                             'pwr_el_del_fcsys',
                                             'dXdt:mass_fuel'])

        self.add_subsystem(name='power_delivery_constraint', subsys=TotalPowerConstraintComp(num_nodes=nn),
                           promotes_inputs=['pwr_el_del_fcsys', 'pwr_el_in_motors', 'pwr_el_inout_batsys'],
                           promotes_outputs=['con_fcbatsys_1'])
        self.add_constraint('con_fcbatsys_1', upper=100, lower=0, scaler=1e-8)

        self.add_subsystem('con_fcsys2', om.ExecComp('con_fcsys_2 = pwr_el_del_fcsys - pwr_el_max_fcsys', shape=nn,
                                                     con_fcsys_2={'units': None}, pwr_el_max_fcsys={'units': None}, pwr_el_del_fcsys={'units': None}),
                                                     promotes_inputs=['pwr_el_max_fcsys', 'pwr_el_del_fcsys'], promotes_outputs=['con_fcsys_2'])
        self.add_constraint('con_fcsys_2', upper=0, scaler=1e-7)

        self.add_subsystem(name='battery_system_state', subsys=BatSysStateComp(num_nodes=nn),
                           promotes_inputs=['egy_batsys', 'pwr_el_inout_batsys'], promotes_outputs=['dXdt:SoC'])

        self.add_subsystem('max_fcsysmodulegroup', maxFCModuleGroup(num_nodes=nn),
                           promotes_inputs=['current_maxfcstack'], promotes_outputs=['pwr_el_del_per_maxfcsysmodule', 'pwr_aircmprsr_maxfcstack','pwr_el_maxfcstack']) #add pwr_del_max later as output

        self.add_subsystem('fuel_cell_system_sizing', FCSysSizingComp(num_nodes=nn),
                           promotes_inputs=['pwr_el_del_per_maxfcsysmodule', 'pwr_el_max_fcsys'], promotes_outputs=['n_fcsysmodules'])

        self.add_subsystem('con_fcsys3', om.ExecComp('con_fcsys_3 = (nminus1_active_fcsysmodules + 1) - (n_fcsysmodules)', shape=nn,
                                                     con_fcsys_3={'units': None}, n_fcsysmodules={'units': None}, nminus1_active_fcsysmodules={'units': None}),
                                                     promotes_inputs=['n_fcsysmodules', 'nminus1_active_fcsysmodules'], promotes_outputs=['con_fcsys_3'])
        self.add_constraint('con_fcsys_3', upper=0,scaler=1e-3)

        self.add_subsystem('fuel_cell_system_module_properties', FCSysModulePropertiesComp(num_nodes=nn),
                           promotes_inputs=[ 'pwr_aircmprsr_maxfcstack', 'pwr_dens_fcstack','pwr_el_maxfcstack'], #changes made here 
                           promotes_outputs=['mass_per_fcsysmodule'])

        self.add_subsystem('fuel_cell_system_properties', FCSysPropertiesComp(num_nodes=nn),
                           promotes_inputs=['n_fcsysmodules', 'mass_per_fcsysmodule'], promotes_outputs=['mass_fcsys'])

        self.add_subsystem(name='battery_system_properties', subsys=BatSysPropertiesComp(num_nodes=nn),
                           promotes_inputs=['egy_batsys', 'egy_dens_batsys'], promotes_outputs=['mass_batsys'])

        self.add_subsystem('powertrain_takeoff_mass_power', TakeOffMassComp(num_nodes=nn),
                           promotes_inputs=['mass_fcsys', 'mass_fuel', 'grav_eff_h2', 'mass_batsys'], promotes_outputs=['tot_takeoff_mass'])


        # //XXX: No solver required. May not even a linear solver. LNBGS may be enough. Non-Linear Newton solver might just do as well.
        iterations = 50

        # nlsolver = self.nonlinear_solver = om.NonlinearBlockGS()
        # nlsolver = self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        # nlsolver = self.nonlinear_so/lver = om.NewtonSolver(solve_subsystems=True)

        # lsolver = self.linear_solver = om.LinearBlockGS()
        # lsolver = self.linear_solver = om.ScipyKrylov()
        lsolver = self.linear_solver = om.DirectSolver()
        # lsolver.options['maxiter'] = iterations
        lsolver.options['iprint'] = -1

        # nlsolver.options['maxiter'] = iterations
        # nlsolver.options['iprint'] = -1
        # # nlsolver.options['rtol'] = 1e-16
        # # nlsolver.options['atol'] = 1e-16
        # nlsolver.options['err_on_non_converge'] = True
        # nlsolver.options['debug_print'] = True
        # if nlsolver == om.NonlinearBlockGS():
        #     nlsolver.options['use_aitken'] = True