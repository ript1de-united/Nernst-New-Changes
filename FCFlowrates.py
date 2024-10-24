import math
import openmdao.api as om
import numpy as np

class FCFlowratesComp(om.ExplicitComponent):
    """
    This component calculates the rates of change for the mass of fuel, oxidant, and water
    in the fuel cell system.

    Inputs
    ------
    nminus1_active_fcsysmodules : float
        Number of active n-1 fuel cell system modules (vector, dimensionless).
    nminus1stack_hydrogenusage_rate : float
        Hydrogen usage rate of the n-1 fuel cell stack (vector, kg/s).
    nthstack_hydrogenusage_rate : float
        Hydrogen usage rate of the nth fuel cell stack (vector, kg/s).
    nminus1stack_airusage_rate : float
        Air usage rate of the n-1 fuel cell stack (vector, kg/s).
    nthstack_airusage_rate : float
        Air usage rate of the nth fuel cell stack (vector, kg/s).
    nminus1stack_waterprodn_rate : float
        Water production rate of the n-1 fuel cell stack (vector, kg/s).
    nthstack_waterprodn_rate : float
        Water production rate of the nth fuel cell stack (vector, kg/s).

    Outputs
    -------
    dXdt:mass_fuel : float
        Rate of change of mass of fuel (vector, kg/s).
    dXdt:mass_oxidant : float
        Rate of change of mass of oxidant (vector, kg/s).
    dXdt:mass_water : float
        Rate of change of mass of water (vector, kg/s).

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
        self.add_input('nminus1_active_fcsysmodules', val=1.0*np.ones(nn))#, units=None
        self.add_input('nminus1stack_hydrogenusage_rate', val=1.0*np.ones(nn))#, units='kg/s'
        self.add_input('nthstack_hydrogenusage_rate', val=1.0*np.ones(nn))#, units='kg/s'
        self.add_input('nminus1stack_airusage_rate', val=1.0*np.ones(nn))#, units='kg/s'
        self.add_input('nthstack_airusage_rate', val=1.0*np.ones(nn))#, units='kg/s'
        self.add_input('nminus1stack_waterprodn_rate', val=1.0*np.ones(nn))#, units='kg/s'
        self.add_input('nthstack_waterprodn_rate', val=1.0*np.ones(nn))#, units='kg/s'

        # Outputs
        self.add_output('dXdt:mass_fuel', val=1.0*np.ones(nn))#, units='kg/s'
        self.add_output('dXdt:mass_oxidant', val=1.0*np.ones(nn))#, units='kg/s'
        self.add_output('dXdt:mass_water', val=1.0*np.ones(nn))#, units='kg/s'

        # Partials
        ar = np.arange(self.options['num_nodes'])
        self.declare_partials(of='dXdt:mass_fuel', wrt='nminus1_active_fcsysmodules', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:mass_fuel', wrt='nminus1stack_hydrogenusage_rate', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:mass_fuel', wrt='nthstack_hydrogenusage_rate', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:mass_fuel', wrt='nminus1stack_airusage_rate', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:mass_fuel', wrt='nthstack_airusage_rate', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:mass_fuel', wrt='nminus1stack_waterprodn_rate', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:mass_fuel', wrt='nthstack_waterprodn_rate', rows=ar, cols=ar)

        self.declare_partials(of='dXdt:mass_oxidant', wrt='nminus1_active_fcsysmodules', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:mass_oxidant', wrt='nminus1stack_hydrogenusage_rate', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:mass_oxidant', wrt='nthstack_hydrogenusage_rate', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:mass_oxidant', wrt='nminus1stack_airusage_rate', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:mass_oxidant', wrt='nthstack_airusage_rate', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:mass_oxidant', wrt='nminus1stack_waterprodn_rate', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:mass_oxidant', wrt='nthstack_waterprodn_rate', rows=ar, cols=ar)

        self.declare_partials(of='dXdt:mass_water', wrt='nminus1_active_fcsysmodules', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:mass_water', wrt='nminus1stack_hydrogenusage_rate', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:mass_water', wrt='nthstack_hydrogenusage_rate', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:mass_water', wrt='nminus1stack_airusage_rate', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:mass_water', wrt='nthstack_airusage_rate', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:mass_water', wrt='nminus1stack_waterprodn_rate', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:mass_water', wrt='nthstack_waterprodn_rate', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        nminus1_active_fcsysmodules = inputs['nminus1_active_fcsysmodules']
        nminus1stack_hydrogenusage_rate = inputs['nminus1stack_hydrogenusage_rate']
        nthstack_hydrogenusage_rate = inputs['nthstack_hydrogenusage_rate']
        nminus1stack_airusage_rate = inputs['nminus1stack_airusage_rate']
        nthstack_airusage_rate = inputs['nthstack_airusage_rate']
        nminus1stack_waterprodn_rate = inputs['nminus1stack_waterprodn_rate']
        nthstack_waterprodn_rate = inputs['nthstack_waterprodn_rate']

        nth_active_fcsysmodule = 1

        outputs['dXdt:mass_fuel'] = rate_total_hydrogen_in_ = -1 * ((nminus1stack_hydrogenusage_rate * nminus1_active_fcsysmodules) + (nthstack_hydrogenusage_rate * nth_active_fcsysmodule))

        outputs['dXdt:mass_oxidant'] = rate_total_air_in_ = -1 * ((nminus1stack_airusage_rate * nminus1_active_fcsysmodules) + (nthstack_airusage_rate * nth_active_fcsysmodule))

        outputs['dXdt:mass_water'] = rate_total_fcsyswater_out_ = (nminus1stack_waterprodn_rate * nminus1_active_fcsysmodules) + (nthstack_waterprodn_rate * nth_active_fcsysmodule)

    def compute_partials(self, inputs, partials):
        nminus1_active_fcsysmodules = inputs['nminus1_active_fcsysmodules']
        nminus1stack_hydrogenusage_rate = inputs['nminus1stack_hydrogenusage_rate']
        nthstack_hydrogenusage_rate = inputs['nthstack_hydrogenusage_rate']
        nminus1stack_airusage_rate = inputs['nminus1stack_airusage_rate']
        nthstack_airusage_rate = inputs['nthstack_airusage_rate']
        nminus1stack_waterprodn_rate = inputs['nminus1stack_waterprodn_rate']
        nthstack_waterprodn_rate = inputs['nthstack_waterprodn_rate']

        nth_active_fcsysmodule = 1

        partials['dXdt:mass_fuel', 'nminus1_active_fcsysmodules'] = -1 * nminus1stack_hydrogenusage_rate
        partials['dXdt:mass_fuel', 'nminus1stack_hydrogenusage_rate'] = -1 * nminus1_active_fcsysmodules
        partials['dXdt:mass_fuel', 'nthstack_hydrogenusage_rate'] = -1 * nth_active_fcsysmodule
        partials['dXdt:mass_fuel', 'nminus1stack_airusage_rate'] = 0
        partials['dXdt:mass_fuel', 'nthstack_airusage_rate'] = 0
        partials['dXdt:mass_fuel', 'nminus1stack_waterprodn_rate'] = 0
        partials['dXdt:mass_fuel', 'nthstack_waterprodn_rate'] = 0

        partials['dXdt:mass_oxidant', 'nminus1_active_fcsysmodules'] = -1 * nminus1stack_airusage_rate
        partials['dXdt:mass_oxidant', 'nminus1stack_hydrogenusage_rate'] = 0
        partials['dXdt:mass_oxidant', 'nthstack_hydrogenusage_rate'] = 0
        partials['dXdt:mass_oxidant', 'nminus1stack_airusage_rate'] = -1 * nminus1_active_fcsysmodules
        partials['dXdt:mass_oxidant', 'nthstack_airusage_rate'] = -1 * nth_active_fcsysmodule
        partials['dXdt:mass_oxidant', 'nminus1stack_waterprodn_rate'] = 0
        partials['dXdt:mass_oxidant', 'nthstack_waterprodn_rate'] = 0

        partials['dXdt:mass_water', 'nminus1_active_fcsysmodules'] = nminus1stack_waterprodn_rate
        partials['dXdt:mass_water', 'nminus1stack_hydrogenusage_rate'] = 0
        partials['dXdt:mass_water', 'nthstack_hydrogenusage_rate'] = 0
        partials['dXdt:mass_water', 'nminus1stack_airusage_rate'] = 0
        partials['dXdt:mass_water', 'nthstack_airusage_rate'] = 0
        partials['dXdt:mass_water', 'nminus1stack_waterprodn_rate'] = nminus1_active_fcsysmodules
        partials['dXdt:mass_water', 'nthstack_waterprodn_rate'] = nth_active_fcsysmodule