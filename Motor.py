import numpy as np

import openmdao.api as om

EFF_MOTOR = 1.0
EFF_DCACCONV = 1.0

class MotorComp(om.ExplicitComponent):
    """
    Motor model that calculates the electrical power input to the motors based on the
    power input to the gearbox, considering motor and DC-AC converter efficiencies.

    Global Variables
    ----------------
    EFF_MOTOR : float
        Efficiency of the motor (dimensionless).
    EFF_DCACCONV : float
        Efficiency of the DC-AC converter (dimensionless).

    Inputs
    ------
    pwr_in_gearbox : float
        Power input at the gearbox (vector, J/s).
    pwr_max_gearbox : float
        Maximum power input at the gearbox (vector, J/s).

    Outputs
    -------
    pwr_el_in_motors : float
        Electrical power input at the motors (vector, J/s).
    pwr_el_max_in_motors : float
        Maximum electrical power input at the motors (vector, J/s).

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
        self.add_input('pwr_in_gearbox', val=1.0*np.ones(nn))#, units='J/s'
        self.add_input('pwr_max_gearbox', val=1.0*np.ones(nn))#, units='J/s'


        # Outputs
        self.add_output('pwr_el_in_motors', val=1.0*np.ones(nn))#, units='J/s')
        self.add_output('pwr_el_max_in_motors', val=1.0*np.ones(nn))#, units='J/s')

        # Partials
        row_col = np.arange(nn)
        self.declare_partials(of='pwr_el_in_motors', wrt='pwr_in_gearbox', rows=row_col, cols=row_col)
        self.declare_partials(of='pwr_el_max_in_motors', wrt='pwr_max_gearbox', rows=row_col, cols=row_col)

    def compute(self, inputs, outputs):
        pwr_in_gearbox = inputs['pwr_in_gearbox']
        pwr_max_gearbox = inputs['pwr_max_gearbox']

        outputs['pwr_el_in_motors'] = pwr_in_gearbox / (EFF_MOTOR * EFF_DCACCONV)

        outputs['pwr_el_max_in_motors'] = pwr_max_gearbox / (EFF_MOTOR * EFF_DCACCONV)

    def compute_partials(self, inputs, partials):

        partials['pwr_el_in_motors', 'pwr_in_gearbox'] = 1.0 / (EFF_MOTOR * EFF_DCACCONV)

        partials['pwr_el_max_in_motors', 'pwr_max_gearbox'] = 1.0 / (EFF_MOTOR * EFF_DCACCONV)