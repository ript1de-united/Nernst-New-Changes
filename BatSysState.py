import numpy as np
import openmdao.api as om


class BatSysStateComp(om.ExplicitComponent):
    """
    Battery state model computes how much energy is lost or gained by the
    battery system in terms of values 0 to 1, where 0 is fully discharged
    and 1 is fully charged.

    Inputs
    ------
    pwr_el_inout_batsys : float
        Electrical power discharge/charge from/to the battery system (vector, J/s).
    egy_batsys : float
        Energy available in the battery system (vector, J).

    Outputs
    -------
    dXdt:SoC : float
        Rate of change of State of Charge of the battery system (vector, 1/s).

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
        self.add_input('pwr_el_inout_batsys', val=1.0*np.ones(nn))#, units='J/s')
        self.add_input('egy_batsys', val=1.0*np.ones(nn))#, tags=['dymos.static_target']), units='J'

        # Outputs
        self.add_output('dXdt:SoC', val=1.0*np.ones(nn))#, units='1/s')

        # Partials
        row_col = np.arange(nn)
        self.declare_partials(of='dXdt:SoC', wrt='pwr_el_inout_batsys', rows=row_col, cols=row_col)
        self.declare_partials(of='dXdt:SoC', wrt='egy_batsys', rows=row_col, cols=row_col)

    def compute(self, inputs, outputs):
        pwr_el_inout_batsys = inputs['pwr_el_inout_batsys']
        egy_batsys = inputs['egy_batsys']

        outputs['dXdt:SoC'] = - pwr_el_inout_batsys / egy_batsys

    def compute_partials(self, inputs, partials):
        pwr_el_inout_batsys = inputs['pwr_el_inout_batsys']
        egy_batsys = inputs['egy_batsys']

        partials['dXdt:SoC', 'pwr_el_inout_batsys'] = -1 / egy_batsys
        partials['dXdt:SoC', 'egy_batsys'] = pwr_el_inout_batsys / egy_batsys**2