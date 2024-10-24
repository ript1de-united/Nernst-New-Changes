import openmdao.api as om
import numpy as np

def ConvertJtoWh(J):
    return J * (1//3600)

EFF_DCDCCONV = 1


class BatFCOperationComp(om.ExplicitComponent):
    """
    Component that calculated the power discharge/charge from/to the battery system,
    considering the fuel cell system and the DC-DC converter efficiency.

    Additionally, the maximum power to be delivered by the fuel cell system is
    calculated based on a ratio that determines how much of the peak power input
    at the motor is met by the fuel cell system.

    Global Variables
    ----------------
    EFF_DCDCCONV : float
        Efficiency of the DC-DC converter (dimensionless).

    Inputs
    ------
    pwr_el_in_motors : float
        Electrical power input at the motors (vector, J/s).
    pwr_el_max_in_motors : float
        Maximum electrical power input at the motors (vector, J/s).
    split_ratio : float
        Ratio determining the split of power demand between the fuel cell system
        and the battery system (vector, dimensionless).
    egy_batsys : float
        Energy available in the battery system (vector, J).
    C_rate : float
        Discharge/charge rate of the battery system (vector, 1/h).

    Outputs
    -------
    pwr_el_max_fcsys : float
        Maximum electrical power delivered from the fuel cell system (vector, J/s).
    pwr_el_inout_batsys : float
        Electrical power discharge/charge from/to the battery system (vector, J/s).

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
        self.add_input('pwr_el_max_in_motors', val=1.0*np.ones(nn))#, units='J/s'
        self.add_input('split_ratio', val=1.0*np.ones(nn))#, tags=['dymos.static_target']), units=None
        self.add_input('egy_batsys', val=1.0*np.ones(nn))#, tags=['dymos.static_target']), units='J'
        self.add_input('C_rate', val=1.0*np.ones(nn))#, units='1/h'

        # Outputs
        self.add_output('pwr_el_max_fcsys', val=1.0*np.ones(nn))#, units='J/s'
        self.add_output('pwr_el_inout_batsys', val=1.0*np.ones(nn))#, units='J/s'


        # Partials
        row_col = np.arange(nn)
        self.declare_partials(of='pwr_el_max_fcsys', wrt='pwr_el_max_in_motors', rows=row_col, cols=row_col)
        self.declare_partials(of='pwr_el_max_fcsys', wrt='split_ratio', rows=row_col, cols=row_col)
        self.declare_partials(of='pwr_el_max_fcsys', wrt='egy_batsys', rows=row_col, cols=row_col)
        self.declare_partials(of='pwr_el_max_fcsys', wrt='C_rate', rows=row_col, cols=row_col)

        self.declare_partials(of='pwr_el_inout_batsys', wrt='pwr_el_max_in_motors', rows=row_col, cols=row_col)
        self.declare_partials(of='pwr_el_inout_batsys', wrt='split_ratio', rows=row_col, cols=row_col)
        self.declare_partials(of='pwr_el_inout_batsys', wrt='egy_batsys', rows=row_col, cols=row_col)
        self.declare_partials(of='pwr_el_inout_batsys', wrt='C_rate', rows=row_col, cols=row_col)

    def compute(self, inputs, outputs):
        split_ratio = inputs['split_ratio']
        pwr_el_max_in_motors = inputs['pwr_el_max_in_motors']
        egy_batsys = inputs['egy_batsys']
        C_rate = inputs['C_rate']

        # Calculations
        outputs['pwr_el_max_fcsys'] =  split_ratio * pwr_el_max_in_motors

        outputs['pwr_el_inout_batsys'] = C_rate * egy_batsys * (1/3600)

    def compute_partials(self, inputs, partials):
        split_ratio = inputs['split_ratio']
        pwr_el_max_in_motors = inputs['pwr_el_max_in_motors']
        egy_batsys = inputs['egy_batsys']
        C_rate = inputs['C_rate']

        partials['pwr_el_max_fcsys', 'split_ratio'] = pwr_el_max_in_motors
        partials['pwr_el_max_fcsys', 'pwr_el_max_in_motors'] = split_ratio
        partials['pwr_el_max_fcsys', 'egy_batsys'] = 0
        partials['pwr_el_max_fcsys', 'C_rate'] = 0

        partials['pwr_el_inout_batsys', 'C_rate'] = egy_batsys * (1/3600)
        partials['pwr_el_inout_batsys', 'egy_batsys'] = C_rate * (1/3600)
        partials['pwr_el_inout_batsys', 'split_ratio'] = 0
        partials['pwr_el_inout_batsys', 'pwr_el_max_in_motors'] = 0