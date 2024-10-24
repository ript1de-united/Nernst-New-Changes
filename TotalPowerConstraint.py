import openmdao.api as om
import numpy as np

def ConvertJtoWh(J):
    return J * (1//3600)

EFF_DCDCCONV = 1


class TotalPowerConstraintComp(om.ExplicitComponent):
    """
    Constraint 'one' model that calculates the power delivery constraint to ensure
    that the power delivered by the fuel cell system, power consumed/delivered
    by the battery system, and the power input at the motors is balanced out
    to be zero.

    Global Variables
    ----------------
    EFF_DCDCCONV : float
        Efficiency of the DC-DC converter (dimensionless).

    Inputs
    ------
    pwr_el_del_fcsys : float
        Electrical power delivered by the fuel cell system (vector, J/s).
    pwr_el_in_motors : float
        Electrical power input at the motors (vector, J/s).
    pwr_el_inout_batsys : float
        Electrical power discharge/charge from/to the battery system (vector, J/s).

    Outputs
    -------
    con_fcbatsys_1 : float
        Constraint value representing the difference between the combined power of the
        fuel cell and battery systems and the power required by the motors (vector, J/s).

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
        self.add_input('pwr_el_del_fcsys', val=1.0*np.ones(nn)) #, units='J/s'
        self.add_input('pwr_el_in_motors', val=1.0*np.ones(nn))#, units='J/s'
        self.add_input('pwr_el_inout_batsys', val=1.0*np.ones(nn))#, tags=['dymos.static_target']), units=None

        # Outputs
        self.add_output('con_fcbatsys_1', val=1.0*np.ones(nn))#, units='J/s'

        # Partials
        row_col = np.arange(nn)

        self.declare_partials(of='con_fcbatsys_1', wrt='pwr_el_del_fcsys', rows=row_col, cols=row_col)
        self.declare_partials(of='con_fcbatsys_1', wrt='pwr_el_in_motors', rows=row_col, cols=row_col)
        self.declare_partials(of='con_fcbatsys_1', wrt='pwr_el_inout_batsys', rows=row_col, cols=row_col)

    def compute(self, inputs, outputs):
        pwr_el_del_fcsys = inputs['pwr_el_del_fcsys']
        pwr_el_in_motors = inputs['pwr_el_in_motors']
        pwr_el_inout_batsys = inputs['pwr_el_inout_batsys']

        # Calculations
        con_fcbatsys_1 =  pwr_el_in_motors - (pwr_el_del_fcsys + pwr_el_inout_batsys)

        outputs['con_fcbatsys_1'] = con_fcbatsys_1

    def compute_partials(self, inputs, partials):

        partials['con_fcbatsys_1', 'pwr_el_del_fcsys'] = -1
        partials['con_fcbatsys_1', 'pwr_el_in_motors'] = 1
        partials['con_fcbatsys_1', 'pwr_el_inout_batsys'] = -1

        