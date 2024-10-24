import math
import openmdao.api as om
import numpy as np


class FCSysOperationComp(om.ExplicitComponent):
    """
    This component calculates the total electrical power delivered by the fuel cell system.

    Inputs
    ------
    pwr_el_del_per_nminus1fcsysmodule : float
        Electrical power delivered per n-1 fuel cell system module (vector, J/s).
    nminus1_active_fcsysmodules : float
        Number of active n-1 fuel cell system modules (vector, dimensionless).
    pwr_el_del_per_nthfcsysmodule : float
        Electrical power delivered per nth fuel cell system module (vector, J/s).

    Outputs
    -------
    pwr_el_del_fcsys : float
        Total electrical power delivered by the fuel cell system (vector, J/s).

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
        self.add_input('pwr_el_del_per_nminus1fcsysmodule', val=1.0*np.ones(nn))#, units='J/s'
        self.add_input('nminus1_active_fcsysmodules', val=1.0*np.ones(nn))#, units=None
        self.add_input('pwr_el_del_per_nthfcsysmodule', val=1.0*np.ones(nn))#, units='J/s'

        #Outputs
        self.add_output('pwr_el_del_fcsys', val=1.0*np.ones(nn))#, units='J/s'

        # Partials
        ar = np.arange(self.options['num_nodes'])
        self.declare_partials(of='pwr_el_del_fcsys', wrt='pwr_el_del_per_nminus1fcsysmodule', rows=ar, cols=ar)
        self.declare_partials(of='pwr_el_del_fcsys', wrt='nminus1_active_fcsysmodules', rows=ar, cols=ar)
        self.declare_partials(of='pwr_el_del_fcsys', wrt='pwr_el_del_per_nthfcsysmodule', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        pwr_el_del_per_nminus1fcsysmodule = inputs['pwr_el_del_per_nminus1fcsysmodule']
        nminus1_active_fcsysmodules = inputs['nminus1_active_fcsysmodules']
        pwr_el_del_per_nthfcsysmodule = inputs['pwr_el_del_per_nthfcsysmodule']

        nth_active_fcsysmodule = 1

        outputs['pwr_el_del_fcsys'] = (pwr_el_del_per_nminus1fcsysmodule * nminus1_active_fcsysmodules) + (pwr_el_del_per_nthfcsysmodule * nth_active_fcsysmodule)

    def compute_partials(self, inputs, partials):
        nth_active_fcsysmodule = 1

        partials['pwr_el_del_fcsys', 'pwr_el_del_per_nminus1fcsysmodule'] = inputs['nminus1_active_fcsysmodules']
        partials['pwr_el_del_fcsys', 'nminus1_active_fcsysmodules'] = inputs['pwr_el_del_per_nminus1fcsysmodule']
        partials['pwr_el_del_fcsys', 'pwr_el_del_per_nthfcsysmodule'] = nth_active_fcsysmodule