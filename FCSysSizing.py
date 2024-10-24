import math
import openmdao.api as om
import numpy as np


class FCSysSizingComp(om.ExplicitComponent):
    """
    Fuel cell system sizing model that calculates the number of fuel cell system modules
    required based on maximum electrical power deliverable by a single fuel cell system
    module and the maximum electrical power deliverable by the fuel cell system.

    Inputs
    ------
    pwr_el_del_per_maxfcsysmodule : float
        Maximum electrical power delivered by a single fuel cell system module (vector, J/s).
    pwr_el_max_fcsys : float
        Maximum electrical power delivered from the fuel cell system (vector, J/s).

    Outputs
    -------
    n_fcsysmodules : float
        Number of fuel cell system modules required (vector, dimensionless).

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
        self.add_input('pwr_el_del_per_maxfcsysmodule', val=1.0*np.ones(nn))#, units='J/s'
        self.add_input('pwr_el_max_fcsys', val=1.0*np.ones(nn))#, units='J/s'

        #Outputs
        self.add_output('n_fcsysmodules', val=1.0*np.ones(nn))#, units=None

        # Partials
        ar = np.arange(nn)
        self.declare_partials(of='n_fcsysmodules', wrt='pwr_el_del_per_maxfcsysmodule', rows=ar, cols=ar)
        self.declare_partials(of='n_fcsysmodules', wrt='pwr_el_max_fcsys', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        pwr_el_max_fcsys = inputs['pwr_el_max_fcsys']
        pwr_el_del_per_maxfcsysmodule = inputs['pwr_el_del_per_maxfcsysmodule']

        outputs['n_fcsysmodules'] = n_fcsysmodules = (pwr_el_max_fcsys / pwr_el_del_per_maxfcsysmodule)

    def compute_partials(self, inputs, partials):
        pwr_el_max_fcsys = inputs['pwr_el_max_fcsys']
        pwr_el_del_per_maxfcsysmodule = inputs['pwr_el_del_per_maxfcsysmodule']

        partials['n_fcsysmodules', 'pwr_el_del_per_maxfcsysmodule'] = - (pwr_el_max_fcsys/(pwr_el_del_per_maxfcsysmodule ** 2))
        partials['n_fcsysmodules', 'pwr_el_max_fcsys'] = (1/pwr_el_del_per_maxfcsysmodule)
