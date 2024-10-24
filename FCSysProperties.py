import math
import openmdao.api as om
import numpy as np


class FCSysPropertiesComp(om.ExplicitComponent):
    """
    Fuel cell system mass model that calculates the mass of the fuel cell
    system based on the number of fuel cell system modules and the mass of a single
    fuel cell system module.

    Inputs
    ------
    mass_per_fcsysmodule : float
        Mass of a single fuel cell system module (vector, kg).
    n_fcsysmodules : float
        Number of fuel cell system modules required (vector, dimensionless).

    Outputs
    -------
    mass_fcsys : float
        Mass of fuel cell system (vector, kg).

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
        self.add_input('mass_per_fcsysmodule', val=400000.0*np.ones(nn))#, units='kg'
        self.add_input('n_fcsysmodules', val=1.0*np.ones(nn))#, units=None

        # Outputs
        self.add_output('mass_fcsys', val=1.0*np.ones(nn))#, units='kg'

        # Partials
        ar = np.arange(nn)
        self.declare_partials(of='mass_fcsys', wrt='mass_per_fcsysmodule', rows=ar, cols=ar)
        self.declare_partials(of='mass_fcsys', wrt='n_fcsysmodules', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        mass_per_fcsysmodule = inputs['mass_per_fcsysmodule']
        n_fcsysmodules = inputs['n_fcsysmodules']

        outputs['mass_fcsys'] = mass_per_fcsysmodule * n_fcsysmodules

    def compute_partials(self, inputs, partials):
        mass_per_fcsysmodule = inputs['mass_per_fcsysmodule']
        n_fcsysmodules = inputs['n_fcsysmodules']

        partials['mass_fcsys', 'mass_per_fcsysmodule'] = n_fcsysmodules
        partials['mass_fcsys', 'n_fcsysmodules'] = mass_per_fcsysmodule