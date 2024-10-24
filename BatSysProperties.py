import numpy as np
import openmdao.api as om


class BatSysPropertiesComp(om.ExplicitComponent):
    """
    Battery system mass model that calculates the mass of the fuel cell
    system based on the number of fuel cell system modules and the mass of a single
    fuel cell system module.

    Inputs
    ------
    egy_batsys : float
        Energy available in the battery system (vector, J).
    egy_dens_batsys : float
        Energy density of the battery system (vector, kW/kg).

    Outputs
    -------
    mass_batsys : float
        Mass of battery system (vector, kg).

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
        self.add_input('egy_batsys', val=np.ones(nn))#, tags=['dymos.static_target']), units='J'
        self.add_input('egy_dens_batsys', val=np.ones(nn))#, units='kWh/kg'

        # Outputs
        self.add_output('mass_batsys', val=np.ones(nn))#, units='kg'

        # Partials
        ar = np.arange(nn)
        self.declare_partials(of='mass_batsys', wrt='egy_batsys', rows=ar, cols=ar)
        self.declare_partials(of='mass_batsys', wrt='egy_dens_batsys', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        egy_batsys = inputs['egy_batsys']
        egy_dens_batsys = inputs['egy_dens_batsys']

        outputs['mass_batsys'] = (egy_batsys * (1/3600) * (1/1000)) / egy_dens_batsys # kWh/[kWh/kg]

    def compute_partials(self, inputs, partials):
        egy_dens_batsys = inputs['egy_dens_batsys']
        egy_batsys = inputs['egy_batsys']

        partials['mass_batsys', 'egy_batsys'] = ((1/3600) * (1/1000)) / egy_dens_batsys
        partials['mass_batsys', 'egy_dens_batsys'] = -1 * (egy_batsys * (1/3600) * (1/1000)) / (egy_dens_batsys**2)