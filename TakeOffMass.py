import openmdao.api as om
import numpy as np

class TakeOffMassComp(om.ExplicitComponent):
    """
    Powetrain mass model that calculates the mass of the powertrain system
    at every instant of time. Generally named as take-off mass.

    Inputs
    ------
    mass_fcsys : float
        Mass of fuel cell system (vector, kg).
    mass_fuel : float
        Mass of the hydrogen fuel (vector, kg).
    grav_eff_h2 : float
        Gravimetric efficiency of the hydrogen storage tank (vector, dimensionless).
    mass_batsys : float
        Mass of battery system (vector, kg).

    Outputs
    -------
    tot_takeoff_mass : float
        Total takeoff mass of the powertrain system (vector, kg).

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
        self.add_input('mass_fcsys', val=1.0*np.ones(nn))
        self.add_input('mass_fuel', val=1.0*np.ones(nn))
        self.add_input('grav_eff_h2', val=1.0*np.ones(nn))
        self.add_input('mass_batsys', val=1.0*np.ones(nn))

        # Outputs
        self.add_output('tot_takeoff_mass', val=1.0*np.ones(nn))

        # Partials
        ar = np.arange(nn)
        self.declare_partials(of='tot_takeoff_mass', wrt='mass_fcsys', rows=ar, cols=ar)
        self.declare_partials(of='tot_takeoff_mass', wrt='mass_fuel', rows=ar, cols=ar)
        self.declare_partials(of='tot_takeoff_mass', wrt='grav_eff_h2', rows=ar, cols=ar)
        self.declare_partials(of='tot_takeoff_mass', wrt='mass_batsys', rows=ar, cols=ar)


    def compute(self, inputs, outputs):
        mass_fcsys = inputs['mass_fcsys']
        mass_batsys = inputs['mass_batsys']
        mass_fuel = inputs['mass_fuel']
        grav_eff_h2 = inputs['grav_eff_h2']

        mass_tank = (mass_fuel * ((1 - grav_eff_h2)/grav_eff_h2))

        tot_takeoff_mass = mass_fcsys + mass_fuel + mass_tank + mass_batsys

        outputs['tot_takeoff_mass'] = tot_takeoff_mass

    def compute_partials(self, inputs, partials):
        mass_fcsys = inputs['mass_fcsys']
        mass_batsys = inputs['mass_batsys']
        mass_fuel = inputs['mass_fuel']
        grav_eff_h2 = inputs['grav_eff_h2']

        partials['tot_takeoff_mass', 'mass_fcsys'] = 1
        partials['tot_takeoff_mass', 'mass_fuel'] = 1 + 1/grav_eff_h2 - 1
        partials['tot_takeoff_mass', 'grav_eff_h2'] = - mass_fuel / (grav_eff_h2**2)
        partials['tot_takeoff_mass', 'mass_batsys'] = 1
