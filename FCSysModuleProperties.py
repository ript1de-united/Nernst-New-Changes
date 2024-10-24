import openmdao.api as om
import numpy as np


class FCSysModulePropertiesComp(om.ExplicitComponent):
    """
    Fuel cell system module mass model that calculates the mass of a single
    fuel cell system module containing a fuel cell stack, fuel cell stack
    components, humidifier, compressor, and cooling system.

    Inputs
    ------
    pwr_el_del_per_maxfcsysmodule : float
        Maximum electrical power delivered by a single fuel cell system module (vector, J/s).
    pwr_aircmprsr_maxfcstack : float
        Maximum electrical power required by the air compressor for a single fuel cell system module (vector, J/s).
    pwr_dens_fcstack : float
        Power density of the fuel cell stack (vector, kW/kg).

    Outputs
    -------
    mass_per_fcsysmodule : float
        Mass of a single fuel cell system module (vector, kg).

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
        #self.add_input('pwr_el_del_per_maxfcsysmodule', val=1.0*np.ones(nn))#, units='W'
        self.add_input('pwr_aircmprsr_maxfcstack', val=1.0*np.ones(nn))# , units='W'
        self.add_input('pwr_dens_fcstack', val=1.0*np.ones(nn))# , units='kW/kg'

        # New Input
        self.add_input('pwr_el_maxfcstack', val=1.0*np.ones(nn)) #added

        # Outputs
        self.add_output('mass_per_fcsysmodule', val=1.0*np.ones(nn))#, units='kg'

        # Partials
        ar = np.arange(nn)
        #self.declare_partials(of='mass_per_fcsysmodule', wrt='pwr_el_del_per_maxfcsysmodule', rows=ar, cols=ar)
        self.declare_partials(of='mass_per_fcsysmodule', wrt='pwr_aircmprsr_maxfcstack', rows=ar, cols=ar)
        self.declare_partials(of='mass_per_fcsysmodule', wrt='pwr_dens_fcstack', rows=ar, cols=ar)
        # New Partial
        self.declare_partials(of='mass_per_fcsysmodule', wrt='pwr_el_maxfcstack', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        #pwr_el_del_per_maxfcsysmodule = inputs['pwr_el_del_per_maxfcsysmodule']/1
        pwr_aircmprsr_maxfcstack = inputs['pwr_aircmprsr_maxfcstack']/1
        pwr_dens_fcstack = inputs['pwr_dens_fcstack']
        pwr_el_maxfcstack=inputs['pwr_el_maxfcstack']   #added

        n_actuators = 2 #-
        mass_per_actuator = 2 #kg
        n_valves = 1 #-
        mass_per_valve = 3 #kg

        pwr_spec_coolingmodule = 5.56 #[kW/kg]

        # Mass of Fuel Cell Stack
        #mass_fcstack = (pwr_el_del_per_maxfcsysmodule/1000) / pwr_dens_fcstack #[kW]/[kW/kg] 

        mass_fcstack = (pwr_el_maxfcstack/1000) / pwr_dens_fcstack #[kW]/[kW/kg] remember to change partial when running it  (added)

        # Mass of Fuel Cell Stack Valves, Actuators, and other Components
        mass_fcstackcomponents = (n_actuators * mass_per_actuator) + (n_valves * mass_per_valve)

        # Mass of Air Compressor
        mass_cmprsr = (0.0401 * (pwr_aircmprsr_maxfcstack/1000)) + 5.1724

        # Mass of Humidifier
        #mass_humdfr = (1.3669 * np.log(pwr_el_del_per_maxfcsysmodule/1000)) + 0.2644
        mass_humdfr = (1.3669 * np.log(pwr_el_maxfcstack/1000)) + 0.2644

        # Mass of Cooling System
        #mass_cool = (pwr_el_del_per_maxfcsysmodule/1000)/pwr_spec_coolingmodule #[kW]/[kW/kg]
        mass_cool = (pwr_el_maxfcstack/1000)/pwr_spec_coolingmodule #[kW]/[kW/kg]

        # Total Mass
        mass_per_fcsysmodule = mass_fcstack + mass_fcstackcomponents + mass_cmprsr + mass_humdfr + mass_cool
        # mass_per_fcsysmodule = (pwr_el_maxfcstack/1000) / pwr_dens_fcstack +                 
                                # (n_actuators * mass_per_actuator) + (n_valves * mass_per_valve) + 
                                # (0.0401 * (pwr_aircmprsr_maxfcstack/1000)) + 5.1724 +
                                # (1.3669 * np.log(pwr_el_maxfcstack/1000)) + 0.2644 + 
                                # (pwr_el_maxfcstack/1000)/pwr_spec_coolingmodule 

        outputs['mass_per_fcsysmodule'] = mass_per_fcsysmodule

    def compute_partials(self, inputs, partials):
        #pwr_el_del_per_maxfcsysmodule = inputs['pwr_el_del_per_maxfcsysmodule']/1
        pwr_dens_fcstack = inputs['pwr_dens_fcstack']
        pwr_el_maxfcstack = inputs['pwr_el_maxfcstack']
        
        n_actuators = 2 #-
        mass_per_actuator = 2 #kg
        n_valves = 1 #-
        mass_per_valve = 3 #kg

        pwr_spec_coolingmodule = 5.56 #[kW/kg]
        pwr_spec_atmrmodule = 1 #[kW/kg]

        #partials['mass_per_fcsysmodule', 'pwr_el_del_per_maxfcsysmodule'] = 1/(1000*pwr_dens_fcstack) + 1/(1000*pwr_spec_coolingmodule)  + (1.3669/(pwr_el_del_per_maxfcsysmodule/1))
        partials['mass_per_fcsysmodule', 'pwr_aircmprsr_maxfcstack'] = (0.0401/1000)
        #partials['mass_per_fcsysmodule', 'pwr_dens_fcstack'] = - 1 * (pwr_el_del_per_maxfcsysmodule/1000) / (pwr_dens_fcstack ** 2)
        
        partials['mass_per_fcsysmodule', 'pwr_dens_fcstack'] = - 1 * (pwr_el_maxfcstack/1000) / (pwr_dens_fcstack ** 2)
        partials['mass_per_fcsysmodule', 'pwr_el_maxfcstack'] = 1/(1000*pwr_dens_fcstack) + 1/(1000*pwr_spec_coolingmodule)  + (1.3669/(pwr_el_maxfcstack/1))

        #Verify partials
        #Write down what we decided to change and what is enabled and what is disabled or changed
                                                                                                                                
        
        