import numpy as np
import openmdao.api as om

from nthFCModule import nthFCModuleGroup
from nminus1FCModule import nminus1FCModuleGroup
from FCSysOperation import FCSysOperationComp
from FCFlowrates import FCFlowratesComp


class FCSysGroup(om.Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem('nth_fcsysmodulegroup', nthFCModuleGroup(num_nodes=nn), promotes_inputs=['current_nthfcstack'], promotes_outputs=['pwr_el_del_per_nthfcsysmodule', 'nthstack_airusage_rate', 'nthstack_hydrogenusage_rate', 'nthstack_waterprodn_rate'])

        self.add_subsystem('nminus1_fcsysmodulegroup', nminus1FCModuleGroup(num_nodes=nn), promotes_inputs=['current_nminus1fcstack'], promotes_outputs=['pwr_el_del_per_nminus1fcsysmodule', 'nminus1stack_airusage_rate', 'nminus1stack_hydrogenusage_rate', 'nminus1stack_waterprodn_rate'])

        self.add_subsystem('fuel_cell_system_operation', FCSysOperationComp(num_nodes=nn), promotes_inputs=['nminus1_active_fcsysmodules', 'pwr_el_del_per_nthfcsysmodule', 'pwr_el_del_per_nminus1fcsysmodule'], promotes_outputs=['pwr_el_del_fcsys'])

        self.add_subsystem('fuel_cell_system_flowrates', FCFlowratesComp(num_nodes=nn), promotes_inputs=['nminus1_active_fcsysmodules', 'nthstack_airusage_rate', 'nthstack_hydrogenusage_rate', 'nthstack_waterprodn_rate', 'nminus1stack_airusage_rate', 'nminus1stack_hydrogenusage_rate', 'nminus1stack_waterprodn_rate'], promotes_outputs=['dXdt:mass_fuel'])