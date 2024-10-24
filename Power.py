import numpy as np
import openmdao.api as om
from scipy.interpolate import interp1d

# Import powerprofile in J/s
pwroutgearbox = values = np.loadtxt("power-values.csv", delimiter=",", dtype=float)
# Import time in s corresponding to the power from the power profile
time_series = points = np.loadtxt("time-points.csv", delimiter=",", dtype=float)

PWR_MAX_GEARBOX = np.max(pwroutgearbox)


class PowerComp(om.ExplicitComponent):
    """
    Model that takes the time input from Dymos, and matches the time step
    with the respective power from the interpolated power profile.

    Global Variables
    ----------------
    PWR_MAX_GEARBOX : float
        Maximum power input at the gearbox from the power profile (scalar, J/s).

    Inputs
    ------
    time_phase : float
        Current time for which power needs to be interpolated (vector, s).

    Outputs
    -------
    pwr_in_gearbox : float
        Power input at the gearbox using linear interpolation (vector, J/s).
    pwr_max_gearbox : float
        Maximum power input at the gearbox (scalar, J/s).

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless).
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']

        # Inputs
        self.add_input('time_phase', val=np.ones(num_nodes), desc='Current time for which power needs to be interpolated')#, units='s'

        # Outputs
        self.add_output('pwr_in_gearbox', val=np.ones(num_nodes), desc='Power required at the gearbox input')#, units='J/s'
        self.add_output('pwr_max_gearbox', val=np.ones(num_nodes))#, units='J/s'

        # Derivatives
        row_col = np.arange(num_nodes)

        self.declare_partials(of='pwr_in_gearbox', wrt=['time_phase'], rows=row_col, cols=row_col)
        self.declare_partials(of='pwr_max_gearbox', wrt=['time_phase'], rows=row_col, cols=row_col)

        # Create interpolated data
        self.linear_interp = interp1d(points, values, kind='linear')

    def compute(self, inputs, outputs):
        time_phase = inputs['time_phase']

        outputs['pwr_in_gearbox'] = self.linear_interp(time_phase)

        outputs['pwr_max_gearbox'] = PWR_MAX_GEARBOX

    def compute_partials(self, inputs, partials):

        partials['pwr_max_gearbox', 'time_phase'] = 0