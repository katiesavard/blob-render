import os

CWD = os.getcwd()
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
HOME = os.path.expanduser('~')

TOOLS = os.path.join(BASE_DIR, 'tools')
CONTAINERS = os.path.join(BASE_DIR, 'containers')
SIM_DAT = os.path.join(BASE_DIR, 'sim_data')
TEL_INFO = os.path.join(BASE_DIR, 'telescope_info')
PLOTS = os.path.join(BASE_DIR, 'plots')
CONFIGS = os.path.join(BASE_DIR, 'configs')