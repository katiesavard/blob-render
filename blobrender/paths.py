import os
import yaml

CWD = os.getcwd()
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
HOME = os.path.expanduser('~')

TOOLS = os.path.join(BASE_DIR, 'tools')
CONTAINERS = os.path.join(BASE_DIR, 'containers')
TEL_INFO = os.path.join(BASE_DIR, 'telescope_info')
CONFIGS = os.path.join(BASE_DIR, 'configs')

# Data/output directories: default to living inside the repo, but can be
# pointed elsewhere (e.g. scratch storage on a cluster) by editing
# configs/paths.yaml, or with an env var for one-off/job-script overrides.
# Precedence: env var > paths.yaml > in-repo default.
_paths_yaml_file = os.path.join(CONFIGS, 'paths.yaml')
with open(_paths_yaml_file, 'r') as _f:
    _path_overrides = yaml.safe_load(_f) or {}

def _resolve_dir(env_var, yaml_key, default):
    return os.environ.get(env_var) or _path_overrides.get(yaml_key) or default

SIM_DAT = _resolve_dir('BLOBRENDER_SIM_DAT', 'sim_dat', os.path.join(BASE_DIR, 'sim_data'))
PLOTS = _resolve_dir('BLOBRENDER_PLOTS', 'plots', os.path.join(BASE_DIR, 'plots'))
RESULTS = _resolve_dir('BLOBRENDER_RESULTS', 'results', os.path.join(BASE_DIR, 'results'))

for _dir in (SIM_DAT, PLOTS, RESULTS):
    os.makedirs(_dir, exist_ok=True)