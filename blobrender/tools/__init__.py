_EXPORTS = {
    "lum_unit_si": ".pluto_luminosity_conversion",
    "gamma_to_beta": ".sim_analysis_tools",
    "theta_from_beta": ".sim_analysis_tools",
    "cyl_to_cart": ".sim_analysis_tools",
    "load_data_obj": ".sim_analysis_tools",
    "doppler_boost_lum": ".sim_analysis_tools",
    "interpolate_cyl_to_cart": ".sim_analysis_tools",
    "m_to_arcseconds": ".sim_analysis_tools",
    "angle_to_boost": ".sim_analysis_tools",
    "beta_to_gamma": ".sim_analysis_tools",
    "rgb2gray": ".sim_analysis_tools",
    "plot_basic": ".plotting",
    "plot_radio": ".plotting",
    "save_fig": ".plotting",
    "save_list": ".basics",
    "load_list": ".basics",
    "loader_bar": ".basics",
    "get_arguments": ".basics",
    "update_yaml": ".basics",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module = import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
