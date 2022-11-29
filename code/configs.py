configs = dict()

configs["beta_gamma"] = {
    "betas": [0.55, 0.5],
    "gammas": [0.5, 0.55],
    "beta_sharpness": None,
    "phase": None,
    "pac": None,
    "cross_pac": None,
    "phase_shift": None,
    "burst_length": None,
}

configs["beta_sharpness"] = {
    "betas": None,
    "gammas": None,
    "beta_sharpness": [0.1, 0.9],
    "phase": None,
    "pac": None,
    "cross_pac": None,
    "phase_shift": None,
    "burst_length": None,
}

configs["phase"] = {
    "betas": None,
    "gammas": None,
    "beta_sharpness": None,
    "phase": [0.45, 0.6],
    "pac": None,
    "cross_pac": None,
    "phase_shift": None,
    "burst_length": None,
}

configs["pac"] = {
    "betas": None,
    "gammas": None,
    "beta_sharpness": None,
    "phase": None,
    "pac": [0.3, 0.7],
    "cross_pac": None,
    "phase_shift": None,
    "burst_length": None,
}

configs["cross_pac"] = {
    "betas": None,
    "gammas": None,
    "beta_sharpness": None,
    "phase": None,
    "pac": None,
    "cross_pac": [0.4, 0.6],
    "phase_shift": None,
    "burst_length": None,
}

configs["phase_shift"] = {
    "betas": None,
    "gammas": None,
    "beta_sharpness": None,
    "phase": None,
    "pac": None,
    "cross_pac": None,
    "phase_shift": [0.3, 0.7],
    "burst_length": None,
}

configs["burst_length"] = {
    "betas": None,
    "gammas": None,
    "beta_sharpness": None,
    "phase": None,
    "pac": None,
    "cross_pac": None,
    "phase_shift": None,
    "burst_length": [0.3, 0.7],
}
