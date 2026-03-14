"""
thermal_params.py
-----------------
Building thermal parameters for the 5-zone Saudi villa (Table 1 in paper).
All values derived from standard RC thermal model calibration.

Units:
  K_i   : kW/K  (conductance to outdoors)
  C_i   : kJ/K  (thermal capacitance)
  K_ij  : kW/K  (inter-zone wall conductance)
  Q_int : kW    (internal heat gains)
  Q_cool: kW    (cooling capacity = 5.3 kW = 18,500 BTU / 3.517)

All zones use 18,500 BTU On/Off split units:
  Electrical input:  1.8 kW
  Cooling output:    5.3 kW
  EER:               10.25 (BTU/Wh)
"""

# 5-zone villa configuration
VILLA_5Z = {
    'zones': [
        {
            'name':     'Dining Room',
            'area_m2':  30,
            'K_i':      0.28,    # kW/K outdoor conductance
            'C_i':      3600,    # kJ/K thermal mass
            'Q_int':    0.40,    # kW internal gains (appliances + people)
            'adjacent': ['Living Room'],
            'K_ij':     {'Living Room': 0.05}
        },
        {
            'name':     'Living Room',
            'area_m2':  25,
            'K_i':      0.25,
            'C_i':      3000,
            'Q_int':    0.30,
            'adjacent': ['Dining Room', 'Master Bedroom'],
            'K_ij':     {'Dining Room': 0.05, 'Master Bedroom': 0.05}
        },
        {
            'name':     'Master Bedroom',
            'area_m2':  25,
            'K_i':      0.24,
            'C_i':      3000,
            'Q_int':    0.30,
            'adjacent': ['Living Room', 'Boys Bedroom'],
            'K_ij':     {'Living Room': 0.05, 'Boys Bedroom': 0.04}
        },
        {
            'name':     'Boys Bedroom',
            'area_m2':  20,
            'K_i':      0.22,
            'C_i':      2400,
            'Q_int':    0.25,
            'adjacent': ['Master Bedroom', 'Girls Bedroom'],
            'K_ij':     {'Master Bedroom': 0.04, 'Girls Bedroom': 0.04}
        },
        {
            'name':     'Girls Bedroom',
            'area_m2':  20,
            'K_i':      0.22,
            'C_i':      2400,
            'Q_int':    0.25,
            'adjacent': ['Boys Bedroom'],
            'K_ij':     {'Boys Bedroom': 0.04}
        }
    ],
    'ac_specs': {
        'btu':               18500,
        'electrical_kw':     1.8,
        'cooling_kw':        5.3,
        'eer':               10.25
    }
}


def get_params_arrays(config: dict = VILLA_5Z) -> dict:
    """
    Convert zone config dict to flat numpy-ready arrays for simulation.

    Returns:
        dict with keys: K, C, Q_int, K_ij_matrix, zone_names
    """
    import numpy as np
    zones = config['zones']
    n = len(zones)
    zone_names = [z['name'] for z in zones]
    name_to_idx = {name: i for i, name in enumerate(zone_names)}

    K     = np.array([z['K_i']   for z in zones])
    C     = np.array([z['C_i']   for z in zones])
    Q_int = np.array([z['Q_int'] for z in zones])

    # Inter-zone coupling matrix
    K_ij = np.zeros((n, n))
    for i, z in enumerate(zones):
        for adj_name, k_val in z.get('K_ij', {}).items():
            j = name_to_idx.get(adj_name)
            if j is not None:
                K_ij[i, j] = k_val
                K_ij[j, i] = k_val  # symmetric

    return {
        'K':          K,
        'C':          C,
        'Q_int':      Q_int,
        'K_ij':       K_ij,
        'zone_names': zone_names
    }


def get_scaled_params(n_zones: int) -> dict:
    """
    Generate thermal parameters for n_zones compound
    by replicating the villa unit cell.
    Used for scalability experiments (Table 4 in paper).
    """
    import numpy as np
    base = get_params_arrays(VILLA_5Z)
    n_reps = (n_zones + 4) // 5

    K     = np.tile(base['K'],     n_reps)[:n_zones]
    C     = np.tile(base['C'],     n_reps)[:n_zones]
    Q_int = np.tile(base['Q_int'], n_reps)[:n_zones]

    # Block-diagonal coupling matrix
    K_ij = np.zeros((n_zones, n_zones))
    for rep in range(n_reps):
        start = rep * 5
        end   = min(start + 5, n_zones)
        size  = end - start
        K_ij[start:end, start:end] = base['K_ij'][:size, :size]

    return {'K': K, 'C': C, 'Q_int': Q_int, 'K_ij': K_ij}


if __name__ == '__main__':
    import numpy as np
    p = get_params_arrays()
    print('5-Zone Villa Thermal Parameters')
    print('Zone Names:', p['zone_names'])
    print('K  (kW/K):', p['K'])
    print('C  (kJ/K):', p['C'])
    print('Q_int(kW):', p['Q_int'])
    print('K_ij matrix (kW/K):')
    print(np.round(p['K_ij'], 3))
