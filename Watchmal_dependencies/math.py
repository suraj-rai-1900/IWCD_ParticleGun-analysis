import numpy as np


DEFAULT_TANK_AXIS = 1


def towall(position, angle, tank_half_height=300, tank_radius=400, tank_axis=None):
    """
    Calculate towall: distance from position to detector wall, in particle direction

    Parameters
    ----------
    position : array_like
        vector of (x, y, z) position of a particle or (N,3) array of positions for N particles
    angle : array_like
        vector of (zenith, azimuth) direction of particle or (N, 2) array of directions for N particles
    tank_half_height : float, default: 300
        half-height of the detector ID
    tank_radius : float, default: 400
        Radius of the detector ID
    tank_axis : int, optional
        Axis along which the tank cylinder is oriented. By default, use the y-axis.

    Returns
    -------
    np.ndarray or scalar
        array of towall values for each position, or scalar if only one position
    """
    if tank_axis is None:
        tank_axis = DEFAULT_TANK_AXIS
    pos_trans = np.delete(position, tank_axis, axis=-1)
    pos_along = position[..., tank_axis]
    dir_along, dir_trans = polar_to_cartesian(angle)
    a = np.linalg.norm(dir_trans, axis=-1) ** 2
    b = np.sum(pos_trans * dir_trans, axis=-1)
    c = np.linalg.norm(pos_trans, axis=-1) ** 2 - tank_radius ** 2
    towall_barrel = (-b + np.sqrt(b ** 2 - a * c)) / a
    towall_endcap = tank_half_height / abs(dir_along) - pos_along / dir_along
    return np.minimum(towall_barrel, towall_endcap)


def dwall(position, tank_half_height=300, tank_radius=400, tank_axis=None):
    """
    Calculate dwall: distance from position to the nearest detector wall

    Parameters
    ----------
    position : array_like
        vector of (x, y, z) position of an event or (N,3) array of (x, y, z) position coordinates for N events
    tank_half_height : float, default: 300
        half-height of the detector ID
    tank_radius : float, default: 400
        Radius of the detector ID
    tank_axis : int, optional
        Axis along which the tank cylinder is oriented. By default, use y-axis

    Returns
    -------
    np.ndarray or scalar
        array of dwall values for each position, or scalar if only one position
    """
    if tank_axis is None:
        tank_axis = DEFAULT_TANK_AXIS
    pos_along = position[..., tank_axis]
    pos_trans = np.delete(position, tank_axis, axis=-1)
    dwall_barrel = tank_radius - np.linalg.norm(pos_trans, axis=-1)
    dwall_endcap = tank_half_height - np.abs(pos_along)
    return np.minimum(dwall_barrel, dwall_endcap)


def momentum_from_energy(energy, label, particle_masses=np.array((0, 0.511, 105.7, 134.98))):
    """
    Calculate momentum of particle from total energy and particle type (label)
    Default labels are 0:gamma, 1:electron, 2:muon, 3:pi0

    Parameters
    ----------
    energy : array_like
        energy of particle or vector of energies of particles
    label : array_like
        integer label of particle type or vector of labels of particles
    particle_masses : array_like
        array of particle masses indexed by label

    Returns
    -------
    np.ndarray or scalar
        array of momentum values for each energy, or scalar if only one energy
    """
    mass = particle_masses[label]
    return np.sqrt(energy**2 - mass**2)


def polar_to_cartesian(angles):
    """
    Calculate (x,y,z) unit vector from azimuth and zenith angles

    Parameters
    ----------
    angles : array_like
        vector of (zenith, azimuth) of a direction or (N,2) array of (zenith, azimuth) angles for N directions

    Returns
    -------
    dir_along: np.ndarray or scalar
        array of the component along zenith direction for unit vector of each direction, or scalar if only one direction
    dir_trans: np.ndarray
        array of the components transverse to zenith direction for unit vector of each direction
    """
    zenith = angles[..., 0]
    azimuth = angles[..., 1]
    dir_along = np.cos(zenith)
    dir_trans = np.column_stack((np.sin(zenith) * np.cos(azimuth), np.sin(zenith) * np.sin(azimuth)))
    return dir_along, dir_trans


def direction_from_angles(angles, zenith_axis=None):
    """
    Calculate unit vector from azimuth and zenith angles

    Parameters
    ----------
    angles : array_like
        vector of (zenith, azimuth) of a direction or (N,2) array of (zenith, azimuth) angles for N directions
    zenith_axis : int, optional
        Axis along which the zenith angle is relative to (i.e. the axis the tank is oriented). By default, use y-axis.

    Returns
    -------
    np.ndarray
        array of unit vectors of each direction
    """
    dir_along, dir_trans = polar_to_cartesian(angles)
    if zenith_axis is None:
        zenith_axis = DEFAULT_TANK_AXIS
    return np.insert(dir_trans, zenith_axis, dir_along, axis=1)


def angles_from_direction(direction, zenith_axis=None):
    """
    Calculate azimuth and zenith angles from unit vector

    Parameters
    ----------
    direction : array_like
        vector of (x,y,z) components of a unit vector of a direction, or (N,3) array of (x,y,z) unit vector directions
    zenith_axis : int, optional
        Axis along which the zenith angle is relative to (i.e. the axis the tank is oriented). By default, use y-axis.

    Returns
    -------
    np.ndarray
        array of (zenith, azimuth) angles of each direction

    """
    if zenith_axis is None:
        zenith_axis = DEFAULT_TANK_AXIS
    dir_along = direction[..., zenith_axis]
    dir_trans = np.delete(direction, zenith_axis, axis=-1)
    zenith = np.arccos(dir_along)
    azimuth = np.arctan2(dir_trans[..., 1], dir_trans[..., 0])
    return np.column_stack((zenith, azimuth))
