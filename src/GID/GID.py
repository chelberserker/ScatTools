import math
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def calculate_cell_parameters(qxy1, qxy2, qxy3, qz1, qz2, qz3):
    """
    Calculates crystallographic parameters based on scattering vectors.
    Args:
        qxy1 (float): q_{xy} position of the first peak
        qxy2 (float): q_{xy} position of the second peak
        qxy3 (float): q_{xy} position of the third peak
        qz1 (float): q_{z} position of the first peak
        qz2 (float): q_{z} position of the second peak
        qz3 (float): q_{z} position of the third peak
    Returns:

    """

    vectors = sorted([
        (qxy1, qz1),
        (qxy2, qz2),
        (qxy3, qz3)
    ], key=lambda v: v[0])

    qx10, qz10 = vectors[0]
    qx01, qz01 = vectors[1]
    qx11, qz11 = vectors[2]

    cos_g = (qx10 ** 2 + qx01 ** 2 - qx11 ** 2) / (2 * qx10 * qx01)
    if abs(cos_g) > 1.0:
        logger.warning('Error: Check input values (cos G > 1).')
        return None

    gam_rad = np.acos(cos_g)  # Angle between A and B in real space

    a = 2 * np.pi / (qx10 * np.sin(gam_rad))
    b = 2 * np.pi / (qx01 * np.sin(gam_rad))


    if abs(qz10 + qz01 - qz11) > 0.03:
        qz10 = -qz10

    if abs(qz10 + qz01 - qz11) > 0.03:
        qz11 = -qz11

    # Normal components
    nz = qx10 * qx01 * np.sin(gam_rad)
    nx = -qz10 * qx01 * np.sin(gam_rad)
    ny = -(qx10 * qz01 + qz10 * qx01 * np.cos(gam_rad))
    nmod = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)

    if nmod == 0 or abs(nz / nmod) > 1.0:
        logger.warning('Error: Check input values (Nz/Nmod > 1).')
        return None

    t3_rad = math.acos(nz / nmod)

    # Azimuth: Angle between projection of N on XOY where OX==a* and OY==b*
    if t3_rad == 0.0:
        azimuth3_rad = 0.0
    else:
        if nx == 0.0:
            azimuth3_rad = 0.0
        else:
            azimuth3_rad = np.atan(ny / nx)

    area = a * b * np.sin(gam_rad)

    # Convert angles to degrees
    gam_deg = np.rad2deg(gam_rad)
    if gam_deg > 180.0:
        gam_deg = 360.0 - gam_deg

    t3_deg = np.rad2deg(t3_rad)

    # Azimuth = counter-clockwise rotation from real space A-vector
    if azimuth3_rad != 0.0:
        azimuth3_deg = np.rad2deg(azimuth3_rad) + (gam_deg - 90.0)
    else:
        azimuth3_deg = 0.0

    return {
        "a": np.round(a, 5),
        "b": np.round(b, 5),
        "gamma": np.round(gam_deg,2),
        "tilt": np.round(t3_deg,2),
        "azimuth": np.round(azimuth3_deg,2),
        "area": np.round(area, 3)
    }

def calculate_q_values(a, b, gamma, tilt, azimuth):
    """
    Calculates the scattering vectors (qxy, qz) based on 2D crystallographic parameters.
    Args:
        a (float): Unit cell parameter a
        b (float): Unit cell parameter b
        gamma (float): Angle between a and b in real space (degrees)
        tilt (float): Tilt angle of the lattice (degrees)
        azimuth (float): Azimuthal angle of the tilt (degrees)
    Returns:
        dict: The qxy and qz scattering vectors for the first three peaks
    """
    gam_rad = np.deg2rad(gamma)
    t3_rad = np.deg2rad(tilt)

    # 1. Reverse A and B reciprocal relationships to get qxy1 (qx10) and qxy2 (qx01)
    qx10 = 2 * np.pi / (a * np.sin(gam_rad))
    qx01 = 2 * np.pi / (b * np.sin(gam_rad))

    # 2. Calculate qxy3 (qx11) using the reciprocal space law of cosines
    qx11 = np.sqrt(qx10**2 + qx01**2 - 2 * qx10 * qx01 * np.cos(gam_rad))

    # 3. Reverse the azimuth logic
    # azimuth3_deg = azimuth_internal_deg + (gam_deg - 90.0)
    azimuth_internal_deg = azimuth - (gamma - 90.0)
    azimuth_internal_rad = np.deg2rad(azimuth_internal_deg)

    # 4. Reconstruct the normal vector components
    nz = qx10 * qx01 * np.sin(gam_rad)

    # Guard against perfectly vertical tilt dividing by zero
    if np.cos(t3_rad) == 0.0:
        return None

    nmod = nz / np.cos(t3_rad)

    # Extract nx and ny using the inverted azimuthal angles
    nxy = np.sqrt(max(0, nmod**2 - nz**2))
    nx = nxy * np.cos(azimuth_internal_rad)
    ny = nxy * np.sin(azimuth_internal_rad)

    # 5. Reverse the cross product equations to solve for qz10 and qz01
    # nx = -qz10 * qx01 * sin(gamma)
    qz10 = nx / (qx01 * np.sin(gam_rad))

    # ny = -(qx10 * qz01 + qz10 * qx01 * cos(gamma))
    qz01 = (-ny - qz10 * qx01 * np.cos(gam_rad)) / qx10

    # 6. Reconstruct qz11 (assuming ideal constructive addition for the 1,1 peak)
    qz11 = qz10 + qz01

    return {
        "qxy1": np.round(qx10, 5),
        "qz1": np.round(qz10, 5),
        "qxy2": np.round(qx01, 5),
        "qz2": np.round(qz01, 5),
        "qxy3": np.round(qx11, 5),
        "qz3": np.round(qz11, 5)
    }