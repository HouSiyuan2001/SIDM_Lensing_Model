import os
import re
import numpy as np
import pickle
import jax.numpy as jnp
from jax import vmap, jit
from jax.scipy.integrate import trapezoid
from Lensing_tool import Da


def find_closest_tr_file(tr0, folder):
    """
    Find the 'data_tr_*.pickle' file closest to tr0 in the specified folder.
    :param tr0: Input tr value
    :param folder: Data folder path
    :return: Closest matching file path and tr value
    """
    tr_values = []
    filenames = []
    
    tr_pattern = re.compile(r"data_tr_(\d+\.\d+)\.pickle")  
    
    for file in os.listdir(folder):
        match = tr_pattern.match(file)
        if match:
            tr = float(match.group(1))
            tr_values.append(tr)
            filenames.append(file)
    
    if not tr_values:
        raise FileNotFoundError("No matching 'data_tr_*.pickle' files found in the specified folder.")
    
    tr_values = np.array(tr_values)
    closest_idx = np.argmin(np.abs(tr_values - tr0))
    closest_file = filenames[closest_idx]
    closest_tr = tr_values[closest_idx]
    
    print(f"Input tr0: {tr0}")
    print(f"Closest found tr: {closest_tr}")
    print(f"Corresponding file: {closest_file}")
    
    return os.path.join(folder, closest_file), closest_tr


def load_density_data(tr0):
    """
    Load the r and rho data needed for interpolation based on tr0.
    """
    
    data_folder = "../lib/Processed_data"
    file_path, tr = find_closest_tr_file(tr0, data_folder)
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    r = jnp.array(data['r'])
    rho = jnp.array(data['rho'])
    
    return r, rho, tr

@jit
def rho_interp(rx, r_grid, rho_grid):
    """
    Perform 1D linear interpolation using JAX
    :param rx: Points to interpolate
    :param r_grid: Radius grid
    :param rho_grid: Density value grid
    """
    return jnp.interp(rx, r_grid, rho_grid, left=0.0, right=0.0)

@jit
def frhoRZ(R, z, r_grid, rho_grid, rhos0, rs0):
    """
    Calculate the density at a given radius and height.
    :param R: Horizontal radius
    :param z: Vertical position
    :param r_grid: Radius grid
    :param rho_grid: Density value grid
    """
    rx = jnp.sqrt(R**2 + z**2)
    return rho_interp(rx, r_grid, rho_grid) * rhos0 * rs0

@jit
def integrate_density_over_z(R, r_grid, rho_grid, rss, rhoss, chunk_size=100):
    """
    Integrate over the Z direction.
    :param R: Horizontal radius
    :param r_grid: Radius grid
    :param rho_grid: Density value grid
    :param rss: Scale factor
    :param chunk_size: Number of Z points per integration
    """
    rvir = 50 * rss
    z_samples = 20000  # Total number of sampling points
    z = jnp.linspace(-rvir, rvir, z_samples)
    dz = z[1] - z[0]
    
    total_density = 0.0
    for i in range(0, z_samples, chunk_size):
        z_chunk = z[i:i + chunk_size]
        rho_values = vmap(lambda z_i: frhoRZ(R, z_i, r_grid, rho_grid, rhoss, rss))(z_chunk)
        total_density += trapezoid(rho_values, dx=dz)
    
    return total_density

@jit
def Numerical_2d_density(R, r_grid, rho_grid, rss, rhoss):
    """
    Calculate the 2D density distribution.
    :param R: Radius array
    :param r_grid: Radius grid
    :param rho_grid: Density value grid
    :param rss: Scale factor
    """
    if R.ndim == 0:  # Scalar
        return integrate_density_over_z(R, r_grid, rho_grid, rss, rhoss)
    elif R.ndim == 1:  # 1D array
        compute_density = vmap(lambda R_ij: integrate_density_over_z(R_ij, r_grid, rho_grid, rss, rhoss))
        return compute_density(R)
    elif R.ndim == 2:  # 2D array
        compute_density = vmap(vmap(lambda R_ij: integrate_density_over_z(R_ij, r_grid, rho_grid, rss, rhoss)))
        return compute_density(R)
    else:
        raise ValueError("Unsupported dimension for R")
    
def refrence_Sigma(tr0, xi1, xi2, rs0, rhos0, zlens, posx=0, posy=0):
    """
    Calculate the reference kappa density.
    :param tr0: Input parameter tr0
    :param xi1, xi2: Spatial coordinates
    :param rs0: Scale factor
    """
    # Load r and rho data in a non-JAX environment
    r_grid, rho_grid, tr = load_density_data(tr0)
    apr = 1.0 / np.pi * 180.0 * 3600  # rad to arcsec
    X = (xi1 - posx) / apr * Da(zlens)
    Y = (xi2 - posy) / apr * Da(zlens)
    
    R = jnp.sqrt(X**2 + Y**2) / rs0
    
    # Perform integration
    # density_2D = Numerical_2d_density(R, r_grid, rho_grid, rs0, rhos0)
    # Extract unique radial distances (from minimum to maximum)
    max_radius = R.max()
    unique_radii = jnp.linspace(0, max_radius, num=R.shape[0])  # 1D array from center to edge

    # Calculate 1D radial density
    density_1D = Numerical_2d_density(unique_radii, r_grid, rho_grid, rs0, rhos0)

    # Map radial density back to 2D grid
    # Use vectorized approach to directly map each R to the corresponding density_1D value
    r_idx = jnp.searchsorted(unique_radii, R, side="right") - 1  # Find the index of R in unique_radii
    density_2D = density_1D[r_idx]  # Map according to index
    
    return density_2D, tr

def refrence_Sigma_givefun(xi1, xi2, rs0, rhos0, zlens, funrho, posx, posy, **funrho_kwargs):
    """
    Calculate the reference kappa density.
    :param tr0: Input parameter tr0
    :param xi1, xi2: Spatial coordinates
    :param rs0: Scale factor
    """
    # Load r and rho data in a non-JAX environment
    
    apr = 1.0 / np.pi * 180.0 * 3600  # rad to arcsec
    X = (xi1 - posx) / apr * Da(zlens)
    Y = (xi2 - posy) / apr * Da(zlens)
    
    R = jnp.sqrt(X**2 + Y**2) / rs0

    # Perform integration
    # density_2D = Numerical_2d_density(R, r_grid, rho_grid, rs0, rhos0)
    # Extract unique radial distances (from minimum to maximum)
    max_radius = R.max()
    unique_radii = jnp.logspace(np.log10(0.01), np.log10(10), num=R.shape[0])  # 1D array from center to edge
    rho_grid = funrho(unique_radii, **funrho_kwargs)

    # Calculate 1D radial density
    density_1D = Numerical_2d_density(unique_radii, unique_radii, rho_grid, rs0, rhos0)

    # Map radial density back to 2D grid
    # Use vectorized approach to directly map each R to the corresponding density_1D value
    r_idx = jnp.searchsorted(unique_radii, R, side="right") - 1  # Find the index of R in unique_radii
    density_2D = density_1D[r_idx]  # Map according to index
    
    return density_2D