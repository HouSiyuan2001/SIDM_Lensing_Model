import jax.numpy as jnp
# Precompute cosmological distances using Astropy
# Cosmology, Units: Msun, Mpc/h, Gyr
import numpy as np
from astropy import units as u 
import astropy.constants as const 
from astropy.cosmology import FlatwCDM 

vc = const.c.to(u.km/u.s).value 
G =  const.G.to(u.Mpc/u.solMass *(u.km/u.second)*(u.km/u.second)).value 
apr =  1.0/np.pi*180.0*3600  # arcsec per rad


cosmo = FlatwCDM(H0=70.0, Om0=0.3, Ob0=0.05, w0=-1.000000)

z_values = np.linspace(0, 5.0, 10000)
Dc_values = cosmo.comoving_distance(z_values).value * cosmo.h  # Mpc/h
Da_values = cosmo.angular_diameter_distance(z_values).value * cosmo.h  # Mpc/h

# Convert z_values and Dc_values to JAX arrays
z_values_jax = jnp.array(z_values)
Dc_values_jax = jnp.array(Dc_values)
Da_values_jax = jnp.array(Da_values)
#维里过密度
def dv(z): 
    ov = 1.0/cosmo.Om(z)-1.0
    res = 18.8*np.pi*np.pi*(1.0+0.4093*ov**0.9052)
    return res
#r200
def rvir_mvir(m,z, stype="vir"):
    if stype=="vir":
        res = (3.0*m/4.0/np.pi/rho_crit(z)/dv(z))**(1.0/3.0)
    elif stype=="200":
        res = (3.0*m/4.0/np.pi/rho_crit(z)/200.0)**(1.0/3.0)
    else:
        print("wrong stype!!!")
    return res

def SigmaCrit(z1,z2):
    '''
        Critical surface density for the case of lens plane at z1 and source plane at z2.
    '''
    res = (vc*vc/4.0/jnp.pi/G*Dc(z2)/(Dc(z1)/(1.0+z1))/Dc2(z1,z2))
    return res

def rho_crit(z, densType="crit"): 
    if densType=="matter":
        # matter density 
        res = cosmo.Om(z)*cosmo.critical_density(z).to(u.solMass/u.Mpc/u.Mpc/u.Mpc).value/cosmo.h/cosmo.h #M_sun Mpc^-3 *h*h
    elif densType=="crit":
        # critical density
        res = cosmo.critical_density(z).to(u.solMass/u.Mpc/u.Mpc/u.Mpc).value/cosmo.h/cosmo.h #M_sun Mpc^-3 *h*h
    else:
        print("error!!!")
    return res

# 计算共动距离
def Dc0(z):
    res = cosmo.comoving_distance(z).value*cosmo.h #Mpc/h
    return res

#两点间的共动距离
def Dc20(z1,z2):
    Dcz1 = (cosmo.comoving_distance(z1).value*cosmo.h) #Mpc/h
    Dcz2 = (cosmo.comoving_distance(z2).value*cosmo.h) #Mpc/h
    res = Dcz2-Dcz1+1e-8
    return res

# Define distance functions using interpolation
def Dc(z):
    return jnp.interp(z, z_values_jax, Dc_values_jax)

def Dc2(z1, z2):
    return Dc(z2) - Dc(z1) + 1e-8

#角直径距离
def Da0(z):
    res = cosmo.angular_diameter_distance(z).value*cosmo.h #Mpc/h
    return res


def Da20(z1,z2):
    res = cosmo.angular_diameter_distance_z1z2(z1, z2).value*cosmo.h #Mpc/h
    return res

def Da(z):
    return jnp.interp(z, z_values_jax, Da_values_jax)

def Da2(z1, z2):
    return Da(z2) - Da(z1) + 1e-8


zs0 = 10.0
def rev_dangle(zl_lens, zs, al1_ref, al2_ref):
    rescale = Da(zs0)/Da2(zl_lens, zs0) * Da2(zl_lens, zs) / Da(zs)
    al1_zs = al1_ref * rescale
    al2_zs = al2_ref * rescale
    return al1_zs, al2_zs

def rev_potential(zl_lens, zs, pot):
    rescale = Da(zs0)/Da2(zl_lens, zs0) * Da2(zl_lens, zs) / Da(zs)
    pot_zs =pot * rescale
    return pot_zs
def alphas_to_mu(alpha1_in, alpha2_in, dsx_arc, xi1, xi2):
    al11_tmp, al12_tmp = jnp.gradient(alpha1_in, dsx_arc)
    al21_tmp, al22_tmp = jnp.gradient(alpha2_in, dsx_arc)
    kappa = 0.5*(al11_tmp + al22_tmp)
    gamma1 = 0.5*(al22_tmp - al11_tmp)
    gamma2 = al12_tmp
    gamma_sq = gamma1**2.0 + gamma2**2.0 
    # mu_out = 1.0/(1.0 - (al11_tmp + al22_tmp) + al11_tmp*al22_tmp - al12_tmp*al21_tmp)
    mu_out = 1.0/((1.0 - kappa)**2 - gamma_sq)
    y1_out = xi1-alpha1_in
    y2_out = xi2-alpha2_in
    return y1_out, y2_out, mu_out, kappa, gamma1, gamma2



def Mass_c_to_rhos_rs(Mvir, cvir,zlens):
    Rvir = rvir_mvir(Mvir,zlens)
    rs = Rvir / cvir  # Mpc
    rhos = rho_crit(zlens)*dv(zlens)/3.0*cvir**3.0/(jnp.log(1.0+cvir)-cvir/(1+cvir))
    rhos = rhos.item()  # 将 JAX Array 转换为 float
    return rhos, rs

# Ensure that make_c_coor and other utility functions are also using jax.numpy
def make_c_coor(bs,nc):
    '''
        Draw the mesh grids for a bs*bs box with nc*nc pixels
    '''
    ds = bs/nc
    xx01 = np.linspace(-bs/2.0,bs/2.0-ds,nc)+0.5*ds
    xx02 = np.linspace(-bs/2.0,bs/2.0-ds,nc)+0.5*ds
    xg1,xg2 = np.meshgrid(xx01,xx02)
    return xg1,xg2

def Einstein_angle(M,D_L,D_S,D_LS):
    return np.sqrt((4 * G * M * D_LS) / (vc**2 * D_L * D_S))

apr =  1.0/np.pi*180.0*3600  # arcsec per rad
def Rad_to_arcsec(rad):
    return rad * apr
def arcsec_to_Rad(arcsec):
    return arcsec / apr



# fft

def zero_padding(in_arr, Nx, Ny):
    out = np.zeros((2*Nx, 2*Ny), dtype=in_arr.dtype)
    out[:Nx,:Ny] = in_arr
    return out

def corner_matrix(in_arr, Nx, Ny):
    return in_arr[:Nx,:Ny]

def roll_a_matrix(in_arr, roll_nx1, roll_nx2):
    # 使用 numpy.roll 进行平移
    return np.roll(np.roll(in_arr, roll_nx1, axis=0), roll_nx2, axis=1)

def kernel_alphas_iso_I(Ncc, dsx):
    # I (原始)边界条件的 kernel
    alpha1_iso = np.zeros((Ncc, Ncc))
    alpha2_iso = np.zeros((Ncc, Ncc))
    half = Ncc // 2
    for i in range(Ncc):
        for j in range(Ncc):
            if i <= half and j <= half:
                x = (i)*dsx+0.5*dsx
                y = (j)*dsx+0.5*dsx
                r = np.sqrt(x*x+y*y)
                if r > dsx*(Ncc/2.0):
                    alpha1_iso[i,j] = 0.0
                    alpha2_iso[i,j] = 0.0
                else:
                    val = 1.0/(np.pi*r*r)
                    alpha1_iso[i,j] = x*val
                    alpha2_iso[i,j] = y*val
            else:
                # 利用对称性填充
                if i <= half and j > half:
                    alpha1_iso[i,j] =  alpha1_iso[i, Ncc-j]
                    alpha2_iso[i,j] = -alpha2_iso[i, Ncc-j]
                if i > half and j <= half:
                    alpha1_iso[i,j] = -alpha1_iso[Ncc-i, j]
                    alpha2_iso[i,j] =  alpha2_iso[Ncc-i, j]
                if i > half and j > half:
                    alpha1_iso[i,j] = -alpha1_iso[Ncc-i, Ncc-j]
                    alpha2_iso[i,j] = -alpha2_iso[Ncc-i, Ncc-j]
    return alpha1_iso, alpha2_iso

def kernel_alphas_iso_P(Ncc, dsx):
    # P (周期)边界条件的 kernel
    alpha1_iso = np.zeros((Ncc, Ncc))
    alpha2_iso = np.zeros((Ncc, Ncc))
    for i in range(Ncc):
        for j in range(Ncc):
            x = ((i - Ncc//2) + 0.5)*dsx
            y = ((j - Ncc//2) + 0.5)*dsx
            r = np.sqrt(x*x + y*y)
            if r > dsx*(Ncc/2.0):
                alpha1_iso[i,j] = 0.0
                alpha2_iso[i,j] = 0.0
            else:
                alpha1_iso[i,j] = x/(np.pi*r*r)
                alpha2_iso[i,j] = y/(np.pi*r*r)
    return alpha1_iso, alpha2_iso

def convolve_fft(in1, in2, dx, dy):
    f1 = np.fft.rfft2(in1)
    f2 = np.fft.rfft2(in2)
    out_fft = f1 * f2
    out = np.fft.irfft2(out_fft, s=in1.shape)
    out = out * dx * dy
    return out

def call_kappa_to_alphas(Kappa, Bsz, Ncc, boundary_type='I'):
    Kappa = np.array(Kappa, dtype=np.float64)
    dsx = Bsz / Ncc

    if boundary_type == 'I':
        # 使用I类型kernel
        alpha1_iso, alpha2_iso = kernel_alphas_iso_I(2*Ncc, dsx)
        # 零填充
        kappa = zero_padding(Kappa, Ncc, Ncc)
        # 卷积
        alpha1_tmp = convolve_fft(kappa, alpha1_iso, dsx, dsx)
        alpha2_tmp = convolve_fft(kappa, alpha2_iso, dsx, dsx)
        alpha1 = corner_matrix(alpha1_tmp, Ncc, Ncc)
        alpha2 = corner_matrix(alpha2_tmp, Ncc, Ncc)
    
    elif boundary_type == 'P':
        # 使用P类型kernel (不需要zero padding,与C代码保持一致)
        alpha1_iso, alpha2_iso = kernel_alphas_iso_P(Ncc, dsx)
        # 直接对Kappa做卷积
        alpha1_tmp = np.zeros_like(Kappa)
        alpha2_tmp = np.zeros_like(Kappa)

        alpha1_tmp = convolve_fft(Kappa, alpha1_iso, dsx, dsx)
        alpha2_tmp = convolve_fft(Kappa, alpha2_iso, dsx, dsx)

        # roll处理
        alpha1 = roll_a_matrix(alpha1_tmp, Ncc//2, Ncc//2)
        alpha2 = roll_a_matrix(alpha2_tmp, Ncc//2, Ncc//2)
    
    else:
        raise ValueError("boundary_type must be 'I' or 'P'.")

    return alpha1, alpha2
