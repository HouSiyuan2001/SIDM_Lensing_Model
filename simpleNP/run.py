import sys 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.integrate import quad
from parametricLensing import *

zl = 0.439
zs = 3
arcsec = 1.0 / np.pi * 180.0 * 3600  # arcsec per rad

# NFW parameters 
rhosh=437553.
rsh=1332.76
rhoss=1.1378e6
rss=107.342
dist = 0.24*rsh
sigmacr=(3e5)**2/(4*np.pi*G)*dA(zs)/(dA(zl)*DLS(zl, zs))

ah,bh,ch,ph,sh = [0.758150, 3.585676, 1.941042, 2.573823, 1.993227] # host halo is NFW
trx=1.08 # the gravothermal phase tau
a,b,c,p,s = [fited_a(trx), fited_b(trx), fited_c(trx), fited_p(trx), fited_s(trx)]

# create maps, size of the scale radii 
nc=1000
rsh_arc = rsh/dA(zl)*arcsec
rss_arc = rss/dA(zl)*arcsec

# host
bs=rsh_arc
ds=bs/nc
xx01=np.linspace(-bs/2.0,bs/2.0-ds,nc)+0.5*ds
xx02=np.linspace(-bs/2.0,bs/2.0-ds,nc)+0.5*ds
xi2,xi1 = np.meshgrid(xx01,xx02)

map_RH=np.sqrt(pow(xi1,2)+pow(xi2,2))
map_RS=np.sqrt(pow(xi1-dist/dA(zl)*arcsec,2)+pow(xi2,2))

map_alphaH_hat = fit_alpha(map_RH*dA(zl)/arcsec/rsh,ah,bh,ch,ph,sh)
map_alphaH = map_alphaH_hat * (rhosh * rsh * rsh)/dA(zl) / sigmacr *arcsec
map_alphaH1 = map_alphaH*(xi1/map_RH)
map_alphaH2 = map_alphaH*(xi2/map_RH)

map_alphas_hat = fit_alpha(map_RS*dA(zl)/arcsec/rss,a,b,c,p,s)
map_alphas = map_alphas_hat * (rhoss * rss * rss)/dA(zl) / sigmacr *arcsec
map_alphas1 = map_alphas*((xi1-dist/dA(zl)*arcsec)/map_RS)
map_alphas2 = map_alphas*(xi2/map_RS)

# compute lensing shear matrix
dsx_arc = bs / nc
alpha1_in=map_alphaH1+map_alphas1
alpha2_in=map_alphaH2+map_alphas2
# np.gradient runs in linear time as evaluating equations on grid points
# no need to evaluate analytic equations 
al11_tmp, al12_tmp = np.gradient(alpha1_in, dsx_arc)
al21_tmp, al22_tmp = np.gradient(alpha2_in, dsx_arc)
kappa = 0.5*(al11_tmp + al22_tmp)
gamma1 = 0.5*(al22_tmp - al11_tmp)
gamma2 = al12_tmp
gamma_sq = gamma1**2.0 + gamma2**2.0
mu_out = 1.0/((1.0 - kappa)**2 - gamma_sq)
yi1 = xi1-alpha1_in
yi2 = xi2-alpha2_in

gamma = np.sqrt(gamma1**2 + gamma2**2)
lambdat_global = 1 - kappa - gamma
lambdar_global = 1 - kappa + gamma

fig_combined, ax = plt.subplots(1, 2, figsize=(10, 6))

ax_ = ax[0]
lambdar_contour1 = ax_.contour(xi1, xi2, lambdar_global, levels=[0.0], colors='red', linewidths=2, alpha=0.4, linestyles='dashed')
lambdat_contour1 = ax_.contour(xi1, xi2, lambdat_global, levels=[0.0], colors='red', linewidths=2, alpha=0.9, linestyles='-')

ax_.set_aspect('equal')
ax_.set_xlabel("X (arcsec)")
ax_.set_ylabel("Y (arcsec)")
ax_.set_xlim(-85, 85)
ax_.set_ylim(-85, 85)

ax_ = ax[1]
lambdar_contour2 = ax_.contour(yi1, yi2, lambdar_global, levels=[0.0], colors='green', linewidths=2, alpha=0.4, linestyles='dashed')
lambdat_contour2 = ax_.contour(yi1, yi2, lambdat_global, levels=[0.0], colors='green', linewidths=2, alpha=0.9, linestyles='-')

ax_.set_aspect('equal')
ax_.set_xlabel("X (arcsec)")
ax_.set_xlim(-20,20)
ax_.set_ylim(-20,20)

solid_line = mlines.Line2D([], [], color='red', linestyle='-', label='Tangential Critical')
dashed_line = mlines.Line2D([], [], color='red', linestyle='--', label='Radial Critical',alpha=0.4)
solid_line2 = mlines.Line2D([], [], color='green', linestyle='-', label='Tangential Caustics')
dashed_line2 = mlines.Line2D([], [], color='green', linestyle='--', label='Radial Caustics',alpha=0.4)

fig_combined.legend(handles=[solid_line, dashed_line, solid_line2, dashed_line2], loc='upper center', bbox_to_anchor=(0.5, 0.06), ncol=4, fontsize=16, frameon=False, columnspacing=1.0, handletextpad=0.7) 
fig_combined.tight_layout()
plt.savefig("critical_curves.jpg", bbox_inches='tight')
plt.show()
