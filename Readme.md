# A Universal Analytic Model for Gravitational Lensing by Self-Interacting Dark Matter Halos
[![arXiv](https://img.shields.io/badge/arXiv-25XX.XXXXX%20-green.svg)](https://arxiv.org/abs/25XX.XXXXX)
This code implements a **parametric model for gravitational lensing** by self-interacting dark matter (SIDM) halos. It extends the parametric framework introduced in arXiv:2305.16176 (JCAP), enabling more comprehensive lensing studies. The code allows for the analytic calculation of the lensing potential, deflection angle, and convergence (κ). It includes example scripts to compute critical curves and caustics for SIDM halos, both in isolation and within a main halo, and track their evolution through the gravothermal phase. For broader applicability, we also provide efficient FFT-based tools for numerically computing lensing signatures.

# File Structure

- **example**
  - `example/Accuracy.ipynb`: This Jupyter notebook computes the density and lensing-related profiles, comparing the model's predicted results with those obtained numerically.
  - `example/Lensing.ipynb`: This Jupyter notebook provides instructions for using the parametric lensing model, along with examples of lensing-related maps, critical curves, and caustics.

- **lib**
  - `Lensing_tool.py`: Contains utility functions for additional lensing-related calculations, such as finding critical curves and other lensing features.
  - `SIDM_density_fluid.py`: Implements the density profile of SIDM halos.
  - `SIDM_Parametric_Model_jax.py`:  The core parametric model for gravitational lensing by SIDM halos.

- **Processed_data**: This directory contains data from fluid simulations used by the model for running simulations and analyses.


---

# Using Package

- **JAX web**: https://jax.readthedocs.io/en/latest/

Some necessary packages (possibly not complete):

```shell
pip install jupyter
pip install numpy
pip install matplotlib
pip install lenstronomy
pip install astropy
pip install sys
pip install pickle
pip install tqdm
pip install scipy
```
---

# Unit 

The unit system used in this model is the following:

Mass: $M_\odot/h$  
$r_{s,0}$:  $Mpc/h$   
$\rho_{s,0}$:  $h^2 M_\odot/Mpc^3$  

# Contributions

Contributions are welcome! If you have suggestions or improvements for the model, feel free to fork the repository and submit a pull request.

---

# Citation

If you use this model in your research, please cite it appropriately. You may use the following citation format:

[1] S. Hou, D. Yang, N. Li, and G. Li, A Universal Analytic Model for Gravitational Lensing by Self-Interacting Dark Matter Halos, arXiv:25???.?????.  
[2] D. Yang, E. O. Nadler, H.-B. Yu, and Y.-M. Zhong, A Parametric Model for Self-Interacting Dark Matter Halos, J. Cosmol. Astropart. Phys. 2024, 032 (2024).  
