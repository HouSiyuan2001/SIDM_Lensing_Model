# Lensing Parametric Model

This project provides a **parametric model for gravitational lensing** designed to study lensing phenomena in the context of self-interacting dark matter (SIDM). The code calculates several key lensing parameters, including convergence (κ), lensing potential, deflection angle, and critical curves. This model helps users better understand the gravitational lensing phenomena caused by self-interacting dark matter.

# File Structure

- **example**
  - `Accuracy.ipynb`: This Jupyter notebook demonstrates the accuracy of the model, including two main components: the density part and the lensing model part.
  - `Lensing.ipynb`: This Jupyter notebook provides a detailed guide on how to use the parametric lensing model, calculate the parameters, and generate plots related to lensing phenomena.

- **lib**
  - `Lensing_tool.py`: Contains utility functions to perform additional lensing-related calculations, such as finding critical curves and other lensing features.
  - `SIDM_density_fluid.py`: Implements the density profile related to self-interacting dark matter for the lensing model.
  - `SIDM_Parametric_Model_jax.py`: The core parametric model for gravitational lensing due to self-interacting dark matter.

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
Distance: Mpc/h 
Mass: M_sun

# Contributions

Contributions are welcome! If you have suggestions or improvements for the model, feel free to fork the repository and submit a pull request.

---

# Citation

If you use this model in your research, please cite it appropriately. You may use the following citation format:

[1] S. Hou, D. Yang, N. Li, and G. Li, A Universal Analytic Model for Gravitational Lensing by Self-Interacting Dark Matter Halos, arXiv:25???.?????.  
[2] D. Yang, Exploring Self-Interacting Dark Matter Halos with Diverse Baryonic Distributions: A Parametric Approach, arXiv:2405.03787.  
[3] D. Yang, E. O. Nadler, H.-B. Yu, and Y.-M. Zhong, A Parametric Model for Self-Interacting Dark Matter Halos, J. Cosmol. Astropart. Phys. 2024, 032 (2024).  
[4] D. Yang and H.-B. Yu, Self-Interacting Dark Matter and Small-Scale Gravitational Lenses in Galaxy Clusters, Phys. Rev. D 104, 103031 (2021).  
