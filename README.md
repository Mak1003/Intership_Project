# Intership_Project

"""
water_synthesis.py

Add water effect to an in-air RGB image given a depth map and water parameters.
This uses Sea-thru's forward formation model:
    I_c(z) = J_c * exp(-betaD_c(z) * z) + B_inf_c * (1 - exp(-betaB_c * z))

Inputs:
- J_rgb: HxWx3 float32 image in linear RGB, values in [0,1]
- z: HxW depth map in meters (absolute)
- params: dictionary with water parameters (see defaults)

Outputs:
- I_out: HxWx3 float32 synthesized underwater image in [0,1]
"""