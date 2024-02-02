import yaml
import warnings
import functools

# NOTE: it is important to import the librarys that you use to model the intrinsic parameters here,
#       some general ones have already been imported below.

import numpy as np
from numpy.random import normal as normrnd
from scipy.stats import norm
from scipy.stats import skewnorm

class IntrinsicParameters:

    """
        Class that obtains all the intrinsic parameter values from the IntrinsicParameters.yaml file
    """

    belly_br_distribution = lambda: np.random.normal(0.405766627860074, 0.08510657640471547) * 400
    belly_br_u = lambda: 0.405766627860074 * 400
    belly_l_distribution = lambda: np.random.normal(1.0414934216489478, 0.0497660715368968)
    belly_l_u = lambda: 1.0414934216489478
    belly_w_distribution = lambda: np.random.normal(0.4509085008872612, 0.04318248156070455)
    belly_w_u = lambda: 0.4509085008872612
    c_belly_distribution = lambda: np.random.normal(0.8976990388969368, 0.046408648027029396)
    c_belly_u = lambda: 0.8976990388969368
    c_eye_distribution = lambda: np.random.normal(1.7175761389389386, 0.20589859105033523)
    c_eye_u = lambda: 1.7175761389389386
    c_head_distribution = lambda: skewnorm.rvs(13.160772264490056, 0.9469252531374381, 0.3251749094426817)
    c_head_u = lambda: 0.9469252531374381
    d_eye_distribution = lambda: np.random.normal(1.144396567090266, 0.04442945643226941)
    d_eye_u = lambda: 1.144396567090266
    eye_br_distribution = lambda: skewnorm.rvs(-8.231518385739356, 0.6969923960604942, 0.1853606969178898) * 400
    eye_br_u = lambda: 0.6969923960604942
    eye_l_distribution = lambda: np.random.normal(0.5122961600634638, 0.07434852108369809)
    eye_l_u = lambda: 0.5122961600634638
    eye_w_distribution = lambda: np.random.normal(0.4209238403150402, 0.040920763968052117)
    eye_w_u = lambda: 0.4209238403150402
    head_br_distribution = lambda: np.random.normal(0.27670705295942777, 0.05571972183069393) * 400
    head_br_u = lambda: 0.27670705295942777
    head_l_distribution = lambda: np.random.normal(0.7889611717422198, 0.10205659701587598)
    head_l_u = lambda: 0.7889611717422198
    head_w_distribution = lambda: np.random.normal(0.5077558460625518, 0.032475097562161506)
    head_w_u = lambda: 0.5077558460625518
    seglen_distribution = lambda: 2 + np.random.rand()
    seglen_u = lambda: 2.5

    def __init__(self, pathToYamlFile):

        """
            Essentially just a function to update the variables accordingly
        """

        static_vars = list(vars(IntrinsicParameters))[2:-3]

        file = open(pathToYamlFile, 'r')
        config = yaml.safe_load(file)
        keys = config.keys()
        list_of_vars_in_config = list(keys)

        # Updating the static variables
        for var in list_of_vars_in_config:
            if var in static_vars:
                value = config[var]
                line = 'IntrinsicParameters.' + var + ' = lambda : '

                line += str(value)
                try:
                    exec(line)
                except:
                    warnings.warn('\n' + line + ' could not be executed. \nYou might have forgotten to import the library in the Programs/IntrinsicParameters.py file \n' )
            else:
                warnings.warn(var + ' is not a valid variable, could be a spelling issue')


IntrinsicParameters('inputs/IntrinsicParameters.yaml')








