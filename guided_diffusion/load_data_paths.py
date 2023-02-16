import os

os.environ["EVALUATION_SYSTEM_CONFIG_FILE"] = "/work/bm1159/XCES/freva/evaluation_system.conf"
os.environ["EVALUATION_SYSTEM_CONFIG_DIR"] = "/work/bm1159/XCES/freva"

import freva


def load_paths(project='cmip6', model='mpi-esm1-2-lr', experiment='historical', time_frequency='mon', variable='tas',
               ensemble='r1i2000p1f1', realm='atmos'):
    #paths = ['/home/joe/PycharmProjects/volai/data/input/{}/{}_ens{}.nc'.format(variable, variable, ensemble)]
    #return paths
    return freva.databrowser(project=project, model=model, experiment=experiment, time_frequency=time_frequency,
                             variable=variable, ensemble=ensemble, realm=realm)
