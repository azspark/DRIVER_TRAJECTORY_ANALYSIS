from .vis_obj import Viser
import numpy as np

class OD(Viser):

    def __init__(self, trajectories, df_od):
        Viser.__init__(self, trajectories, df_od)

