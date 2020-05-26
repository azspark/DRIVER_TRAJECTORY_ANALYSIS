from .vis_obj import Viser
import numpy as np

class OD(Viser):

    def __init__(self, trajectories, df_od):
        Viser.__init__(self, trajectories, df_od)

    def show_passed_driver(self):
        return self.df.groupby('LABEL').size()

    def vis_two_driver(self, driver_id1, driver_id2, max_num1, max_num2, with_marker=False, mark_od_point=True):
        df_driver1 = self.df[self.df['LABEL'] == driver_id1]
        df_driver2 = self.df[self.df['LABEL'] == driver_id2]
        idxs1 = list(df_driver1.index)
        idxs2 = list(df_driver2.index)
        if len(idxs1) > max_num1:
            idxs1 = np.random.choice(idxs1, max_num1, replace=False)
        if len(idxs2) > max_num2:
            idxs2 = np.random.choice(idxs2, max_num2, replace=False)
        return self._vis_trajs(idxs1, with_marker=with_marker, mark_od_point=mark_od_point, idxs2=idxs2)
