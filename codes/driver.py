import folium
import numpy as np 
import datetime

class Driver:
    """Record and analyse driving behaviors of one driver"""
    
    def __init__(self, trajectories, df_driver, df_features):
        self.trajectories = trajectories
        self.df_driver = df_driver
        self.df_driver['DATE'] = df_driver.TIME.dt.date
        self.df_driver['HOUR'] = df_driver.TIME.dt.hour
        self.df_driver['StartPoint'] = df_driver.apply(lambda df: df['POLYLINE'][0], axis=1)

    
    def start_point_distribution_visulization(self, date=None, hour=None):
        pass

    def driver_date_records(self):
        return self.df_driver.groupby('DATE')['DATE'].count()

    def vis_trajs_date(self, max_shown_number, date, with_marker=False):
        """Visualization of driver trajectories of given date"""
        if not isinstance(date, datetime.date):
            if isinstance(date, list):
                date = datetime.date(date[0], date[1], date[2])
            else:
                raise TypeError("date should be either datetime.date object or list of date")
        df_given_date = self.df_driver[self.df_driver['DATE'] == date]
        idxs = list(df_given_date.index)
        if len(idxs) > max_shown_number:
            idxs = np.random.choice(idxs, max_shown_number, replace=False)
        return self._vis_trajs(idxs, with_marker)

    def vis_trajs_hour(self, max_shown_number, hour_range, with_marker=False):
        """Visualization of driver trajectories of given hours from different date"""
        s, e = hour_range
        df_given_hours = self.df_driver[(self.df_driver['HOUR'] >= s) & (self.df_driver['HOUR'] <= e)]
        idxs = list(df_given_hours.index)
        print(idxs)
        if len(idxs) > max_shown_number:
            idxs = np.random.choice(idxs, max_shown_number, replace=False)
        return self._vis_trajs(idxs, with_marker)

    
    def _vis_trajs(self, idxs, with_marker):
        start_coords = self.df_driver.loc[idxs]['StartPoint'].values
        start_coords = np.array([[i[0], i[1]] for i in start_coords])
        middle_point = start_coords.mean(axis=0)

        m = folium.Map(location=(middle_point[1], middle_point[0]), zoom_start=13)
        for idx in idxs:
            self.trajectories[idx].vis(with_marker, osm_map=m, time=self.df_driver.loc[idx]['TIME'])
        return m