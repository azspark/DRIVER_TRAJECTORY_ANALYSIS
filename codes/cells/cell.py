from .point import Point

class Cell:
    def __init__(self, point):
        self.point = point
        self.drivers = dict()  # record the appearance num of the driver_id
        self.trajectories = dict()
        self.sorted_drivers = None
        self.sorted_trajectories = None

    def add(self, driver_id, traj_id):
        if driver_id not in self.drivers:
            self.drivers[driver_id] = 0
        if traj_id not in self.trajectories:
            self.trajectories[traj_id] = 0
        self.drivers[driver_id] += 1
        self.trajectories[traj_id] += 1

        if self.sorted_drivers is not None or self.sorted_trajectories is not None:
            self.sorted_drivers = None
            self.sorted_trajectories = None

    def add_bulk(self, driver_ids, traj_ids):
        for driver_id, traj_id in zip(driver_ids, traj_ids):
            self.add(driver_id, traj_id)

    def topk_drivers(self, k=10):
        """

        :param k:
        :return: (list) topk driver_ids, in the format: [('driver1', 28), ('driver2', 15), ...]
        """
        if self.sorted_drivers is None or self.sorted_trajectories is None:
            self.sorted_drivers = sorted(self.drivers.items(), key=lambda x: x[1], reverse=True)
            # self.sorted_trajectories = sorted(self.trajectories.items(), key=lambda x: x[1], reverse=True)
        k = min(k, len(self.sorted_drivers))
        drivers = [self.sorted_drivers[i] for i in range(k)]
        # trajs = [self.sorted_trajectories[i][0] for i in range(k)]
        return drivers

    def __str__(self):
        return str(self.point)

    def __repr__(self):
        return str(self.point)

    def __eq__(self, other):
        if self.point == other.point:
            return True
        else:
            return False

    def __hash__(self):
        return hash(self.point)
