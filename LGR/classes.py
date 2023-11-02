
import numpy as np


class SimpleParticle:

    def __init__(self, pos, time, idx=[], vel=[]) -> None:

        # Set Attributes
        self.idx = idx              # Particle Index Number
        self.t = np.array(time)     # vector of times of snapshots
        self.pos = np.array(pos)    # vector of locations of the particle
        self.vel = np.array(vel)    # vector of particle velocities

    def update(self, pos, time, vel=None):

        # Update time and position
        self.pos = np.append(self.pos, pos, axis=0)
        self.t = np.vstack((self.t, time))
        if vel is not None:
            self.vel = np.append(self.vel, vel, axis=0)
