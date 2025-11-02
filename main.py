import time
import numpy as np
from power import PowerSupply


class MagneticControl:
    def __init__(self, 
                 port_x='/dev/ttyUSB0', port_y='/dev/ttyUSB1', 
                 run_time=10, x=1.0, y=0, B0=5e-3, B_bias=1e-3, f=2,
                 alpha_x=0.5e-3, alpha_y=0.5e-3, alpha_z=0.5e-3):
        self.power_x = PowerSupply(port_x)
        self.power_y = PowerSupply(port_y)
        self.run_time = run_time
        self.x = x
        self.y = y
        self.B0 = B0
        self.B_bias = B_bias
        self.f = f
        self.alpha_x = alpha_x
        self.alpha_y = alpha_y
        self.alpha_z = alpha_z

    def send_current_x(self, Ix, Iy, Iz):
        print(f"Ix={Ix:.2f} A, Iy={Iy:.2f} A, Iz={Iz:.2f} A")
        self.power_x.set_current(abs(Ix))
        self.power_y.set_current(abs(Iy))

    def run(self,):
        w = 2*np.pi*self.f

        ux, uy = self.x/np.hypot(self.x, self.y), self.y/np.hypot(self.x, self.y)
        t0 = time.time()

        while time.time() - t0 < self.run_time:
            t = time.time() - t0

            Bx = self.B0 * uy * np.sin(w*t) + self.B_bias * ux
            By = -self.B0 * ux * np.sin(w*t) + self.B_bias * uy
            Bz = self.B0 * np.cos(w*t)

            self.send_current(Bx/self.alpha_x, By/self.alpha_y, Bz/self.alpha_z)
            time.sleep(0.001)


if __name__ == "__main__":
    ctrl = MagneticControl()
    ctrl.run()
