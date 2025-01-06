import numpy as np
import os
import jax.numpy as jnp
from jax import jit
import sys
from jax import config 
from binned_pzeta import Binned_P_zeta

config.update("jax_enable_x64", True)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    model_name = "log-normal-in-Pz"
    model_label = "Log Normal in Pz"
    model = Binned_P_zeta(model_name, model_label, norm="RD")
    f = np.geomspace(2e-5, 1., 100)

    def LISA_noise(f,
        # acceleration noise
        da_rms=3e-15, #m/s^2
        # optical metrology noise
        dx_rms=1.5e-11#m
                   ):
        # from LISA for Cosmologists,  https://arxiv.org/pdf/1908.00546.pdf
        f1 = 0.4e-3 # Hz
        c = 299792458.0 # m/s
        L = 2.5e9 # m
        fstar = c / (2*np.pi*L)

        # use 100 because OmegaGW is really Ω*h^2
        H0 = 100 # km/s / Mpc
        H0 *= 3.24078e-20 # convert to Hz
        S1 = 4*(da_rms/L)**2 * (1 + f1/f)**2 #s^(-4) Hz^(-1). prefactor should be 5.76e-48
        S2 = (dx_rms/L)**2 # should be 3.6e-41 Hz^-1
        SI = np.sqrt(2)*20/3*(S1/(2*np.pi*f)**4 + S2)*(1 + (f/(4*fstar/3))**2)
        return SI*4*np.pi**2*f**3 / (3*H0**2) # ΣΩ(f)

    lisa_noise = LISA_noise(f)
    lisa_mission_time = 4*365.25*24*3600
    import matplotlib.pyplot as plt
    import numpy as np

    class InteractivePlot:
        def __init__(self, x_values):
            self.xs = x_values
            self.ys = 1e-4*np.ones_like(x_values)

            self.fig, (self.ax_left, self.ax_right) = plt.subplots(1, 2, figsize=(10, 5))
            #self.line_left, = self.ax_left.loglog(self.xs, self.ys, 'ro-')
            self.line_left, = self.ax_left.loglog([], [], 'ro-')
            #self.line_right, = self.ax_right.loglog(f, model.template(f,np.log10(self.ys)), 'bo-')
            self.line_right, = self.ax_right.loglog([], [], 'bo-')
            self.noise_curve, = self.ax_right.loglog(f,lisa_noise/np.sqrt(lisa_mission_time))

            self.is_drawing = False

            self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.onpress)
            self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.onrelease)
            self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.onmove)
            self.kid = self.fig.canvas.mpl_connect('key_press_event', self.onkeypress)

            self.setup_plot()
            self.fig.canvas.draw()
            self.background_left = self.fig.canvas.copy_from_bbox(self.ax_left.bbox)
            self.background_right = self.fig.canvas.copy_from_bbox(self.ax_right.bbox)

        def setup_plot(self):
            self.ax_left.set_xlim(min(self.xs), max(self.xs))
            self.ax_left.set_ylim(1e-6, 1e-1)  # Adjust y-axis limits as needed
            self.ax_left.set_title('Pz')

            self.ax_right.set_xlim(min(f), max(f))
            self.ax_right.set_ylim(1e-16, 1e-7)  # Adjust y-axis limits as needed
            self.ax_right.set_title('OmegaGW h^2')

        def onpress(self, event):
            if event.inaxes == self.ax_left:
                self.is_drawing = True
                self.add_or_update_point(event.xdata, event.ydata)

        def onrelease(self, event):
            self.is_drawing = False

        def onmove(self, event):
            if self.is_drawing and event.inaxes == self.ax_left:
                self.add_or_update_point(event.xdata, event.ydata)

        def add_or_update_point(self, xdata, ydata):
            closest_x = min(self.xs, key=lambda x: abs(x - xdata))
            idx = np.where(self.xs == closest_x)[0][0]
            self.ys[idx] = ydata
            self.update_plot()

        def update_plot(self):
            self.line_left.set_data(self.xs, self.ys)
            self.update_right_plot()

            self.fig.canvas.restore_region(self.background_left)
            self.fig.canvas.restore_region(self.background_right)

            self.ax_left.draw_artist(self.line_left)
            self.ax_right.draw_artist(self.noise_curve)
            self.ax_right.draw_artist(self.line_right)

            self.fig.canvas.blit(self.ax_left.bbox)
            self.fig.canvas.blit(self.ax_right.bbox)

        def update_right_plot(self):
            ogw = model.template(f,np.log10(self.ys))
            snr = np.sqrt( np.trapz( (ogw/lisa_noise)**2 * lisa_mission_time, f))
            print(f"SNR (4yr): {snr}")
            self.line_right.set_data(f, ogw)
            #self.ax_right.set_title(f'OmegaGW h^2 SNR (4 yr): {snr}')

        def onkeypress(self, event):
            if event.key == 'enter':
                self.fig.canvas.mpl_disconnect(self.cid_press)
                self.fig.canvas.mpl_disconnect(self.cid_release)
                self.fig.canvas.mpl_disconnect(self.cid_motion)
                self.fig.canvas.mpl_disconnect(self.kid)
                plt.close(self.fig)

        def show(self):
            plt.show()

    x_values = model.fp
    interactive_plot = InteractivePlot(x_values)
    interactive_plot.show()

