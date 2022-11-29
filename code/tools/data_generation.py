import numpy as np
from scipy import signal
from numba import njit
import colorednoise as cn


class ConfigGenerator:
    names = [
        "betas",
        "gammas",
        "beta_sharpness",
        "phase",
        "pac",
        "cross_pac",
        "phase_shift",
        "burst_length",
    ]

    def __init__(self):
        pass

    @staticmethod
    def generate_empty_config():
        config = {}
        for el in ConfigGenerator.names:
            config[el] = None

        return config

    @staticmethod
    def generate_random_config():

        activate = np.random.binomial(1, 0.2, len(ConfigGenerator.names)) > 0.5

        config = {}
        for jj, active in enumerate(list(activate)):
            if active:
                vals = np.random.uniform(0, 1, 2)
                config[ConfigGenerator.names[jj]] = list(vals)
            else:
                config[ConfigGenerator.names[jj]] = None
        if np.array([val == None for val in config.values()]).all():
            return ConfigGenerator.generate_random_config()
        return config


@njit()
def hann_sample(x, intensity=1.0):
    return intensity * 0.6 * np.sin(2 * np.pi * (x - 0.75) / 4) ** 3 * -1


@njit()
def phase_distort(x, intensity=1.0):
    y = np.zeros_like(x)
    for jj in range(x.size):
        y[jj] = x[jj] + hann_sample(x[jj], intensity=intensity)
    return y


class DataGenerator:
    def __init__(
        self,
        cfg,
        snr=10.0,
        transition_width=0.05,
        T: int = 60 * 20,
        Fs=2048,
        phase_method="window",
        gamma_mode="burst",
    ):
        """Object that generates a signals that probabilistically transitions between two states whose features are defined by a configuration. Data is generated on __init__.

        Args:
            cfg (dict[str,list[float]|None]): Configuration object defining whether signals features are included and how their intensity differs across states.
            snr (float, optional): Signal-to-noise ratio (signals power / noise power), noise is 1/f additive colored noise. Defaults to 10.
            transition_width (float, optional): Duration of linear transition period between states [in seconds], avoids abrupt changes in the signal. Defaults to 0.05.
            T (int, optional): Total duration of generated data [in seconds]. Defaults to 60*20 (20 minutes). Each state has a minimum duration of 5 seconds.
            Fs (int, optional): Sampling rate of data. Defaults to 2048.
            phase_method (str, optional): "sine" or "window". Defaults to "window".
            gamma_mode (str, optional): "burst" or "continuous". Defaults to "burst".

        """
        if T % 5 != 0:
            raise ValueError("T must be a multiple of 10 seconds (10, 20, 30, ...)")

        self.cfg = cfg
        self.fs = Fs
        self.T = T
        self.transition_width = transition_width

        self.t = np.linspace(0, T, T * self.fs)
        self.envelope = np.zeros_like(self.t)

        # Create states as 1st order Markov process:
        self.inertia0, self.inertia1 = np.random.uniform(0.6, 0.7), np.random.uniform(
            0.6, 0.7
        )
        self.transitions = np.array(
            [[self.inertia0, 1 - self.inertia0], [1 - self.inertia1, self.inertia1]]
        )
        self.generate_states()

        # Generate waveforms:
        self.sigs, self.sigs2 = [], []

        self.generate_beta()
        self.generate_gamma()
        self.generate_sharp_beta()
        self.generate_phase_distorted(phase_method)
        self.generate_pac(gamma_mode)
        self.generate_cross_ch_pac(gamma_mode)
        self.generate_phase_shifted()
        self.generate_bursts()

        # Merge signals into final sig & add noise:
        self.signal_clean = np.stack(self.sigs).sum(0)
        noise_level = (
            np.sqrt((self.signal_clean**2).sum() / self.signal_clean.size) / snr
        )
        noise = cn.powerlaw_psd_gaussian(1, self.signal_clean.shape) * noise_level
        self.signals = self.signal_clean + noise

        if self.sigs2:
            self.signal2_clean = np.stack(self.sigs2).sum(0)
            noise_level = (
                np.sqrt((self.signal2_clean**2).sum() / self.signal2_clean.size) / snr
            )
            noise = cn.powerlaw_psd_gaussian(1, self.signal2_clean.shape) * noise_level
            self.signal2 = self.signal2_clean + noise

            self.signals = np.stack((self.signals, self.signal2))

        if self.signals.ndim < 2:
            self.signals = self.signals[np.newaxis, ...]

        self.label = (self.envelope > 0.5).astype(float)

    def generate_beta(self):

        if self.cfg["betas"] is not None:
            beta0 = np.sin(2 * np.pi * self.t * 25) * self.cfg["betas"][0]
            beta1 = np.sin(2 * np.pi * self.t * 25) * self.cfg["betas"][1]

            self.sigs.append(beta1 * self.envelope + beta0 * (1 - self.envelope))

    def generate_gamma(self):

        if self.cfg["gammas"] is not None:
            gamma0 = np.sin(2 * np.pi * self.t * 70) * self.cfg["gammas"][0]
            gamma1 = np.sin(2 * np.pi * self.t * 70) * self.cfg["gammas"][1]

            self.sigs.append(gamma1 * self.envelope + gamma0 * (1 - self.envelope))

    def generate_sharp_beta(self):

        if self.cfg["beta_sharpness"] is not None:
            triangle = signal.sawtooth(2 * np.pi * 21 * self.t + np.pi / 2, width=0.5)
            sinusoid = np.sin(2 * np.pi * self.t * 21)

            sharp0 = triangle * self.cfg["beta_sharpness"][0] + sinusoid * (
                1 - self.cfg["beta_sharpness"][0]
            )
            sharp1 = triangle * self.cfg["beta_sharpness"][1] + sinusoid * (
                1 - self.cfg["beta_sharpness"][1]
            )

            w0, h0 = signal.periodogram(sharp0, fs=self.fs)
            w1, h1 = signal.periodogram(sharp1, fs=self.fs)

            assert np.array_equal(w0, w1), "Error generating sharp waveforms"

            beta_ix = np.argmin(np.abs(w0 - 21))
            ratio = h0[beta_ix] / h1[beta_ix]

            sharp1 *= np.sqrt(ratio)

            self.sigs.append(sharp1 * self.envelope + sharp0 * (1 - self.envelope))

    def generate_phase_distorted(self, phase_method: str):

        if self.cfg["phase"] is not None:
            base = signal.sawtooth(2 * np.pi * 28 * self.t, width=1)
            if phase_method == "sine":
                base_sine = 0.25 * np.sin(0.5 * 2 * np.pi * self.t + np.pi)
                bases = [
                    base + intensity * base_sine for intensity in self.cfg["phase"]
                ]

            elif phase_method == "window":
                bases = [
                    phase_distort(base, intensity=intensity)
                    for intensity in self.cfg["phase"]
                ]
            else:
                raise ValueError("phase_method not recognized")

            bases = [(base + 1) * np.pi for base in bases]

            phase0, phase1 = np.sin(bases[0]), np.sin(bases[1])

            w0, h0 = signal.periodogram(phase0, fs=self.fs)
            w1, h1 = signal.periodogram(phase1, fs=self.fs)

            assert np.array_equal(w0, w1), "Error generating phase distortions"

            beta_ix = np.argmin(np.abs(w0 - 28))

            ratio = h0[beta_ix] / h1[beta_ix]
            phase1 *= np.sqrt(ratio)

            self.sigs.append(phase1 * self.envelope + phase0 * (1 - self.envelope))

    def generate_pac(self, gamma_mode: str):

        if self.cfg["pac"] is not None:

            theta = np.sin(2 * np.pi * 8 * self.t)
            gamma = 0.7 * np.sin(2 * np.pi * 70 * self.t)

            # Generate Gamma waveform:
            gamma_pac = gamma * theta * (theta > 0).astype(float)

            if gamma_mode == "burst":
                b, a = signal.butter(4, [7, 15], "bandpass", fs=self.fs)
                gamma_burst = signal.filtfilt(b, a, 5 * np.random.randn(*self.t.shape))
                gamma_burst += 0
                gamma_burst = gamma * (gamma_burst * (gamma_burst > 0).astype(float))
            elif gamma_mode == "continuous":
                gamma_burst = gamma
            else:
                raise ValueError("gamma_mode (burst or continuous) must bu specified")

            w0, h0 = signal.periodogram(gamma_pac, fs=self.fs)
            _, h1 = signal.periodogram(gamma_burst, fs=self.fs)

            gamma_ix = np.argmin(np.abs(w0 - 70))
            ratio = h1[gamma_ix] / h0[gamma_ix]

            gamma_pac *= np.sqrt(ratio)

            pac = theta + gamma_pac
            no_pac = theta + gamma_burst

            pac0 = pac * self.cfg["pac"][0] + no_pac * (1 - self.cfg["pac"][0])
            pac1 = pac * self.cfg["pac"][1] + no_pac * (1 - self.cfg["pac"][1])

            self.sigs.append(pac1 * self.envelope + pac0 * (1 - self.envelope))

    def generate_cross_ch_pac(self, gamma_mode: str):

        if self.cfg["cross_pac"] is not None:
            theta = np.sin(2 * np.pi * 8 * self.t)
            gamma = 0.6 * np.sin(2 * np.pi * 70 * self.t)

            gamma_pac = gamma * theta * (theta > 0).astype(float)

            if gamma_mode == "burst":
                b, a = signal.butter(4, [7, 15], "bandpass", fs=self.fs)
                gamma_burst = signal.filtfilt(b, a, 5 * np.random.randn(*self.t.shape))
                gamma_burst += 0
                gamma_burst = gamma * (gamma_burst * (gamma_burst > 0).astype(float))
            elif gamma_mode == "continuous":
                gamma_burst = gamma
            else:
                raise ValueError("gamma_mode (burst or continuous) must bu specified")

            w0, h0 = signal.periodogram(gamma_pac, fs=self.fs)
            _, h1 = signal.periodogram(gamma_burst, fs=self.fs)

            gamma_ix = np.argmin(np.abs(w0 - 70))
            ratio = h1[gamma_ix] / h0[gamma_ix]

            gamma_pac *= np.sqrt(ratio)

            gamma0 = (
                self.cfg["cross_pac"][0] * gamma_pac
                + (1 - self.cfg["cross_pac"][0]) * gamma_burst
            )
            gamma1 = (
                self.cfg["cross_pac"][1] * gamma_pac
                + (1 - self.cfg["cross_pac"][1]) * gamma_burst
            )

            self.sigs.append(theta)
            self.sigs2.append(self.envelope * gamma1 + (1 - self.envelope) * gamma0)

    def generate_phase_shifted(self):

        if self.cfg["phase_shift"] is not None:

            b, a = signal.butter(2, 6, fs=self.fs)
            f = signal.filtfilt(b, a, 10 * np.random.rand(*self.t.shape))

            base = np.sin(2 * np.pi * self.t * f)

            v0 = np.sin(2 * np.pi * self.t * f + np.pi * self.cfg["phase_shift"][0])
            v1 = np.sin(2 * np.pi * self.t * f + np.pi * self.cfg["phase_shift"][1])

            self.sigs.append(base)
            self.sigs2.append(self.envelope * v1 + (1 - self.envelope) * v0)

    def generate_bursts(self):

        if self.cfg["burst_length"] is not None:
            beta = np.sin(2 * np.pi * 22 * self.t)

            # Bursts::
            burst0 = self.cfg["burst_length"][0]
            b, a = signal.butter(
                2, [1.25 - burst0, (1.25 - burst0) * 5], "bandpass", fs=self.fs
            )
            burst_envelope_0 = signal.filtfilt(b, a, 7 * np.random.randn(*self.t.shape))
            burst_envelope_0 += 0.15 * burst0
            bursty_beta_0 = beta * (
                burst_envelope_0 * (burst_envelope_0 > 0).astype(float)
            )

            burst1 = self.cfg["burst_length"][1]
            b, a = signal.butter(
                2, [1.25 - burst1, (1.25 - burst1) * 5], "bandpass", fs=self.fs
            )
            burst_envelope_1 = signal.filtfilt(b, a, 7 * np.random.randn(*self.t.shape))
            burst_envelope_1 += 0.15 * burst1
            bursty_beta_1 = beta * (
                burst_envelope_1 * (burst_envelope_1 > 0).astype(float)
            )

            w0, h0 = signal.periodogram(bursty_beta_0, fs=self.fs)
            _, h1 = signal.periodogram(bursty_beta_1, fs=self.fs)

            beta_ix = np.argmin(np.abs(w0 - 22))
            ratio = h1[beta_ix] / h0[beta_ix]

            bursty_beta_0 *= np.sqrt(ratio)

            self.sigs.append(
                bursty_beta_1 * self.envelope + (1 - self.envelope) * bursty_beta_0
            )

    def generate_states(self):

        state = 0
        for n in range(1, self.T // 5):
            if state == 0:
                new_state = np.random.binomial(1, self.inertia0) > 0.5
            elif state == 1:
                new_state = np.random.binomial(1, self.inertia1) > 0.5
            else:
                raise RuntimeError("Error generating Markov chain")

            self.envelope[n * 5 * self.fs : (n + 1) * 5 * self.fs] = new_state

            if new_state > state:
                self.envelope[
                    n * 5 * self.fs
                    - int(self.transition_width / 2 * self.fs) : n * 5 * self.fs
                    + int(self.transition_width / 2 * self.fs)
                ] = np.linspace(0, 1, int(self.transition_width * self.fs))
            elif new_state < state:
                self.envelope[
                    n * 5 * self.fs
                    - int(self.transition_width / 2 * self.fs) : n * 5 * self.fs
                    + int(self.transition_width / 2 * self.fs)
                ] = np.linspace(1, 0, int(self.transition_width * self.fs))

            state = new_state
