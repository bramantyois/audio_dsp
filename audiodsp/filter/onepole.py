import numpy as np

from ..dsp_module import DSPModule
from ..data_model import Parameters, Coefficients


class OnePoleLP(DSPModule):
    def __init__(self, sample_rate=44100, fc=1000, name="ExampleOnePoleLP"):
        super().__init__(sample_rate)

        self.parameters.name = name
        self.parameters.module_name = "OnePoleLP"
        self.parameters.values = {"cutoff_freq": fc}

        self.coefficients.name = name
        self.coefficients.module_name = "OnePoleLP"
        self.coefficients.values = {"a": 0, "b": 0}

        self.reset_buffer()
        self.calc_coefficients()

    def reset_buffer(self):
        self.buffer = 0

    def calc_coefficients(self):
        fc = self.parameters.values["cutoff_freq"]
        if fc < 0:
            fc = 0
        elif fc > 20000:
            fc = 20000

        a = np.tan(np.pi * fc / self.sample_rate)
        b = 1.0 / (1.0 + a)

        self.coefficients.values["a"] = a
        self.coefficients.values["b"] = b

    def process_sample(self, x: float):
        output = (
            self.buffer + self.coefficients.values["a"] * x
        ) * self.coefficients.values["b"]
        self.buffer = self.coefficients.values["a"] * (x - output) + output

        return output

    def process_block(self, x: np.ndarray) -> np.ndarray:
        y = np.zeros_like(x)
        for i in range(len(x)):
            y[i] = self.process_sample(x[i])

        return y

    def process_block_with_param(
        self, x: np.ndarray, parameters: Parameters
    ) -> np.ndarray:
        # check if parameters are different
        if self.parameters.values != parameters.parameters:
            self.parameters.values = parameters.parameters
            self.calc_coefficients()
        self.reset_buffer()

        return self.process_block(x)


class OnePoleHP(DSPModule):
    def __init__(self, sample_rate=44100, fc=1000, name="ExampleOnePoleHP"):
        super().__init__(sample_rate)

        self.parameters.name = name
        self.parameters.module_name = "OnePoleHP"
        self.parameters.values = {"cutoff_freq": fc}

        self.coefficients.name = name
        self.coefficients.module_name = "OnePoleHP"
        self.coefficients.values = {"a": 0, "b": 0}

        self.reset_buffer()
        self.calc_coefficients()

    def reset_buffer(self):
        self.buffer = 0

    def calc_coefficients(self):
        fc = self.parameters.values["cutoff_freq"]
        if fc < 0:
            fc = 0
        elif fc > 20000:
            fc = 20000

        a = np.tan(np.pi * fc / self.sample_rate)
        b = 1.0 / (1.0 + a)

        self.coefficients.values["a"] = a
        self.coefficients.values["b"] = b

    def process_sample(self, x: float):
        output = (
            self.buffer + self.coefficients.values["a"] * x
        ) * self.coefficients.values["b"]
        self.buffer = self.coefficients.values["a"] * (x - output) + output

        return x - output

    def process_block(self, x: np.ndarray) -> np.ndarray:
        y = np.zeros_like(x)
        for i in range(len(x)):
            y[i] = self.process_sample(x[i])

        return y

    def process_block_with_param(
        self, x: np.ndarray, parameters: Parameters
    ) -> np.ndarray:
        assert parameters.module_name == self.parameters.module_name

        old_params = self.parameters.values
        self.parameters.values = parameters.values

        self.calc_coefficients()
        self.reset_buffer()

        y = self.process_block(x)

        self.parameters.values = old_params

        return y


if __name__ == "__main__":
    from utils import generate_sine_sweep
    import matplotlib.pyplot as plt

    sr = 44100

    input_x = generate_sine_sweep(duration=0.5)

    hp = OnePoleHP(sr)

    filtered = hp.process_block_w_param(input_x, fc=80)

    plt.plot(filtered, label="filtered")
    plt.plot(input_x, alpha=0.2, label="input")
    plt.legend()
    plt.show()
