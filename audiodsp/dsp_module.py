import numpy as np
from .data_model import Parameters, Coefficients


class DSPModule:
    """Base class for all DSP modules."""
    def __init__(
            self, 
            sample_rate: int = 44100,
            name: str = 'ExampleModuleName',
            module_name: str = 'ModuleName') -> None:
        self.sample_rate = sample_rate
        self.parameters = Parameters(name=name, module_name=module_name)
        self.coefficients = Coefficients(name=name, module_name=module_name)  

    def reset_buffer(self) -> None:
        raise NotImplementedError   
    
    def process_sample(self, x: float) -> float:    
        raise NotImplementedError
    
    def process_block(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def process_block_with_param(self, x: np.ndarray, parameters: Parameters) -> np.ndarray:
        raise NotImplementedError
    
    def get_parameters(self) -> Parameters:
        return self.parameters 