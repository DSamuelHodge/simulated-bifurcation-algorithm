from typing import Optional, Tuple, Union

import numpy as np
import torch

from .abc_model import ABCModel


class QUBO(ABCModel):

    """
    Quadratic Unconstrained Binary Optimization

    Given a matrix `Q` the value to minimize is the quadratic form
    `ΣΣ Q(i,j)b(i)b(j)` where the `b(i)`'s values are either `0` or `1`.
    """

    input_type = "binary"

    def __init__(
        self,
        Q: Union[torch.Tensor, np.ndarray],
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> None:
        super().__init__(Q, dtype=dtype, device=device)
        self.Q = self[2]
