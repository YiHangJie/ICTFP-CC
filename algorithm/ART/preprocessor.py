# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements the abstract base class for defences that pre-process input data.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import abc
from typing import List, Optional, Tuple, Any, TYPE_CHECKING

import numpy as np

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

import algorithm.ART.config

class Preprocessor(abc.ABC):
    """
    Abstract base class for preprocessing defences.

    By default, the gradient is estimated using BPDA with the identity function.
        To modify, override `estimate_gradient`
    """

    params: List[str] = []

    def __init__(self, is_fitted: bool = False, apply_fit: bool = True, apply_predict: bool = True) -> None:
        """
        Create a preprocessing object.

        Optionally, set attributes.
        """
        self._is_fitted = bool(is_fitted)
        self._apply_fit = bool(apply_fit)
        self._apply_predict = bool(apply_predict)

    @property
    def is_fitted(self) -> bool:
        """
        Return the state of the preprocessing object.

        :return: `True` if the preprocessing model has been fitted (if this applies).
        """
        return self._is_fitted

    @property
    def apply_fit(self) -> bool:
        """
        Property of the defence indicating if it should be applied at training time.

        :return: `True` if the defence should be applied when fitting a model, `False` otherwise.
        """
        return self._apply_fit

    @property
    def apply_predict(self) -> bool:
        """
        Property of the defence indicating if it should be applied at test time.

        :return: `True` if the defence should be applied at prediction time, `False` otherwise.
        """
        return self._apply_predict

    @abc.abstractmethod
    def __call__(self, x: np.ndarray, y: Optional[Any] = None) -> Tuple[np.ndarray, Optional[Any]]:
        """
        Perform data preprocessing and return preprocessed data as tuple.

        :param x: Dataset to be preprocessed.
        :param y: Labels to be preprocessed.
        :return: Preprocessed data.
        """
        raise NotImplementedError

    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> None:
        """
        Fit the parameters of the data preprocessor if it has any.

        :param x: Training set to fit the preprocessor.
        :param y: Labels for the training set.
        :param kwargs: Other parameters.
        """
        pass

    def estimate_gradient(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:  # pylint: disable=W0613,R0201
        """
        Provide an estimate of the gradients of the defence for the backward pass. If the defence is not differentiable,
        this is an estimate of the gradient, most often replacing the computation performed by the defence with the
        identity function (the default).

        :param x: Input data for which the gradient is estimated. First dimension is the batch size.
        :param grad: Gradient value so far.
        :return: The gradient (estimate) of the defence.
        """
        return grad

    def set_params(self, **kwargs) -> None:  # pragma: no cover
        """
        Take in a dictionary of parameters and apply checks before saving them as attributes.
        """
        for key, value in kwargs.items():
            if key in self.params:
                setattr(self, key, value)
        self._check_params()

    def _check_params(self) -> None:  # pragma: no cover
        pass

    def forward(self, x: Any, y: Any = None) -> Tuple[Any, Any]:
        """
        Perform data preprocessing and return preprocessed data.

        :param x: Dataset to be preprocessed.
        :param y: Labels to be preprocessed.
        :return: Preprocessed data.
        """
        raise NotImplementedError