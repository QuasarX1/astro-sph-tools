# SPDX-FileCopyrightText: 2025-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: LicenseRef-NotYetLicensed

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, ParamSpec
from collections.abc import Callable, Sequence

from .._Interface import Interface, ensure_not_interface

import numpy as np
from scipy.interpolate import RegularGridInterpolator



P = ParamSpec("P", bound = np.ndarray)

class IIonisationTable(Interface, Generic[P], ABC):
    """
    Interface for types that provide ionisation state.
    """
    def __new__(cls, *args, **kwargs):
        ensure_not_interface(cls, IIonisationTable)
        return super().__new__(cls)
    
    @abstractmethod
    def __call__(self, gas_state: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Attempted to call an abstract method.")

class IonisationTableBase(IIonisationTable[P]):
    def __init__(self, table: np.ndarray, *table_positions: P.args, redshift_input_index: int = -1) -> None:
        self.__n_input_dimensions = len(table_positions)
        self.__dimension_index_of_redshift = redshift_input_index if redshift_input_index >= 0 else (self.__n_input_dimensions - redshift_input_index)

        if self.__n_input_dimensions == 0:
            raise IndexError("No input dimensions were specified for table interpolation construction.")
        
        if len(table.shape) != self.__n_input_dimensions:
            raise IndexError(f"Interpolation table has {len(table.shape)} dimensions but {self.__n_input_dimensions} arrays were used to specify the table positions.")

        self.__table_positions: tuple[np.ndarray] = table_positions
        self.__table: np.ndarray = table

        self.__interpolator = RegularGridInterpolator(
            self.__table_positions,
            self.__table,
            bounds_error = False,
            fill_value = -np.inf
        )

    def __call__(self, gas_state: np.ndarray) -> np.ndarray:
        return self.__interpolator(gas_state)

    def evaluate_at_redshift(self, gas_state: np.ndarray, redshift: float) -> np.ndarray:
        formatted_gas_state = np.empty(shape = (gas_state.shape[0], gas_state.shape[1] + 1), dtype = float)
        formatted_gas_state[:, np.arange(self.__n_input_dimensions) != self.__dimension_index_of_redshift] = gas_state[:, :]
        formatted_gas_state[:, self.__dimension_index_of_redshift] = redshift
        return self.__interpolator(formatted_gas_state)

    @property
    def number_of_input_dimensions(self) -> int:
        return self.__n_input_dimensions

    @property
    def ionisation_fraction_table(self) -> np.ndarray:
        return self.__table.copy()

    def get_table_dimension(self, dimension: int) -> np.ndarray:
        return self.__table_positions[dimension].copy()
