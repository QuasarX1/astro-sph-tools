# SPDX-FileCopyrightText: 2025-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: LicenseRef-NotYetLicensed

from ._SupportedIons import SupportedIons
from ...data_structures._IonisationTable import IonisationTableBase

import os
from functools import singledispatchmethod

import numpy as np
from unyt import unyt_array, unyt_quantity
import h5py as h5
from scipy.interpolate import CubicSpline



class IonisationTable_HM01(IonisationTableBase[[np.ndarray, np.ndarray, np.ndarray]]):
    supported_ions: tuple[SupportedIons, ...] = (
        SupportedIons.H_I,
        SupportedIons.He_I,
        SupportedIons.He_II,
        SupportedIons.C_I,
        SupportedIons.C_II,
        SupportedIons.C_III,
        SupportedIons.C_IV,
        SupportedIons.C_V,
        SupportedIons.C_VI,
        SupportedIons.N_II,
        SupportedIons.N_III,
        SupportedIons.N_IV,
        SupportedIons.N_V,
        SupportedIons.N_VI,
        SupportedIons.N_VII,
        SupportedIons.O_I,
        SupportedIons.O_III,
        SupportedIons.O_IV,
        SupportedIons.O_V,
        SupportedIons.O_VI,
        SupportedIons.O_VII,
        SupportedIons.O_VIII,
        SupportedIons.Ne_VIII,
        SupportedIons.Ne_IX,
        SupportedIons.Ne_X,
        SupportedIons.Mg_I,
        SupportedIons.Mg_II,
        SupportedIons.Al_I,
        SupportedIons.Al_II,
        SupportedIons.Al_III,
        SupportedIons.Si_II,
        SupportedIons.Si_III,
        SupportedIons.Si_IV,
        SupportedIons.Si_XIII,
        SupportedIons.S_V,
        SupportedIons.Fe_II,
        SupportedIons.Fe_III,
        SupportedIons.Fe_XVII
    )

    def __init__(self, ion: SupportedIons, directory: str) -> None:
        """
        gas_state -> 2D numpy array where each row contains the following:

            [0] = log10(hydrogen number density [cm**-3])

            [1] = log10(gas temperature [K])

            [2] = redshift
        """

        if ion not in IonisationTable_HM01.supported_ions:
            raise NotImplementedError(f"The HM01 table set contains no ionisation table for {ion}.")

        with h5.File(os.path.join(directory, ion.value) + ".hdf5") as file:
            table_axis_log10_temp: np.ndarray = file["logt"][:]
            table_axis_log10_hydrogen_numberdensity: np.ndarray = file["logd"][:]
            table_axis_redshift: np.ndarray = file["redshift"][:]
            table_values_ion_frac: np.ndarray = file["ionbal"][:]

            self.__cloudy_version: str = file["header"].attrs["cloudy_version"]
            self.__model_name: str = file["header/spectrum"].attrs["model_name"]
            self.__model_description: str = file["header/spectrum"].attrs["model_description"]

            self.__evaluation_redshifts: np.ndarray = file["header/spectrum/redshift"][:] # These appear to be the same as the redshifts from the "redshift" dataset!?
            self.__gammahi: unyt_array = unyt_array(file["header/spectrum/gammahi"][:], units = "s**-1") # H I Photoionisation rate as a function of redshift
            self.__energy: unyt_array = unyt_array(10**file["header/spectrum/logenergy_ryd"][:], units = "Ry")
            self.__flux: unyt_array = unyt_array(10**file["header/spectrum/logflux"][:], units = "erg/s/cm**2/Hz") # Flux as a function of energy and redshift

        super().__init__(
            table_values_ion_frac,
            table_axis_log10_hydrogen_numberdensity, table_axis_log10_temp, table_axis_redshift,
            redshift_input_index = 2
        )

        self.__gammahi_interpolator = CubicSpline(self.__evaluation_redshifts, self.__gammahi.value)

    @property
    def cloudy_version(self) -> str:
        return self.__cloudy_version

    @property
    def model_name(self) -> str:
        return self.__model_name

    @property
    def model_description(self) -> str:
        return self.__model_description

    @property
    def evaluation_redshifts(self) -> np.ndarray:
        return self.__evaluation_redshifts.copy()

    @property
    def gammahi(self) -> unyt_array:
        return self.__gammahi.copy()

    @property
    def energies(self) -> unyt_array:
        return self.__energy.copy()

    @property
    def fluxes(self) -> unyt_array:
        return self.__flux.copy()

    @singledispatchmethod
    def interpolate_gammahi(self, redshift: np.ndarray|float) -> unyt_array|unyt_quantity:
        raise TypeError(f"Unrecognised input type for redshift parameter: \"{type(redshift)}\".")
    @interpolate_gammahi.register(np.ndarray)
    def _(self, redshift: np.ndarray) -> unyt_array:
        return unyt_array(self.__gammahi_interpolator(redshift), units = "s**-1")
    @interpolate_gammahi.register(float|int)
    def _(self, redshift: float) -> unyt_quantity:
        return unyt_quantity(self.__gammahi_interpolator(np.array([redshift], dtype = float))[0], units = "s**-1")
