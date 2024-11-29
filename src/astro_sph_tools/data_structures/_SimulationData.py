# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: LicenseRef-NotYetLicensed

from typing import Generic, TypeVar

from .._Interface import Interface, ensure_not_interface



class ISimulation(Interface):
    """
    Interface for types that specify data belonging to a named simulation dataset (e.g. EAGLE).
    """
    def __new__(cls, *args, **kwargss):
        ensure_not_interface(cls, ISimulation)
        return super().__new__(cls)

T_ISimulation = TypeVar("T_ISimulation", bound = ISimulation)



class ISimulationData(Interface):
    """
    Interface indicating types used for reading simulation data.
    """
    def __new__(cls, *args, **kwargs):
        ensure_not_interface(cls, ISimulationData)
        return super().__new__(cls, *args, **kwargs)

T_ISimulationData = TypeVar("T_ISimulationData", bound = ISimulationData)

class SimulationDataBase(Generic[T_ISimulation], ISimulationData):
    pass
