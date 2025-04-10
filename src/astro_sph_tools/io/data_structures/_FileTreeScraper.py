# SPDX-FileCopyrightText: 2025-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: LicenseRef-NotYetLicensed

from abc import ABC, abstractmethod
import os
import re
from collections.abc import Iterable, Iterator, Collection
from typing import TypeVar, Generic
from functools import singledispatchmethod

import numpy as np
from QuasarCode import Console

from ...data_structures._SimulationData import T_ISimulation, SimulationDataBase#, T_ISimulationData
from ._tree_structures import SimulationFileTreeBase, SimulationFileTreeLeafBase



class FileTreeScraperBase(ABC, Generic[T_ISimulation]):
    def __init__(self, directories: dict[str, str], skip_snapshot_numbers: list[str]|None = None, skip_snipshot_numbers: list[str]|None = None) -> None:
        self.__root_directories:      dict[str, str]  = directories.copy()
        self.__skip_snapshot_numbers: tuple[str, ...] = tuple(skip_snapshot_numbers) if skip_snapshot_numbers is not None else tuple()
        self.__skip_snipshot_numbers: tuple[str, ...] = tuple(skip_snipshot_numbers) if skip_snipshot_numbers is not None else tuple()

    @property
    def root_directories(self) -> dict[str, str]:
        return self.__root_directories.copy()

    @property
    def skipped_snapshot_numbers(self) -> tuple[str, ...]:
        return self.__skip_snapshot_numbers

    @property
    def skipped_snipshot_numbers(self) -> tuple[str, ...]:
        return self.__skip_snipshot_numbers

    @property
    @abstractmethod
    def snapshots(self) -> SimulationFileTreeBase[SimulationDataBase[T_ISimulation]]:
        raise NotImplementedError("Attempted to call an abstract method.")

    @property
    @abstractmethod
    def snipshots(self) -> SimulationFileTreeBase[SimulationDataBase[T_ISimulation]]:
        raise NotImplementedError("Attempted to call an abstract method.")

    @property
    @abstractmethod
    def catalogues(self) -> SimulationFileTreeBase[SimulationDataBase[T_ISimulation]]:
        raise NotImplementedError("Attempted to call an abstract method.")

    @property
    @abstractmethod
    def snipshot_catalogues(self) -> SimulationFileTreeBase[SimulationDataBase[T_ISimulation]]:
        raise NotImplementedError("Attempted to call an abstract method.")
