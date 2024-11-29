# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: LicenseRef-NotYetLicensed

from abc import abstractmethod
from collections.abc import Iterable, Iterator, Sized
from typing import Generic
import os

from ..._Interface import Interface, ensure_not_interface
from ...data_structures._SimulationData import ISimulationData, T_ISimulationData



class ISimulationFileTreeLeaf(Interface, Sized):
    def __new__(cls, *args, **kwargs):
        ensure_not_interface(cls, ISimulationFileTreeLeaf)
        return super().__new__(cls, *args, **kwargs)
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()
    @abstractmethod
    def load(self) -> ISimulationData:
        raise NotImplementedError()
    @property
    @abstractmethod
    def number(self) -> str:
        raise NotImplementedError()
    @property
    @abstractmethod
    def number_numerical(self) -> int:
        raise NotImplementedError()
    @property
    @abstractmethod
    def filepaths(self) -> tuple[str, ...]:
        raise NotImplementedError()
class SimulationFileTreeLeafBase(Generic[T_ISimulationData], ISimulationFileTreeLeaf):
    @abstractmethod
    def load(self) -> T_ISimulationData:
        raise NotImplementedError()



class ISimulationFileTree(Interface, Iterable, Sized):
    def __new__(cls, *args, **kwargs):
        ensure_not_interface(cls, ISimulationFileTree)
        return super().__new__(cls, *args, **kwargs)
    @abstractmethod
    def __iter__(self) -> Iterator[ISimulationFileTreeLeaf]:
        raise NotImplementedError()
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()
    @abstractmethod
    def __get_item__(self, key: int|slice) -> ISimulationFileTreeLeaf|tuple[ISimulationFileTreeLeaf, ...]:
        raise NotImplementedError()
    @abstractmethod
    def get_info(self) -> tuple[ISimulationFileTreeLeaf, ...]:
        raise NotImplementedError()
    @abstractmethod
    def get_numbers(self) -> tuple[str, ...]:
        raise NotImplementedError()
    def get_by_number(self, number: str) -> ISimulationFileTreeLeaf:
        raise NotImplementedError()
    def get_by_numbers(self, number: Iterable[str]) -> tuple[ISimulationFileTreeLeaf, ...]:
        raise NotImplementedError()
class SimulationFileTreeBase(Generic[T_ISimulationData], ISimulationFileTree):
    def __init__(self, directory: str):
        self.__directory = os.path.realpath(directory)
    @property
    def directory(self) -> str:
        return self.__directory
    @abstractmethod
    def __iter__(self) -> Iterator[SimulationFileTreeLeafBase[T_ISimulationData]]:
        raise NotImplementedError()
    @abstractmethod
    def __get_item__(self, key: int|slice) -> SimulationFileTreeLeafBase[T_ISimulationData]|tuple[SimulationFileTreeLeafBase[T_ISimulationData], ...]:
        raise NotImplementedError()
    @abstractmethod
    def get_info(self) -> tuple[SimulationFileTreeLeafBase[T_ISimulationData], ...]:
        raise NotImplementedError()
    @abstractmethod
    def get_by_number(self, number: str) -> SimulationFileTreeLeafBase[T_ISimulationData]:
        raise NotImplementedError()
    def get_by_numbers(self, number: Iterable[str]) -> tuple[SimulationFileTreeLeafBase[T_ISimulationData], ...]:
        return tuple([self.get_by_number(n) for n in number])
    @abstractmethod
    def find_file_number_from_redshift(self, redshift: float) -> str:
        raise NotImplementedError()
