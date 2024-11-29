# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: LicenseRef-NotYetLicensed

from abc import ABC
from typing import TypeVar

class Interface(ABC):
    """
    Base class for interface types.
    """
    def __new__(cls, *args, **kwargss):
        ensure_not_interface(cls, Interface)
        return super().__new__(cls)

T_Interface = TypeVar("T_Interface", bound = Interface)
U_Interface = TypeVar("U_Interface", bound = Interface)

def check_interface(cls: type[U_Interface], interface_type: type[T_Interface]) -> bool:
    return cls is interface_type
def ensure_not_interface(cls: type[U_Interface], interface_type: type[T_Interface]):
    if check_interface(cls, interface_type):
        raise TypeError(f"Abstract interface type {interface_type.__name__} cannot be instantiated. To use this type, create an instance of a subclass.")
