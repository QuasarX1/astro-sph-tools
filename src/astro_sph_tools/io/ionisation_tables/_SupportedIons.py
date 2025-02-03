# SPDX-FileCopyrightText: 2025-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: LicenseRef-NotYetLicensed

from enum import Enum

from mendeleev import element as get_element, H, He, C, N, O, Ne, Mg, Al, Si, S, Fe
from mendeleev.models import Element

_ions_by_element: dict[Element, dict[int, "SupportedIons"]] = {}
_ion_info: dict["SupportedIons", tuple[Element, int, str]] = {}

class SupportedIons(Enum):
    H_I     = "h1"
    He_I    = "he1"
    He_II   = "he2"
    C_I     = "c1"
    C_II    = "c2"
    C_III   = "c3"
    C_IV    = "c4"
    C_V     = "c5"
    C_VI    = "c6"
    N_II    = "n2"
    N_III   = "n3"
    N_IV    = "n4"
    N_V     = "n5"
    N_VI    = "n6"
    N_VII   = "n7"
    O_I     = "o1"
    O_III   = "o3"
    O_IV    = "o4"
    O_V     = "o5"
    O_VI    = "o6"
    O_VII   = "o7"
    O_VIII  = "o8"
    Ne_VIII = "ne8"
    Ne_IX   = "ne9"
    Ne_X    = "ne10"
    Mg_I    = "mg1"
    Mg_II   = "mg2"
    Al_I    = "al1"
    Al_II   = "al2"
    Al_III  = "al3"
    Si_II   = "si2"
    Si_III  = "si3"
    Si_IV   = "si4"
    Si_XIII = "si13"
    S_V     = "s5"
    Fe_II   = "fe2"
    Fe_III  = "fe3"
    Fe_XVII = "fe17"

    @property
    def element(self) -> Element:
        return _ion_info[self][0]

    @property
    def ionisation_state(self) -> int:
        return _ion_info[self][1]

    @property
    def symbol(self) -> str:
        return _ion_info[self][2]

    def __str__(self) -> str:
        return self.symbol

    def __repr__(self) -> str:
        return f"Ion {self.value}"

    @staticmethod
    def get_ions_of_element(element: Element|int|str) -> dict[int, "SupportedIons"]:
        return _ions_by_element[element if isinstance(element, Element) else get_element(element)].copy()

_ions_by_element[H] = {
    1:  SupportedIons.H_I,
}
_ions_by_element[He] = {
    1:  SupportedIons.He_I,
    2:  SupportedIons.He_II,
}
_ions_by_element[C] = {
    1:  SupportedIons.C_I,
    2:  SupportedIons.C_II,
    3:  SupportedIons.C_III,
    4:  SupportedIons.C_IV,
    5:  SupportedIons.C_V,
    6:  SupportedIons.C_VI,
}
_ions_by_element[N] = {
    2:  SupportedIons.N_II,
    3:  SupportedIons.N_III,
    4:  SupportedIons.N_IV,
    5:  SupportedIons.N_V,
    6:  SupportedIons.N_VI,
    7:  SupportedIons.N_VII,
}
_ions_by_element[O] = {
    2:  SupportedIons.O_I,
    3:  SupportedIons.O_III,
    4:  SupportedIons.O_IV,
    5:  SupportedIons.O_V,
    6:  SupportedIons.O_VI,
    7:  SupportedIons.O_VII,
    8:  SupportedIons.O_VIII,
}
_ions_by_element[Ne] = {
    8:  SupportedIons.Ne_VIII,
    9:  SupportedIons.Ne_IX,
    10: SupportedIons.Ne_X,
}
_ions_by_element[Mg] = {
    1:  SupportedIons.Mg_I,
    2:  SupportedIons.Mg_II,
}
_ions_by_element[Al] = {
    1:  SupportedIons.Al_I,
    2:  SupportedIons.Al_II,
    3:  SupportedIons.Al_III,
}
_ions_by_element[Si] = {
    2:  SupportedIons.Si_II,
    3:  SupportedIons.Si_III,
    4:  SupportedIons.Si_IV,
    13: SupportedIons.Si_XIII,
}
_ions_by_element[S] = {
    5:  SupportedIons.S_V,
}
_ions_by_element[Fe] = {
    2:  SupportedIons.Fe_II,
    3:  SupportedIons.Fe_III,
    17: SupportedIons.Fe_XVII,
}

_ion_info = {
    SupportedIons.H_I     : (H,  1,  "H_I"),
    SupportedIons.He_I    : (He, 1,  "He_I"),
    SupportedIons.He_II   : (He, 2,  "He_II"),
    SupportedIons.C_I     : (C,  1,  "C_I"),
    SupportedIons.C_II    : (C,  2,  "C_II"),
    SupportedIons.C_III   : (C,  3,  "C_III"),
    SupportedIons.C_IV    : (C,  4,  "C_IV"),
    SupportedIons.C_V     : (C,  5,  "C_V"),
    SupportedIons.C_VI    : (C,  6,  "C_VI"),
    SupportedIons.N_II    : (N,  2,  "N_II"),
    SupportedIons.N_III   : (N,  3,  "N_III"),
    SupportedIons.N_IV    : (N,  4,  "N_IV"),
    SupportedIons.N_V     : (N,  5,  "N_V"),
    SupportedIons.N_VI    : (N,  6,  "N_VI"),
    SupportedIons.N_VII   : (N,  7,  "N_VII"),
    SupportedIons.O_I     : (O,  1,  "O_I"),
    SupportedIons.O_III   : (O,  3,  "O_III"),
    SupportedIons.O_IV    : (O,  4,  "O_IV"),
    SupportedIons.O_V     : (O,  5,  "O_V"),
    SupportedIons.O_VI    : (O,  6,  "O_VI"),
    SupportedIons.O_VII   : (O,  7,  "O_VII"),
    SupportedIons.O_VIII  : (O,  8,  "O_VIII"),
    SupportedIons.Ne_VIII : (Ne, 8,  "Ne_VIII"),
    SupportedIons.Ne_IX   : (Ne, 9,  "Ne_IX"),
    SupportedIons.Ne_X    : (Ne, 10, "Ne_X"),
    SupportedIons.Mg_I    : (Mg, 1,  "Mg_I"),
    SupportedIons.Mg_II   : (Mg, 2,  "Mg_II"),
    SupportedIons.Al_I    : (Al, 1,  "Al_I"),
    SupportedIons.Al_II   : (Al, 2,  "Al_II"),
    SupportedIons.Al_III  : (Al, 3,  "Al_III"),
    SupportedIons.Si_II   : (Si, 2,  "Si_II"),
    SupportedIons.Si_III  : (Si, 3,  "Si_III"),
    SupportedIons.Si_IV   : (Si, 4,  "Si_IV"),
    SupportedIons.Si_XIII : (Si, 13, "Si_XIII"),
    SupportedIons.S_V     : (S,  5,  "S_V"),
    SupportedIons.Fe_II   : (Fe, 2,  "Fe_II"),
    SupportedIons.Fe_III  : (Fe, 3,  "Fe_III"),
    SupportedIons.Fe_XVII : (Fe, 17, "Fe_XVII")
}
