# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: LicenseRef-NotYetLicensed

from enum import Enum

class ParticleType(Enum):
    gas = 0
    dark_matter = 1
    star = 4
    black_hole = 5

    @property
    def common_hdf5_name(self) -> str:
        return f"PartType{self.value}"

    @property
    def name(self) -> str:
        return "gas" if self == ParticleType.gas else "dark matter" if self == ParticleType.dark_matter else "star" if self == ParticleType.star else "black hole"

    @staticmethod
    def get_all():
        return (ParticleType.gas, ParticleType.star, ParticleType.black_hole, ParticleType.dark_matter)
    
    def __str__(self) -> str:
        return self.common_hdf5_name
    
    def get_SWIFT_dataset(self, dataset: object):#TODO: find type
        return getattr(
            dataset,
                 "gas"         if self == ParticleType.gas
            else "dark_matter" if self == ParticleType.dark_matter
            else "stars"        if self == ParticleType.star
            else "black_holes"
        )
