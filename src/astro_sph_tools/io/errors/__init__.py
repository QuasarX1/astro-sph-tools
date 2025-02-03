# SPDX-FileCopyrightText: 2025-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: LicenseRef-NotYetLicensed

class SnipshotError(NotImplementedError):
    """
    Base class for errors related to the current dataset being a snipshot.
    """

class SnipshotOperationError(SnipshotError):
    """
    Operation is invalid as target data is a snipshot.
    """
    def __init__(self, operation_name: str, message: str|None = None) -> None:
        super().__init__(
            f"Operation \"{operation_name}\" not supported for snipshots."
            + (f"\n    {message}" if message is not None else "")
        )

class SnipshotFieldError(SnipshotError):
    """
    Data field is not available in snipshots.
    """
    def __init__(self, field_name: str, message: str|None = None) -> None:
        super(SnipshotError, self).__init__(
            f"Particle field \"{field_name}\" not available in snipshots."
            + (f"\n    {message}" if message is not None else "")
        )

class HaloDefinitionNotSupportedError(NotImplementedError):
    """
    Halo definition not supported by catalogue.
    """
    def __init__(self, definition_type: type, catalogue_type: type, definition_detail: str|None = None, message: str|None = None) -> None:
        super().__init__(
            f"Halo definition of type \"{definition_type.__name__}\"{(' ' + definition_detail) if definition_detail is not None else ''} not supported by catalogue type \"{catalogue_type.__name__}\"."
            + (f"\n    {message}" if message is not None else "")
        )
