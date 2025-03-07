from pydantic import Field
from datetime import date
from typing import Union, Any, List
from .field_validators import DateConversionMixin
from enum import Enum

# Pydantic Models for Documents
class Passport(DateConversionMixin):
    passport_number: str
    full_name: str
    nationality: str
    date_of_birth: date
    expiry_date: date


class PAN(DateConversionMixin):
    permanent_account_number: str
    name: str
    fathers_name: str
    date_of_birth: date

class LicenseClass(str, Enum):
    lmv = "LMV"
    mcwg = "MCWG"
    mcwog = "MCWOG"
    hgmv = "HGMV"
    hmpv = "HPMV"
    fvg = "FVG"
    mc_ex50cc = "MC EX50CC"
    lmv_nt = "LMV-NT"


class DrivingLicense(DateConversionMixin):
    license_number: Union[str, None] = None
    full_name: Union[str, None] = None
    issue_date: Union[date, None] = None
    expiry_date: Union[date, None] = None
    category: Union[List[LicenseClass], None] = Field(
        None, description="can have multiple classes."
    )
    address: Union[str, None] = None
    bloodgroup: Union[str, None] = Field(None, description="must be valid blood type. Sometimes represented as B.G.")
    son_of: Union[str, None] = Field(None, description="given as s/o in credentials")


class Aadhaar(DateConversionMixin):
    aadhaar_number: str
    full_name: str
    dob: date
    gender: str

