from pydantic import BaseModel, field_validator
from typing import Any
from datetime import date, datetime

# Pydantic Models for Documents
date_formats = (
    "%Y-%m-%d",  # 2024-06-25 (ISO 8601 Standard)
    "%d/%m/%Y",  # 25/06/2024 (Common in India, UK)
    "%m/%d/%Y",  # 06/25/2024 (US Format)
    "%d-%m-%Y",  # 25-06-2024 (Common in India, UK)
    "%Y/%m/%d",  # 2024/06/25 (Rare but possible)
    "%m-%d-%Y",  # 06-25-2024 (US Format)
    "%d %b %Y",  # 25 Jun 2024 (Short Month Name)
    "%d %B %Y",  # 25 June 2024 (Full Month Name)
    "%b %d, %Y",  # Jun 25, 2024 (US, Passport Style)
    "%B %d, %Y",  # June 25, 2024 (Long-form US, UK)
    "%d.%m.%Y",  # 25.06.2024 (German, European style)
    "%m.%d.%Y",  # 06.25.2024 (Alternative US style)
    "%Y.%m.%d",  # 2024.06.25 (Database formats)
    "%d%m%Y",  # 25062024 (No separator, found in OCR errors)
    "%Y%m%d",  # 20240625 (Machine-readable)
    "%d-%b-%Y",  # 25-Jun-2024 (Found in some Indian credentials)
    "%d-%B-%Y",  # 25-June-2024 (Long form, uncommon)
    "%Y %b %d",  # 2024 Jun 25 (Seen in some passport formats)
    "%Y %B %d",  # 2024 June 25
    "%b-%d-%Y",  # Jun-25-2024
    "%B-%d-%Y",  # June-25-2024
    "%b %d %Y",  # Jun 25 2024
    "%B %d %Y",  # June 25 2024
    "%d/%b/%Y",  # 25/Jun/2024 (Common in travel documents)
    "%d/%B/%Y",  # 25/June/2024
    "%m-%Y-%d",  # 06-2024-25
)

class DateConversionMixin(BaseModel):
    """Mixin that attempts to convert values to dates, but passes them through if conversion fails."""

    @field_validator("*", mode="before", check_fields=False)
    @classmethod
    def try_convert_to_date(cls, value: Any) -> Any:
        """Attempt to convert the value to a date. If unsuccessful, return the original value."""

        # If already a date, return as is
        if isinstance(value, date):
            return value

        # Try parsing valid date strings
        if isinstance(value, str):
            for fmt in date_formats:
                try:
                    return str(datetime.strptime(value, fmt).date())  # Convert to date
                except ValueError:
                    continue  # Try next format
            return value  # If no formats match, return the original string

        # Handle integer timestamps (convert to date)
        if isinstance(value, int):
            try:
                return str(datetime.fromtimestamp(value).date())
            except ValueError:
                return value  # Return as is if not a valid timestamp

        return value  # Return original value if it can't be converted
