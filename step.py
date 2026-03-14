# steps.py
from enum import Enum

class Step(str, Enum):
    FIND_LOGIN_PAGE     = "find-login-page"
    LOGIN               = "login"
    OPEN_EMAIL_SETTINGS = "open-email-settings"
    CHANGE_EMAIL        = "change-email"
    DONE                = "done"

    @classmethod
    def values_list(cls) -> str:
        """Retourne la liste formatée pour injection dans un prompt."""
        return "\n- ".join(f"{s.value}" for s in cls)
