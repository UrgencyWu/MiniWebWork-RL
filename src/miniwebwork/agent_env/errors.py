"""Environment error types."""


class EnvironmentError(Exception):
    """Base error for environment issues."""
    pass


class EnvironmentClosedError(EnvironmentError):
    """Raised when operating on a closed environment."""
    pass


class EpisodeFinishedError(EnvironmentError):
    """Raised when step() is called after terminal state."""
    pass


class InvalidActionError(EnvironmentError):
    """Raised when an action is malformed or invalid."""
    def __init__(self, error_code: str, message: str = ""):
        self.error_code = error_code
        self.message = message
        super().__init__(f"[{error_code}] {message}")


class NavigationBlockedError(EnvironmentError):
    """Raised when navigation to an unauthorized URL is attempted."""
    pass


class BrowserError(EnvironmentError):
    """Raised when Playwright encounters an unrecoverable error."""
    pass
