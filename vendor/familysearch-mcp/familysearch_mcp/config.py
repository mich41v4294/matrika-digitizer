"""Configuration from environment variables.

===================  =========================================================
``FS_CLIENT_ID``     Client id of *your* registered FamilySearch application.
``FS_ACCESS_TOKEN``  Bearer token from your application's OAuth flow.
``FS_ENV_FILE``      The env file to read settings from, and to re-read a
                     refreshed token from. Defaults to the nearest ``.env``
                     from the working directory upward.
``FS_ENVIRONMENT``   ``production`` (default) or ``integration``.
``FS_TIMEOUT``       HTTP timeout in seconds.
===================  =========================================================

There is deliberately no bundled client id and no username or password
setting. A production credential is issued to an application, and borrowing
another project's id is not a foundation anyone should build on. The place
gazetteer answers anonymously, so it works with no configuration at all.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from dotenv import dotenv_values, find_dotenv, load_dotenv

#: API hosts by environment name.
HOSTS = {
    "production": "https://api.familysearch.org",
    "integration": "https://api-integ.familysearch.org",
}

logger = logging.getLogger("familysearch_mcp.config")


def find_env_file() -> str | None:
    """Locate the env file, the same way at startup and on a 401.

    ``FS_ENV_FILE`` wins; otherwise the nearest ``.env`` from the working
    directory upward. Startup and token recovery used to search from
    different places -- dotenv's default walks up from the *package's*
    source file, not the working directory -- so a server could load its
    token from one file and look for the refreshed one in another, or in
    none.

    Returns
    -------
    str or None
        Absolute path, or None when nothing is set and nothing is found. An
        ``FS_ENV_FILE`` is returned even if it does not exist, so the
        mistake can be reported rather than silently skipped.
    """
    if explicit := os.environ.get("FS_ENV_FILE"):
        # Client configs are JSON, and JSON does not expand ~.
        return os.path.abspath(os.path.expanduser(explicit))
    return find_dotenv(usecwd=True) or None


def token_from_env_file(path: str | None = None) -> str | None:
    """Read ``FS_ACCESS_TOKEN`` straight out of the env file on disk.

    A token lives about an hour, and a server process reads the environment
    once at startup -- so a long session outlives its token and cannot pick
    up a refreshed one without being restarted. This is what makes recovery
    possible without a restart.

    It deliberately reads the FILE rather than the environment.
    ``load_dotenv`` does not override a variable already set in the process,
    which is precisely the case here: the launcher exported the token before
    starting the server.

    Parameters
    ----------
    path : str, optional
        Env file to read. Defaults to :func:`find_env_file`.

    Returns
    -------
    str or None
        The token in the file, or None if there is no file or no token.
    """
    target = path or find_env_file()
    if not target:
        return None
    try:
        return dotenv_values(target).get("FS_ACCESS_TOKEN") or None
    except OSError:
        return None


class ConfigError(RuntimeError):
    """Raised when configuration is missing or invalid."""


class AuthRequiredError(RuntimeError):
    """Raised when a tool needs a token and none is configured."""


@dataclass
class Config:
    """Resolved server configuration.

    Attributes
    ----------
    access_token : str or None
        Bearer token. None leaves authenticated tools unavailable.
    env_file : str or None
        Env file a refreshed token is read back from after a 401.
    client_id : str or None
        Your application's client id, reported by ``auth_status``.
    environment : str
        Key into :data:`HOSTS`.
    timeout : float
        HTTP timeout in seconds.
    """

    access_token: str | None = None
    client_id: str | None = None
    environment: str = "production"
    timeout: float = 60.0
    env_file: str | None = None

    @property
    def base_url(self) -> str:
        """str: API host for the configured environment."""
        return HOSTS[self.environment]

    @property
    def authenticated(self) -> bool:
        """bool: Whether a token is available for the authenticated tools."""
        return bool(self.access_token)


def load_config() -> Config:
    """Load configuration from the environment.

    The env file from :func:`find_env_file` is read if there is one; real
    environment variables win. A missing token is not an error here: the
    gazetteer still works, and the authenticated tools report what is
    missing when called.

    Returns
    -------
    Config
        Fully resolved configuration.

    Raises
    ------
    ConfigError
        If ``FS_ENVIRONMENT`` names an unknown environment.
    """
    env_file = find_env_file()
    if env_file:
        load_dotenv(env_file)

    environment = os.environ.get("FS_ENVIRONMENT", "production").strip().lower()
    if environment not in HOSTS:
        raise ConfigError(f"FS_ENVIRONMENT must be one of {sorted(HOSTS)}, not {environment!r}.")

    cfg = Config(
        access_token=os.environ.get("FS_ACCESS_TOKEN") or None,
        client_id=os.environ.get("FS_CLIENT_ID") or None,
        environment=environment,
        env_file=env_file,
    )
    if env_file and not os.path.exists(env_file):
        logger.warning("FS_ENV_FILE names %s, which does not exist", env_file)
    elif cfg.access_token and not env_file:
        logger.warning(
            "FS_ACCESS_TOKEN came from the environment and no env file was "
            "found, so an expired token will need a restart. Set FS_ENV_FILE "
            "to the file the token is written to."
        )
    if raw := os.environ.get("FS_TIMEOUT"):
        cfg.timeout = float(raw)
    return cfg
