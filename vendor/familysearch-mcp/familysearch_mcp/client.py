"""Async HTTP client for the FamilySearch API.

Accepts a bearer token; it never performs a login. Obtaining a token is your
registered application's job, through its own OAuth flow -- see ``docs/AUTH.md``.

The place gazetteer (``/platform/places/search``) answers anonymous requests.
Everything else requires a token, and this client raises
:class:`~familysearch_mcp.config.AuthRequiredError` rather than sending an
unauthenticated request that would fail confusingly.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlsplit

import httpx

from .config import AuthRequiredError, Config, find_env_file, token_from_env_file

logger = logging.getLogger("familysearch_mcp.client")

#: GEDCOM X is the API's native representation.
GEDCOMX_JSON = "application/x-gedcomx-v1+json"

#: The Atom flavour. Search and change-history resources negotiate this one;
#: the places search in particular does not offer the plain GEDCOM X type.
GEDCOMX_ATOM_JSON = "application/x-gedcomx-atom+json"

#: FamilySearch's own extension of GEDCOM X, offered by the place description
#: routes and carrying the ``display`` block those reads are worth reading for.
FS_JSON = "application/x-fs-v1+json"

#: Paths that answer with no Authorization header at all.
#:
#: Only the routes this server actually calls are listed, because every entry
#: is a claim that a request sent with no Authorization header will be
#: answered rather than rejected -- and getting that wrong replaces a local
#: refusal naming FS_ACCESS_TOKEN with a confusing 401 from the server.
#:
#: Verified live on 2026-09-23 against production, header-less: every route
#: below answered 200. The Places documentation says only that no
#: *authenticated session* is required, and every worked example still sends
#: a token, so this was worth confirming rather than assuming. The catalogue
#: routes (collections, waypoints, the image resource) are not documented as
#: open at all.
#:
#: Note that places/search rejects ``application/x-gedcomx-v1+json`` with
#: HTTP 406; it is a feed and negotiates :data:`GEDCOMX_ATOM_JSON`.
ANONYMOUS_PATHS = (
    "/platform/places/search",
    "/platform/places/description",
    "/platform/records/collections",
    "/platform/records/waypoints",
    "/platform/records/images",
)

#: Exceptions carved back out of :data:`ANONYMOUS_PATHS`. The image resource
#: is open, but the records indexed from an image are not -- verified live
#: 2026-09-23: ``/platform/records/images/{ark}`` answers 200 with no header
#: while ``.../{ark}/records`` answers 401.
#:
#: A plain prefix match cannot express that, and getting it wrong is not a
#: harmless over-permission: the server would stop refusing locally and send
#: a bare request, turning a clear "set FS_ACCESS_TOKEN" into a confusing
#: 401 from FamilySearch.
AUTHENTICATED_SUBPATHS = ("/records",)

#: The cheapest read that tells a live token from a dead one: 200 for a good
#: token, 401 for none or a bad one. Verified live 2026-09-28. Needed because
#: the image resource answers a bad token as if it were anonymous.
CURRENT_USER_PATH = "/platform/users/current"


def needs_token(path: str) -> bool:
    """Whether ``path`` requires an Authorization header.

    Parameters
    ----------
    path : str
        An API path below the host.

    Returns
    -------
    bool
        True when the path must carry a token.
    """
    if not path.startswith(ANONYMOUS_PATHS):
        return True
    return path.rstrip("/").endswith(AUTHENTICATED_SUBPATHS)


class FamilySearchApiError(RuntimeError):
    """Raised on a non-2xx response, carrying status and server detail.

    Attributes
    ----------
    status : int
        HTTP status. 401, 403 and 429 are each read differently by the tool
        layer, so the raw number is kept rather than collapsed to a message.
    detail : str
        The server's own explanation, if it gave one.
    retry_after : int or None
        Seconds from a ``Retry-After`` header, present on a 429.
    """

    def __init__(self, status: int, detail: str, *, path: str, retry_after: int | None = None):
        """Record the failing request and the server's explanation."""
        self.status = status
        self.detail = detail
        self.retry_after = retry_after
        super().__init__(f"GET {path} -> {status}: {detail}")


class FamilySearchClient:
    """Async client for one FamilySearch environment.

    Parameters
    ----------
    config : Config
        Resolved configuration, carrying the token and environment.
    """

    def __init__(self, config: Config):
        self._config = config
        self._http = httpx.AsyncClient(base_url=config.base_url, timeout=config.timeout)
        #: Catalog entries read recently, by environment and id, with when.
        self._catalog: dict[tuple[str, str], tuple[float, dict]] = {}

    def _reload_token(self) -> bool:
        """Pick up a token refreshed on disk since this process started.

        A token lasts about an hour and the environment is read once at
        startup, so a long session outlives its credential. Re-reading the
        env file is what lets a running server recover from that without
        being reconnected.

        This reads a token someone else obtained. It does not authenticate,
        and the server still never handles a password.

        Returns
        -------
        bool
            True when a different token was found and adopted.
        """
        if not self._config.env_file:
            # Nothing was found at startup. Look again: a file placed in the
            # working directory since is the one way to rescue a running
            # server whose launcher did not say where the token lives.
            self._config.env_file = find_env_file()
        if not self._config.env_file:
            return False
        fresh = token_from_env_file(self._config.env_file)
        if fresh and fresh != self._config.access_token:
            self._config.access_token = fresh
            logger.info("adopted a refreshed FS_ACCESS_TOKEN from the env file")
            return True
        return False

    @property
    def authenticated(self) -> bool:
        """bool: Whether a token is configured -- not whether it is still good."""
        return self._config.authenticated

    async def confirm_token(self) -> bool:
        """Check the token is still accepted, adopting a refreshed one if not.

        Some routes never say 401. The image resource answers an expired or
        invalid token exactly as it answers no token at all -- 200, with the
        image links left out -- so the recovery in :meth:`get` cannot trigger
        there. The current-user read does answer 401, and asking it through
        :meth:`get` brings that recovery with it. Verified live 2026-09-28.

        Returns
        -------
        bool
            True when the token had expired and a refreshed one was adopted,
            so whatever came back thin is worth asking for again.

        Raises
        ------
        AuthRequiredError
            If no token is configured.
        FamilySearchApiError
            401 when the token is rejected and no different one is on disk.
        """
        before = self._config.access_token
        await self.get(CURRENT_USER_PATH)
        return self._config.access_token != before

    async def download(self, url: str) -> bytes:
        """Fetch image bytes, recovering from an expired token as :meth:`get` does.

        The image hosts answer an expired token with 401 -- unlike the image
        resource that hands out their URLs -- so this path can recover the
        ordinary way. It needs its own retry because it does not go through
        :meth:`get`. Verified live 2026-09-28.

        Parameters
        ----------
        url : str
            An image URL from the image resource's link relations.

        Returns
        -------
        bytes
            The image bytes.

        Raises
        ------
        FamilySearchApiError
            On any status of 400 or above that one retry did not cure.
        """
        try:
            return await fetch_binary(url, self._config.access_token)
        except FamilySearchApiError as exc:
            if exc.status == 401 and self._reload_token():
                return await fetch_binary(url, self._config.access_token)
            if exc.status == 429 and (wait := retry_wait(exc.retry_after)) is not None:
                await pause(wait)
                return await fetch_binary(url, self._config.access_token)
            raise

    async def _send(self, attempt: Callable[[], Awaitable[httpx.Response]]) -> httpx.Response:
        """Send a request, retrying once after an expired token or a short throttle.

        Parameters
        ----------
        attempt : callable
            Sends the request. Called again for a retry, so it must build its
            headers each time: a retry after a 401 carries the new token.

        Returns
        -------
        httpx.Response
            The last response, whatever its status.
        """
        resp = await attempt()

        # A 401 usually means the token expired mid-session. If a refreshed
        # one has been written to the env file, adopt it and retry once
        # rather than making the caller reconnect the server.
        if resp.status_code == 401 and self._reload_token():
            resp = await attempt()

        if resp.status_code == 429 and (wait := retry_wait(_retry_after(resp))) is not None:
            logger.info("throttled; retrying once in %s seconds", wait)
            await pause(wait)
            resp = await attempt()
        return resp

    async def aclose(self) -> None:
        """Close the underlying HTTP transport."""
        await self._http.aclose()

    async def __aenter__(self) -> FamilySearchClient:
        """Return the client."""
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Close the transport."""
        await self.aclose()

    def _headers(self, path: str, accept: str = GEDCOMX_JSON) -> dict[str, str]:
        """Build request headers, requiring a token outside the anonymous paths.

        Raises
        ------
        AuthRequiredError
            If the path needs a token and none is configured.
        """
        headers = {"Accept": accept}
        if self._config.access_token:
            headers["Authorization"] = f"Bearer {self._config.access_token}"
        elif needs_token(path):
            raise AuthRequiredError(
                f"{path} requires a FamilySearch access token. Set "
                "FS_ACCESS_TOKEN from your own registered application's OAuth "
                "flow; see docs/AUTH.md. The place gazetteer works without one."
            )
        return headers

    async def search(self, params: dict) -> dict:
        """Run a record search against the website's search service.

        Parameters
        ----------
        params : dict
            Query parameters. None values are dropped.

        Returns
        -------
        dict
            The search response: ``results``, ``index``, ``entries``.

        Raises
        ------
        AuthRequiredError
            If no token is configured.
        FamilySearchApiError
            On any status of 400 or above.
        """
        return await self._website(SEARCH_URLS, params, "Record search")

    async def fulltext(self, params: dict) -> dict:
        """Run a search of the machine-read page text on the website's service.

        Parameters
        ----------
        params : dict
            Query parameters. None values are dropped.

        Returns
        -------
        dict
            The response: ``results``, ``index``, ``entries``, ``facets``,
            ``links``.

        Raises
        ------
        AuthRequiredError
            If no token is configured.
        FamilySearchApiError
            On any status of 400 or above.
        """
        return await self._website(FULLTEXT_URLS, params, "Full-text search")

    async def catalog_entry(self, catalog_id: str) -> dict:
        """Read one FamilySearch Catalog entry from the website's service.

        An entry for a county's probate files can list thousands of films and
        weigh most of a megabyte, and a caller narrowing it asks several
        times in a row, so an entry is kept for :data:`CATALOG_CACHE_SECONDS`.

        Parameters
        ----------
        catalog_id : str
            The catalog's title number, digits only.

        Returns
        -------
        dict
            The entry's ``source`` object: title, authors, notes, subjects
            and ``film_note``, one per film or DGS. Empty if there is none.

        Raises
        ------
        AuthRequiredError
            If no token is configured.
        FamilySearchApiError
            On any status of 400 or above; 404 for an unknown id.
        """
        key = (self._config.environment, catalog_id)
        cached = self._catalog.get(key)
        if cached and time.monotonic() - cached[0] < CATALOG_CACHE_SECONDS:
            return cached[1]
        payload = await self._website(CATALOG_URLS, {}, "The catalog", path=f"/{catalog_id}")
        source = payload.get("source") if isinstance(payload, dict) else None
        entry = source if isinstance(source, dict) else {}
        if len(self._catalog) >= CATALOG_CACHE_ENTRIES:
            self._catalog.pop(min(self._catalog, key=lambda k: self._catalog[k][0]))
        self._catalog[key] = (time.monotonic(), entry)
        return entry

    async def catalog_for_film(self, dgs: str) -> list[dict]:
        """The Catalog entries that hold one film, by its DGS or microfilm number.

        The website's catalog search takes ``+film_number:<n>`` and answers
        with each entry that lists the film. Verified live 2026-10-11: a DGS
        finds its entry (004110393 -> 2770955), and an unknown number answers
        with no hits rather than an error. It needs the token and a browser
        User-Agent, as the entry service does.

        Returns
        -------
        list of dict
            ``{"catalog_id", "title"}`` for each entry, in the order given.
        """
        payload = await self._website(
            CATALOG_SEARCH_URLS,
            {"count": 20, "query": f"+film_number:{dgs}"},
            "The catalog search",
        )
        found = []
        for hit in payload.get("searchHits") or []:
            metadata = ((hit or {}).get("metadataHit") or {}).get("metadata") or {}
            ident = str((metadata.get("identifier") or {}).get("value") or "")
            number = ident.rsplit("olib:", 1)[-1] if "olib:" in ident else ""
            if number.isdigit():
                titles = [
                    t.get("value", "") for t in metadata.get("title") or [] if isinstance(t, dict)
                ]
                found.append({"catalog_id": number, "title": " ".join(t for t in titles if t)})
        return found

    async def image_group(self, dgs: str) -> tuple[int, dict]:
        """Ask the storage host about one image group (DGS), as this account.

        The group's node answers a token that may see the images with 200
        and ``childCount``, the number of images; one that may not with 403;
        an unknown or unpadded number with 404. Verified live 2026-10-05.
        It needs the token but not a browser User-Agent.

        Parameters
        ----------
        dgs : str
            The DGS number, nine digits with its leading zeros.

        Returns
        -------
        tuple of (int, dict)
            The status, and the decoded body (empty unless it was JSON).

        Raises
        ------
        AuthRequiredError
            If no token is configured.
        """
        url = f"{DAS_HOST}/dgs:{dgs}"
        resp = await self._send(
            lambda: self._http.get(url, headers=self._headers(url, "application/json"))
        )
        try:
            body = resp.json()
        except ValueError:
            body = {}
        return resp.status_code, body if isinstance(body, dict) else {}

    async def _website(
        self, urls: dict[str, str], params: dict, what: str, *, path: str = ""
    ) -> dict:
        """GET one of the website's services with the token.

        Each takes the API's bearer token, but only alongside a browser
        User-Agent: the token alone gets 403 and the User-Agent alone 401.
        """
        if not self._config.access_token:
            raise AuthRequiredError(
                f"{what} requires a FamilySearch access token. Set "
                "FS_ACCESS_TOKEN; see docs/AUTH.md."
            )
        clean = {k: v for k, v in params.items() if v is not None}
        # A sandbox token means nothing to the production website.
        url = urls[self._config.environment] + path
        resp = await self._send(
            lambda: self._http.get(
                url,
                params=clean,
                # Built per attempt, so a retry carries a refreshed token.
                headers={
                    "Authorization": f"Bearer {self._config.access_token}",
                    "Accept": "application/json",
                    # Required. Without it this service answers 403 even
                    # with a valid token.
                    "User-Agent": BROWSER_UA,
                },
            )
        )
        if resp.status_code >= 400:
            raise FamilySearchApiError(
                resp.status_code,
                _detail(resp),
                path=url,
                retry_after=_retry_after(resp),
            )
        try:
            return resp.json()
        except ValueError:
            return {}

    async def get_text(self, url: str) -> str:
        """GET a resource that answers with bare text rather than JSON.

        An image's ``image-name`` relation answers ``application/json`` with a
        body such as ``dgs:008190429.008190429_00580`` -- not a JSON string,
        so it cannot be decoded as one. Verified live 2026-10-05.

        Parameters
        ----------
        url : str
            The full URL, from a link relation.

        Returns
        -------
        str
            The body, stripped.

        Raises
        ------
        FamilySearchApiError
            On any status of 400 or above.
        """
        resp = await self._send(
            lambda: self._http.get(url, headers=self._headers(url, "application/json"))
        )
        if resp.status_code >= 400:
            raise FamilySearchApiError(
                resp.status_code, _detail(resp), path=url, retry_after=_retry_after(resp)
            )
        return resp.text.strip()

    async def get(
        self,
        path: str,
        *,
        accept: str = GEDCOMX_JSON,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict:
        """Send a GET and decode the response.

        Parameters
        ----------
        path : str
            Path below the API host.
        accept : str, optional
            Accept header. Feed-shaped resources such as a change history
            negotiate :data:`GEDCOMX_ATOM_JSON` instead of the default.
        params : dict, optional
            Query parameters whose names are not Python identifiers, such as
            the ``q.spouseSurname`` family used by record search.
        **kwargs
            Further query parameters. None values are dropped from both.

        Returns
        -------
        dict
            The decoded body, or an empty dict for a 204.

        Raises
        ------
        AuthRequiredError
            If the path needs a token and none is configured.
        FamilySearchApiError
            On any status of 400 or above.
        """
        body, _ = await self.get_with_validators(path, accept=accept, params=params, **kwargs)
        return body

    async def get_with_validators(
        self,
        path: str,
        *,
        accept: str = GEDCOMX_JSON,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[dict, dict[str, str | None]]:
        """Send a GET, returning the body and the resource's cache validators.

        A tree person carries a weak ``ETag`` and a ``Last-Modified`` that
        change whenever anyone edits the profile. A comparison reports them so
        a later reader can tell whether the profile has changed since. The GET
        itself carries both, so no separate HEAD is needed -- verified live
        2026-10-05: ``GET /platform/tree/persons/{id}`` and ``HEAD`` on the
        same path answered the same weak ETag.

        Parameters
        ----------
        path, accept, params, **kwargs
            As for :meth:`get`.

        Returns
        -------
        tuple of (dict, dict)
            The decoded body (empty for a 204), and ``{"etag",
            "last_modified"}``, either of which may be None.

        Raises
        ------
        AuthRequiredError
            If the path needs a token and none is configured.
        FamilySearchApiError
            On any status of 400 or above.
        """
        merged = {**(params or {}), **kwargs}
        clean = {k: v for k, v in merged.items() if v is not None}
        resp = await self._send(
            lambda: self._http.get(path, params=clean, headers=self._headers(path, accept))
        )
        validators = {
            "etag": resp.headers.get("ETag"),
            "last_modified": resp.headers.get("Last-Modified"),
        }
        if resp.status_code == 204:
            return {}, validators
        if resp.status_code >= 400:
            raise FamilySearchApiError(
                resp.status_code,
                _detail(resp),
                path=path,
                retry_after=_retry_after(resp),
            )
        try:
            return resp.json(), validators
        except ValueError:
            return {}, validators


def _retry_after(resp: httpx.Response) -> int | None:
    """Read a ``Retry-After`` delay in seconds, ignoring a date-form header."""
    raw = resp.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return int(float(raw.strip()))
    except ValueError:
        return None


#: The longest ``Retry-After`` this client waits out on its own, in seconds.
#: A tool call that blocks for longer than this looks hung to the client and
#: may hit its timeout, so a longer wait is handed back to the caller.
MAX_RETRY_WAIT = 15

#: The wait before retrying a 429 that named no delay, in seconds.
DEFAULT_RETRY_WAIT = 2


def retry_wait(retry_after: int | None) -> float | None:
    """How long to wait before the one retry of a throttled request.

    Parameters
    ----------
    retry_after : int or None
        The server's ``Retry-After``, in seconds, if it sent one.

    Returns
    -------
    float or None
        Seconds to wait, or None when the server asked for longer than
        :data:`MAX_RETRY_WAIT` and the caller should be told instead.
    """
    if retry_after is None:
        return DEFAULT_RETRY_WAIT
    if retry_after > MAX_RETRY_WAIT:
        return None
    return max(0, retry_after)


async def pause(seconds: float) -> None:
    """Wait before a retry. A module-level function so tests can replace it."""
    await asyncio.sleep(seconds)


def is_familysearch_url(url: str) -> bool:
    """Whether ``url`` is HTTPS on a FamilySearch host, so may carry the token.

    Parameters
    ----------
    url : str
        Any URL.

    Returns
    -------
    bool
        True for ``https://familysearch.org/...`` and its subdomains.
    """
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return False
    host = (parts.hostname or "").lower()
    return parts.scheme == "https" and (
        host == "familysearch.org" or host.endswith(".familysearch.org")
    )


#: The website search service rejects a default library user agent.
BROWSER_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:128.0) Gecko/20100101 Firefox/128.0"

#: The record search the FamilySearch website itself uses.
#:
#: ``/platform/records/personas`` on the API host answers from a partial
#: index: verified live 2026-09-23, a required-name search for "John Smith"
#: returned 53,043 hits that were ALL passenger and crew lists, and
#: ``f.collectionId`` for a collection outside that index was silently
#: dropped rather than refused. The same search here returns 5,835,812
#: across the full catalogue, and the collection filter works.
#:
#: It takes the same bearer token, but only with a browser User-Agent: the
#: token alone gets 403 and the User-Agent alone gets 401.
SEARCH_URL = "https://www.familysearch.org/service/search/hr/v2/personas"

#: The same search service for each environment in ``config.HOSTS``. The
#: sandbox's answers 401 to a browser User-Agent with no token, exactly as
#: production's does, where an unknown host would not answer at all --
#: verified 2026-09-29. It has not been verified with a sandbox token.
SEARCH_URLS = {
    "production": SEARCH_URL,
    "integration": "https://integration.familysearch.org/service/search/hr/v2/personas",
}

#: The website's search over handwriting-recognised page text.
#:
#: Undocumented, like :data:`SEARCH_URL`, and on the same footing: verified
#: live 2026-10-05, it answers the API's bearer token with a browser
#: User-Agent (200), the token alone with 403 and the User-Agent alone with
#: 401. Terms in ``q.text`` are OR'd unless each carries ``+``, even with
#: ``m.queryRequireDefault=on``: "Hannah Ball" matched 19,707,871 pages and
#: "+Hannah +Ball" 270,195.
FULLTEXT_URL = "https://www.familysearch.org/service/search/fulltext/search"

#: The same service per environment. The sandbox answers 401 to a browser
#: User-Agent with no token, as production does -- checked 2026-10-05; not
#: checked with a sandbox token.
FULLTEXT_URLS = {
    "production": FULLTEXT_URL,
    "integration": "https://integration.familysearch.org/service/search/fulltext/search",
}

#: One FamilySearch Catalog entry, as the website's catalog page reads it:
#: ``{CATALOG_URL}/{catalog_id}``. Found 2026-10-05 in the page's own
#: scripts. Undocumented, and on the same footing as :data:`SEARCH_URL`:
#: verified live that day, the token with a browser User-Agent gets 200, the
#: token alone 403 and the User-Agent alone 401; an unknown id gets 404 with
#: no body.
CATALOG_URL = "https://www.familysearch.org/service/search/catalog/item"

#: The same service per environment. The sandbox answers 401 to a browser
#: User-Agent with no token, as production does -- checked 2026-10-05; not
#: checked with a sandbox token.
CATALOG_URLS = {
    "production": CATALOG_URL,
    "integration": "https://integration.familysearch.org/service/search/catalog/item",
}

#: The website's catalog search, which finds the entries holding a film:
#: ``?query=+film_number:<n>``. Found 2026-10-11 in the catalog search page.
CATALOG_SEARCH_URLS = {
    "production": "https://www.familysearch.org/service/search/catalog/v3/search",
    "integration": "https://integration.familysearch.org/service/search/catalog/v3/search",
}

#: How long a catalog entry is reused, in seconds, and how many are kept.
CATALOG_CACHE_SECONDS = 600
CATALOG_CACHE_ENTRIES = 4

#: Storage host serving digital-artifact images.
DAS_HOST = "https://sg30p0.familysearch.org/service/records/storage/dascloud/das/v2"


def film_image_node(film: str, image: int) -> str:
    """Build the storage node id for a film image.

    A citation often names a film and an image number rather than an ark --
    the ``FS_DIGITAL_FILM_NBR`` and ``FS_IMAGE_NBR`` fields on an indexed
    record. Those address the image directly on the storage host, with no
    API resource to ask.

    Parameters
    ----------
    film : str
        Digital film (DGS) number, e.g. ``"004893581"``. Kept as a string
        because the leading zeros are significant.
    image : int
        Image number within the film, 1-based.

    Returns
    -------
    str
        The node id, e.g. ``"dgs:004893581_00213"``.

    Notes
    -----
    Verified live 2026-09-23: both ``dgs:{film}_{image:05d}`` and the longer
    ``dgs:{film}.{film}_{image:05d}`` resolve. The short form is used here.
    """
    return f"dgs:{film.strip()}_{int(image):05d}"


async def film_image_exists(node: str, *, timeout: float = 30.0) -> bool:
    """Check whether a storage node has an image, using its thumbnail.

    The thumbnail is served without a token while the full page is not, so
    this confirms a film and image number are real before a caller is handed
    a URL that would only fail later.

    Parameters
    ----------
    node : str
        A storage node id from :func:`film_image_node`.
    timeout : float, optional
        Request timeout in seconds.

    Returns
    -------
    bool
        True when a thumbnail came back.
    """
    url = f"{DAS_HOST}/{node}/thumb_p200.jpg"
    try:
        async with httpx.AsyncClient(timeout=timeout) as http:
            resp = await http.get(url, headers={"Accept": "image/*"})
    except httpx.HTTPError:
        return False
    return resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image/")


async def fetch_binary(url: str, token: str | None, *, timeout: float = 90.0) -> bytes:
    """Download a FamilySearch image, handling its storage redirect.

    The image routes answer with a 302 to a **presigned S3 URL**. Carrying the
    bearer header onto that request fails with ``InvalidArgument: Only one
    auth mechanism allowed`` -- the signature in the query string is already
    the credential. So the header is sent on the first hop and dropped on the
    second. Verified live 2026-09-23.

    Parameters
    ----------
    url : str
        An image URL from the image resource's link relations.
    token : str or None
        Bearer token for the first hop. Sent only when ``url`` is HTTPS on a
        FamilySearch host; see :func:`is_familysearch_url`.
    timeout : float, optional
        Per-request timeout in seconds.

    Returns
    -------
    bytes
        The image bytes.

    Raises
    ------
    FamilySearchApiError
        On any status of 400 or above.
    """
    headers = {"Accept": "*/*"}
    # The URL is a tool argument, so it is whatever the model was persuaded
    # to pass. A token sent to any host it names would leak to that host.
    if token and is_familysearch_url(url):
        headers["Authorization"] = f"Bearer {token}"
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as http:
        resp = await http.get(url, headers=headers)
        if resp.status_code in (301, 302, 303, 307, 308):
            location = resp.headers.get("location")
            if not location:
                raise FamilySearchApiError(
                    resp.status_code, "redirect without a location", path=url
                )
            # No Authorization here: the presigned URL is its own credential.
            resp = await http.get(location, headers={"Accept": "*/*"})
        if resp.status_code >= 400:
            raise FamilySearchApiError(
                resp.status_code, _detail(resp), path=url, retry_after=_retry_after(resp)
            )
        return resp.content


def _detail(resp: httpx.Response) -> str:
    """Pull a human-readable explanation out of an error response body."""
    try:
        body = resp.json()
        if isinstance(body, dict):
            message = body.get("error") or body.get("message")
            # The full-text service explains a 400 in a list of strings:
            # "Validation failed." alone does not say what to fix.
            errors = [e for e in body.get("errors") or [] if isinstance(e, str)]
            if message and errors:
                return f"{message} {'; '.join(errors)}"
            return str(message or body)
        return str(body)
    except Exception:
        return resp.text[:300]
