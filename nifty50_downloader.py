"""Standalone downloader for current NIFTY 50 daily price histories.

The script downloads the official constituent roster first and refuses to
start price downloads unless the roster is a valid 50-symbol NSE CSV. Price
histories are saved separately under a run-specific directory in Windows
TEMP; this module does not use the dashboard or SQLite storage layer.
"""

from __future__ import annotations

import hashlib
from html.parser import HTMLParser
from io import BytesIO
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import gettempdir
from typing import Callable, Mapping
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
import yfinance as yf


OFFICIAL_ROSTER_PAGE_URL = "https://www.nseindia.com/static/products-services/indices-nifty50-index"
EXPECTED_ROSTER_CSV_HOST = "nsearchives.nseindia.com"
EXPECTED_ROSTER_CSV_PATH = "/content/indices/ind_nifty50list.csv"
ROSTER_COLUMNS = {"company name", "symbol"}
EXPECTED_CONSTITUENT_COUNT = 50
DOWNLOAD_PERIOD = "5y"
DOWNLOAD_INTERVAL = "1d"
REQUEST_TIMEOUT_SECONDS = 20
TEMP_OUTPUT_FOLDER = "stock_market_prediction_nifty50"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"

# Deliberately empty until a verified NSE-to-Yahoo exception is needed.
# Values are recorded in the manifest whenever an override is used.
YAHOO_TICKER_OVERRIDES: dict[str, str] = {}

MANIFEST_FIELDS = [
    "company_name",
    "nse_symbol",
    "yahoo_ticker",
    "mapping_exception",
    "retrieval_time_utc",
    "row_count",
    "first_date",
    "last_date",
    "returned_columns",
    "status",
    "error_details",
    "validation_problems",
]


class RosterError(RuntimeError):
    """Raised when the official NSE constituent roster cannot be verified."""


class _AnchorParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._current: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() == "a":
            self._current = [dict(attrs).get("href", ""), ""]

    def handle_data(self, data: str) -> None:
        if self._current is not None:
            self._current[1] += data

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._current is not None:
            self.links.append((self._current[0], self._current[1].strip()))
            self._current = None


@dataclass(frozen=True)
class OfficialRoster:
    raw_csv: bytes
    csv_url: str
    retrieved_at_utc: str
    frame: pd.DataFrame


def _official_csv_link(html: str, page_url: str = OFFICIAL_ROSTER_PAGE_URL) -> str:
    parser = _AnchorParser()
    parser.feed(html)
    for href, text in parser.links:
        if "list of nifty 50 stocks" not in text.casefold():
            continue
        candidate = urljoin(page_url, href)
        parsed = urlparse(candidate)
        if parsed.scheme != "https" or parsed.hostname != EXPECTED_ROSTER_CSV_HOST:
            continue
        if parsed.path != EXPECTED_ROSTER_CSV_PATH:
            continue
        return candidate
    raise RosterError("Official NSE page did not provide the expected NIFTY 50 constituent CSV link.")


def validate_official_roster(
    raw_csv: bytes,
    csv_url: str,
    retrieved_at_utc: str,
    expected_count: int = EXPECTED_CONSTITUENT_COUNT,
) -> OfficialRoster:
    try:
        frame = pd.read_csv(BytesIO(raw_csv))
    except Exception as exc:
        raise RosterError(f"Official NSE constituent CSV could not be parsed: {exc}") from exc

    normalized = {str(column).strip().casefold(): column for column in frame.columns}
    missing = ROSTER_COLUMNS.difference(normalized)
    if missing:
        raise RosterError("Official NSE constituent CSV is missing required columns: " + ", ".join(sorted(missing)))
    company_column = normalized["company name"]
    symbol_column = normalized["symbol"]
    companies = frame[company_column].astype("string").str.strip()
    symbols = frame[symbol_column].astype("string").str.strip().str.upper()
    if frame.empty or companies.isna().any() or companies.eq("").any():
        raise RosterError("Official NSE constituent CSV contains missing company names.")
    if symbols.isna().any() or symbols.eq("").any():
        raise RosterError("Official NSE constituent CSV contains missing NSE symbols.")
    if symbols.duplicated().any():
        raise RosterError("Official NSE constituent CSV contains duplicate NSE symbols.")
    if len(frame) != expected_count:
        raise RosterError(
            f"Official NSE constituent CSV contains {len(frame)} rows; expected "
            f"{expected_count}. Refusing an unverified roster."
        )
    return OfficialRoster(raw_csv, csv_url, retrieved_at_utc, frame)


def fetch_official_roster() -> OfficialRoster:
    """Retrieve the linked official NSE roster and validate its 50 symbols."""
    session = requests.Session()
    # The roster is public. Avoid inheriting a broken process proxy; this is
    # limited to this in-process session and does not alter machine settings.
    session.trust_env = False
    headers = {"User-Agent": USER_AGENT}
    try:
        page = session.get(OFFICIAL_ROSTER_PAGE_URL, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
        page.raise_for_status()
        csv_url = _official_csv_link(page.text, page.url)
        csv_response = session.get(
            csv_url,
            headers={**headers, "Accept": "text/csv,*/*"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        csv_response.raise_for_status()
    except RosterError:
        raise
    except Exception as exc:
        raise RosterError(f"Could not retrieve the official NSE constituent roster: {type(exc).__name__}: {exc}") from exc

    content_type = csv_response.headers.get("Content-Type", "").casefold()
    if "csv" not in content_type and not csv_response.url.casefold().endswith(".csv"):
        raise RosterError(f"Official NSE roster response was not identified as CSV (Content-Type: {content_type or 'missing'}).")
    retrieved_at = datetime.now(timezone.utc).isoformat()
    return validate_official_roster(csv_response.content, csv_response.url, retrieved_at)


def map_nse_to_yahoo(nse_symbol: str, overrides: Mapping[str, str] | None = None) -> tuple[str, str | None]:
    """Map an NSE symbol to Yahoo's NSE suffix and report any override."""
    symbol = str(nse_symbol).strip().upper()
    if not symbol:
        raise ValueError("NSE symbol cannot be empty.")
    override_map = {str(k).strip().upper(): str(v).strip().upper() for k, v in (overrides or YAHOO_TICKER_OVERRIDES).items()}
    if symbol in override_map:
        return override_map[symbol], f"Explicit Yahoo ticker override: {symbol} -> {override_map[symbol]}"
    return f"{symbol}.NS", None


def _flatten_yfinance_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame.copy()
    known = {"open", "high", "low", "close", "adj close", "volume", "dividends", "stock splits", "capital gains"}
    flattened = []
    for column in frame.columns:
        parts = [str(part).strip() for part in column if str(part).strip()]
        price_name = next((part for part in parts if part.casefold() in known), None)
        flattened.append(price_name if price_name is not None else " | ".join(parts))
    result = frame.copy()
    result.columns = flattened
    return result


def _date_series(frame: pd.DataFrame) -> tuple[pd.Series | None, bool]:
    date_column = next((column for column in frame.columns if str(column).strip().casefold() == "date"), None)
    if date_column is not None:
        return pd.to_datetime(frame[date_column], errors="coerce"), True
    if isinstance(frame.index, pd.DatetimeIndex):
        return pd.Series(pd.to_datetime(frame.index, errors="coerce"), index=frame.index), True
    return None, False


def validate_history(frame: pd.DataFrame | None) -> tuple[pd.DataFrame | None, list[str], str | None, str | None]:
    """Return flattened history, validation issues, and date bounds."""
    if frame is None:
        return None, ["Yahoo Finance returned no DataFrame."], None, None
    result = _flatten_yfinance_columns(frame)
    if result.empty:
        return result, ["Yahoo Finance returned an empty history."], None, None

    problems: list[str] = []
    dates, has_dates = _date_series(result)
    if not has_dates or dates is None:
        problems.append("Missing dates: expected a DatetimeIndex or Date column.")
        first_date = last_date = None
    else:
        invalid_dates = dates.isna()
        if invalid_dates.any():
            problems.append(f"{int(invalid_dates.sum())} row(s) have invalid or missing dates.")
        valid_dates = dates[~invalid_dates]
        duplicate_count = int(valid_dates.duplicated(keep=False).sum())
        if duplicate_count:
            problems.append(f"{duplicate_count} row(s) have duplicate dates.")
        first_date = valid_dates.min().date().isoformat() if not valid_dates.empty else None
        last_date = valid_dates.max().date().isoformat() if not valid_dates.empty else None

    columns_by_name = {str(column).strip().casefold(): column for column in result.columns}
    if "close" not in columns_by_name:
        problems.append("Missing required Close column.")
    price_names = ("open", "high", "low", "close", "adj close")
    for price_name in price_names:
        if price_name not in columns_by_name:
            continue
        column = columns_by_name[price_name]
        original = result[column]
        numeric = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & numeric.isna()
        nonfinite = numeric.notna() & ~numeric.map(lambda value: pd.notna(value) and abs(value) != float("inf"))
        nonpositive = numeric.notna() & (numeric <= 0)
        missing = numeric.isna()
        count = int(invalid.sum() + nonfinite.sum() + nonpositive.sum() + missing.sum())
        if count:
            problems.append(f"{column} contains {count} missing, invalid, non-finite, or nonpositive price value(s).")
    return result, problems, first_date, last_date


def build_manifest_row(
    company_name: str,
    nse_symbol: str,
    yahoo_ticker: str,
    mapping_exception: str | None,
    retrieval_time_utc: str,
    frame: pd.DataFrame | None,
    status: str,
    error_details: str | None = None,
    validation_problems: list[str] | None = None,
    first_date: str | None = None,
    last_date: str | None = None,
) -> dict[str, object]:
    return {
        "company_name": company_name,
        "nse_symbol": nse_symbol,
        "yahoo_ticker": yahoo_ticker,
        "mapping_exception": mapping_exception or "",
        "retrieval_time_utc": retrieval_time_utc,
        "row_count": 0 if frame is None else len(frame),
        "first_date": first_date or "",
        "last_date": last_date or "",
        "returned_columns": json.dumps([] if frame is None else [str(column) for column in frame.columns]),
        "status": status,
        "error_details": error_details or "",
        "validation_problems": json.dumps(validation_problems or []),
    }


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows, columns=MANIFEST_FIELDS).to_csv(temp_path, index=False, encoding="utf-8")
    temp_path.replace(path)


def _write_history(path: Path, frame: pd.DataFrame) -> None:
    export = frame.copy()
    temp_path = path.with_suffix(path.suffix + ".tmp")
    if isinstance(export.index, pd.DatetimeIndex):
        export.index.name = "Date"
        export.to_csv(temp_path, index=True, index_label="Date", encoding="utf-8")
    elif any(str(column).strip().casefold() == "date" for column in export.columns):
        export.to_csv(temp_path, index=False, encoding="utf-8")
    else:
        export.to_csv(temp_path, index=True, index_label="source_index", encoding="utf-8")
    temp_path.replace(path)


def _default_output_dir(now: datetime | None = None) -> Path:
    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    temp_root = Path(gettempdir()).resolve()
    project_root = Path(__file__).resolve().parent
    run_dir = temp_root / TEMP_OUTPUT_FOLDER / f"run_{timestamp}"
    resolved = run_dir.resolve()
    if resolved == project_root or project_root in resolved.parents:
        raise RuntimeError("Resolved TEMP output path is inside the project; refusing to create output there.")
    if temp_root != resolved and temp_root not in resolved.parents:
        raise RuntimeError("Resolved output path is not inside the Windows TEMP directory.")
    return run_dir


def _new_yfinance_session():
    """Create a per-ticker yfinance session that ignores environment proxies."""
    try:
        from curl_cffi import requests as curl_requests
    except ImportError as exc:
        raise RuntimeError("yfinance's curl_cffi session support is unavailable in this environment.") from exc
    session = curl_requests.Session(impersonate="chrome")
    session.trust_env = False
    return session


def _download_one(ticker: str) -> pd.DataFrame | None:
    session = _new_yfinance_session()
    try:
        return yf.download(
            ticker,
            period=DOWNLOAD_PERIOD,
            interval=DOWNLOAD_INTERVAL,
            auto_adjust=False,
            multi_level_index=False,
            progress=False,
            threads=False,
            timeout=REQUEST_TIMEOUT_SECONDS,
            session=session,
        )
    finally:
        session.close()


def run_download(
    output_dir: Path | str | None = None,
    roster_fetcher: Callable[[], OfficialRoster] | None = None,
    download_one: Callable[[str], pd.DataFrame | None] | None = None,
    clock: Callable[[], datetime] | None = None,
    ticker_overrides: Mapping[str, str] | None = None,
) -> Path:
    """Fetch roster, then independently download and record each constituent."""
    roster = (roster_fetcher or fetch_official_roster)()
    now = clock or (lambda: datetime.now(timezone.utc))
    destination = Path(output_dir) if output_dir is not None else _default_output_dir(now())
    destination.mkdir(parents=True, exist_ok=False)

    roster_path = destination / "nifty50_constituents_source.csv"
    roster_path.write_bytes(roster.raw_csv)
    metadata = {
        "source_page_url": OFFICIAL_ROSTER_PAGE_URL,
        "source_csv_url": roster.csv_url,
        "retrieved_at_utc": roster.retrieved_at_utc,
        "sha256": hashlib.sha256(roster.raw_csv).hexdigest(),
        "row_count": len(roster.frame),
    }
    (destination / "nifty50_constituents_source.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    history_dir = destination / "history"
    history_dir.mkdir()

    downloader = download_one or _download_one
    manifest_rows: list[dict[str, object]] = []
    manifest_path = destination / "manifest.csv"
    for roster_index, (_, constituent) in enumerate(roster.frame.iterrows(), start=1):
        company_name = str(constituent[next(c for c in roster.frame.columns if str(c).strip().casefold() == "company name")]).strip()
        nse_symbol = str(constituent[next(c for c in roster.frame.columns if str(c).strip().casefold() == "symbol")]).strip().upper()
        yahoo_ticker, mapping_exception = map_nse_to_yahoo(nse_symbol, ticker_overrides)
        retrieved_at = ""
        try:
            returned = downloader(yahoo_ticker)
            retrieved_at = now().astimezone(timezone.utc).isoformat()
            normalized, problems, first_date, last_date = validate_history(returned)
            if normalized is None or normalized.empty:
                status = "failed"
                error = "; ".join(problems)
            else:
                # Preserve a nonempty response even when validation flags it;
                # its manifest status prevents it from being mistaken as usable.
                filename = f"{roster_index:02d}_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", yahoo_ticker) + ".csv"
                _write_history(history_dir / filename, normalized)
                status = "success" if not problems else "validation_failed"
                error = "" if not problems else "; ".join(problems)
            row = build_manifest_row(
                company_name, nse_symbol, yahoo_ticker, mapping_exception, retrieved_at,
                normalized, status, error, problems, first_date, last_date,
            )
        except Exception as exc:
            if not retrieved_at:
                retrieved_at = now().astimezone(timezone.utc).isoformat()
            row = build_manifest_row(
                company_name, nse_symbol, yahoo_ticker, mapping_exception, retrieved_at,
                None, "failed", f"{type(exc).__name__}: {exc}",
            )
        manifest_rows.append(row)
        _write_manifest(manifest_path, manifest_rows)
        print(f"{nse_symbol} ({yahoo_ticker}): {row['status']}")

    print(f"Dataset output: {destination}")
    print(f"Manifest: {manifest_path}")
    return destination


def main() -> int:
    try:
        run_download()
    except RosterError as exc:
        print(f"Roster verification failed; no ticker downloads were started: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Downloader stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
