"""ClinicalTrials.gov API v2 client with pagination, retry, and caching.

Fetches clinical trial records for a given condition, validates them
through the TrialRecord schema, and writes JSONL to data/raw/.
"""

from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from neuroguard.evidence.schema import TrialRecord, parse_study
from neuroguard.logging_setup import get_logger

logger = get_logger(__name__)

CTGOV_API_BASE = "https://clinicaltrials.gov/api/v2/studies"
DEFAULT_PAGE_SIZE = 100


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
async def _fetch_page(
    client: httpx.AsyncClient,
    condition: str,
    page_size: int,
    page_token: str | None = None,
) -> dict:
    """Fetch a single page from the CT.gov API v2.

    Args:
        client: httpx async client.
        condition: Condition search term (e.g. "Alzheimer Disease").
        page_size: Number of studies per page.
        page_token: Pagination token for next page (None for first).

    Returns:
        Raw JSON response dict.

    Raises:
        httpx.HTTPStatusError: On non-2xx responses after retries.
    """
    params: dict = {
        "query.cond": condition,
        "pageSize": page_size,
        "countTotal": "true",
    }
    if page_token:
        params["pageToken"] = page_token

    response = await client.get(CTGOV_API_BASE, params=params)
    response.raise_for_status()
    return response.json()


async def fetch_trials(
    condition: str = "Alzheimer Disease",
    max_studies: int = 500,
    page_size: int = DEFAULT_PAGE_SIZE,
    output_dir: Path | None = None,
) -> list[TrialRecord]:
    """Fetch clinical trials from CT.gov API v2 with pagination.

    Writes a JSONL cache to ``output_dir`` on completion.

    Args:
        condition: Condition to search for.
        max_studies: Maximum number of studies to fetch.
        page_size: Studies per API page (max 1000).
        output_dir: Directory to write JSONL cache. Defaults to data/raw/.

    Returns:
        List of validated TrialRecord objects.
    """
    if output_dir is None:
        output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[TrialRecord] = []
    page_token: str | None = None
    total_available: int | None = None

    logger.info(
        "fetch_trials_start",
        condition=condition,
        max_studies=max_studies,
        page_size=page_size,
    )

    async with httpx.AsyncClient(timeout=30.0) as client:
        with tqdm(total=max_studies, desc="Fetching trials", unit="study") as pbar:
            while len(records) < max_studies:
                remaining = max_studies - len(records)
                this_page_size = min(page_size, remaining)

                data = await _fetch_page(client, condition, this_page_size, page_token)

                # Update total on first page
                if total_available is None:
                    total_available = data.get("totalCount", 0)
                    effective_total = min(max_studies, total_available)
                    pbar.total = effective_total
                    logger.info(
                        "total_available",
                        total=total_available,
                        fetching=effective_total,
                    )

                studies = data.get("studies", [])
                if not studies:
                    break

                for raw_study in studies:
                    if len(records) >= max_studies:
                        break
                    try:
                        record = parse_study(raw_study)
                        records.append(record)
                        pbar.update(1)
                    except Exception as exc:
                        nct = (
                            raw_study.get("protocolSection", {})
                            .get("identificationModule", {})
                            .get("nctId", "UNKNOWN")
                        )
                        logger.warning("parse_error", nct_id=nct, error=str(exc))

                page_token = data.get("nextPageToken")
                if not page_token:
                    break

    # Write JSONL cache
    safe_condition = condition.lower().replace(" ", "_")
    output_path = output_dir / f"clinicaltrials_{safe_condition}.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")

    logger.info(
        "fetch_trials_complete",
        count=len(records),
        output=str(output_path),
    )
    return records


def load_cached_trials(path: Path) -> list[TrialRecord] | None:
    """Load previously fetched trials from a JSONL file.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of TrialRecord objects, or None if file doesn't exist.
    """
    if not path.exists():
        return None

    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(TrialRecord.model_validate_json(line))

    logger.info("cache_loaded", path=str(path), count=len(records))
    return records
