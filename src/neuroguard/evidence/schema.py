"""Pydantic schemas for ClinicalTrials.gov trial records.

Defines the flat TrialRecord model and a parser that converts the
nested CT.gov API v2 response into our schema.
"""

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class TrialRecord(BaseModel):
    """A single clinical trial record (Dataset D1).

    Flat representation derived from the nested CT.gov API v2 response.
    One record per trial. Primary key: nct_id.
    """

    nct_id: str = Field(description="NCT identifier, e.g. NCT00000001")
    brief_title: str = Field(description="Short title of the study")
    official_title: str | None = Field(default=None, description="Full official title")
    conditions: list[str] = Field(
        default_factory=list, description="Conditions studied"
    )
    interventions: list[str] = Field(
        default_factory=list, description="Intervention names"
    )
    phase: str | None = Field(default=None, description="Trial phase (e.g. PHASE3)")
    overall_status: str | None = Field(default=None, description="Recruitment status")
    study_type: str | None = Field(
        default=None, description="INTERVENTIONAL or OBSERVATIONAL"
    )
    enrollment: int | None = Field(default=None, description="Number of participants")
    primary_outcomes: list[str] = Field(
        default_factory=list, description="Primary outcome measure names"
    )
    secondary_outcomes: list[str] = Field(
        default_factory=list, description="Secondary outcome measure names"
    )
    eligibility_criteria: str | None = Field(
        default=None, description="Free-text eligibility criteria"
    )
    lead_sponsor: str | None = Field(default=None, description="Lead sponsor name")
    has_results: bool = Field(default=False, description="Whether results are posted")
    start_date: str | None = Field(default=None, description="Study start date")
    completion_date: str | None = Field(
        default=None, description="Study completion date"
    )
    source_url: str = Field(description="Direct URL to the study on CT.gov")
    fetched_at: str = Field(description="ISO timestamp of when this record was fetched")
    api_version: str = Field(default="v2", description="CT.gov API version used")


def _safe_get(d: dict, *keys, default=None):
    """Safely traverse nested dicts without raising KeyError."""
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key, default)
    return d


def parse_study(raw: dict) -> TrialRecord:
    """Parse a single CT.gov API v2 study object into a TrialRecord.

    Args:
        raw: A study object from the CT.gov API v2 ``studies`` array.

    Returns:
        A validated TrialRecord with fields extracted and flattened.
    """
    protocol = raw.get("protocolSection", {})
    ident = protocol.get("identificationModule", {})
    status = protocol.get("statusModule", {})
    conditions_mod = protocol.get("conditionsModule", {})
    design = protocol.get("designModule", {})
    arms = protocol.get("armsInterventionsModule", {})
    outcomes = protocol.get("outcomesModule", {})
    eligibility = protocol.get("eligibilityModule", {})
    sponsor = protocol.get("sponsorCollaboratorsModule", {})

    nct_id = ident.get("nctId", "UNKNOWN")

    # Extract intervention names
    intervention_list = arms.get("interventions", [])
    intervention_names = [i.get("name", "") for i in intervention_list if i.get("name")]

    # Extract outcome measure names
    primary = outcomes.get("primaryOutcomes", [])
    primary_names = [o.get("measure", "") for o in primary if o.get("measure")]
    secondary = outcomes.get("secondaryOutcomes", [])
    secondary_names = [o.get("measure", "") for o in secondary if o.get("measure")]

    # Phases may be a list like ["PHASE2", "PHASE3"]
    phases = design.get("phases", [])
    phase_str = ", ".join(phases) if phases else None

    # Date structs
    start_struct = status.get("startDateStruct", {})
    completion_struct = status.get("completionDateStruct", {})

    return TrialRecord(
        nct_id=nct_id,
        brief_title=ident.get("briefTitle", ""),
        official_title=ident.get("officialTitle"),
        conditions=conditions_mod.get("conditions", []),
        interventions=intervention_names,
        phase=phase_str,
        overall_status=status.get("overallStatus"),
        study_type=design.get("studyType"),
        enrollment=_safe_get(design, "enrollmentInfo", "count"),
        primary_outcomes=primary_names,
        secondary_outcomes=secondary_names,
        eligibility_criteria=eligibility.get("eligibilityCriteria"),
        lead_sponsor=_safe_get(sponsor, "leadSponsor", "name"),
        has_results=raw.get("hasResults", False),
        start_date=start_struct.get("date") if start_struct else None,
        completion_date=completion_struct.get("date") if completion_struct else None,
        source_url=f"https://clinicaltrials.gov/study/{nct_id}",
        fetched_at=datetime.now(timezone.utc).isoformat(),
    )
