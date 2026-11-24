"""Search across many independently anchored historical source families.

This is evidence discovery, not a source-truth oracle. Unknown keys, unreviewed
catalogs and ambiguous matches never become science. Every discovered candidate
is later rehashed by VerifiedSourceAcquirer against its recorded source SHA.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from hephaestus.schemas.research_objective import (
    ResearchAdmissionError,
    ResearchObjective, SourceCatalog, SourceDescriptor,
)


def _tokens(value: str) -> frozenset[str]:
    # Search tokens are NOT evidence, confidence or independent labels.
    stop={"research","study","investigate","diagnostician","diagnostic",
          "reliability","and","from","into","the","with","original",
          "evidence","experiment","evaluate","evaluating","comparison"}
    return frozenset(x for x in re.findall(r"[a-z0-9]{3,}",value.lower()) if x not in stop)


@dataclass(frozen=True, slots=True)
class ReviewedSearchResult:
    catalog: SourceCatalog
    original: SourceDescriptor
    candidate: SourceDescriptor
    matched_terms: tuple[str,...]
    selection_kind: str


@dataclass(slots=True)
class ReviewedEvidenceDiscovery:
    """Allowlisted multiple versioned catalogs, each with its own trust anchor."""

    catalogs: tuple[SourceCatalog, ...]
    maximum_catalogs: int = 24

    @classmethod
    def open(cls, admitted_sources: tuple[tuple[Path,str],...]) -> "ReviewedEvidenceDiscovery":
        if not 1<=len(admitted_sources)<=24:
            raise ResearchAdmissionError("reviewed source index must contain 1..24 manifests")
        indexes=tuple(SourceCatalog.load(path,independently_pinned_sha256=digest)
                      for path,digest in admitted_sources)
        if len(set(x.catalog_sha256 for x in indexes))!=len(indexes):
            raise ResearchAdmissionError("duplicate reviewed catalog is not an independent source")
        identities=set()
        for index in indexes:
            for record in index.sources:
                if record.source_id in identities:
                    raise ResearchAdmissionError("same source identity in conflicting catalogs")
                identities.add(record.source_id)
        return cls(indexes)

    def search(self, question: str, *, required_scope: str | None = None) -> ReviewedSearchResult:
        objective=ResearchObjective.from_text(question)
        objective.validate()
        candidates=[]
        for catalog in self.catalogs:
            try:
                original,candidate=catalog.select(objective)
            except ResearchAdmissionError:
                # A catalog with more than one family is ambiguous by contract.
                continue
            if original.study_scope != candidate.study_scope:
                continue
            if required_scope is not None and original.study_scope != required_scope:
                continue
            search_terms=_tokens(original.study_scope+" "+original.subject+" "+candidate.subject)
            matched=tuple(sorted(search_terms & _tokens(question)))
            candidates.append((len(matched),catalog.catalog_sha256,catalog,
                               original,candidate,matched))
        if not candidates:
            raise ResearchAdmissionError("no source-verified matched study available")
        if len(candidates)==1:
            _,_,catalog,original,candidate,matched=candidates[0]
            return ReviewedSearchResult(catalog,original,candidate,matched,"unique_reviewed_scope")
        candidates.sort(key=lambda x:(-x[0],x[1]))
        if candidates[0][0]<1 or candidates[0][0]==candidates[1][0]:
            raise ResearchAdmissionError(
                "historical evidence discovery ambiguous; exact reviewed study scope required")
        _,_,catalog,original,candidate,matched=candidates[0]
        return ReviewedSearchResult(catalog,original,candidate,matched,
                                    "deterministic_topic_match_not_scientific_gold")
