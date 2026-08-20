#!/usr/bin/env python3
"""Which phrases the detector could actually emit, read from the run that ran it.

A closed-vocabulary detector cannot produce a class its source vocabulary does
not contain. Under the YOLO11x provider (C23) four of the taxonomy's ten
phrases have no COCO source class at all — "a road barrier", "a traffic cone",
"a construction vehicle", "a trailer" — together **21.3 % of this substrate's
ground truth**. Recall on them is 0 BY CONSTRUCTION.

Scoring those four as misses does not measure the pipeline; it measures the
gap between two vocabularies, and it does it silently. A ten-class recall is
then four structural zeros averaged with six real numbers, and every summary
built on it understates the pipeline by a fixed amount nobody can see.

**The reachable set is a property of the RUN, not of the checkout.** It is read
from Stage 3's `run_manifest.json` (`class_map.phrases_in_use`), which records
the class map actually loaded, with its sha256. Reading the config file on disk
instead would describe whatever the working tree happens to hold now, which is
how an eval comes to describe a run that never happened. The config file is a
declared fallback, used only when no manifest exists (a viz script pointed at a
tree with no Stage 3), and the choice is recorded in `source`.

Suppression is DECLARED AND COUNTED, never silent (the rule the Stage 6
provenance dicts state): every consumer reports how many GT objects it set
aside and why, so a recall number and the objects excluded from its denominator
are always readable together.

    from pipeline.common.class_space import load_detectable_classes
    detectable = load_detectable_classes(paths.work_root)
    if not detectable.is_reachable(phrase):
        n_suppressed += 1
        continue
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

import yaml

SPEC = "dhakascenes-pilot/class_space/v1"
DEFAULT_CLASS_MAP = "configs/coco_to_phrase_nuscenes.yaml"
DEFAULT_TAXONOMY = "configs/taxonomy_pilot_nuscenes.yaml"
PROPOSAL_STAGE = "stage3_proposals"

# An open-vocabulary provider is prompted with the phrases themselves, so every
# phrase in the taxonomy is reachable and there is no class map to read. Listed
# by name rather than assumed: a provider absent from both this set and the
# class-map path is a provider whose class space nobody has established.
OPEN_VOCABULARY_PROVIDERS: tuple[str, ...] = ("llmdet_hf", "grounding_dino_hf")


class ClassSpaceError(RuntimeError):
    """The reachable class space could not be established for this run."""


@dataclass(frozen=True)
class DetectableClasses:
    """The phrases a given run's detector could emit, and the ones it could not."""

    reachable: tuple[str, ...]
    unreachable: tuple[str, ...]
    provider: str
    source: str
    class_map_path: str = ""
    class_map_sha256: str = ""
    _reachable_set: frozenset = field(default_factory=frozenset, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_reachable_set", frozenset(self.reachable))

    def is_reachable(self, phrase: str | None) -> bool:
        """True if the detector could have produced this phrase at all.

        `None` — a GT category the taxonomy does not map — is NOT reachable, but
        it is a different exclusion (out of the class space entirely, C21) and
        callers must count it separately.
        """
        return phrase is not None and phrase in self._reachable_set

    def as_dict(self) -> dict:
        return {
            "spec": SPEC,
            "provider": self.provider,
            "source": self.source,
            "reachable_phrases": list(self.reachable),
            "unreachable_phrases": list(self.unreachable),
            "class_map_path": self.class_map_path,
            "class_map_sha256": self.class_map_sha256,
        }

    def describe(self) -> str:
        if not self.unreachable:
            return f"all {len(self.reachable)} phrases reachable ({self.provider})"
        return (
            f"{len(self.reachable)} of {len(self.reachable) + len(self.unreachable)} phrases "
            f"reachable under {self.provider}; unreachable: {', '.join(self.unreachable)}"
        )


def _taxonomy_phrases(taxonomy_path: str) -> tuple[str, ...]:
    with open(taxonomy_path, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    mapping = doc.get("prompt_phrase") or {}
    if not mapping:
        raise ClassSpaceError(f"{taxonomy_path}: no `prompt_phrase` block")
    seen: dict[str, None] = {}
    for phrase in mapping.values():
        seen.setdefault(str(phrase).strip(), None)
    return tuple(seen)


def _from_manifest(manifest_path: str, taxonomy_path: str) -> DetectableClasses | None:
    """Read the class space Stage 3 recorded, or None if it recorded none."""
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    provider = str(manifest.get("provider", "") or "")
    block = manifest.get("class_map") or {}
    reachable = tuple(block.get("phrases_in_use") or ())
    if not reachable:
        # An open-vocabulary run records no class map: every phrase is reachable.
        if provider in OPEN_VOCABULARY_PROVIDERS:
            return DetectableClasses(
                reachable=_taxonomy_phrases(taxonomy_path),
                unreachable=(),
                provider=provider,
                source=f"{manifest_path} (open-vocabulary provider, no class map)",
            )
        return None
    unreachable = tuple(block.get("unreachable_phrases") or ())
    if not unreachable:
        # Derive it rather than trust its absence: an older manifest may predate
        # the field while still carrying the mapping that determines it.
        in_use = set(reachable)
        unreachable = tuple(p for p in _taxonomy_phrases(taxonomy_path) if p not in in_use)
    return DetectableClasses(
        reachable=reachable,
        unreachable=unreachable,
        provider=provider or "unknown",
        source=manifest_path,
        class_map_path=str(block.get("path", "")),
        class_map_sha256=str(block.get("sha256", "")),
    )


def _from_config(class_map_path: str, taxonomy_path: str) -> DetectableClasses:
    """Declared fallback: the class map as it sits on disk right now."""
    if not os.path.isfile(class_map_path):
        raise ClassSpaceError(
            f"no Stage 3 manifest and no class map at {class_map_path}: the set of phrases the "
            "detector could emit is unknown, and scoring every phrase would silently count "
            "unreachable classes as misses"
        )
    with open(class_map_path, "rb") as fh:
        raw = fh.read()
    doc = yaml.safe_load(raw.decode("utf-8")) or {}
    mapping = doc.get("coco_to_phrase") or {}
    if not mapping:
        raise ClassSpaceError(f"{class_map_path}: no `coco_to_phrase` block")
    seen: dict[str, None] = {}
    for phrase in mapping.values():
        seen.setdefault(str(phrase).strip(), None)
    reachable = tuple(seen)
    unreachable = tuple(p for p in _taxonomy_phrases(taxonomy_path) if p not in seen)
    return DetectableClasses(
        reachable=reachable,
        unreachable=unreachable,
        provider="unknown (no manifest)",
        source=f"{os.path.realpath(class_map_path)} (fallback: no Stage 3 manifest)",
        class_map_path=os.path.realpath(class_map_path),
        class_map_sha256=hashlib.sha256(raw).hexdigest(),
    )


def load_detectable_classes(
    work_root: str,
    *,
    taxonomy_path: str = DEFAULT_TAXONOMY,
    class_map_path: str = DEFAULT_CLASS_MAP,
    stage: str = PROPOSAL_STAGE,
) -> DetectableClasses:
    """The class space of the run under `work_root`, from its own Stage 3 manifest."""
    manifest_path = os.path.join(work_root, stage, "run_manifest.json")
    if os.path.isfile(manifest_path):
        found = _from_manifest(manifest_path, taxonomy_path)
        if found is not None:
            return found
    return _from_config(class_map_path, taxonomy_path)


def all_phrases_reachable(taxonomy_path: str = DEFAULT_TAXONOMY) -> DetectableClasses:
    """Every phrase reachable — what `--include-unreachable` selects.

    Not a default anywhere: it is the explicit opt-out, so that a report scoring
    the four structural zeros says so in its own `class_space.source`.
    """
    return DetectableClasses(
        reachable=_taxonomy_phrases(taxonomy_path),
        unreachable=(),
        provider="all (suppression disabled by flag)",
        source="--include-unreachable",
    )
