"""Human-authored diagnostic plans for development-only benchmark ablations.

Nothing in this module is a production question compiler.  In particular, the
Aster Prism compiler below was written after inspecting that synthetic fixture
and encodes its relation schema, source-type layout, and lexical forms.  It is
useful for separating two questions:

* can the Witness executor carry exact bindings through a multi-hop program?
* can an automatic compiler infer that program from a previously unseen query?

Only the first question may be evaluated with this module.  Results from these
plans are deliberately marked ineligible for headline or fair baseline claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from witness_engine import (
    DependencySpec,
    PredicateSpec,
    QuestionCompiler,
    ScanSpec,
    WitnessProgram,
)

from .witness_system import WitnessOfflineSystem


@dataclass(frozen=True, slots=True)
class _Hop:
    scan_id: str
    predicate_id: str
    description: str
    source_types: tuple[str, ...]
    must_terms: tuple[str, ...]
    binding_name: str
    binding_pattern: str
    dependency_placeholder: str | None = None
    depends_on: str | None = None
    upstream_binding: str | None = None


# These forms intentionally mirror the visible development fixture.  They do
# not contain scorer-only evidence IDs or expected answer values, but they are
# still oracle-like because a human inspected the fixture before writing them.
_ASTER_HOPS: tuple[_Hop, ...] = (
    _Hop(
        scan_id="designer",
        predicate_id="aster_designer",
        description="Find the person credited with designing the Aster Prism optical core.",
        source_types=("minutes",),
        must_terms=("Aster Prism", "optical core", "design"),
        binding_name="person",
        binding_pattern=r"\b[A-Z][a-z]+ [A-Z][a-z]+(?=—not\b)",
    ),
    _Hop(
        scan_id="founded_organization",
        predicate_id="organization_founded_by_designer",
        description="Find the organization whose founder is {person}.",
        source_types=("registry",),
        must_terms=("{person}", "incorporation ledger", "sole founder"),
        binding_name="organization",
        binding_pattern=r"(?<=The )[A-Z][a-z]+ [A-Z][a-z]+(?= incorporation ledger)",
        dependency_placeholder="person",
        depends_on="designer",
        upstream_binding="person",
    ),
    _Hop(
        scan_id="acquirer",
        predicate_id="company_acquiring_organization",
        description="Find the company that acquired {organization}.",
        source_types=("registry",),
        must_terms=("{organization}", "completed its acquisition"),
        binding_name="company",
        binding_pattern=(
            r"\b[A-Z][a-z]+ [A-Z][a-z]+"
            r"(?= completed its acquisition of {organization}\b)"
        ),
        dependency_placeholder="organization",
        depends_on="founded_organization",
        upstream_binding="organization",
    ),
    _Hop(
        scan_id="controller",
        predicate_id="controller_of_acquirer",
        description="Find the controlling company of {company}.",
        source_types=("registry",),
        must_terms=("{company}", "controlling company"),
        binding_name="controller",
        binding_pattern=(
            r"(?<=identify )[A-Z][a-z]+ [A-Z][a-z]+"
            r"(?= as the controlling company of {company}\b)"
        ),
        dependency_placeholder="company",
        depends_on="acquirer",
        upstream_binding="company",
    ),
    _Hop(
        scan_id="buyer",
        predicate_id="buyer_of_controller",
        description="Find the buyer of {controller}.",
        source_types=("registry",),
        must_terms=("{controller}", "purchased all outstanding shares"),
        binding_name="buyer",
        binding_pattern=(
            r"\b[A-Z][a-z]+ [A-Z][a-z]+"
            r"(?= purchased all outstanding shares of {controller}\b)"
        ),
        dependency_placeholder="controller",
        depends_on="controller",
        upstream_binding="controller",
    ),
    _Hop(
        scan_id="ultimate_owner",
        predicate_id="ultimate_owner_of_buyer",
        description="Find the ultimate owner of {buyer}.",
        source_types=("registry",),
        must_terms=("{buyer}", "ultimate owner"),
        binding_name="owner",
        binding_pattern=(
            r"(?<=records )[A-Z][a-z]+ [A-Z][a-z]+"
            r"(?= as the ultimate owner of {buyer}\b)"
        ),
        dependency_placeholder="buyer",
        depends_on="buyer",
        upstream_binding="buyer",
    ),
)


def aster_prism_depth(question: str) -> int | None:
    """Return the fixture hop depth, or ``None`` for an unsupported question."""

    normalized = " ".join(question.casefold().replace("-", " ").split())
    if "aster prism" not in normalized:
        return None
    if "ultimate owner" in normalized:
        return 6
    if "purchased" in normalized:
        return 5
    if "controls" in normalized or "controlling company" in normalized:
        return 4
    if "acquired" in normalized or "acquirer" in normalized:
        return 3
    if "founded" in normalized:
        return 2
    return None


def compile_aster_prism_development_plan(
    question: str,
    *,
    as_of: str | None = None,
    recorded_as_of: str | None = None,
    inputs: Mapping[str, Any] | None = None,
) -> WitnessProgram | None:
    """Build a fixture-specific dependency program for the supported dev tasks.

    Returning ``None`` rather than guessing makes its narrow scope explicit.
    The exact final entity is never embedded in the plan; it must be extracted
    from the final source row after every upstream binding succeeds.
    """

    depth = aster_prism_depth(question)
    if depth is None:
        return None
    selected = _ASTER_HOPS[:depth]
    scans = tuple(
        ScanSpec(
            scan_id=hop.scan_id,
            source_types=hop.source_types,
            predicate=PredicateSpec(
                predicate_id=hop.predicate_id,
                description=hop.description,
                must_terms=hop.must_terms,
                binding_patterns={hop.binding_name: hop.binding_pattern},
                binding_types={hop.binding_name: "entity"},
                required_bindings=(hop.binding_name,),
                # The handwritten lexical rule is precise for this fixture,
                # but lexical absence is not a general semantic proof.
                closed_world=False,
                quote_window=360,
                modality="synthetic_registry_statement",
            ),
        )
        for hop in selected
    )
    dependencies = tuple(
        DependencySpec(
            scan_id=hop.scan_id,
            depends_on=str(hop.depends_on),
            bindings={str(hop.dependency_placeholder): str(hop.upstream_binding)},
        )
        for hop in selected
        if hop.depends_on is not None
    )
    final_binding = selected[-1].binding_name
    program = WitnessProgram(
        question=question,
        scans=scans,
        dependencies=dependencies,
        inputs=dict(inputs or {}),
        answer_contract={
            "mode": "evidence_rows",
            "final_relation": selected[-1].scan_id,
            "final_binding": final_binding,
            "expected_hop_depth": depth,
            "must_cite_witness_ids": True,
            "development_oracle": True,
            "headline_eligible": False,
        },
        as_of=as_of,
        recorded_as_of=recorded_as_of,
    )
    QuestionCompiler.validate(program)
    return program


class AsterPrismDevelopmentOracleCompiler(QuestionCompiler):
    """Fixture-specific human planner with conservative fallback elsewhere."""

    COMPILER_VERSION = "aster-human-development-oracle-0.1.0"

    def compile(
        self,
        question: str,
        *,
        as_of: str | None = None,
        recorded_as_of: str | None = None,
        inputs: Mapping[str, Any] | None = None,
    ) -> WitnessProgram:
        program = compile_aster_prism_development_plan(
            question,
            as_of=as_of,
            recorded_as_of=recorded_as_of,
            inputs=inputs,
        )
        if program is not None:
            return program
        return super().compile(
            question,
            as_of=as_of,
            recorded_as_of=recorded_as_of,
            inputs=inputs,
        )


class AsterPrismDevelopmentOracleSystem(WitnessOfflineSystem):
    """Non-headline adapter that executes the handwritten Aster plan."""

    name = "witness-aster-human-plan-development-oracle"

    def __init__(self, **kwargs: Any) -> None:
        if "compiler" in kwargs:
            raise TypeError("this diagnostic system fixes its compiler")
        if kwargs.pop("counter_search", False):
            raise ValueError("counter search is disabled in this execution-only ablation")
        super().__init__(
            compiler=AsterPrismDevelopmentOracleCompiler(),
            counter_search=False,
            **kwargs,
        )
        self.name = "witness-aster-human-plan-development-oracle"

    @property
    def metadata(self) -> dict[str, Any]:
        output = super().metadata
        output.update(
            {
                "benchmark_role": "development_oracle_execution_diagnostic",
                "human_authored_plan": True,
                "fixture": "synthetic_aster_prism_only",
                "headline_eligible": False,
                "fair_baseline_comparison": False,
            }
        )
        output["limitations"] = [
            "A human inspected the synthetic Aster Prism fixture before writing this plan.",
            "The plan encodes fixture relation names, source types, and lexical forms.",
            "It tests bound-variable execution, not automatic question compilation.",
            "Its result must not be used as evidence that Witness beats a baseline.",
            "The local evaluator remains lexical rather than a production semantic model.",
        ]
        return output


__all__ = [
    "AsterPrismDevelopmentOracleCompiler",
    "AsterPrismDevelopmentOracleSystem",
    "aster_prism_depth",
    "compile_aster_prism_development_plan",
]
