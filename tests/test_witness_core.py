from __future__ import annotations

import tempfile
import unittest

from witness_engine import (
    ClaimLedgerEntry,
    ClaimStatus,
    CorpusStore,
    CounterTestSpec,
    CoverageCertificate,
    DependencySpec,
    DeterministicReducer,
    Document,
    EvaluationStatus,
    EvidenceRole,
    LexicalEvidenceEvaluator,
    MaterializedWitnessCache,
    MechanicalVerifier,
    PredicateSpec,
    QuestionCompiler,
    ReducerSpec,
    ScanSpec,
    ValidationError,
    WitnessEngine,
    WitnessProgram,
    WitnessRow,
)


class CorpusTests(unittest.TestCase):
    def test_lossless_versions_stable_offsets_and_permissions(self):
        text = "alpha " * 90 + "decisive exception" + " omega" * 40
        store = CorpusStore(block_size_chars=200, overlap_chars=20)
        document = store.add_document(
            {
                "id": "contract",
                "version": "1",
                "content": text,
                "permissions": ["legal"],
                "metadata": {"kind": "agreement"},
            }
        )
        self.assertEqual(document.raw_content, text)
        self.assertEqual(store.blocks(principals=["sales"]), ())
        blocks = store.blocks(principals=["legal"])
        self.assertGreater(len(blocks), 1)
        covered = [False] * len(text)
        for block in blocks:
            self.assertEqual(text[block.start_offset : block.end_offset], block.content)
            self.assertTrue(store.verify_block(block))
            covered[block.start_offset : block.end_offset] = [True] * len(block.content)
        self.assertTrue(all(covered))
        with self.assertRaises(ValidationError):
            store.add_document(
                {"id": "contract", "version": "1", "content": text + " changed", "permissions": ["legal"]}
            )

    def test_persistent_store_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            first = CorpusStore(storage_path=directory)
            first.add_document({"id": "a", "text": "verbatim bytes", "version": "v1"})
            second = CorpusStore(storage_path=directory)
            self.assertEqual(second.get_document("a", "v1").raw_content, "verbatim bytes")
            self.assertEqual(first.snapshot_hash(), second.snapshot_hash())


class CompilerAndEvaluatorTests(unittest.TestCase):
    def test_compiler_is_conservative_and_rejects_executable_ir(self):
        compiler = QuestionCompiler()
        plan = compiler.compile("Who approved Project Zephyr?", as_of="2025-03-14")
        self.assertEqual(plan.as_of, "2025-03-14")
        self.assertFalse(plan.scans[0].predicate.closed_world)
        raw = plan.to_dict()
        raw["python"] = "__import__('os').system('false')"
        with self.assertRaises(ValidationError):
            compiler.compile_structured(raw)
        nested = plan.to_dict()
        nested["scans"][0]["predicate"]["tool"] = "network"
        with self.assertRaises(ValidationError):
            compiler.compile_structured(nested)

    def test_lexical_miss_is_uncertain_unless_closed_world(self):
        store = CorpusStore()
        store.add_document({"id": "d", "text": "The policy discusses leave."})
        block = store.blocks()[0]
        evaluator = LexicalEvidenceEvaluator()
        open_predicate = PredicateSpec(
            "semantic", "materially related evidence", should_terms=("approval",)
        )
        closed_predicate = PredicateSpec(
            "literal", "literal approval token", should_terms=("approval",), closed_world=True
        )
        self.assertEqual(evaluator.evaluate(block, open_predicate).status, EvaluationStatus.UNCERTAIN)
        self.assertEqual(evaluator.evaluate(block, closed_predicate).status, EvaluationStatus.NO_MATCH)

    def test_exact_binding_and_mechanical_verification(self):
        store = CorpusStore()
        document = store.add_document({"id": "r", "text": "Final revenue was 14.5 million."})
        block = store.blocks()[0]
        predicate = PredicateSpec(
            "revenue",
            "reported final revenue",
            must_terms=("revenue",),
            should_terms=("final",),
            binding_patterns={"amount": r"14\.5"},
            binding_types={"amount": "float"},
            required_bindings=("amount",),
            closed_world=True,
        )
        evaluation = LexicalEvidenceEvaluator().evaluate(block, predicate)
        self.assertEqual(evaluation.status, EvaluationStatus.MATCH)
        row = evaluation.witnesses[0]
        self.assertEqual(row.bindings["amount"].value, 14.5)
        self.assertTrue(MechanicalVerifier().verify(row, store).accepted)
        forged = WitnessRow(
            witness_id=row.witness_id,
            predicate_id=row.predicate_id,
            source_id=row.source_id,
            source_version=row.source_version,
            source_hash=document.content_hash,
            block_id=row.block_id,
            start_offset=row.start_offset,
            end_offset=row.end_offset,
            quote=row.quote.replace("14.5", "99.9"),
            bindings=row.bindings,
            evaluator_version=row.evaluator_version,
        )
        self.assertFalse(MechanicalVerifier().verify(forged, store).accepted)


class EngineTests(unittest.TestCase):
    def test_public_workflow_is_reproducible_and_warm_cache_is_used(self):
        engine = WitnessEngine(max_workers=2)
        engine.add_documents(
            [
                {"id": "policy", "text": "Manager approval is required."},
                {"id": "unrelated", "text": "The cafeteria opens at noon."},
            ]
        )
        plan = engine.compile("Is manager approval required?", as_of="2026-01-01")
        cold = engine.execute(plan)
        warm = engine.execute(plan)
        self.assertTrue(any("approval" in row.quote.casefold() for row in cold.rows))
        self.assertEqual(
            [row.to_dict() for row in cold.rows], [row.to_dict() for row in warm.rows]
        )
        self.assertEqual(
            cold.telemetry.execution_fingerprint, warm.telemetry.execution_fingerprint
        )
        self.assertEqual(cold.telemetry.cache_hits, 0)
        self.assertEqual(warm.telemetry.cache_hits, 2)
        self.assertEqual(warm.telemetry.evaluator_calls, 0)
        self.assertEqual(warm.coverage.uncertain_shards, 1)

    def test_dependency_bindings_drive_later_exhaustive_scans(self):
        engine = WitnessEngine(max_workers=2)
        engine.add_documents(
            [
                {"id": "design", "text": "Mira designed product Orion."},
                {"id": "founding", "text": "Mira founded Nova Labs."},
                {"id": "acquisition", "text": "Apex acquired Nova Labs."},
            ]
        )
        program = WitnessProgram(
            question="Who acquired the company founded by the designer of Orion?",
            scans=(
                ScanSpec(
                    "designer",
                    PredicateSpec(
                        "p_designer",
                        "person who designed Orion",
                        must_terms=("designed",),
                        should_terms=("Orion",),
                        binding_patterns={"person": r"Mira"},
                        required_bindings=("person",),
                        closed_world=True,
                    ),
                ),
                ScanSpec(
                    "organization",
                    PredicateSpec(
                        "p_org",
                        "organization founded by {person}",
                        must_terms=("founded",),
                        should_terms=("{person}",),
                        binding_patterns={"organization": r"Nova Labs"},
                        required_bindings=("organization",),
                        closed_world=True,
                    ),
                ),
                ScanSpec(
                    "acquirer",
                    PredicateSpec(
                        "p_acquirer",
                        "company that acquired {organization}",
                        must_terms=("acquired",),
                        should_terms=("{organization}",),
                        binding_patterns={"company": r"Apex"},
                        required_bindings=("company",),
                        closed_world=True,
                    ),
                ),
            ),
            dependencies=(
                DependencySpec("organization", "designer", {"person": "person"}),
                DependencySpec("acquirer", "organization", {"organization": "organization"}),
            ),
        )
        result = engine.execute(program)
        companies = [row.bindings["company"].value for row in result.rows if "company" in row.bindings]
        self.assertEqual(companies, ["Apex"])
        self.assertEqual(result.coverage.unresolved_shards, 0)
        self.assertEqual(result.telemetry.dependency_expansions, 3)

    def test_explicit_counter_test_qualifies_claim(self):
        program = WitnessProgram(
            question="Does policy require manager approval?",
            scans=(
                ScanSpec(
                    "support",
                    PredicateSpec(
                        "support_required",
                        "support for requirement",
                        must_terms=("approval",),
                        should_terms=("required",),
                        closed_world=True,
                    ),
                ),
            ),
            counter_tests=(
                CounterTestSpec(
                    "optional_exception",
                    "c1",
                    PredicateSpec(
                        "counter_optional",
                        "approval is optional or excepted",
                        must_terms=("approval",),
                        should_terms=("optional", "exception"),
                        closed_world=True,
                        polarity="refutes",
                    ),
                    "An optional rule would falsify an unconditional requirement.",
                ),
            ),
        )
        engine = WitnessEngine(max_workers=2)
        engine.add_documents(
            [
                {"id": "policy", "text": "Manager approval is required."},
                {"id": "amendment", "text": "Amendment: manager approval is optional."},
            ]
        )
        preliminary = engine.execute(program)
        support_id = next(row.witness_id for row in preliminary.rows if "required" in row.quote)
        result = engine.execute(
            program,
            claims=[ClaimLedgerEntry("c1", "Manager approval is required.", (support_id,))],
        )
        self.assertEqual(result.claims[0].status, ClaimStatus.QUALIFIED)
        self.assertTrue(result.claims[0].counter_witness_ids)
        self.assertTrue(any("optional" in row.quote for row in result.counter_rows))

    def test_generated_counter_requires_claim_anchor_and_falsifier_cue(self):
        engine = WitnessEngine(max_workers=1)
        engine.add_documents(
            [
                {"id": "base", "text": "Manager approval is required."},
                {
                    "id": "override",
                    "text": "A later override voided the manager approval requirement.",
                },
            ]
        )
        program = WitnessProgram(
            question="Is manager approval required?",
            scans=(
                ScanSpec(
                    "support",
                    PredicateSpec(
                        "required",
                        "manager approval requirement",
                        must_terms=("manager approval",),
                        should_terms=("required",),
                        closed_world=True,
                    ),
                ),
            ),
        )
        preliminary = engine.execute(program)
        self.assertEqual(preliminary.counter_phase, "awaiting_claims")
        self.assertIn("awaits provisional claims", preliminary.unresolved[-1])
        support_id = next(row.witness_id for row in preliminary.rows)
        checked = engine.execute(
            program,
            claims=[ClaimLedgerEntry("claim", "Manager approval is required", (support_id,))],
        )
        self.assertEqual(checked.counter_phase, "completed")
        self.assertEqual(checked.claims[0].status, ClaimStatus.QUALIFIED)
        self.assertEqual(len(checked.counter_rows), 1)
        self.assertIn("voided", checked.counter_rows[0].quote)

    def test_incremental_update_only_misses_changed_shard(self):
        engine = WitnessEngine(max_workers=1)
        engine.add_documents([{"id": "one", "text": "Issue ALPHA occurred."}])
        program = WitnessProgram(
            question="Where did ALPHA occur?",
            scans=(
                ScanSpec(
                    "evidence",
                    PredicateSpec(
                        "alpha", "literal ALPHA evidence", should_terms=("ALPHA",), closed_world=True
                    ),
                ),
            ),
        )
        engine.execute(program)
        warm = engine.execute(program)
        self.assertEqual((warm.telemetry.cache_hits, warm.telemetry.cache_misses), (1, 0))
        engine.add_documents([{"id": "two", "text": "Issue ALPHA recurred."}])
        updated = engine.execute(program)
        self.assertEqual((updated.telemetry.cache_hits, updated.telemetry.cache_misses), (1, 1))
        self.assertEqual(len(updated.rows), 2)

    def test_versioned_disk_view_survives_engine_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_one = MaterializedWitnessCache(directory)
            first = WitnessEngine(cache=cache_one, max_workers=1)
            first.add_documents([{"id": "d", "text": "Beacon status is amber."}])
            program = WitnessProgram(
                question="What is the Beacon status?",
                scans=(
                    ScanSpec(
                        "status",
                        PredicateSpec(
                            "beacon", "Beacon status", should_terms=("Beacon",), closed_world=True
                        ),
                    ),
                ),
            )
            cold = first.execute(program)
            self.assertEqual(cold.telemetry.cache_misses, 1)
            restarted = WitnessEngine(
                cache=MaterializedWitnessCache(directory), max_workers=1
            )
            restarted.add_documents([{"id": "d", "text": "Beacon status is amber."}])
            warm = restarted.execute(program)
            self.assertEqual((warm.telemetry.cache_hits, warm.telemetry.evaluator_calls), (1, 0))
            self.assertEqual(cold.rows[0].to_dict(), warm.rows[0].to_dict())

    def test_explicit_source_type_filter_is_a_sound_skip(self):
        engine = WitnessEngine()
        engine.add_documents([{"id": "p", "text": "Approval is required.", "source_type": "policy"}])
        program = WitnessProgram(
            question="Did any email mention an outage?",
            scans=(
                ScanSpec(
                    "emails",
                    PredicateSpec("outage", "literal outage", should_terms=("outage",), closed_world=True),
                    source_types=("email",),
                ),
            ),
        )
        result = engine.execute(program)
        self.assertEqual(result.coverage.scanned_shards, 0)
        self.assertEqual(result.coverage.soundly_skipped_shards, 1)
        self.assertIn("1 conclusively evaluated", result.coverage.negative_claim_language())


class ReducerAndCoverageTests(unittest.TestCase):
    def test_inner_join_does_not_match_missing_or_null_keys(self):
        relations = {
            "left": [
                {"name": "missing-left", "__witness_ids": ["l-missing"]},
                {"account_id": None, "name": "null-left", "__witness_ids": ["l-null"]},
                {"account_id": "acct-1", "name": "known-left", "__witness_ids": ["l-known"]},
            ],
            "right": [
                {"detail": "missing-right", "__witness_ids": ["r-missing"]},
                {"account_id": None, "detail": "null-right", "__witness_ids": ["r-null"]},
                {"account_id": "acct-1", "detail": "known-right", "__witness_ids": ["r-known"]},
            ],
        }
        spec = ReducerSpec(
            "join_accounts",
            "JOIN",
            ("left", "right"),
            "joined",
            {"on": ["account_id"]},
        )

        result = DeterministicReducer().execute(relations, (spec,))["joined"]

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "known-left")
        self.assertEqual(result[0]["detail"], "known-right")
        self.assertEqual(result[0]["__witness_ids"], ["l-known", "r-known"])

    def test_composite_join_requires_every_key_component(self):
        relations = {
            "left": [
                {"account_id": "acct-1", "region": None, "label": "unknown region"},
                {"account_id": "acct-1", "region": "eu", "label": "fully bound"},
            ],
            "right": [
                {"customer": "acct-1", "territory": None, "value": "null territory"},
                {"customer": "acct-1", "territory": "eu", "value": "matched"},
            ],
        }
        spec = ReducerSpec(
            "join_composite",
            "JOIN",
            ("left", "right"),
            "joined",
            {"on": {"account_id": "customer", "region": "territory"}},
        )

        result = DeterministicReducer().execute(relations, (spec,))["joined"]

        self.assertEqual([(row["label"], row["value"]) for row in result], [("fully bound", "matched")])

    def test_left_join_retains_unknown_key_without_spurious_right_values(self):
        relations = {
            "left": [{"name": "unbound", "__witness_ids": ["left"]}],
            "right": [{"detail": "must not join", "__witness_ids": ["right"]}],
        }
        spec = ReducerSpec(
            "left_join",
            "JOIN",
            ("left", "right"),
            "joined",
            {"on": ["account_id"], "how": "left"},
        )

        result = DeterministicReducer().execute(relations, (spec,))["joined"]

        self.assertEqual(result, relations["left"])

    def test_join_rejects_empty_or_duplicate_key_specification(self):
        reducer = DeterministicReducer()
        relations = {"left": [{"id": 1}], "right": [{"id": 1}]}
        invalid_keys = ([], {}, ["id", "id"], [""])
        for index, on in enumerate(invalid_keys):
            with self.subTest(on=on), self.assertRaises(ValidationError):
                spec = ReducerSpec(
                    f"bad_join_{index}",
                    "JOIN",
                    ("left", "right"),
                    f"joined_{index}",
                    {"on": on},
                )
                reducer.execute(relations, (spec,))

    def test_join_preserves_colliding_right_columns_without_overwrite(self):
        relations = {
            "left": [{"id": 7, "right.id": "left-owned", "__witness_ids": ["left"]}],
            "right": [{"id": 7, "right.id": "right-owned", "__witness_ids": ["right"]}],
        }
        spec = ReducerSpec(
            "collision_join",
            "JOIN",
            ("left", "right"),
            "joined",
            {"on": ["id"]},
        )

        row = DeterministicReducer().execute(relations, (spec,))["joined"][0]

        self.assertEqual(row["id"], 7)
        self.assertEqual(row["right.id"], "left-owned")
        self.assertEqual(row["right.right.id"], 7)
        self.assertEqual(row["right.right.right.id"], "right-owned")

    def test_deterministic_group_and_sum_preserve_provenance(self):
        relations = {
            "events": [
                {"reason": "price", "amount": 2, "__witness_ids": ["w1"]},
                {"reason": "price", "amount": 3, "__witness_ids": ["w2"]},
                {"reason": "support", "amount": 1, "__witness_ids": ["w3"]},
            ]
        }
        specs = (
            ReducerSpec(
                "group",
                "GROUP_BY",
                ("events",),
                "groups",
                {
                    "fields": ["reason"],
                    "aggregates": {"count": {"op": "COUNT"}, "total": {"op": "SUM", "field": "amount"}},
                },
            ),
            ReducerSpec("top", "ARGMAX", ("groups",), "top_reason", {"field": "count"}),
        )
        result = DeterministicReducer().execute(relations, specs)
        self.assertEqual(result["top_reason"][0]["reason"], "price")
        self.assertEqual(result["top_reason"][0]["total"], 5)
        self.assertEqual(result["top_reason"][0]["__witness_ids"], ["w1", "w2"])

    def test_negative_claim_counts_sound_skips_as_conclusive(self):
        certificate = CoverageCertificate(
            corpus_snapshot="sha256:x",
            authorized_shards=10,
            scanned_shards=6,
            soundly_skipped_shards=3,
            unresolved_shards=1,
            uncertain_shards=2,
            positive_witnesses=0,
            counter_witnesses=0,
            program_version="wql-test",
            program_hash="p",
            evaluator_version="e",
            verifier_version="v",
        )
        self.assertEqual(certificate.conclusive_fraction, 0.7)
        self.assertIn("7 conclusively evaluated of 10", certificate.negative_claim_language())

    def test_coverage_distinguishes_plan_adequacy_and_obligation_execution(self):
        engine = WitnessEngine(max_workers=1)
        engine.add_documents(
            [
                {"id": "a", "text": "Beacon status is amber."},
                {"id": "b", "text": "No beacon appears here."},
            ]
        )
        program = WitnessProgram(
            question="What is the Beacon status?",
            scans=(
                ScanSpec(
                    "status",
                    PredicateSpec(
                        "beacon",
                        "Literal Beacon evidence",
                        should_terms=("Beacon",),
                        closed_world=False,
                    ),
                ),
            ),
        )

        certificate = engine.execute(program).coverage
        payload = certificate.to_dict()

        self.assertEqual(payload["metric_name"], "program_execution_coverage")
        self.assertEqual(payload["plan_adequacy"], "not_evaluated")
        self.assertEqual(certificate.scheduled_obligations, 2)
        self.assertEqual(
            certificate.conclusive_obligations
            + certificate.uncertain_obligations
            + certificate.failed_obligations,
            certificate.scheduled_obligations,
        )

    def test_temporal_resolve_honors_valid_interval_and_record_cutoff(self):
        relations = {
            "policies": [
                {
                    "policy": "approval",
                    "rule": "manager",
                    "valid_time": "2025-01-01",
                    "valid_to": "2025-06-01",
                    "record_time": "2025-01-02",
                    "__witness_ids": ["old"],
                },
                {
                    "policy": "approval",
                    "rule": "director",
                    "valid_time": "2025-06-01",
                    "valid_to": None,
                    "record_time": "2025-06-03",
                    "__witness_ids": ["new"],
                },
            ]
        }
        old_spec = ReducerSpec(
            "historical",
            "TEMPORAL_RESOLVE",
            ("policies",),
            "historical_policy",
            {"fields": ["policy"], "as_of": "2025-05-15", "recorded_as_of": "2025-07-01"},
        )
        new_spec = ReducerSpec(
            "current",
            "TEMPORAL_RESOLVE",
            ("policies",),
            "current_policy",
            {"fields": ["policy"], "as_of": "2025-07-15", "recorded_as_of": "2025-07-15"},
        )
        reduced = DeterministicReducer().execute(relations, (old_spec, new_spec))
        self.assertEqual(reduced["historical_policy"][0]["rule"], "manager")
        self.assertEqual(reduced["current_policy"][0]["rule"], "director")


if __name__ == "__main__":
    unittest.main()
