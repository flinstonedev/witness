"""Run a tiny fictional policy example with in-memory storage and no API keys."""

from witness_engine import WitnessEngine


def main() -> None:
    engine = WitnessEngine()
    engine.add_documents([
        {
            "id": "example-policy",
            "version": "1",
            "source_type": "policy",
            "text": "Manager approval is optional for emergency remediation.",
            "permissions": ["auditor"],
        },
        {
            "id": "example-unrelated",
            "text": "The cafeteria opens at noon.",
            "permissions": ["auditor"],
        },
    ])
    program = engine.compile("Is manager approval always required?")
    denied = engine.execute(program, principals=("visitor",))
    assert not denied.rows and denied.coverage.authorized_shards == 0
    result = engine.execute(program, principals=("auditor",))
    assert result.rows, "the lexical fallback should find the explicit approval terms"
    for row in result.rows:
        document = engine.store.get_document(row.source_id, row.source_version)
        assert document.raw_content[row.start_offset:row.end_offset] == row.quote
        print(f"{row.source_id}@{row.source_version} [{row.start_offset}:{row.end_offset}]: {row.quote}")
    coverage = result.coverage.to_dict()
    assert coverage["plan_adequacy"] == "not_evaluated"
    print(f"Authorized shards: {result.coverage.authorized_shards}")
    print(f"Uncertain shards: {coverage['uncertain_shards']}")
    print("Exact source spans verified; semantic entailment and plan adequacy are not evaluated.")


if __name__ == "__main__":
    main()
