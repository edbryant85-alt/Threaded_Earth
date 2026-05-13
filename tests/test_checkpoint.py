from __future__ import annotations

from tools.checkpoint import ChangedFile, generate_commit_message, parse_status


def test_parse_status_handles_untracked_and_modified_files():
    parsed = parse_status(" M README.md\n?? tools/checkpoint.py\nR  old.py -> threaded_earth/new.py\n")
    assert parsed == [
        ChangedFile(status="M", path="README.md"),
        ChangedFile(status="??", path="tools/checkpoint.py"),
        ChangedFile(status="R", path="threaded_earth/new.py"),
    ]


def test_generate_commit_message_mentions_tooling():
    message = generate_commit_message(
        [
            ChangedFile(status="??", path="tools/checkpoint.py"),
            ChangedFile(status="M", path="Makefile"),
            ChangedFile(status="M", path="README.md"),
        ]
    )
    assert "checkpoint" in message.lower() or "tooling" in message.lower()
