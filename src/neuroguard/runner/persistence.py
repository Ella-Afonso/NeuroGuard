"""JSONL persistence for sessions."""

from pathlib import Path

from neuroguard.runner.schema import Session


def write_session(session: Session, path: Path) -> None:
    """Append a session to a JSONL file. Creates parent directories if needed.

    Args:
        session: The Session to persist.
        path: JSONL file path. Sessions are appended one per line.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(session.model_dump_json() + "\n")


def read_sessions(path: Path) -> list[Session]:
    """Load all sessions from a JSONL file.

    Args:
        path: JSONL file path.

    Returns:
        List of Session objects. Empty list if the file does not exist.
    """
    if not path.exists():
        return []
    sessions: list[Session] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            sessions.append(Session.model_validate_json(stripped))
    return sessions
