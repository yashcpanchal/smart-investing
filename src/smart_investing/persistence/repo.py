"""State persistence: strategies, proposals, executions, portfolio, audit log.

DuckDB-backed JSON document store — enough to make the API stateful across
restarts (Phase 7). Pydantic models serialize to JSON; we re-validate on read.
A Postgres backend can replace this behind the same interface for multi-user.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import duckdb

from smart_investing.config import settings
from smart_investing.domain.types import AccountState, Proposal

_SCHEMA = """
create sequence if not exists audit_seq start 1;
create table if not exists proposals (
    id varchar primary key,
    prompt varchar,
    created_at varchar,
    json varchar
);
create table if not exists executions (
    id varchar primary key,
    proposal_id varchar,
    created_at varchar,
    account_json varchar
);
create table if not exists portfolio (
    id varchar primary key,        -- single row 'current'
    account_json varchar,
    realized_pnl double,
    updated_at varchar
);
create table if not exists audit (
    seq bigint default nextval('audit_seq') primary key,
    ts varchar,
    event varchar,
    detail varchar
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def default_state_path() -> str:
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    return str(Path(settings.data_dir) / "state.duckdb")


class StateRepo:
    def __init__(self, path: str | None = None) -> None:
        self.con = duckdb.connect(path or default_state_path())
        self.con.execute(_SCHEMA)

    # ---- proposals ----
    def save_proposal(self, proposal: Proposal) -> str:
        self.con.execute(
            "insert or replace into proposals values (?,?,?,?)",
            [proposal.id, proposal.spec.raw_prompt, proposal.generated_at, proposal.model_dump_json()],
        )
        self.audit("proposal_saved", proposal.id)
        return proposal.id

    def get_proposal(self, proposal_id: str) -> Proposal | None:
        row = self.con.execute("select json from proposals where id = ?", [proposal_id]).fetchone()
        return Proposal.model_validate_json(row[0]) if row else None

    def list_proposals(self, limit: int = 50) -> list[dict]:
        rows = self.con.execute(
            "select id, prompt, created_at from proposals order by created_at desc limit ?", [limit]
        ).fetchall()
        return [{"id": r[0], "prompt": r[1], "created_at": r[2]} for r in rows]

    # ---- executions ----
    def save_execution(self, proposal_id: str, account: AccountState) -> None:
        self.con.execute(
            "insert into executions values (?,?,?,?)",
            [f"{proposal_id}:{_now()}", proposal_id, _now(), account.model_dump_json()],
        )
        self.audit("execution", proposal_id)

    # ---- portfolio (single current snapshot) ----
    def save_portfolio(self, account: AccountState, realized_pnl: float = 0.0) -> None:
        self.con.execute(
            "insert or replace into portfolio values ('current', ?, ?, ?)",
            [account.model_dump_json(), realized_pnl, _now()],
        )

    def get_portfolio(self) -> tuple[AccountState, float] | None:
        row = self.con.execute(
            "select account_json, realized_pnl from portfolio where id = 'current'"
        ).fetchone()
        if not row:
            return None
        return AccountState.model_validate_json(row[0]), float(row[1])

    # ---- audit ----
    def audit(self, event: str, detail: str = "") -> None:
        self.con.execute("insert into audit (ts, event, detail) values (?,?,?)", [_now(), event, detail])

    def audit_log(self, limit: int = 100) -> list[dict]:
        rows = self.con.execute(
            "select ts, event, detail from audit order by seq desc limit ?", [limit]
        ).fetchall()
        return [{"ts": r[0], "event": r[1], "detail": r[2]} for r in rows]

    def close(self) -> None:
        self.con.close()
