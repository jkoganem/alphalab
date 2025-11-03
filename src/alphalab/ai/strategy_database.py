"""Strategy database for storing and retrieving strategy examples.

This module provides a database to store strategies with their performance
metrics and validation status, enabling dynamic few-shot learning by showing
the LLM examples of successful and failed strategies.
"""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class StrategyRecord:
    """Record of a strategy attempt."""

    name: str
    spec: dict[str, Any]
    sharpe: float | None
    returns: float | None
    evaluation_score: float | None
    validation_passed: bool
    validation_errors: list[str]
    timestamp: str
    model: str  # LLM model used (e.g., "gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet")
    metadata: dict[str, Any]


class StrategyDatabase:
    """Database for storing and retrieving strategy examples."""

    def __init__(self, db_path: str = ".alphalab/strategies.db"):
        """Initialize database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Create database schema if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                spec TEXT NOT NULL,
                sharpe REAL,
                returns REAL,
                evaluation_score REAL,
                validation_passed INTEGER NOT NULL,
                validation_errors TEXT,
                timestamp TEXT NOT NULL,
                model TEXT NOT NULL,
                metadata TEXT,
                UNIQUE(name, timestamp)
            )
        """)

        # Create indexes for fast queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sharpe
            ON strategies(sharpe DESC)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_score
            ON strategies(evaluation_score DESC)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_validation
            ON strategies(validation_passed, timestamp DESC)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_model
            ON strategies(model, sharpe DESC)
        """)

        conn.commit()
        conn.close()

    def store_strategy(self, record: StrategyRecord) -> int:
        """Store a strategy record in the database.

        Args:
            record: Strategy record to store

        Returns:
            ID of stored record
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO strategies (
                name, spec, sharpe, returns, evaluation_score,
                validation_passed, validation_errors, timestamp, model, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.name,
            json.dumps(record.spec),
            record.sharpe,
            record.returns,
            record.evaluation_score,
            1 if record.validation_passed else 0,
            json.dumps(record.validation_errors) if record.validation_errors else None,
            record.timestamp,
            record.model,
            json.dumps(record.metadata) if record.metadata else None,
        ))

        record_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return record_id

    def get_best_by_sharpe(self, limit: int = 5) -> list[StrategyRecord]:
        """Get top strategies by Sharpe ratio.

        Args:
            limit: Maximum number of strategies to return

        Returns:
            List of strategy records
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT name, spec, sharpe, returns, evaluation_score,
                   validation_passed, validation_errors, timestamp, model, metadata
            FROM strategies
            WHERE validation_passed = 1 AND sharpe IS NOT NULL
            ORDER BY sharpe DESC
            LIMIT ?
        """, (limit,))

        records = [self._row_to_record(row) for row in cursor.fetchall()]
        conn.close()

        return records

    def get_best_by_score(self, limit: int = 5) -> list[StrategyRecord]:
        """Get top strategies by evaluation score.

        Args:
            limit: Maximum number of strategies to return

        Returns:
            List of strategy records
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT name, spec, sharpe, returns, evaluation_score,
                   validation_passed, validation_errors, timestamp, model, metadata
            FROM strategies
            WHERE validation_passed = 1 AND evaluation_score IS NOT NULL
            ORDER BY evaluation_score DESC
            LIMIT ?
        """, (limit,))

        records = [self._row_to_record(row) for row in cursor.fetchall()]
        conn.close()

        return records

    def get_recent_failures(self, limit: int = 10) -> list[StrategyRecord]:
        """Get recent validation failures with error messages.

        Args:
            limit: Maximum number of failures to return

        Returns:
            List of failed strategy records
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT name, spec, sharpe, returns, evaluation_score,
                   validation_passed, validation_errors, timestamp, model, metadata
            FROM strategies
            WHERE validation_passed = 0 AND validation_errors IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        records = [self._row_to_record(row) for row in cursor.fetchall()]
        conn.close()

        return records

    def get_common_errors(self, limit: int = 5) -> list[tuple[str, int]]:
        """Get most common validation errors.

        Args:
            limit: Maximum number of error types to return

        Returns:
            List of (error_message, count) tuples
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT validation_errors, COUNT(*) as count
            FROM strategies
            WHERE validation_passed = 0 AND validation_errors IS NOT NULL
            GROUP BY validation_errors
            ORDER BY count DESC
            LIMIT ?
        """, (limit,))

        results = []
        for row in cursor.fetchall():
            errors = json.loads(row[0]) if row[0] else []
            count = row[1]
            if errors:
                # Take first error as representative
                results.append((errors[0], count))

        conn.close()
        return results

    def get_statistics(self) -> dict[str, Any]:
        """Get database statistics.

        Returns:
            Dictionary with statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total strategies
        cursor.execute("SELECT COUNT(*) FROM strategies")
        total = cursor.fetchone()[0]

        # Validation pass rate
        cursor.execute("""
            SELECT COUNT(*) FROM strategies WHERE validation_passed = 1
        """)
        passed = cursor.fetchone()[0]

        # Best Sharpe
        cursor.execute("""
            SELECT MAX(sharpe) FROM strategies
            WHERE validation_passed = 1 AND sharpe IS NOT NULL
        """)
        best_sharpe = cursor.fetchone()[0]

        # Best score
        cursor.execute("""
            SELECT MAX(evaluation_score) FROM strategies
            WHERE validation_passed = 1 AND evaluation_score IS NOT NULL
        """)
        best_score = cursor.fetchone()[0]

        conn.close()

        return {
            "total_strategies": total,
            "passed_validation": passed,
            "pass_rate": passed / total if total > 0 else 0.0,
            "best_sharpe": best_sharpe,
            "best_score": best_score,
        }

    def get_model_comparison(self) -> list[dict[str, Any]]:
        """Compare performance across different LLM models.

        Returns:
            List of dictionaries with model statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                model,
                COUNT(*) as total,
                SUM(CASE WHEN validation_passed = 1 THEN 1 ELSE 0 END) as passed,
                AVG(CASE WHEN validation_passed = 1 THEN sharpe ELSE NULL END) as avg_sharpe,
                MAX(CASE WHEN validation_passed = 1 THEN sharpe ELSE NULL END) as max_sharpe,
                AVG(CASE WHEN validation_passed = 1 THEN evaluation_score ELSE NULL END) as avg_score,
                MAX(CASE WHEN validation_passed = 1 THEN evaluation_score ELSE NULL END) as max_score
            FROM strategies
            GROUP BY model
            ORDER BY avg_sharpe DESC NULLS LAST
        """)

        results = []
        for row in cursor.fetchall():
            model = row[0]
            total = row[1]
            passed = row[2]
            results.append({
                "model": model,
                "total_strategies": total,
                "passed_validation": passed,
                "pass_rate": passed / total if total > 0 else 0.0,
                "avg_sharpe": row[3],
                "max_sharpe": row[4],
                "avg_score": row[5],
                "max_score": row[6],
            })

        conn.close()
        return results

    def _row_to_record(self, row: tuple) -> StrategyRecord:
        """Convert database row to StrategyRecord."""
        return StrategyRecord(
            name=row[0],
            spec=json.loads(row[1]),
            sharpe=row[2],
            returns=row[3],
            evaluation_score=row[4],
            validation_passed=bool(row[5]),
            validation_errors=json.loads(row[6]) if row[6] else [],
            timestamp=row[7],
            model=row[8],
            metadata=json.loads(row[9]) if row[9] else {},
        )


class ExampleSelector:
    """Selects good and bad examples for few-shot learning."""

    def __init__(self, database: StrategyDatabase):
        """Initialize example selector.

        Args:
            database: Strategy database to query
        """
        self.db = database

    def get_few_shot_examples(
        self,
        n_good: int = 3,
        n_bad: int = 2,
    ) -> dict[str, list[dict[str, Any]]]:
        """Get examples for few-shot learning.

        Args:
            n_good: Number of good examples to retrieve
            n_bad: Number of bad examples to retrieve

        Returns:
            Dictionary with 'good' and 'bad' example lists
        """
        # Get good examples (mix of high Sharpe and high score)
        good_by_sharpe = self.db.get_best_by_sharpe(limit=n_good)
        good_by_score = self.db.get_best_by_score(limit=n_good)

        # Combine and deduplicate
        good_examples = []
        seen_names = set()

        for record in good_by_sharpe + good_by_score:
            if record.name not in seen_names:
                good_examples.append({
                    "name": record.name,
                    "spec": record.spec,
                    "sharpe": record.sharpe,
                    "returns": record.returns,
                    "score": record.evaluation_score,
                })
                seen_names.add(record.name)

                if len(good_examples) >= n_good:
                    break

        # Get bad examples (recent failures with errors)
        bad_records = self.db.get_recent_failures(limit=n_bad * 2)

        bad_examples = []
        seen_errors = set()

        for record in bad_records:
            error_key = tuple(record.validation_errors)
            if error_key not in seen_errors:
                bad_examples.append({
                    "name": record.name,
                    "spec": record.spec,
                    "errors": record.validation_errors,
                })
                seen_errors.add(error_key)

                if len(bad_examples) >= n_bad:
                    break

        return {
            "good": good_examples,
            "bad": bad_examples,
        }

    def format_examples_for_prompt(
        self,
        examples: dict[str, list[dict[str, Any]]],
    ) -> str:
        """Format examples for inclusion in LLM prompt.

        Args:
            examples: Dictionary with 'good' and 'bad' examples

        Returns:
            Formatted string for prompt
        """
        parts = []

        # Good examples
        if examples["good"]:
            parts.append("="*80)
            parts.append("SUCCESSFUL STRATEGIES (Learn from these)")
            parts.append("="*80)

            for i, ex in enumerate(examples["good"], 1):
                parts.append(f"\n[GOOD EXAMPLE {i}] {ex['name']}")
                parts.append(f"  Performance: Sharpe={ex['sharpe']:.2f}, "
                           f"Return={ex['returns']:.1f}%, Score={ex['score']:.1f}/100")
                parts.append(f"  Spec:")
                parts.append(f"  {json.dumps(ex['spec'], indent=2)}")

        # Bad examples
        if examples["bad"]:
            parts.append("\n" + "="*80)
            parts.append("FAILED STRATEGIES (Avoid these patterns)")
            parts.append("="*80)

            for i, ex in enumerate(examples["bad"], 1):
                parts.append(f"\n[BAD EXAMPLE {i}] {ex['name']}")
                parts.append(f"  Validation Errors:")
                for error in ex["errors"]:
                    parts.append(f"    - {error}")
                parts.append(f"  Spec (DO NOT COPY THIS):")
                parts.append(f"  {json.dumps(ex['spec'], indent=2)}")

        if parts:
            parts.append("\n" + "="*80)
            return "\n".join(parts)

        return ""
