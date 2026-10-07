"""
Profiling v2 P2: measured per-node FLOPs of real runs.

Creates ``profiling_runs`` (one row per finished prompt: totals, mode, run
facts) and ``profiling_node_stats`` (one row per node: measured FLOPs, op
count, census JSON, demotion flag).  Written by the executor hook via the
app-layer persistence callback; read by ``/comfydl/profiling/history`` and the
estimate calibration display.

Revision ID: 0007_profiling_measured_runs
Revises: 0006_add_loader_path
Create Date: 2026-10-07
"""

from alembic import op
import sqlalchemy as sa

revision = "0007_profiling_measured_runs"
down_revision = "0006_add_loader_path"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "profiling_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("prompt_id", sa.String(36), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("mode", sa.String(16), nullable=True),
        sa.Column("total_flops", sa.Integer(), nullable=True),
        sa.Column("total_ops", sa.Integer(), nullable=True),
        sa.Column("node_count", sa.Integer(), nullable=True),
        sa.Column("unattributed_flops", sa.Integer(), nullable=True),
        sa.Column("sampled", sa.Boolean(), nullable=True),
        sa.Column("suppressed_errors", sa.Integer(), nullable=True),
    )
    op.create_index(
        "ix_profiling_runs_prompt_id", "profiling_runs", ["prompt_id"])

    op.create_table(
        "profiling_node_stats",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("run_id", sa.String(36), nullable=True),
        sa.Column("node_id", sa.String(64), nullable=True),
        sa.Column("class_type", sa.String(128), nullable=True),
        sa.Column("flops", sa.Integer(), nullable=True),
        sa.Column("op_count", sa.Integer(), nullable=True),
        sa.Column("ops", sa.JSON(), nullable=True),
        sa.Column("demoted", sa.Boolean(), nullable=True),
    )
    op.create_index(
        "ix_profiling_node_stats_run", "profiling_node_stats", ["run_id"])
    op.create_index(
        "ix_profiling_node_stats_node_id", "profiling_node_stats", ["node_id"])
    op.create_index(
        "ix_profiling_node_stats_class_type", "profiling_node_stats", ["class_type"])


def downgrade() -> None:
    op.drop_table("profiling_node_stats")
    op.drop_table("profiling_runs")
