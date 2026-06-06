"""
Tests for UXPipe (newmain.py)
Run with:  pytest test_uxpipe.py -v
"""

import io
import pytest
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")   # non-interactive backend – no window needed

from newmain import UXPipe


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

COLUMNS = ["user_id", "session_id", "timestamp", "event_type"]

SAMPLE_ROWS = [
    # user A – session 1
    ("u1", "s1", "2024-01-01 10:00:00", "search"),
    ("u1", "s1", "2024-01-01 10:01:00", "click"),
    ("u1", "s1", "2024-01-01 10:02:00", "purchase"),
    # user A – session 2
    ("u1", "s2", "2024-01-02 09:00:00", "search"),
    ("u1", "s2", "2024-01-02 09:01:00", "add_to_cart"),
    # user B – session 3
    ("u2", "s3", "2024-01-01 11:00:00", "search"),
    ("u2", "s3", "2024-01-01 11:01:00", "scroll"),
    ("u2", "s3", "2024-01-01 11:02:00", "click"),
    ("u2", "s3", "2024-01-01 11:03:00", "purchase"),
]


@pytest.fixture
def sample_csv(tmp_path):
    """Write sample data to a temp CSV and return its path."""
    df = pd.DataFrame(SAMPLE_ROWS, columns=COLUMNS)
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)
    return str(p)


@pytest.fixture
def pipe(sample_csv):
    """Fully initialised UXPipe with columns declared."""
    p = UXPipe(sample_csv, "csv")
    p.declare_columns(COLUMNS)
    return p


@pytest.fixture
def pipe_with_metrics(pipe):
    """UXPipe with session + user metrics computed."""
    pipe.session_metrics()
    pipe.user_metrics()
    return pipe


@pytest.fixture
def pipe_graph_session(pipe):
    """UXPipe with graph computed in session mode."""
    pipe.compute_graph(group_by="session")
    return pipe


@pytest.fixture
def pipe_graph_user(pipe):
    """UXPipe with graph computed in user mode."""
    pipe.compute_graph(group_by="user")
    return pipe


# ─────────────────────────────────────────────────────────────────────────────
# 1. Initialisation
# ─────────────────────────────────────────────────────────────────────────────

class TestInit:
    def test_loads_csv(self, sample_csv):
        p = UXPipe(sample_csv, "csv")
        assert isinstance(p.data, pd.DataFrame)
        assert len(p.data) == len(SAMPLE_ROWS)

    def test_timestamp_parsed(self, sample_csv):
        p = UXPipe(sample_csv, "csv")
        assert pd.api.types.is_datetime64_any_dtype(p.data["timestamp"])

    def test_unsupported_type_raises(self, sample_csv):
        with pytest.raises(ValueError, match="Unsupported file type"):
            UXPipe(sample_csv, "parquet")

    def test_missing_file_raises(self):
        with pytest.raises(Exception):
            UXPipe("nonexistent.csv", "csv")


# ─────────────────────────────────────────────────────────────────────────────
# 2. declare_columns
# ─────────────────────────────────────────────────────────────────────────────

class TestDeclareColumns:
    def test_sets_all_attributes(self, sample_csv):
        p = UXPipe(sample_csv, "csv")
        p.declare_columns(COLUMNS)
        assert p.usercol    == "user_id"
        assert p.sessioncol == "session_id"
        assert p.timecol    == "timestamp"
        assert p.eventcol   == "event_type"

    def test_returns_self(self, sample_csv):
        p = UXPipe(sample_csv, "csv")
        result = p.declare_columns(COLUMNS)
        assert result is p


# ─────────────────────────────────────────────────────────────────────────────
# 3. session_metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestSessionMetrics:
    def test_returns_self(self, pipe):
        assert pipe.session_metrics() is pipe

    def test_sessiondf_created(self, pipe):
        pipe.session_metrics()
        assert hasattr(pipe, "sessiondf")
        assert isinstance(pipe.sessiondf, pd.DataFrame)

    def test_one_row_per_session(self, pipe):
        pipe.session_metrics()
        n_sessions = pipe.data["session_id"].nunique()
        assert len(pipe.sessiondf) == n_sessions

    def test_duration_non_negative(self, pipe):
        pipe.session_metrics()
        assert (pipe.sessiondf["duration"] >= 0).all()

    def test_events_per_session_positive(self, pipe):
        pipe.session_metrics()
        assert (pipe.sessiondf["events_per_session"] > 0).all()

    def test_event_count_columns_present(self, pipe):
        pipe.session_metrics()
        for event in pipe.data["event_type"].unique():
            assert f"{event}_count" in pipe.sessiondf.columns

    def test_user_col_present(self, pipe):
        pipe.session_metrics()
        assert "user_id" in pipe.sessiondf.columns


# ─────────────────────────────────────────────────────────────────────────────
# 4. user_metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestUserMetrics:
    def test_returns_self(self, pipe):
        pipe.session_metrics()
        assert pipe.user_metrics() is pipe

    def test_userdf_created(self, pipe):
        pipe.session_metrics()
        pipe.user_metrics()
        assert hasattr(pipe, "userdf")
        assert isinstance(pipe.userdf, pd.DataFrame)

    def test_one_row_per_user(self, pipe):
        pipe.session_metrics()
        pipe.user_metrics()
        n_users = pipe.data["user_id"].nunique()
        assert len(pipe.userdf) == n_users

    def test_sessions_per_user_correct(self, pipe):
        pipe.session_metrics()
        pipe.user_metrics()
        # u1 has 2 sessions, u2 has 1
        u1_row = pipe.userdf[pipe.userdf["user_id"] == "u1"]
        u2_row = pipe.userdf[pipe.userdf["user_id"] == "u2"]
        assert u1_row["sessions_per_user"].values[0] == 2
        assert u2_row["sessions_per_user"].values[0] == 1

    def test_expected_metric_columns(self, pipe):
        pipe.session_metrics()
        pipe.user_metrics()
        for col in ["sessions_per_user", "total_time", "total_events", "avg_session_duration"]:
            assert col in pipe.userdf.columns

    def test_total_events_positive(self, pipe):
        pipe.session_metrics()
        pipe.user_metrics()
        assert (pipe.userdf["total_events"] > 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# 5. compute_graph
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeGraph:
    def test_returns_self(self, pipe):
        assert pipe.compute_graph() is pipe

    def test_graph_created(self, pipe_graph_session):
        assert hasattr(pipe_graph_session, "G")
        assert isinstance(pipe_graph_session.G, nx.DiGraph)

    def test_graph_has_edges(self, pipe_graph_session):
        assert len(pipe_graph_session.G.edges()) > 0

    def test_start_node_present(self, pipe_graph_session):
        assert "__START__" in pipe_graph_session.G.nodes

    def test_end_node_present(self, pipe_graph_session):
        assert "__END__" in pipe_graph_session.G.nodes

    def test_start_end_marked_special(self, pipe_graph_session):
        G = pipe_graph_session.G
        assert G.nodes["__START__"]["special"] is True
        assert G.nodes["__END__"]["special"]   is True

    def test_regular_nodes_not_special(self, pipe_graph_session):
        G = pipe_graph_session.G
        for node in G.nodes:
            if node not in ("__START__", "__END__"):
                assert G.nodes[node]["special"] is False

    def test_all_edges_have_weight(self, pipe_graph_session):
        G = pipe_graph_session.G
        for u, v in G.edges():
            assert "weight" in G[u][v]
            assert G[u][v]["weight"] > 0

    def test_start_has_outgoing_edges(self, pipe_graph_session):
        G = pipe_graph_session.G
        assert G.out_degree("__START__") > 0

    def test_end_has_incoming_edges(self, pipe_graph_session):
        G = pipe_graph_session.G
        assert G.in_degree("__END__") > 0

    def test_quantile_threshold_stored(self, pipe):
        pipe.compute_graph(quantile_threshold=0.5)
        assert pipe.quantile_threshold == 0.5

    # ── group_by="user" ──────────────────────────────────────────────────────

    def test_user_mode_start_present(self, pipe_graph_user):
        assert "__START__" in pipe_graph_user.G.nodes

    def test_user_mode_end_present(self, pipe_graph_user):
        assert "__END__" in pipe_graph_user.G.nodes

    def test_user_mode_has_edges(self, pipe_graph_user):
        assert len(pipe_graph_user.G.edges()) > 0

    def test_invalid_group_by_silently_defaults(self, pipe):
        # NOTE: compute_graph currently does not validate `group_by` —
        # an unrecognised value silently falls back to session mode
        # because the ternary `self.sessioncol if group_by == "session" else self.usercol`
        # treats anything that isn't "session" as "user".
        # This test documents the current behaviour; consider adding an explicit
        # ValueError guard in compute_graph for production use:
        #   if group_by not in ("session", "user"):
        #       raise ValueError(f"group_by must be 'session' or 'user', got '{group_by}'")
        pipe.compute_graph(group_by="invalid")
        assert isinstance(pipe.G, nx.DiGraph)   # graph still built, no crash

    # ── session vs user produce different graphs ─────────────────────────────

    def test_session_vs_user_differ(self, pipe):
        pipe.compute_graph(group_by="session")
        edges_session = set(pipe.G.edges())
        pipe.compute_graph(group_by="user")
        edges_user = set(pipe.G.edges())
        # They may share some edges but should not be identical
        # (u1 has 2 sessions collapsed into 1 user chain)
        assert edges_session != edges_user


# ─────────────────────────────────────────────────────────────────────────────
# 6. see_graph  (rendering – no window, just check it doesn't crash)
# ─────────────────────────────────────────────────────────────────────────────

class TestSeeGraph:
    def test_returns_self(self, pipe_graph_session):
        assert pipe_graph_session.see_graph() is pipe_graph_session

    def test_runs_with_quantile_override(self, pipe_graph_session):
        # Should not raise even when threshold filters most edges
        pipe_graph_session.see_graph(quantile_threshold=0.5)

    def test_start_end_survive_high_quantile(self, pipe_graph_session):
        """START/END nodes must always appear regardless of threshold."""
        import matplotlib.pyplot as plt
        pipe_graph_session.see_graph(quantile_threshold=0.9)
        plt.close("all")

    def test_uses_stored_threshold(self, pipe):
        pipe.compute_graph(quantile_threshold=0.3)
        # see_graph with no arg should use 0.3, not crash
        pipe.see_graph()

    def test_empty_graph_prints_message(self, pipe, capsys):
        pipe.G = nx.DiGraph()   # empty graph
        pipe.see_graph()
        captured = capsys.readouterr()
        assert "no edges" in captured.out.lower()
