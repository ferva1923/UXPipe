import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.patches import Patch

class UXPipe:
    def __init__(self, filepath, type):
        
        if type == 'csv':
            data = pd.read_csv(filepath)
        elif type == 'json':
            data = pd.read_json(filepath)
        else:
            raise ValueError("Unsupported file type")
        data = pd.DataFrame(data)
        self.data = data
        # Standardize time
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        
        return None
    

    def declare_columns(self, columns):
        self.usercol = columns[0]
        self.sessioncol = columns[1]
        self.timecol = columns[2]
        self.eventcol = columns[3]
        return self
    
    def session_metrics(self):
        self.sessiondf = pd.DataFrame()
        self.sessiondf[self.sessioncol] = self.data[self.sessioncol].unique()

        session_times = self.data.groupby([self.sessioncol,self.usercol])[self.timecol].agg(['min', 'max'])
        print(session_times.head())
        session_times['duration'] = (session_times['max'] - session_times['min']).dt.total_seconds()
        self.sessiondf = self.sessiondf.merge(session_times['duration'], on=self.sessioncol, how='left')

      
            
        events_count = self.data.groupby([self.sessioncol,self.usercol]).count()
        events_count['events_per_session'] = events_count[self.eventcol]
        self.sessiondf = self.sessiondf.merge(events_count['events_per_session'], on=self.sessioncol, how='left')
        

        #event counts
        self.events = self.data[self.eventcol].unique()
        for event in self.events:

            event_count = self.data[self.data[self.eventcol] == event].groupby(self.sessioncol).count()
            event_count[event+'_count'] = event_count[self.eventcol]
            self.sessiondf = self.sessiondf.merge(event_count[event+'_count'], on=self.sessioncol, how='left')

        self.sessiondf = self.sessiondf.merge( self.data[[self.sessioncol, self.usercol]].drop_duplicates(), on=self.sessioncol, how='left')

        return self
    
    def user_metrics(self):

        self.userdf = pd.DataFrame()
        self.userdf[self.usercol] = self.data[self.usercol].unique()


        #sessions per user
        sessions_per_user = self.data.groupby(self.usercol).nunique()[self.sessioncol]
        sessions_per_user.name = 'sessions_per_user'
        self.userdf = self.userdf.merge(sessions_per_user, on=self.usercol, how='left')

        #total time
        user_times = self.sessiondf.groupby(self.usercol)['duration'].sum()
        user_times.name = 'total_time'
        self.userdf = self.userdf.merge(user_times, left_on=self.usercol, right_index=True, how='left')

        #total events
        total_events = self.sessiondf.groupby(self.usercol)['events_per_session'].sum()
        total_events.name = 'total_events'
        self.userdf = self.userdf.merge(total_events, left_on=self.usercol, right_index=True, how='left')
        self.events_count = [event+'_count' for event in self.events]
        for event in self.events_count:
            event_count = self.sessiondf.groupby(self.usercol)[event].sum()
            event_count.name = event
            self.userdf = self.userdf.merge(event_count, left_on=self.usercol, right_index=True, how='left')

        #average session duration
        avg_session_duration = self.sessiondf.groupby(self.usercol)['duration'].mean()
        avg_session_duration.name = 'avg_session_duration'
        self.userdf = self.userdf.merge(avg_session_duration, left_on=self.usercol, right_index=True, how='left')

        return self
    
    

    def add_new_data(self, filepath, type, name, key):
        if type == 'csv':
            newdata = pd.read_csv(filepath)
        elif type == 'json':
            newdata = pd.read_json(filepath)
        else:
            raise ValueError("Unsupported file type")
        newdata = pd.DataFrame(newdata)
        setattr(self, name, newdata)
        self.data = pd.merge(self.data, newdata, on=key, how='left')
        return self
    
    def compute_graph(self, group_by="session", quantile_threshold=0.0):
        self.quantile_threshold = quantile_threshold

        group_col = self.sessioncol if group_by == "session" else self.usercol

        self.G = nx.DiGraph()

        for _, group in self.data.groupby(group_col):
            # Always sort by time so the chain is chronological
            group = group.sort_values(self.timecol)
            events = list(group[self.eventcol])
            chain = ["__START__"] + events + ["__END__"]

            for src, dst in zip(chain[:-1], chain[1:]):
                if self.G.has_edge(src, dst):
                    self.G[src][dst]["weight"] += 1
                else:
                    self.G.add_edge(src, dst, weight=1)

        for node in self.G.nodes:
            self.G.nodes[node]["special"] = node in ("__START__", "__END__")

        return self
    
    
    def see_graph(self, quantile_threshold=None):
        

        if quantile_threshold is None:
            quantile_threshold = getattr(self, "quantile_threshold", 0.0)

        if len(self.G.edges()) == 0:
            print("Graph has no edges to draw.")
            return self

        # ── quantile filtering ───────────────────────────────────────────────
        # Compute cutoff only on non-special edges so START/END are never filtered
        non_special_weights = np.array([
            self.G[u][v]["weight"] for u, v in self.G.edges()
            if u not in ("__START__", "__END__") and v not in ("__START__", "__END__")
        ])
        weight_cutoff = np.quantile(non_special_weights, quantile_threshold) if len(non_special_weights) else 0
        
        visible_edges = [
            (u, v) for u, v in self.G.edges()
            if self.G[u][v]["weight"] >= weight_cutoff
            or u in ("__START__", "__END__")
            or v in ("__START__", "__END__")
        ]
        subG = self.G.edge_subgraph(visible_edges).copy()

        weights = {(u, v): subG[u][v]["weight"] for u, v in subG.edges()}
        if not weights:
            print("No edges survive the quantile filter.")
            return self

        # ── Log scale widths ─────────────────────────────────────────────────
        log_weights = {(u, v): np.log1p(w) for (u, v), w in weights.items()}
        min_log = min(log_weights.values())
        max_log = max(log_weights.values())

        def edge_width(u, v):
            lw = log_weights[(u, v)]
            return 1.0 + 6.0 * (lw - min_log) / (max_log - min_log + 1e-9)

        # ── Layout ───────────────────────────────────────────────────────────
        middle_nodes = [n for n in subG.nodes if n not in ("__START__", "__END__")]
        if middle_nodes:
            pos_middle = nx.spring_layout(subG.subgraph(middle_nodes), seed=42, k=2.5)
            xs = [v[0] for v in pos_middle.values()]
            ys = [v[1] for v in pos_middle.values()]
            x_range = max(xs) - min(xs) or 1
            y_range = max(ys) - min(ys) or 1
            pos_middle = {
                n: (
                    -0.6 + 1.2 * (xy[0] - min(xs)) / x_range,
                    -0.8 + 1.6 * (xy[1] - min(ys)) / y_range,
                )
                for n, xy in pos_middle.items()
            }
        else:
            pos_middle = {}

        pos = {**pos_middle}
        if "__START__" in subG.nodes:
            pos["__START__"] = (-1.2, 0.0)
        if "__END__" in subG.nodes:
            pos["__END__"] = (1.2, 0.0)

        # ── Figure ───────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(14, 9))
        fig.patch.set_facecolor("#1e1e2e")
        ax.set_facecolor("#1e1e2e")

        special_nodes = [n for n in subG.nodes if subG.nodes[n].get("special")]
        regular_nodes = [n for n in subG.nodes if not subG.nodes[n].get("special")]

        nx.draw_networkx_nodes(
            subG, pos, nodelist=regular_nodes,
            node_size=2800, node_color="#7ec8e3",
            edgecolors="#ffffff", linewidths=1.5, ax=ax,
        )
        nx.draw_networkx_nodes(
            subG, pos, nodelist=special_nodes,
            node_size=2000, node_color="#f4a261",
            edgecolors="#ffffff", linewidths=1.5, node_shape="D", ax=ax,
        )

        label_map = {
            n: ("▶ START" if n == "__START__" else "END ◀" if n == "__END__" else n)
            for n in subG.nodes
        }
        nx.draw_networkx_labels(
            subG, pos, labels=label_map,
            font_size=7, font_weight="bold", font_color="#1e1e2e", ax=ax,
        )

        # ── Edges (single color, log-scaled thickness) ────────────────────────
        EDGE_COLOR = "#7ec8e3"

        for u, v in subG.edges():
            width = edge_width(u, v)

            if u == v:
                nx.draw_networkx_edges(
                    subG, pos, edgelist=[(u, v)],
                    width=width, edge_color=[EDGE_COLOR],
                    arrows=True, arrowstyle="-|>", arrowsize=15,
                    connectionstyle="arc3,rad=0.9",
                    min_source_margin=25, min_target_margin=25, ax=ax,
                )
                continue

            rad = 0.25 if subG.has_edge(v, u) else 0.0
            nx.draw_networkx_edges(
                subG, pos, edgelist=[(u, v)],
                width=width, edge_color=[EDGE_COLOR],
                arrows=True, arrowstyle="-|>", arrowsize=15,
                connectionstyle=f"arc3,rad={rad}",
                min_source_margin=20, min_target_margin=25, ax=ax,
            )

        # ── Edge count labels (manually placed to avoid white-box glitch) ─────
        for u, v in subG.edges():
            if u == v:
                # self-loop: place label above the node
                x, y = pos[u]
                ax.text(
                    x, y + 0.18, str(weights[(u, v)]),
                    fontsize=6, color="#ffffff",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="#2a2a3e", ec="none", alpha=0.7),
                )
                continue

            # midpoint between u and v, with a small perpendicular offset
            # so the label doesn't sit on top of the arrow line
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2

            # perpendicular nudge (avoids overlapping bidirectional labels)
            dx, dy = x1 - x0, y1 - y0
            length = (dx**2 + dy**2) ** 0.5 or 1
            perp_x, perp_y = -dy / length, dx / length
            rad = 0.25 if subG.has_edge(v, u) else 0.0
            offset = 0.06 * (1 + rad * 2)
            lx, ly = mx + perp_x * offset, my + perp_y * offset

            ax.text(
                lx, ly, str(weights[(u, v)]),
                fontsize=6, color="#ffffff",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="#2a2a3e", ec="none", alpha=0.7),
            )

        # ── Legend ───────────────────────────────────────────────────────────
        ax.legend(
            handles=[
                Patch(facecolor="#7ec8e3", edgecolor="#fff", label="Event node"),
                Patch(facecolor="#f4a261", edgecolor="#fff", label="START / END"),
            ],
            loc="lower left", facecolor="#2a2a3e",
            edgecolor="#555", labelcolor="#ffffff", fontsize=9,
        )

        ax.set_title(
            f"UX Process Discovery Graph  (edge quantile threshold ≥ {quantile_threshold:.0%})",
            color="#ffffff", fontsize=13, pad=14,
        )
        ax.axis("off")
        plt.tight_layout()
        plt.show()

        return self 