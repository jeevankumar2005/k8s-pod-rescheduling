import pandas as pd
import plotly.express as px


def node_table(nodes, pods):
    """Per-node requested load, pod count and real utilisation."""
    g = pods.groupby("current_node").agg(
        n_pods=("pod_id", "count"),
        cpu_load=("cpu_req", "sum"),
        mem_load=("mem_req", "sum"))
    t = nodes.merge(g, left_on="node_id", right_index=True, how="left")
    t[["n_pods", "cpu_load", "mem_load"]] = \
        t[["n_pods", "cpu_load", "mem_load"]].fillna(0)
    t["cpu_req_util"] = 100 * t["cpu_load"] / t["cpu_cap"]
    t["mem_req_util"] = 100 * t["mem_load"] / t["mem_cap"]
    return t


def summary(nodes, pods, hi=80, lo=30):
    t = node_table(nodes, pods)
    return {
        "nodes": len(nodes),
        "pods": len(pods),
        "apps": pods["app_du"].nunique(),
        "avg_pods_per_node": round(t["n_pods"].mean(), 1),
        "cpu_demand_vs_capacity": round(pods.cpu_req.sum() / nodes.cpu_cap.sum(), 3),
        "mem_demand_vs_capacity": round(pods.mem_req.sum() / nodes.mem_cap.sum(), 3),
        "overloaded_nodes": int((t["cpu_req_util"] > hi).sum()),
        "underused_nodes": int((t["cpu_req_util"] < lo).sum()),
        "cpu_req_util_mean": round(t["cpu_req_util"].mean(), 1),
        "cpu_req_util_std": round(t["cpu_req_util"].std(), 1),
        "cpu_req_util_max": round(t["cpu_req_util"].max(), 1),
        "real_cpu_util_mean": round(t["cpu_util"].mean(), 1),
    }


def util_stats(nodes, pods):
    t = node_table(nodes, pods)
    cols = ["cpu_req_util", "mem_req_util", "cpu_util", "mem_util", "n_pods"]
    return t[cols].describe().round(2)


def charts(nodes, pods, hi=80, lo=30):
    t = node_table(nodes, pods)
    figs = {}
    f = px.histogram(t, x="cpu_req_util", nbins=20,
                     title="Node CPU utilisation (requested), %")
    f.add_vline(x=hi, line_dash="dash", line_color="red")
    f.add_vline(x=lo, line_dash="dash", line_color="orange")
    figs["node_util"] = f
    figs["real_vs_req"] = px.scatter(
        t, x="cpu_req_util", y="cpu_util", hover_name="node_id",
        title="Requested vs real CPU utilisation per node, %")
    figs["pod_size"] = px.histogram(
        pods, x="cpu_req", nbins=30, title="Pod CPU request distribution")
    pc = pods["priority"].value_counts().reset_index()
    pc.columns = ["priority", "pods"]
    figs["priority"] = px.bar(pc, x="priority", y="pods",
                              title="Pods by priority")
    return figs


if __name__ == "__main__":
    nodes = pd.read_csv("data/nodes.csv")
    pods = pd.read_csv("data/pods.csv")
    for k, v in summary(nodes, pods).items():
        print(f"{k}: {v}")
    print(util_stats(nodes, pods))
    for name, fig in charts(nodes, pods).items():
        fig.write_html(f"data/chart_{name}.html")
    print("charts saved in data/")