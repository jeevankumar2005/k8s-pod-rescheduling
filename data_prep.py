import pandas as pd

MACHINE_COLS = ["machine_id", "time_stamp", "fd1", "fd2",
                "cpu_num", "mem_size", "status"]
CONTAINER_COLS = ["container_id", "machine_id", "time_stamp", "app_du",
                  "status", "cpu_request", "cpu_limit", "mem_size"]


def load_raw(machine_path="data/machine_meta.csv",
             container_path="data/container_meta.csv"):
    m = pd.read_csv(machine_path, header=None, names=MACHINE_COLS)
    c = pd.read_csv(container_path, header=None, names=CONTAINER_COLS)
    return m, c


def latest_state(df, id_col, t):
    """Last event of each ID at or before time t."""
    df = df[df["time_stamp"] <= t].sort_values("time_stamp")
    return df.groupby(id_col).tail(1)


def build_snapshot(m, c, t=None, n_nodes=60, seed=42):
    if t is None:
        t = int(c["time_stamp"].median())

    machines = latest_state(m, "machine_id", t)
    machines = machines[machines["status"] == "USING"]

    pods = latest_state(c, "container_id", t)
    pods = pods[pods["status"] == "started"]
    pods = pods[pods["machine_id"].isin(machines["machine_id"])]

    # sample nodes that actually host pods
    hosting = pods["machine_id"].unique()
    n_nodes = min(n_nodes, len(hosting))
    chosen = pd.Series(hosting).sample(n=n_nodes, random_state=seed)

    nodes = machines[machines["machine_id"].isin(chosen)].copy()
    nodes["cpu_cap"] = nodes["cpu_num"] * 100      # 1/100-core units
    nodes["mem_cap"] = nodes["mem_size"]
    nodes = nodes.rename(columns={"machine_id": "node_id"})[
        ["node_id", "cpu_cap", "mem_cap"]].reset_index(drop=True)

    pods = pods[pods["machine_id"].isin(chosen)].copy()
    pods = pods.rename(columns={"container_id": "pod_id",
                                "machine_id": "current_node",
                                "cpu_request": "cpu_req",
                                "mem_size": "mem_req"})

    # ASSUMPTION: priority is not in the trace -> assign per app, seeded
    apps = pd.Series(pods["app_du"].unique())
    prio = apps.sample(frac=1, random_state=seed).reset_index(drop=True)
    cut1, cut2 = int(0.2 * len(prio)), int(0.5 * len(prio))
    mapping = {a: "High" if i < cut1 else "Medium" if i < cut2 else "Low"
               for i, a in enumerate(prio)}
    pods["priority"] = pods["app_du"].map(mapping)

    pods = pods[["pod_id", "app_du", "cpu_req", "mem_req",
                 "current_node", "priority"]].reset_index(drop=True)
    return nodes, pods, t


def add_usage(nodes, t, path="data/machine_usage.csv", window=900):
    """Attach each node's latest real utilisation at or before time t."""
    cols = ["machine_id", "time_stamp", "cpu_util", "mem_util"]
    ids = set(nodes["node_id"])
    keep = []
    for ch in pd.read_csv(path, header=None, names=cols,
                          usecols=[0, 1, 2, 3], chunksize=5_000_000):
        ch = ch[ch["machine_id"].isin(ids)
                & ch["time_stamp"].between(t - window, t)]
        if len(ch):
            keep.append(ch)
    u = pd.concat(keep).dropna(subset=["cpu_util", "mem_util"])
    u = u.sort_values("time_stamp").groupby("machine_id").tail(1)
    u = u.rename(columns={"machine_id": "node_id"})
    return nodes.merge(u[["node_id", "cpu_util", "mem_util"]],
                       on="node_id", how="left")


if __name__ == "__main__":
    m, c = load_raw()
    nodes, pods, t = build_snapshot(m, c)
    nodes = add_usage(nodes, t)
    nodes.to_csv("data/nodes.csv", index=False)
    pods.to_csv("data/pods.csv", index=False)

    print("snapshot time:", t)
    print("nodes:", len(nodes), "| pods:", len(pods))
    print("CPU demand/capacity:",
          round(pods.cpu_req.sum() / nodes.cpu_cap.sum(), 3))
    print("MEM demand/capacity:",
          round(pods.mem_req.sum() / nodes.mem_cap.sum(), 3))
    print(pods.priority.value_counts())
    print("nodes with usage:", nodes.cpu_util.notna().sum(), "/", len(nodes))
    print(nodes[["cpu_util", "mem_util"]].describe())