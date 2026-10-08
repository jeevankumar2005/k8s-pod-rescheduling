import pandas as pd
import plotly.express as px
import streamlit as st

import stats
from model1_consolidation import consolidate, cpu_util
from model2_balancing import balance, add_headroom_nodes
from model3_priority import prioritise_pods, capacity_check

st.set_page_config(page_title="K8s Pod Rescheduling", layout="wide")
st.title("Optimization-Based Kubernetes Pod Rescheduling")
st.caption("Alibaba Cluster Trace 2018 | Model 1 consolidation | "
           "Model 2 load balancing | Model 3 priority-based pod selection")

ss = st.session_state
NODE_COLS = {"node_id", "cpu_cap", "mem_cap"}
POD_COLS = {"pod_id", "app_du", "cpu_req", "mem_req", "current_node", "priority"}


# ---------------------------------------------------------------- helpers
def prep(nodes, pods):
    for c in ("cpu_util", "mem_util"):
        if c not in nodes:
            nodes[c] = float("nan")
    return nodes, pods


def scaled(nodes, f):
    n = nodes.copy()
    n["cpu_cap"] = n["cpu_cap"] * f
    n["mem_cap"] = n["mem_cap"] * f
    return n


def util_chart(n2, pods, after, col, title):
    ids = list(n2["node_id"])
    b = cpu_util(n2, pods, "current_node").reindex(ids).fillna(0)
    a = cpu_util(n2, after, col).reindex(ids).fillna(0)
    d = pd.DataFrame({"node": ids, "Before": b.values, "After": a.values})
    d = d.sort_values("Before", ascending=False)
    return px.bar(d, x="node", y=["Before", "After"], barmode="group",
                  title=title,
                  labels={"value": "CPU requested / capacity (%)"})


def fail(title, log, msg):
    return {"title": title, "log": log, "error": msg}


# ---------------------------------------------------------------- models
def run_m1(n2, pods, target, tl):
    r = consolidate(n2, pods, target=target, time_limit=tl)
    log = [f"Model 1 | status {r['status']} | feasible {r['feasible']} "
           f"| {r['solve_time']} s"]
    if not r["feasible"]:
        return fail("Model 1: node consolidation", log,
                    "No valid placement found (infeasible or time limit). "
                    "Try target 1.0, or run Model 3.")
    plan = r["placement"]
    after = pods.merge(plan[["pod_id", "to_node"]], on="pod_id")
    u0 = cpu_util(n2, pods, "current_node")
    u1 = cpu_util(n2, after, "to_node")
    return {"title": "Model 1: node consolidation", "log": log, "plan": plan,
            "metrics": {
                "Active nodes": f"{pods.current_node.nunique()} -> {after.to_node.nunique()}",
                "Util std": f"{u0.std():.1f} -> {u1.std():.1f}",
                "Max util %": f"{u0.max():.1f} -> {u1.max():.1f}",
                "Pods moved": int(plan["moved"].sum()),
                "Solve time (s)": r["solve_time"]},
            "chart": util_chart(n2, pods, after, "to_node",
                                "CPU utilisation per node, before vs after")}


def run_m2(n2, pods, K, tl):
    start = pods[["pod_id"]].assign(to_node=pods["current_node"])
    r = balance(n2, pods, start, list(n2["node_id"]), K=K, time_limit=tl)
    log = [f"Model 2 | status {r['status']} | feasible {r['feasible']} "
           f"| {r['solve_time']} s"]
    if not r["feasible"]:
        return fail("Model 2: load balancing", log,
                    "No valid plan within the migration limit. Raise K.")
    plan = r["placement"]
    after = pods.merge(plan[["pod_id", "to_node"]], on="pod_id")
    u0 = cpu_util(n2, pods, "current_node")
    u1 = cpu_util(n2, after, "to_node")
    return {"title": "Model 2: load balancing", "log": log, "plan": plan,
            "flows": r["flows"],
            "metrics": {
                "Util std": f"{u0.std():.1f} -> {u1.std():.1f}",
                "Max util %": f"{u0.max():.1f} -> {u1.max():.1f}",
                "Pods moved": int(plan["moved"].sum()),
                "Solve time (s)": r["solve_time"]},
            "chart": util_chart(n2, pods, after, "to_node",
                                "CPU utilisation per node, before vs after")}


def run_m3(n2, pods, weights, tl):
    r3 = prioritise_pods(n2, pods, weights=weights, time_limit=tl)
    plan = r3["plan"]
    kept = pods.merge(plan.dropna(subset=["to_node"])[["pod_id", "to_node"]],
                      on="pod_id")
    u = cpu_util(n2, kept, "to_node")
    table = (plan.groupby(["priority", "decision"]).size()
             .unstack(fill_value=0).reindex(["High", "Medium", "Low"]))
    log = [f"Model 3 | result from: {r3['method']} | MIP status {r3['status']} "
           f"| {r3['solve_time']} s",
           f"Weighted score: final {r3['score']} | heuristic {r3['heuristic_score']} "
           f"| in-place baseline {r3['baseline_score']} | max {r3['max_score']}"]
    return {"title": "Model 3: priority-based pod selection", "log": log,
            "plan": plan, "table": table,
            "metrics": {
                "Pods kept": f"{len(kept)} / {len(pods)}",
                "Evicted": int((plan.decision == 'evict').sum()),
                "Moved": int((plan.decision == 'move').sum()),
                "Score": f"{100 * r3['score'] / r3['max_score']:.1f}%",
                "Max util %": f"{u.max():.1f}"},
            "chart": util_chart(n2, pods, kept, "to_node",
                                "CPU utilisation per node, before vs after")}


def run_full(n2, pods, headroom, K, weights, tl):
    ok, cr, mr = capacity_check(n2, pods)
    log = [f"Capacity check | CPU demand/capacity {cr} | MEM {mr} | "
           f"{'PASS -> Models 1 and 2' if ok else 'FAIL -> Model 3'}"]
    if ok:
        r1 = consolidate(n2, pods, target=1.0, time_limit=tl)
        log.append(f"Model 1 | status {r1['status']} | feasible {r1['feasible']} "
                   f"| {r1['solve_time']} s")
        if r1["feasible"]:
            start = r1["placement"][["pod_id", "to_node"]]
            active = add_headroom_nodes(n2, pods, r1["active_nodes"], headroom)
            r2 = balance(n2, pods, start, active, K=K, time_limit=tl)
            log.append(f"Model 2 | status {r2['status']} | feasible {r2['feasible']} "
                       f"| {r2['solve_time']} s")
            if r2["feasible"]:
                plan = r2["placement"]
                after = pods.merge(plan[["pod_id", "to_node"]], on="pod_id")
                u0 = cpu_util(n2, pods, "current_node")
                u2 = cpu_util(n2, after, "to_node")
                moves = int((after["to_node"] != after["current_node"]).sum())
                return {"title": "Full pipeline: balanced cluster", "log": log,
                        "plan": plan, "flows": r2["flows"],
                        "metrics": {
                            "Active nodes": f"{pods.current_node.nunique()} -> "
                                            f"{len(r1['active_nodes'])} (min) -> "
                                            f"{after.to_node.nunique()}",
                            "Util std": f"{u0.std():.1f} -> {u2.std():.1f}",
                            "Max util %": f"{u0.max():.1f} -> {u2.max():.1f}",
                            "Total moves": moves,
                            "Solve time (s)": round(r1["solve_time"] + r2["solve_time"], 1)},
                        "chart": util_chart(n2, pods, after, "to_node",
                                            "CPU utilisation per node, before vs after")}
        log.append("Models 1/2 gave no valid placement -> Model 3")
    res = run_m3(n2, pods, weights, tl)
    res["log"] = log + res["log"]
    res["title"] = "Full pipeline: capacity insufficient, Model 3"
    return res


# ---------------------------------------------------------------- UI
tab_load, tab_stats, tab_opt, tab_res = st.tabs(
    ["1. Load data", "2. Statistics", "3. Optimisation", "4. Results"])

with tab_load:
    src = st.radio("Data source",
                   ["Use sample (data/nodes.csv, data/pods.csv)",
                    "Upload nodes.csv and pods.csv"], horizontal=True)
    try:
        if src.startswith("Use"):
            ss.nodes, ss.pods = prep(pd.read_csv("data/nodes.csv"),
                                     pd.read_csv("data/pods.csv"))
        else:
            c1, c2 = st.columns(2)
            fn = c1.file_uploader("nodes.csv", type="csv")
            fp = c2.file_uploader("pods.csv", type="csv")
            if fn is not None and fp is not None:
                n, p = pd.read_csv(fn), pd.read_csv(fp)
                if not NODE_COLS <= set(n.columns) or not POD_COLS <= set(p.columns):
                    st.error(f"nodes.csv needs {sorted(NODE_COLS)}; "
                             f"pods.csv needs {sorted(POD_COLS)}")
                else:
                    ss.nodes, ss.pods = prep(n, p)
            else:
                ss.pop("nodes", None)
                ss.pop("pods", None)
    except FileNotFoundError:
        st.error("data/nodes.csv or data/pods.csv not found. Run data_prep.py first.")

    if "nodes" in ss:
        st.success(f"Loaded {len(ss.nodes)} nodes and {len(ss.pods)} pods")
        a, b = st.columns(2)
        a.write("Nodes")
        a.dataframe(ss.nodes.head(20))
        b.write("Pods")
        b.dataframe(ss.pods.head(20))
    else:
        st.info("Load data to continue.")

loaded = "nodes" in ss

with tab_stats:
    if not loaded:
        st.info("Load data first.")
    else:
        if st.button("Generate statistics"):
            ss.stats_done = True
        if ss.get("stats_done"):
            s = stats.summary(ss.nodes, ss.pods)
            st.subheader("Summary")
            st.dataframe(pd.Series(s, name="value").astype(str).to_frame())
            st.subheader("Utilisation statistics")
            st.dataframe(stats.util_stats(ss.nodes, ss.pods))
            figs = list(stats.charts(ss.nodes, ss.pods).values())
            for i in range(0, len(figs), 2):
                cols = st.columns(2)
                for col, fig in zip(cols, figs[i:i + 2]):
                    col.plotly_chart(fig)

with tab_opt:
    if not loaded:
        st.info("Load data first.")
    else:
        choice = st.selectbox("Optimisation", [
            "Full pipeline (capacity check -> Models 1+2, or Model 3)",
            "Model 1: node consolidation",
            "Model 2: load balancing (all nodes, from current placement)",
            "Model 3: priority-based pod selection"])
        c1, c2, c3 = st.columns(3)
        cap = c1.slider("Capacity factor (scale all node capacity)", 0.4, 1.0, 1.0, 0.05,
                        help="Below 1.0 simulates an overload.")
        tl = c2.slider("Solver time limit per model (s)", 30, 300, 90, 10)
        K = c3.slider("Migration limit K (Model 2)", 0, 600, 200, 10)
        c4, c5 = st.columns(2)
        headroom = c4.slider("Target utilisation / headroom (full pipeline)",
                             0.70, 1.00, 0.85, 0.01)
        target = c5.slider("Model 1 target utilisation", 0.95, 1.0, 1.0, 0.01,
                           help="Below 1.0 the solver may time out.")
        w1, w2, w3 = st.columns(3)
        weights = {"High": w1.number_input("Weight High", 1, 1000, 100),
                   "Medium": w2.number_input("Weight Medium", 1, 1000, 10),
                   "Low": w3.number_input("Weight Low", 1, 1000, 1)}

        n2 = scaled(ss.nodes, cap)
        ok, cr, mr = capacity_check(n2, ss.pods)
        (st.info if ok else st.warning)(
            f"CPU demand / capacity = {cr} | memory = {mr} -> "
            f"{'everything fits' if ok else 'capacity is insufficient, Model 3 applies'}")

        if st.button("Run optimisation", type="primary"):
            with st.spinner("Solving... this can take a few minutes"):
                try:
                    if choice.startswith("Full"):
                        ss.result = run_full(n2, ss.pods, headroom, K, weights, tl)
                    elif choice.startswith("Model 1"):
                        ss.result = run_m1(n2, ss.pods, target, tl)
                    elif choice.startswith("Model 2"):
                        ss.result = run_m2(n2, ss.pods, K, tl)
                    else:
                        ss.result = run_m3(n2, ss.pods, weights, tl)
                    st.success("Done. See the Results tab.")
                except Exception as e:
                    st.error(f"Run failed: {e}")

with tab_res:
    res = ss.get("result")
    if not res:
        st.info("Run an optimisation first.")
    else:
        st.subheader(res["title"])
        st.code("\n".join(res["log"]))
        if "error" in res:
            st.error(res["error"])
        else:
            cols = st.columns(len(res["metrics"]))
            for col, (k, v) in zip(cols, res["metrics"].items()):
                col.metric(k, v)
            if res.get("chart") is not None:
                st.plotly_chart(res["chart"])
            if res.get("table") is not None:
                st.write("Pods by priority and decision")
                st.dataframe(res["table"])
            if res.get("flows") is not None and len(res["flows"]):
                st.write("Move plan: source -> destination (top 20 by CPU)")
                st.dataframe(res["flows"].head(20))
            st.download_button("Download plan (CSV)",
                               res["plan"].to_csv(index=False),
                               "plan.csv", "text/csv")
            