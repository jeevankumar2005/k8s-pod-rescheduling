import time
import pandas as pd
import pulp


def consolidate(nodes, pods, target=1.0, lam=0.01, time_limit=60, gap=0.01):
    """Minimise active nodes + lam * moved pods, under CPU and memory caps."""
    N = list(nodes["node_id"])
    P = list(pods["pod_id"])
    cpu_cap = dict(zip(nodes.node_id, nodes.cpu_cap))
    mem_cap = dict(zip(nodes.node_id, nodes.mem_cap))
    cpu = dict(zip(pods.pod_id, pods.cpu_req))
    mem = dict(zip(pods.pod_id, pods.mem_req))
    cur = dict(zip(pods.pod_id, pods.current_node))

    prob = pulp.LpProblem("consolidation", pulp.LpMinimize)
    x = {p: {n: pulp.LpVariable(f"x_{i}_{j}", cat=pulp.LpBinary)
             for j, n in enumerate(N)} for i, p in enumerate(P)}
    y = {n: pulp.LpVariable(f"y_{j}", cat=pulp.LpBinary)
         for j, n in enumerate(N)}
    mv = {p: pulp.LpVariable(f"mv_{i}", 0, 1) for i, p in enumerate(P)}

    prob += (pulp.lpSum(y[n] for n in N)
             + lam * pulp.lpSum(mv[p] for p in P))

    for p in P:
        prob += pulp.lpSum(x[p][n] for n in N) == 1
        prob += mv[p] >= 1 - x[p][cur[p]]
    for n in N:
        prob += (pulp.lpSum(cpu[p] * x[p][n] for p in P)
                 <= target * cpu_cap[n] * y[n])
        prob += (pulp.lpSum(mem[p] * x[p][n] for p in P)
                 <= target * mem_cap[n] * y[n])

    t0 = time.time()
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit, gapRel=gap))
    secs = time.time() - t0

    place = {}
    for p in P:
        for n in N:
            v = x[p][n].value()
            if v is not None and v > 0.5:
                place[p] = n
                break

    feasible = len(place) == len(P)
    if feasible:                       # honest check: caps must really hold
        lc, lm = {}, {}
        for p, n in place.items():
            lc[n] = lc.get(n, 0) + cpu[p]
            lm[n] = lm.get(n, 0) + mem[p]
        feasible = all(lc[n] <= target * cpu_cap[n] + 1e-6
                       and lm[n] <= target * mem_cap[n] + 1e-6 for n in lc)

    result = {"feasible": feasible, "status": pulp.LpStatus[prob.status],
              "solve_time": round(secs, 1), "placement": None,
              "active_nodes": None}
    if feasible:
        plan = pods[["pod_id", "current_node"]].rename(
            columns={"current_node": "from_node"}).copy()
        plan["to_node"] = plan["pod_id"].map(place)
        plan["moved"] = plan["from_node"] != plan["to_node"]
        result["placement"] = plan
        result["active_nodes"] = sorted(plan["to_node"].unique())
    return result


def cpu_util(nodes, pods, node_col):
    load = pods.groupby(node_col)["cpu_req"].sum()
    cap = nodes.set_index("node_id")["cpu_cap"]
    return 100 * load / cap.reindex(load.index)


if __name__ == "__main__":
    nodes = pd.read_csv("data/nodes.csv")
    pods = pd.read_csv("data/pods.csv")

    r = consolidate(nodes, pods, target=1.0)
    print("status:", r["status"], "| feasible:", r["feasible"],
          "| solve time (s):", r["solve_time"])
    if r["feasible"]:
        plan = r["placement"]
        after = pods.merge(plan[["pod_id", "to_node"]], on="pod_id")
        before_u = cpu_util(nodes, pods, "current_node")
        after_u = cpu_util(nodes, after, "to_node")
        print("active nodes before:", pods["current_node"].nunique())
        print("active nodes after:", len(r["active_nodes"]))
        print("pods moved:", int(plan["moved"].sum()), "of", len(plan))
        print("CPU util std before / after:",
              round(before_u.std(), 1), "/", round(after_u.std(), 1))
        print("max CPU util before / after:",
              round(before_u.max(), 1), "/", round(after_u.max(), 1))
        plan.to_csv("data/model1_plan.csv", index=False)