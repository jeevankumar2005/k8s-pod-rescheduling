import time
import pandas as pd
import pulp
from model1_consolidation import consolidate, cpu_util


def balance(nodes, pods, start, active, K=150, mu=1e-5,
            time_limit=60, gap=0.01):
    """Minimise max CPU utilisation over active nodes, with at most K moves
    from the starting placement. start: DataFrame [pod_id, to_node]."""
    A = list(active)
    P = list(pods["pod_id"])
    nd = nodes.set_index("node_id")
    cpu_cap = nd["cpu_cap"].to_dict()
    mem_cap = nd["mem_cap"].to_dict()
    cpu = dict(zip(pods.pod_id, pods.cpu_req))
    mem = dict(zip(pods.pod_id, pods.mem_req))
    cur = dict(zip(start.pod_id, start.to_node))

    prob = pulp.LpProblem("balancing", pulp.LpMinimize)
    x = {p: {n: pulp.LpVariable(f"x_{i}_{j}", cat=pulp.LpBinary)
             for j, n in enumerate(A)} for i, p in enumerate(P)}
    mv = {p: pulp.LpVariable(f"mv_{i}", 0, 1) for i, p in enumerate(P)}
    z = pulp.LpVariable("z", 0, 1)          # max CPU utilisation (fraction)

    prob += z + mu * pulp.lpSum(mv[p] for p in P)

    for p in P:
        prob += pulp.lpSum(x[p][n] for n in A) == 1
        prob += mv[p] >= 1 - x[p][cur[p]]
    prob += pulp.lpSum(mv[p] for p in P) <= K
    for n in A:
        prob += pulp.lpSum(cpu[p] * x[p][n] for p in P) <= cpu_cap[n] * z
        prob += pulp.lpSum(mem[p] * x[p][n] for p in P) <= mem_cap[n]

    # warm start from the Model 1 placement
    for p in P:
        for n in A:
            x[p][n].setInitialValue(1 if cur[p] == n else 0)

    t0 = time.time()
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit,
                                 gapRel=gap, warmStart=True))
    secs = time.time() - t0

    place = {}
    for p in P:
        for n in A:
            v = x[p][n].value()
            if v is not None and v > 0.5:
                place[p] = n
                break
    ok = len(place) == len(P)

    # honest check: capacities must really hold
    if ok:
        lc, lm = {}, {}
        for p, n in place.items():
            lc[n] = lc.get(n, 0) + cpu[p]
            lm[n] = lm.get(n, 0) + mem[p]
        ok = all(lc[n] <= cpu_cap[n] + 1e-6 and lm[n] <= mem_cap[n] + 1e-6
                 for n in lc)

    result = {"feasible": ok, "status": pulp.LpStatus[prob.status],
              "solve_time": round(secs, 1), "placement": None, "flows": None}
    if ok:
        plan = start.rename(columns={"to_node": "from_node"}).copy()
        plan["to_node"] = plan["pod_id"].map(place)
        plan["moved"] = plan["from_node"] != plan["to_node"]
        plan = plan.merge(pods[["pod_id", "cpu_req"]], on="pod_id")
        result["placement"] = plan
        # source -> destination load plan
        result["flows"] = (plan[plan["moved"]]
                           .groupby(["from_node", "to_node"])
                           .agg(pods=("pod_id", "count"),
                                cpu=("cpu_req", "sum"))
                           .reset_index()
                           .sort_values("cpu", ascending=False))
    return result


def add_headroom_nodes(nodes, pods, active, headroom=0.85):
    """Keep extra nodes active until total demand / active capacity <= headroom."""
    need = pods["cpu_req"].sum() / headroom
    act = list(active)
    cap = nodes.set_index("node_id")["cpu_cap"]
    npods = pods.groupby("current_node").size()
    chosen = set(act)
    rest = [n for n in nodes["node_id"] if n not in chosen]
    rest.sort(key=lambda n: (-cap[n], -npods.get(n, 0)))
    while cap[act].sum() < need and rest:
        act.append(rest.pop(0))
    return act


if __name__ == "__main__":
    nodes = pd.read_csv("data/nodes.csv")
    pods = pd.read_csv("data/pods.csv")

    r1 = consolidate(nodes, pods, target=1.0, time_limit=120)
    print("Model 1 | status:", r1["status"], "| feasible:", r1["feasible"],
          "| time:", r1["solve_time"], "s")
    if not r1["feasible"]:
        raise SystemExit("Model 1 found no valid placement -> Model 3 needed")

    start = r1["placement"][["pod_id", "to_node"]]
    min_active = r1["active_nodes"]
    m1_moves = int(r1["placement"]["moved"].sum())
    active = add_headroom_nodes(nodes, pods, min_active, headroom=0.85)

    r2 = balance(nodes, pods, start, active, K=150)
    print("Model 2 | status:", r2["status"], "| feasible:", r2["feasible"],
          "| solve time (s):", r2["solve_time"])
    if r2["feasible"]:
        plan = r2["placement"]
        s1 = pods.merge(start, on="pod_id")
        s2 = pods.merge(plan[["pod_id", "to_node"]], on="pod_id")
        u0 = cpu_util(nodes, pods, "current_node")
        u1 = cpu_util(nodes, s1, "to_node")
        u2 = cpu_util(nodes, s2, "to_node")
        total_moves = int((s2["to_node"] != s2["current_node"]).sum())

        print("active nodes: original", pods["current_node"].nunique(),
              "| M1 minimum", len(min_active), "| kept with headroom", len(active))
        print("CPU util std  original / M1 / M2:",
              round(u0.std(), 1), "/", round(u1.std(), 1), "/", round(u2.std(), 1))
        print("CPU util max  original / M1 / M2:",
              round(u0.max(), 1), "/", round(u1.max(), 1), "/", round(u2.max(), 1))
        print("moves: M1", m1_moves, "| M2 extra", int(plan["moved"].sum()),
              "| total vs original", total_moves)
        print(r2["flows"].head(10).to_string(index=False))
        plan.to_csv("data/model2_plan.csv", index=False)