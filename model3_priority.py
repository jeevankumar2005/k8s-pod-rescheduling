import time
import pandas as pd
import pulp
from model1_consolidation import consolidate, cpu_util
from model2_balancing import balance, add_headroom_nodes

WEIGHTS = {"High": 100, "Medium": 10, "Low": 1}


def capacity_check(nodes, pods, limit=1.0):
    """Rule, not a model: can all pods fit at all (CPU and memory)?"""
    cpu_ratio = pods.cpu_req.sum() / nodes.cpu_cap.sum()
    mem_ratio = pods.mem_req.sum() / nodes.mem_cap.sum()
    return (cpu_ratio <= limit and mem_ratio <= limit,
            round(cpu_ratio, 3), round(mem_ratio, 3))


def inplace_baseline(nodes, pods, weights):
    """Baseline: each node keeps its highest-priority pods; nothing moves."""
    cpu_cap = dict(zip(nodes.node_id, nodes.cpu_cap))
    mem_cap = dict(zip(nodes.node_id, nodes.mem_cap))
    place = {}
    for n, g in pods.groupby("current_node"):
        g = g.assign(w=g["priority"].map(weights)).sort_values(
            ["w", "cpu_req"], ascending=[False, True])
        c = m = 0.0
        for r in g.itertuples():
            if c + r.cpu_req <= cpu_cap[n] and m + r.mem_req <= mem_cap[n]:
                place[r.pod_id] = n
                c += r.cpu_req
                m += r.mem_req
    return place


def priority_bestfit(nodes, pods, weights):
    """Priority-first placement: stay if it fits, else best-fit move, else evict."""
    N = list(nodes["node_id"])
    free_c = dict(zip(nodes.node_id, nodes.cpu_cap.astype(float)))
    free_m = dict(zip(nodes.node_id, nodes.mem_cap.astype(float)))
    order = pods.assign(w=pods["priority"].map(weights)).sort_values(
        ["w", "cpu_req"], ascending=[False, True])
    place = {}
    for r in order.itertuples():
        n = r.current_node
        if free_c[n] >= r.cpu_req and free_m[n] >= r.mem_req:
            best = n
        else:
            best, left = None, None
            for m in N:
                if free_c[m] >= r.cpu_req and free_m[m] >= r.mem_req:
                    rem = free_c[m] - r.cpu_req
                    if left is None or rem < left:
                        best, left = m, rem
        if best is not None:
            place[r.pod_id] = best
            free_c[best] -= r.cpu_req
            free_m[best] -= r.mem_req
    return place, free_c


def score_of(place, w):
    return sum(w[p] for p in place)


def valid(place, cpu, mem, cpu_cap, mem_cap):
    lc, lm = {}, {}
    for p, n in place.items():
        lc[n] = lc.get(n, 0) + cpu[p]
        lm[n] = lm.get(n, 0) + mem[p]
    return all(lc[n] <= cpu_cap[n] + 1e-6 and lm[n] <= mem_cap[n] + 1e-6
               for n in lc)


def prioritise_pods(nodes, pods, weights=WEIGHTS, lam=0.01,
                    time_limit=120, gap=0.01, n_extra=4):
    """Model 3: maximise weighted pods kept (restricted MIP, heuristic start)."""
    P = list(pods["pod_id"])
    cpu_cap = dict(zip(nodes.node_id, nodes.cpu_cap))
    mem_cap = dict(zip(nodes.node_id, nodes.mem_cap))
    cpu = dict(zip(pods.pod_id, pods.cpu_req))
    mem = dict(zip(pods.pod_id, pods.mem_req))
    cur = dict(zip(pods.pod_id, pods.current_node))
    w = {p: weights[pr] for p, pr in zip(pods.pod_id, pods.priority)}

    base = inplace_baseline(nodes, pods, weights)
    heur, free_c = priority_bestfit(nodes, pods, weights)
    emptiest = sorted(free_c, key=lambda n: -free_c[n])[:n_extra]

    cand = {}
    for p in P:
        s = {cur[p]} | set(emptiest)
        if p in heur:
            s.add(heur[p])
        cand[p] = sorted(s)

    prob = pulp.LpProblem("priority", pulp.LpMaximize)
    x = {(p, n): pulp.LpVariable(f"x_{i}_{j}", cat=pulp.LpBinary)
         for i, p in enumerate(P) for j, n in enumerate(cand[p])}
    k = {p: pulp.LpVariable(f"k_{i}", cat=pulp.LpBinary)
         for i, p in enumerate(P)}
    mv = {p: pulp.LpVariable(f"mv_{i}", 0, 1) for i, p in enumerate(P)}

    prob += (pulp.lpSum(w[p] * k[p] for p in P)
             - lam * pulp.lpSum(mv[p] for p in P))
    on_node = {}
    for (p, n), v in x.items():
        on_node.setdefault(n, []).append((p, v))
    for p in P:
        prob += pulp.lpSum(x[(p, n)] for n in cand[p]) == k[p]
        prob += mv[p] >= k[p] - x[(p, cur[p])]
    for n, items in on_node.items():
        prob += pulp.lpSum(cpu[p] * v for p, v in items) <= cpu_cap[n]
        prob += pulp.lpSum(mem[p] * v for p, v in items) <= mem_cap[n]

    for p in P:
        k[p].setInitialValue(1 if p in heur else 0)
        for n in cand[p]:
            x[(p, n)].setInitialValue(1 if heur.get(p) == n else 0)

    t0 = time.time()
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit,
                                 gapRel=gap, warmStart=True))
    secs = time.time() - t0

    mip = {}
    for p in P:
        for n in cand[p]:
            v = x[(p, n)].value()
            if v is not None and v > 0.5:
                mip[p] = n
                break

    final, method = heur, "priority best-fit heuristic"
    if (valid(mip, cpu, mem, cpu_cap, mem_cap)
            and score_of(mip, w) > score_of(heur, w)):
        final, method = mip, "restricted MIP"

    plan = pods[["pod_id", "priority", "current_node"]].copy()
    plan["to_node"] = plan["pod_id"].map(final)
    plan["decision"] = plan.apply(
        lambda r: "evict" if pd.isna(r.to_node)
        else ("keep" if r.to_node == r.current_node else "move"), axis=1)

    return {"plan": plan, "method": method, "status": pulp.LpStatus[prob.status],
            "solve_time": round(secs, 1), "score": score_of(final, w),
            "heuristic_score": score_of(heur, w),
            "baseline_score": score_of(base, w), "max_score": sum(w.values())}


def run_pipeline(nodes, pods, cap_factor=1.0, headroom=0.85, K=150, limit=1.0):
    n2 = nodes.copy()
    n2["cpu_cap"] = n2["cpu_cap"] * cap_factor
    n2["mem_cap"] = n2["mem_cap"] * cap_factor

    ok, cr, mr = capacity_check(n2, pods, limit)
    print(f"Capacity check | CPU demand/capacity {cr} | MEM {mr} | "
          f"{'PASS -> Models 1 and 2' if ok else 'FAIL -> Model 3'}")

    if ok:
        r1 = consolidate(n2, pods, target=1.0, time_limit=120)
        print("Model 1 | status:", r1["status"], "| feasible:", r1["feasible"],
              "| time:", r1["solve_time"], "s")
        if r1["feasible"]:
            start = r1["placement"][["pod_id", "to_node"]]
            active = add_headroom_nodes(n2, pods, r1["active_nodes"], headroom)
            r2 = balance(n2, pods, start, active, K=K)
            print("Model 2 | status:", r2["status"], "| feasible:",
                  r2["feasible"], "| time:", r2["solve_time"], "s")
            if r2["feasible"]:
                plan = r2["placement"]
                s2 = pods.merge(plan[["pod_id", "to_node"]], on="pod_id")
                u0 = cpu_util(n2, pods, "current_node")
                u2 = cpu_util(n2, s2, "to_node")
                print("Balanced cluster | active nodes", s2["to_node"].nunique(),
                      "of", len(n2), "| max CPU util", round(u0.max(), 1), "->",
                      round(u2.max(), 1), "| moves",
                      int((s2["to_node"] != s2["current_node"]).sum()))
                plan.to_csv("data/scenario_normal_plan.csv", index=False)
                return
        print("Models 1/2 could not give a valid placement -> Model 3")

    r3 = prioritise_pods(n2, pods)
    plan = r3["plan"]
    print("Model 3 | result from:", r3["method"], "| MIP status:", r3["status"],
          "| time:", r3["solve_time"], "s")
    print("Weighted score kept: final", r3["score"],
          "| heuristic", r3["heuristic_score"],
          "| in-place baseline", r3["baseline_score"],
          "| max possible", r3["max_score"])
    print(plan.groupby(["priority", "decision"]).size()
          .unstack(fill_value=0).reindex(["High", "Medium", "Low"]))
    kept = pods.merge(plan.dropna(subset=["to_node"])[["pod_id", "to_node"]],
                      on="pod_id")
    u = cpu_util(n2, kept, "to_node")
    print("Kept pods:", len(kept), "of", len(pods), "| max CPU util",
          round(u.max(), 1), "| moves", int((plan.decision == "move").sum()),
          "| evicted", int((plan.decision == "evict").sum()))
    plan.to_csv("data/scenario_overload_plan.csv", index=False)


if __name__ == "__main__":
    nodes = pd.read_csv("data/nodes.csv")
    pods = pd.read_csv("data/pods.csv")
    for name, factor in [("NORMAL LOAD", 1.0), ("OVERLOAD (60% capacity)", 0.6)]:
        print("=" * 60)
        print(name)
        run_pipeline(nodes, pods, factor)