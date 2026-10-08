# Optimization-Based Kubernetes Pod Rescheduling

Operations Research course project. Uses the Alibaba Cluster Trace 2018
(public dataset) with three optimisation models:

1. **Model 1: Node consolidation** (bin packing, minimise active nodes)
2. **Model 2: Load balancing** (minimise maximum node utilisation, migration limit)
3. **Model 3: Priority-based pod selection** (used when capacity is insufficient)

A capacity check decides whether the full pipeline runs Models 1 and 2 or Model 3.

## Run the app
```
pip install -r requirements.txt
streamlit run app.py
```
`data/nodes.csv` and `data/pods.csv` are a ready-made sample (60 nodes, 945 pods),
so the app works straight away. You can also upload your own files in the UI.

## Rebuild the sample from the raw trace (optional)
1. Get `machine_meta`, `container_meta` and `machine_usage` from
   https://github.com/alibaba/clusterdata (cluster-trace-v2018). The download
   link is available after a short survey.
2. Extract the CSVs into `data/`, then run `python data_prep.py`.

## Files
- `data_prep.py` builds the nodes/pods snapshot from the raw trace
- `stats.py` statistics and charts
- `model1_consolidation.py`, `model2_balancing.py`, `model3_priority.py`
- `app.py` Streamlit UI

## Assumptions
- 60 nodes sampled from about 4,000, at one snapshot time
- The trace has no priority field, so priority (High/Medium/Low) is assigned per app
- Pods are placed by requested resources, not measured usage

Dataset: Alibaba Cluster Trace 2018 (Alibaba Group).