from __future__ import annotations
import html
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter
import numpy as np
import pandas as pd
import streamlit as st

import dashboard_data as dd
from dashboard_data import ArtifactError, EXPERIMENTS, LABELS
from traffic_inference import InferenceError, load_predictor

st.set_page_config(page_title="Federated IoT IDS", page_icon="🛡", layout="wide")
st.markdown("""
<style>
.block-container{max-width:1420px;padding-top:1.5rem;padding-bottom:3rem}
h1,h2,h3{color:#10243a;letter-spacing:-.02em}
h1{font-size:1.85rem!important;font-weight:700!important;margin-bottom:.2rem!important}
h2{font-size:1.22rem!important;font-weight:650!important;margin-top:1.25rem!important}
p,label,.stMarkdown{color:#334155}
[data-testid="stMetric"]{background:#fff;border:1px solid #dbe4ee;border-radius:12px;padding:13px 16px}
[data-testid="stMetricLabel"]{color:#52657a!important;font-size:.82rem!important}
[data-testid="stMetricValue"]{color:#10243a!important;font-weight:700!important}
[data-testid="stSidebar"]{border-right:1px solid #dbe4ee}
[data-testid="stSidebar"] h1{font-size:1.12rem!important}
div[data-testid="stExpander"]{border:1px solid #dbe4ee;border-radius:10px;background:#fff}
div[data-testid="stDataFrame"]{border:1px solid #e2e8f0;border-radius:8px}
.small{font-size:.88rem;color:#52657a}
.finding{background:#f0fdfa;border-left:4px solid #0f766e;padding:.8rem 1rem;border-radius:6px;color:#134e4a}
.attack{background:#fff1f2;border-left:4px solid #be123c;padding:.9rem 1rem;border-radius:8px;color:#881337}
.normal{background:#ecfdf5;border-left:4px solid #0f766e;padding:.9rem 1rem;border-radius:8px;color:#134e4a}
.client{background:#fff;border:1px solid #dbe4ee;border-radius:12px;padding:15px;min-height:100px}
.client strong{color:#10243a}.client-value{font-size:1.15rem;font-weight:700;color:#10243a;margin-top:8px}
</style>
""", unsafe_allow_html=True)

PAGES=["Overview","Traffic Classification","Dataset","Model Comparison","Federated Learning",
       "Drift Detection","Selective Retraining","Model Evaluation"]
with st.sidebar:
    st.markdown("### 🛡 Federated IoT IDS")
    st.caption("CICIoT2023 experiment dashboard")
    page=st.radio("Navigate",PAGES,label_visibility="collapsed")
COLORS=["#2563eb","#0f766e","#7c3aed","#d97706","#be123c"]
RARE=["Uploading_Attack","XSS","Backdoor_Malware","SqlInjection","BrowserHijacking","Recon-PingSweep"]

def heading(title,subtitle=""):
    st.title(title)
    if subtitle: st.markdown(f'<div class="small">{subtitle}</div>',unsafe_allow_html=True)
def pct(x):
    try: return "—" if x is None or pd.isna(x) else f"{float(x)*100:.2f}%"
    except (TypeError,ValueError): return "—"
def num(x):
    try:return f"{int(x):,}"
    except (TypeError,ValueError):return "—"
def cards(items):
    cols=st.columns(len(items))
    for c,(k,v) in zip(cols,items): c.metric(k,v)
def safe(loader):
    try:return loader()
    except (ArtifactError,OSError,ValueError,KeyError,TypeError) as exc:
        st.error("A saved artifact could not be loaded."); st.caption(str(exc)); return None
def axes(ax,title,xlabel=None,ylabel=None):
    ax.set_title(title,color="#10243a",fontsize=13,fontweight="bold",pad=11)
    if xlabel:ax.set_xlabel(xlabel,color="#334155",labelpad=8)
    if ylabel:ax.set_ylabel(ylabel,color="#334155",labelpad=8)
    ax.tick_params(axis="both",colors="#475569",labelsize=9)
    ax.grid(axis="x",color="#e2e8f0",linewidth=.7);ax.set_axisbelow(True)
    for s in ax.spines.values():s.set_color("#dbe4ee")
def show(fig):
    fig.patch.set_facecolor("#fff");st.pyplot(fig,width='stretch');plt.close(fig)
def sizes(d):
    return d["metadata"].get("derived_rows",d["sizes"])
def get_result_rows():
    return dd.load_comparison_metrics()
def compare_chart(data,metrics=("accuracy","macro_f1"),title="Final test performance"):
    fig,ax=plt.subplots(figsize=(10.5,4.5),constrained_layout=True)
    names=["Original centralized baseline","Frozen improved centralized","FedAvg"]
    q=data.set_index("Model").reindex(names)
    x=np.arange(len(names));w=.34
    labels={"accuracy":"Accuracy","macro_f1":"Macro-F1","weighted_f1":"Weighted-F1"}
    for j,m in enumerate(metrics):
        bars=ax.bar(x+(j-(len(metrics)-1)/2)*w,q[m],w,label=labels.get(m,m),color=COLORS[j])
        if len(metrics) <= 2:
            ax.bar_label(bars, labels=[f"{v*100:.1f}%" for v in q[m]], fontsize=8, color="#334155", padding=2)
    ax.set_xticks(x,["Centralized baseline","Improved centralized","FedAvg"])
    ax.set_ylim(0,1.08);axes(ax,title,ylabel="Score")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.grid(axis="y",color="#e2e8f0");ax.grid(axis="x",visible=False)
    ax.legend(frameon=False,labelcolor="#334155",ncol=len(metrics))
    show(fig)

def overview():
    heading("Privacy-Preserving Federated IoT IDS","Classify engineered CICIoT2023 traffic-flow features and review the saved federation, drift, and retraining experiments.")
    d=safe(dd.load_dataset_summary);m=safe(get_result_rows);f=safe(dd.load_federated_summary)
    if any(x is None for x in (d,m,f)):return
    sz=sizes(d);candidate=m.loc[m.Model=="Frozen improved centralized"].iloc[0]
    st.caption("Saved final test results. Validation was used for model selection; no raw test records are loaded here.")
    cards([("Training rows",num(sz.get("train"))),("Test rows",num(sz.get("test"))),
           ("Retained features",num(len(d["retained_features"]))),("Classes",num(len(d["classes"])))])
    st.subheader("Centralized candidate and FedAvg")
    compare_chart(m,("accuracy","macro_f1"),"Accuracy and Macro-F1 on the final test split")
    st.markdown(f'<div class="finding"><b>Finding:</b> the frozen improved centralized model test Macro-F1 is {pct(candidate["macro_f1"])}. '
                'The Phase 5 retraining candidate was rejected on validation and did not replace FedAvg.</div>',unsafe_allow_html=True)
    st.subheader("Workflow")
    st.markdown("**Flow features → saved preprocessing → 34-class prediction → normal / attack-like summary → PSI drift monitor → client-specific retraining decision.**")
    st.caption(f"{len(f['clients'])} simulated clients, {int(f['rounds']['round'].max())} FedAvg rounds. Single-computer simulation; no formal privacy guarantee.")

def traffic():
    heading("Traffic Classification", "Classify a training-derived illustrative traffic record.")
    try:
        predictor = load_predictor()
    except (InferenceError, ArtifactError, OSError, ValueError, KeyError) as exc:
        st.error("The frozen model or preprocessing artifacts are missing or incompatible.")
        st.caption(str(exc))
        return

    st.caption("No live packet capture. A model prediction is not a guarantee of malicious intent.")
    try:
        examples = predictor.demo_records()
    except InferenceError as exc:
        st.error("Training-derived demo examples are unavailable.")
        st.caption(str(exc))
        return

    selected = st.selectbox(
        "Illustrative example", range(len(examples)),
        format_func=lambda index: f"Example {index + 1}",
    )
    frame = pd.DataFrame([examples[selected].to_dict()])
    classify = st.button("Classify example", type="primary")

    if classify:
        try:
            result = predictor.predict(frame)
        except InferenceError as exc:
            st.error("The illustrative record could not be classified.")
            st.caption(str(exc))
            return
        label = result["labels"][0]
        status = result["status"][0]
        panel_class = "normal" if label == "BenignTraffic" else "attack"
        st.markdown(
            f'<div class="{panel_class}"><b>{status}</b><br>Predicted class: <b>{label}</b></div>',
            unsafe_allow_html=True,
        )
        st.caption("Top-three outputs are uncalibrated model scores, not confidence.")
        top = pd.DataFrame(result["top3"][0]).rename(
            columns={"label": "Predicted class", "score": "Model score"}
        )
        top["Model score"] = top["Model score"].map(lambda value: f"{value:.4f}")
        st.dataframe(top, hide_index=True, width="stretch")
        st.caption("BenignTraffic maps to NORMAL. Every other class maps to ATTACK-LIKE; this result does not establish malicious intent.")

def dataset():
    heading("Dataset Overview", "A concise summary of CICIoT2023 preparation and data-quality checks.")
    d = safe(dd.load_dataset_summary)
    if d is None:
        return
    sz = sizes(d)
    counts = d["class_counts"]
    n_classes = len(d.get("classes", []))
    cards([
        ("Training records", num(sz.get("train"))),
        ("Validation records", num(sz.get("validation"))),
        ("Test records", num(sz.get("test"))),
        ("Number of classes", num(n_classes)),
    ])

    st.subheader("Dataset split sizes")
    split_names = ["Training", "Validation", "Test"]
    split_values = [sz.get("train"), sz.get("validation"), sz.get("test")]
    if all(value is not None for value in split_values):
        split_values = [int(value) for value in split_values]
        fig, ax = plt.subplots(figsize=(9.5, 2.8), constrained_layout=True)
        bars = ax.barh(split_names, split_values, color=[COLORS[0], COLORS[1], COLORS[2]], height=.58)
        ax.invert_yaxis()
        ax.set_xlim(0, max(split_values) * 1.2)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1_000_000:.1f}M"))
        axes(ax, "Records in each derived split", xlabel="Records")
        ax.grid(axis="y", visible=False)
        ax.bar_label(bars, labels=[num(value) for value in split_values], padding=7, color="#334155", fontsize=9)
        show(fig)
        st.caption("Each bar shows the number of records assigned to a prepared split.")
    else:
        st.info("Saved split-size metadata is incomplete, so the split chart is unavailable.")

    audit = dd.load_dataset_audit()
    integrity = dd.dataset_integrity_items(d, audit)
    st.subheader("Dataset Integrity")
    tone = {
        "Verified": ("#0f766e", "#ccfbf1"),
        "Limitation": ("#a16207", "#fef3c7"),
        "Review": ("#b45309", "#ffedd5"),
        "Unavailable": ("#64748b", "#f1f5f9"),
    }
    cards_html = []
    for item in integrity:
        fg, bg = tone.get(item["status"], tone["Unavailable"])
        cards_html.append(
            '<article class="dataset-status-item">'
            f'<div class="dataset-status-heading">{html.escape(item["title"])}</div>'
            f'<span class="dataset-status-badge" style="color:{fg};background:{bg};">{html.escape(item["status"])}</span>'
            f'<div class="dataset-status-detail">{html.escape(item["detail"])}</div>'
            '</article>'
        )
    st.markdown(
        '<style>'
        '.dataset-status-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:.55rem;margin:.25rem 0 .7rem;}'
        '.dataset-status-item{background:#fff;border:1px solid #dbe4ee;border-radius:10px;padding:.62rem .75rem;min-height:88px;}'
        '.dataset-status-heading{font-weight:650;color:#10243a;margin-bottom:.25rem;}'
        '.dataset-status-badge{display:inline-block;border-radius:999px;padding:.1rem .45rem;font-size:.7rem;font-weight:650;margin-bottom:.2rem;}'
        '.dataset-status-detail{font-size:.8rem;line-height:1.28;color:#52657a;}'
        '</style><div class="dataset-status-grid">' + ''.join(cards_html) + '</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Training class distribution")
    scale = st.radio("Class chart scale", ["Logarithmic", "Linear"], horizontal=True, index=0)
    ordered = counts.sort_values("train", kind="stable")
    fig, ax = plt.subplots(figsize=(11.2, 7.1), constrained_layout=True)
    ax.barh(ordered.label.astype(str), ordered.train.astype(int), color=COLORS[0], height=.72)
    ax.set_xscale("log" if scale == "Logarithmic" else "linear")
    axes(ax, "Training records by class", xlabel="Records (log scale)" if scale == "Logarithmic" else "Records", ylabel="Class")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=8.2)
    show(fig)
    st.caption("The logarithmic option keeps low-support classes visible beside the largest classes.")

    coverage_rows = []
    for split in ("train", "validation", "test"):
        present = int((pd.to_numeric(counts[split], errors="coerce").fillna(0) > 0).sum())
        coverage_rows.append({"Split": split.title(), "Records": num(sz.get(split)),
                              "Classes represented": f"{present} of {n_classes}"})
    st.subheader("Class coverage by split")
    st.dataframe(pd.DataFrame(coverage_rows), hide_index=True, width="stretch")

    rare = counts[counts.test < 100].sort_values("test")
    if not rare.empty:
        st.warning("Some attack classes have fewer than 100 test records; their per-class estimates are less stable.")
    with st.expander("Advanced: Class Details"):
        st.dataframe(counts.sort_values("train", ascending=False), hide_index=True, width="stretch", height=460)



def comparison():
    heading("Model Comparison", "Final test results for the three saved models.")
    m = safe(get_result_rows)
    if m is None:
        return
    view = m.set_index("Model").loc[["Original centralized baseline", "Frozen improved centralized", "FedAvg"]].reset_index()
    compare_chart(view, ("accuracy", "macro_f1", "weighted_f1"), "Final test metric comparison")
    showdf = view.copy()
    cols = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
    for column in cols:
        showdf[column] = showdf[column].map(pct)
    showdf = showdf.rename(columns={"Model": "Approach", "accuracy": "Accuracy", "macro_precision": "Macro precision",
                                    "macro_recall": "Macro recall", "macro_f1": "Macro-F1", "weighted_f1": "Weighted-F1"})
    st.dataframe(showdf, hide_index=True, width="stretch", height=185, column_config={
        "Approach": st.column_config.TextColumn("Approach", width="large"),
        "Accuracy": st.column_config.TextColumn("Accuracy", width="medium"),
        "Macro precision": st.column_config.TextColumn("Macro precision", width="medium"),
        "Macro recall": st.column_config.TextColumn("Macro recall", width="medium"),
        "Macro-F1": st.column_config.TextColumn("Macro-F1", width="medium"),
        "Weighted-F1": st.column_config.TextColumn("Weighted-F1", width="medium"),
    })
    best = view.loc[view.macro_f1.idxmax()]
    st.markdown(f'<div class="finding"><b>Key finding:</b> {html.escape(str(best["Model"]))} has the highest saved test Macro-F1.</div>', unsafe_allow_html=True)
    st.caption("Accuracy and Weighted-F1 can obscure poor performance on rare classes.")

    with st.expander("Advanced: Validation Selection"):
        paths = {"Original centralized baseline": EXPERIMENTS / "centralized" / "best_validation_metrics.json",
                 "FedAvg": EXPERIMENTS / "federated" / "experiment" / "validation_metrics.json"}
        rows = []
        for name, path in paths.items():
            if path.exists():
                x = dd.read_json(path)
                rows.append({"Model": name, "Selected epoch/round": str(x.get("selected_epoch", x.get("selected_round", "—"))),
                             "Validation accuracy": pct(x.get("accuracy")), "Validation Macro-F1": pct(x.get("macro_f1")),
                             "Validation Weighted-F1": pct(x.get("weighted_f1"))})
        frozen = EXPERIMENTS / "model_improvement" / "robustness_evaluation" / "selection_frozen.json"
        if frozen.exists():
            x = dd.read_json(frozen).get("validation_metrics", {})
            rows.append({"Model": "Frozen improved centralized", "Selected epoch/round": "Frozen Stage 2 checkpoint",
                         "Validation accuracy": pct(x.get("accuracy")), "Validation Macro-F1": pct(x.get("macro_f1")),
                         "Validation Weighted-F1": pct(x.get("weighted_f1"))})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.write("Validation metrics selected checkpoints. Test metrics were evaluated only after model selection and were not used for tuning.")

    with st.expander("Advanced: Rare-Class Results"):
        model_names = ["Original centralized baseline", "Frozen improved centralized", "FedAvg"]
        per_model = {name: dd.load_model_evaluation(name)["per_class"] for name in model_names}
        rows = []
        for label in RARE:
            baseline_rows = per_model[model_names[0]].loc[per_model[model_names[0]].label.astype(str) == label]
            row = {"Class": label, "Test support": num(baseline_rows.iloc[0].support) if not baseline_rows.empty else "—"}
            for name, column in zip(model_names, ("Baseline F1", "Improved centralized F1", "FedAvg F1")):
                class_rows = per_model[name].loc[per_model[name].label.astype(str) == label]
                row[column] = "—" if class_rows.empty else pct(class_rows.iloc[0].f1)
            rows.append(row)
        rare_table = pd.DataFrame(rows, columns=["Class", "Test support", "Baseline F1", "Improved centralized F1", "FedAvg F1"])
        st.dataframe(rare_table, hide_index=True, width="stretch", height=270, column_config={
            "Class": st.column_config.TextColumn("Class", width="medium"),
            "Test support": st.column_config.TextColumn("Test support", width="small"),
            "Baseline F1": st.column_config.TextColumn("Baseline F1", width="small"),
            "Improved centralized F1": st.column_config.TextColumn("Improved centralized F1", width="medium"),
            "FedAvg F1": st.column_config.TextColumn("FedAvg F1", width="small"),
        })

def federated():
    heading("Federated Learning")
    f = safe(dd.load_federated_summary)
    if f is None:
        return
    rounds = f["rounds"].sort_values("round").copy()
    validation = f["validation"]
    selected_value = validation.get("selected_round")
    if selected_value is None or pd.isna(selected_value):
        selected_flags = rounds.get("best_checkpoint_selected", pd.Series(False, index=rounds.index)).astype(bool)
        selected_value = rounds.loc[selected_flags, "round"].max() if selected_flags.any() else rounds["round"].max()
    selected_round = int(selected_value)
    try:
        history = dd.prepare_federated_chart_data(rounds, selected_round)
    except (ArtifactError, KeyError, TypeError, ValueError) as exc:
        st.error("Saved FedAvg round history could not be prepared for display.")
        st.caption(str(exc))
        return
    test_metrics = f.get("test_metrics", {})
    cards([
        ("Simulated clients", num(len(f["clients"]))),
        ("Communication rounds", num(rounds["round"].max()) if not rounds.empty else "—"),
        ("Validation Macro-F1", pct(validation.get("macro_f1"))),
        ("FedAvg test Macro-F1", pct(test_metrics.get("macro_f1"))),
    ])

    client_columns = st.columns(3)
    for column, client in zip(client_columns, f["clients"]):
        column.markdown(
            f'<div class="client"><b>Client {int(client["client_id"])}</b>'
            f'<div class="client-value">{num(client.get("sample_count"))} rows</div>'
            f'<div class="small">{num(client.get("represented_classes"))} represented classes</div></div>',
            unsafe_allow_html=True,
        )

    fig, ax = plt.subplots(figsize=(10.8, 3.9), constrained_layout=True)
    ax.plot(history["round"], history["validation_macro_f1"], color=COLORS[1],
            marker="o", markersize=5, linewidth=2.2, label="Validation Macro-F1")
    ax.axvline(selected_round, color=COLORS[3], linestyle="--", linewidth=1.5,
               label=f"Selected checkpoint: round {selected_round}")
    selected_point = history.loc[history["selected_checkpoint"]]
    if not selected_point.empty:
        ax.scatter(selected_point["round"], selected_point["validation_macro_f1"],
                   s=72, color=COLORS[3], zorder=3)
    ax.set_ylim(0, 1.02)
    axes(ax, "Global validation Macro-F1 across communication rounds",
         xlabel="Communication round", ylabel="Macro-F1 (0–1)")
    ax.grid(axis="y", color="#e2e8f0")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, labelcolor="#334155", loc="best")
    show(fig)

    st.markdown("Each client trains locally, and the server combines model parameters using sample-count-weighted averaging.")
    st.caption("The three clients are simulated on one computer. This experiment does not provide a formal privacy guarantee.")

    with st.expander("Advanced: Training Configuration"):
        tabs = st.tabs(["Configuration", "Client Distributions", "Round History"])
        with tabs[0]:
            config_table = dd.format_metadata_table(f["config"])
            st.dataframe(config_table, hide_index=True, width="stretch", height=330)
        with tabs[1]:
            distribution_rows = []
            for client in f["clients"]:
                for label, count in client.get("class_counts", {}).items():
                    distribution_rows.append({"Client": f'Client {int(client["client_id"])}',
                                              "Class": label, "Training rows": int(count)})
            if distribution_rows:
                st.dataframe(pd.DataFrame(distribution_rows), hide_index=True,
                             width="stretch", height=420)
            else:
                st.info("Detailed client class counts are unavailable.")
        with tabs[2]:
            st.dataframe(rounds, hide_index=True, width="stretch", height=390)


def drift():
    heading("Drift Detection", "PSI tracks changes in each simulated client’s feature distribution across stream windows.")
    x = safe(dd.load_drift_summary)
    if x is None:
        return
    summaries = x["summary"].set_index("scenario", drop=False)
    no_drift = summaries.loc["no_drift"] if "no_drift" in summaries.index else {}
    synthetic = summaries.loc["synthetic_shift"] if "synthetic_shift" in summaries.index else {}
    threshold = x["config"].get("threshold")
    threshold_text = f"{float(threshold):.2f}" if threshold is not None else "—"
    shifted = synthetic.get("shift_windows") if isinstance(synthetic, pd.Series) else None
    detected = synthetic.get("detected_shift_windows") if isinstance(synthetic, pd.Series) else None
    shifted_text = f"{num(detected)} / {num(shifted)}" if shifted is not None and detected is not None else "—"
    cards([
        ("PSI threshold", threshold_text),
        ("No-drift alerts", num(no_drift.get("alerts", 0)) if isinstance(no_drift, pd.Series) else "—"),
        ("Synthetic-shift alerts", num(synthetic.get("alerts", 0)) if isinstance(synthetic, pd.Series) else "—"),
        ("Shift windows detected", shifted_text),
    ])
    st.warning("Synthetic distribution shifts are experimental scenarios. Detected drift indicates a feature-distribution change, not proof of a cyberattack.")
    scenario = st.radio(
        "PSI timeline scenario", ["no_drift", "synthetic_shift"], horizontal=True,
        format_func=lambda value: "No-drift control" if value == "no_drift" else "Synthetic shift",
    )
    try:
        chart_data = dd.prepare_drift_chart_data(x["scores"], scenario)
        event_table = dd.prepare_drift_event_table(x["scores"])
    except (ArtifactError, KeyError, TypeError, ValueError) as exc:
        st.error("Saved drift scores could not be prepared for display.")
        st.caption(str(exc))
        return

    if scenario == "synthetic_shift":
        st.caption("Synthetic scenario: Client 2’s Rate feature was shifted from stream window 4 onward.")
    else:
        st.caption("Control scenario: windows were sampled without an injected feature shift.")

    st.subheader("Client PSI by stream window")
    if chart_data.empty:
        st.info("No saved PSI windows are available for this scenario.")
    else:
        fig, ax = plt.subplots(figsize=(11.0, 4.5), constrained_layout=True)
        for index, client_id in enumerate(sorted(chart_data["client_id"].unique())):
            part = chart_data.loc[chart_data["client_id"] == client_id]
            ax.plot(part["window_index"], part["max_feature_psi"], marker="o", markersize=4,
                    linewidth=1.8, label=f"Client {int(client_id)}", color=COLORS[index % len(COLORS)])
        thresholds = chart_data["threshold"].dropna().unique()
        if len(thresholds):
            ax.axhline(float(thresholds[0]), color=COLORS[4], linestyle="--", linewidth=1.6,
                       label=f"PSI threshold {float(thresholds[0]):.2f}")
        if scenario == "synthetic_shift":
            ax.set_yscale("symlog", linthresh=0.01)
            psi_axis_label = "Maximum PSI (symlog; linear near zero)"
        else:
            psi_axis_label = "Maximum PSI (linear scale)"
        axes(ax, "Maximum feature PSI across stream windows", xlabel="Stream window", ylabel=psi_axis_label)
        ax.grid(axis="y", color="#e2e8f0")
        ax.grid(axis="x", visible=False)
        ax.legend(frameon=False, labelcolor="#334155", ncol=min(4, chart_data["client_id"].nunique() + 1))
        show(fig)

    st.subheader("Window alerts")
    st.dataframe(event_table, hide_index=True, width="stretch", height=290)

    with st.expander("Advanced: Drift Event Details", expanded=False):
        tabs = st.tabs(["Detector configuration", "Event records", "Feature PSI audit"])
        with tabs[0]:
            st.dataframe(dd.format_metadata_table(x["config"]), hide_index=True,
                         width="stretch", height=360)
        with tabs[1]:
            events = x.get("events", [])
            event_rows = []
            for index, event in enumerate(events, start=1):
                for row in dd.flatten_audit(event, prefix=f"Event {index}"):
                    event_rows.append(row)
            if event_rows:
                st.dataframe(pd.DataFrame(event_rows), hide_index=True,
                             width="stretch", height=360)
            else:
                st.info("No alert event records were saved.")
        with tabs[2]:
            feature_scores = x["feature_scores"].copy()
            if not feature_scores.empty:
                feature_scores = feature_scores.rename(columns={
                    "scenario": "Scenario", "client_id": "Client", "window_index": "Window",
                    "feature": "Feature", "psi": "PSI", "threshold": "Threshold",
                    "alert": "Alert", "synthetic_shift_applied": "Synthetic shift",
                    "current_bin_counts": "Current bin counts",
                })
                if "PSI" in feature_scores:
                    feature_scores["PSI"] = pd.to_numeric(feature_scores["PSI"], errors="coerce").round(4)
                if "Threshold" in feature_scores:
                    feature_scores["Threshold"] = pd.to_numeric(feature_scores["Threshold"], errors="coerce").round(4)
                st.dataframe(feature_scores, hide_index=True, width="stretch", height=420)
            else:
                st.info("Feature-level PSI audit rows are unavailable.")

def selective():
    heading("Selective Retraining","Saved policy actions and validation decisions from the drift replay.")
    x=safe(dd.load_retraining_summary)
    if x is None:return
    actions=x["actions"];syn=actions[actions.scenario=="synthetic_shift"]
    action=syn.iloc[0] if not syn.empty else None
    runs=int(pd.to_numeric(actions.actual_training_run,errors="coerce").fillna(False).astype(bool).sum())
    cards([("Actual training runs",num(runs)),("Affected client",f'Client {int(action.client_id)}' if action is not None else "None"),
           ("Unaffected clients skipped",str(action.unaffected_clients_skipped) if action is not None else "—"),("Candidate deployed","No")])
    pol=x["config"].get("policy",{})
    st.write(f'**Policy:** {pol.get("required_consecutive_alerts")} consecutive alerts; {pol.get("cooldown_windows")}-window cooldown; maximum {pol.get("max_retrainings_per_client_per_scenario")} action per client/scenario.')
    if actions.empty:st.info("No retraining actions recorded.")
    else:
        table=actions[["scenario","client_id","trigger_window","reason","unaffected_clients_skipped","actual_training_run","accepted_by_validation","deployment_performed"]].rename(columns={
            "scenario":"Scenario","client_id":"Affected client","trigger_window":"Trigger window","reason":"Reason","unaffected_clients_skipped":"Skipped clients",
            "actual_training_run":"Training ran","accepted_by_validation":"Accepted by validation","deployment_performed":"Deployed"})
        table["Scenario"]=table.Scenario.replace({"synthetic_shift":"Synthetic shift","no_drift":"No-drift"})
        st.dataframe(table,hide_index=True,width='stretch')
    st.markdown('<div class="finding"><b>Conclusion:</b> the Client 2 synthetic-shift candidate ran once and scored lower on validation Macro-F1. It was rejected; the original Phase 3 FedAvg checkpoint remained selected and the candidate was not deployed.</div>',unsafe_allow_html=True)
    cand=EXPERIMENTS/"selective_retraining"/"candidates"
    pairs=[("Clean validation","synthetic_shift_client_2_baseline_clean_validation_metrics.json","synthetic_shift_client_2_candidate_clean_validation_metrics.json"),
           ("Shift-target validation","synthetic_shift_client_2_baseline_target_validation_metrics.json","synthetic_shift_client_2_candidate_target_validation_metrics.json")]
    rows=[]
    for label,inc,candidate in pairs:
        a,b=cand/inc,cand/candidate
        if a.exists() and b.exists():
            ma,mb=dd.read_json(a),dd.read_json(b)
            rows.append({"Validation set":label,"Incumbent Macro-F1":pct(ma.get("macro_f1")),"Candidate Macro-F1":pct(mb.get("macro_f1")),"Decision":"Rejected"})
    if rows:st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
    with st.expander("Window-level alert and policy log"):
        st.dataframe(x["decisions"],hide_index=True,width='stretch',height=380)
        st.write("Alert, retraining trigger, completed local training, validation acceptance, and deployment are separate events.")

def confusion_figure(mat,labels,title,full=False):
    side=13 if full else max(9,.85*len(labels)+3)
    fig,ax=plt.subplots(figsize=(side,side*.76),constrained_layout=True)
    image=ax.imshow(np.log10(mat+1),cmap="Blues",aspect="auto")
    ax.set_xticks(range(len(labels)),labels,rotation=65,ha="right",fontsize=7 if full else 8)
    ax.set_yticks(range(len(labels)),labels,fontsize=7 if full else 8)
    axes(ax,title,xlabel="Predicted class",ylabel="Actual class");ax.grid(False)
    cb=fig.colorbar(image,ax=ax,fraction=.035,pad=.03);cb.set_label("log10(count + 1)",color="#334155");cb.ax.tick_params(colors="#475569",labelsize=8)
    if not full and len(labels)<=9:
        for i in range(len(labels)):
            for j in range(len(labels)):
                color="white" if mat[i,j]>mat.max()*.55 else "#10243a"
                ax.text(j,i,f"{int(mat[i,j]):,}",ha="center",va="center",fontsize=7,color=color)
    return fig

def evaluation():
    heading("Model Evaluation","Saved final test metrics only. The dashboard never opens prepared test feature or label arrays.")
    name=st.selectbox("Model",list(dd.MODEL_PATHS))
    b=safe(lambda:dd.load_model_evaluation(name))
    if b is None:return
    m=b["metrics"]
    cards([("Accuracy",pct(m.get("accuracy"))),("Macro precision",pct(m.get("macro_precision"))),
           ("Macro recall",pct(m.get("macro_recall"))),("Macro-F1",pct(m.get("macro_f1"))),("Weighted-F1",pct(m.get("weighted_f1")))])
    pc=b["per_class"].copy();pc["support"]=pd.to_numeric(pc.support,errors="coerce").fillna(0).astype(int)
    st.caption(f'Final test: {num(m.get("evaluated_rows"))} records; {int(m.get("supported_classes",0))} supported classes.')
    defaults=[x for x in ["BenignTraffic","DDoS-ICMP_Flood","DDoS-TCP_Flood","Uploading_Attack","XSS","Backdoor_Malware","SqlInjection","BrowserHijacking"] if x in set(pc.label)]
    selected=st.multiselect("Focused confusion matrix classes",LABELS,default=defaults)
    if selected:
        ids=[LABELS.index(x) for x in selected];sub=b["confusion"][np.ix_(ids,ids)]
        show(confusion_figure(sub,selected,"Focused confusion matrix (selected classes only)"))
        st.caption("Counts routed to unselected classes are omitted from this focused view; expand for the full matrix.")
    else:st.info("Select classes to view their confusion matrix.")
    st.subheader("Per-class F1")
    plot=pc[(pc.support>0)&pc.f1.notna()].sort_values("f1")
    fig,ax=plt.subplots(figsize=(11.4,max(8,.30*len(plot))),constrained_layout=True)
    ax.barh(plot.label,plot.f1,color=COLORS[1],height=.72);ax.set_xlim(0,1.04)
    axes(ax,"F1 by class with test support",xlabel="F1 score (0-1)",ylabel="Class");ax.grid(axis="y",visible=False);ax.tick_params(axis="y",labelsize=8)
    for i,val in enumerate(plot.f1):ax.text(min(float(val)+.014,1),i,f"{val:.2f}",va="center",fontsize=7,color="#334155")
    show(fig)
    rare=pc[pc.label.isin(RARE)][["label","precision","recall","f1","support","note"]].copy()
    for c in ("precision","recall","f1"):rare[c]=rare[c].map(lambda x:"—" if pd.isna(x) else pct(x))
    st.subheader("Rare and weak classes")
    st.dataframe(rare.rename(columns={"label":"Class","precision":"Precision","recall":"Recall","f1":"F1","support":"Test support","note":"Note"}),hide_index=True,width='stretch')
    if name=="Frozen improved centralized":st.warning("Uploading_Attack still has low recall and only a small number of test examples. A good aggregate score does not guarantee dependable rare-attack detection.")
    absent=pc.loc[pc.support==0,"label"].tolist()
    if absent:st.warning("Classes absent from this split have undefined metrics: "+", ".join(absent))
    with st.expander("All per-class metrics"):
        full=pc.copy()
        for c in ("precision","recall","f1"):full[c]=full[c].map(lambda x:"—" if pd.isna(x) else pct(x))
        st.dataframe(full.rename(columns={"label":"Class","precision":"Precision","recall":"Recall","f1":"F1","support":"Test support","note":"Note"}),hide_index=True,width='stretch',height=520)
    with st.expander("Complete confusion matrix"):
        show(confusion_figure(b["confusion"],LABELS,"34-class confusion matrix, log color scale",True))
        st.dataframe(pd.DataFrame(b["confusion"],index=LABELS,columns=LABELS),width='stretch',height=540)

ROUTES={"Overview":overview,"Traffic Classification":traffic,"Dataset":dataset,"Model Comparison":comparison,
        "Federated Learning":federated,"Drift Detection":drift,"Selective Retraining":selective,
        "Model Evaluation":evaluation}
try:
    ROUTES[page]()
except Exception as exc:
    st.error("This page could not be rendered from its saved artifacts.")
    st.caption(f"{type(exc).__name__}: {exc}")

