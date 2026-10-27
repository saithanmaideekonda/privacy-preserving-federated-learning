from __future__ import annotations
import ast
import hashlib
import unittest
from pathlib import Path
from unittest import mock
import numpy as np
import pandas as pd
import dashboard_data as dd
import traffic_inference as ti

class DashboardArtifactTests(unittest.TestCase):
    def test_saved_metric_confusion_consistency(self):
        table=dd.load_comparison_metrics()
        self.assertEqual(set(table.Model),set(dd.MODEL_PATHS))
        for name in dd.MODEL_PATHS:
            b=dd.load_model_evaluation(name);m=b["metrics"];cm=b["confusion"];pc=b["per_class"]
            self.assertEqual(cm.shape,(34,34));self.assertEqual(int(cm.sum()),int(m["evaluated_rows"]))
            self.assertEqual(pc.label.tolist(),dd.LABELS)
            self.assertEqual(pc.support.astype(int).tolist(),cm.sum(axis=1).tolist())
            self.assertAlmostEqual(np.trace(cm)/cm.sum(),m["accuracy"],places=6)
            f=pd.to_numeric(pc.f1,errors="coerce");w=pc.support.astype(float)
            self.assertAlmostEqual(float(f.mean()),m["macro_f1"],places=5)
            self.assertAlmostEqual(float(np.average(f,weights=w)),m["weighted_f1"],places=5)
    def test_missing_artifacts_and_unknown_model(self):
        with self.assertRaises(dd.ArtifactError):
            dd.read_json(Path(__file__).parent / "missing-dashboard-fixture.json")
        with self.assertRaises(dd.ArtifactError):dd.load_model_evaluation("unknown")
    def test_dataset_shape_and_counts(self):
        d=dd.load_dataset_summary()
        self.assertEqual((len(d["input_features"]),len(d["retained_features"]),len(d["classes"])),(46,44,34))
        self.assertEqual(d["class_counts"][["train","validation","test"]].sum().tolist(),[5492542,1176701,1176430])
    def test_dataset_integrity_summary_matches_saved_audits(self):
        d=dd.load_dataset_summary(); audit=dd.load_dataset_audit()
        items={item["title"]:item for item in dd.dataset_integrity_items(d,audit)}
        self.assertEqual(len(items),5)
        self.assertTrue(all(len(item["detail"].split()) <= 14 for item in items.values()))
        self.assertEqual(items["Class coverage"]["status"],"Verified")
        self.assertIn("34 of 34",items["Class coverage"]["detail"])
        self.assertEqual(items["Derived split overlap"]["status"],"Verified")
        self.assertEqual(items["Preprocessing"]["status"],"Verified")
        self.assertEqual(items["Duplicate handling"]["status"],"Verified")
        self.assertEqual(items["Independence limitation"]["status"],"Limitation")
        # The raw source audit has overlaps; the derived-split audit is separately zero.
        source=audit["duplicate_overlap_audit"]["source_fingerprint_audit"]["cross_split_unique_fingerprint_overlap"]
        self.assertGreater(source["train__validation"]["feature_vector"],0)
        self.assertEqual(audit["duplicate_overlap_audit"]["derived_cross_split_feature_vector_overlap_groups"],0)

    def test_dataset_audit_tables_preserve_nested_metadata(self):
        rows=dd.flatten_audit({"assignment":{"seed":20261026},"features":["a","b"]})
        self.assertEqual(rows,[
            {"Field":"assignment.seed","Saved value":"20261026"},
            {"Field":"features[0]","Saved value":"a"},
            {"Field":"features[1]","Saved value":"b"},
        ])
        with mock.patch.object(dd,"read_json",side_effect=dd.ArtifactError("missing fixture")):
            audit=dd.load_dataset_audit()
        self.assertEqual(len(audit["errors"]),3)
        self.assertTrue(all(audit[name] is None for name in ("preprocessing","split_methodology","duplicate_overlap_audit")))
    def test_dataset_page_keeps_audit_details_out_of_default_view(self):
        root=Path(__file__).resolve().parents[1]
        source=(root/"app.py").read_text(encoding="utf-8")
        page=source.split("def dataset():",1)[1].split("def comparison():",1)[0]
        self.assertIn('heading("Dataset Overview"',page)
        self.assertNotIn("Advanced: Data Integrity Audit",page)
        self.assertNotIn("Split and overlap",page)
        self.assertNotIn("Complete metadata",page)
        self.assertNotIn("st.json(",page)
        self.assertIn('st.expander("Advanced: Class Details")',page)
        self.assertNotIn('expanded=True',page)
        self.assertNotIn("st.tabs(",page)
        self.assertIn("Dataset split sizes",page)
        self.assertIn("Records in each derived split",page)
        self.assertIn("Dataset Integrity",page)
        self.assertIn("Training class distribution",page)
        self.assertIn('st.radio("Class chart scale", ["Logarithmic", "Linear"]',page)
        self.assertIn("Class coverage by split",page)
        self.assertIn("st.warning(",page)
        self.assertIn('("Training records"',page)
        self.assertIn('("Validation records"',page)
        self.assertIn('("Test records"',page)
        self.assertIn('("Number of classes"',page)
        self.assertIn("figsize=(11.2, 7.1)",page)
        self.assertNotIn("source_fingerprint_audit",page)
        self.assertNotIn("removed_features",page)
    def test_overview_removes_about_expander_and_keeps_summary(self):
        root=Path(__file__).resolve().parents[1]
        source=(root/"app.py").read_text(encoding="utf-8")
        page=source.split("def overview():",1)[1].split("def traffic():",1)[0]
        self.assertNotIn("About this experiment",page)
        self.assertIn('st.subheader("Workflow")',page)
        self.assertIn("simulated clients",page)
        self.assertIn("FedAvg rounds",page)
        self.assertIn("no formal privacy guarantee",page)
        self.assertIn('("Training rows"',page)
        self.assertIn('("Test rows"',page)
        self.assertIn('("Retained features"',page)
        self.assertIn('("Classes"',page)
        self.assertIn("compare_chart(m,",page)

    def test_navigation_excludes_reports_page_and_branch(self):
        root=Path(__file__).resolve().parents[1]
        source=(root/"app.py").read_text(encoding="utf-8")
        tree=ast.parse(source)
        assigned={target.id:node.value for node in tree.body if isinstance(node,ast.Assign)
                  for target in node.targets if isinstance(target,ast.Name)}
        pages=ast.literal_eval(assigned["PAGES"])
        route_node=assigned["ROUTES"]
        routes=[ast.literal_eval(key) for key in route_node.keys]
        expected=["Overview","Traffic Classification","Dataset","Model Comparison",
                  "Federated Learning","Drift Detection","Selective Retraining","Model Evaluation"]
        self.assertEqual(pages,expected)
        self.assertEqual(routes,expected)
        self.assertNotIn("Reports & Reproducibility",source)
        self.assertNotIn("def reports(",source)
        self.assertNotIn("st.json(",source)
        self.assertFalse(hasattr(dd,"existing_reports"))
        federated=source.split("def federated():",1)[1].split("def drift():",1)[0]
        drift=source.split("def drift():",1)[1].split("def selective():",1)[0]
        self.assertIn('st.expander("Advanced: Training Configuration")',federated)
        self.assertIn("dd.prepare_federated_chart_data",federated)
        self.assertNotIn("st.json(",federated)
        self.assertIn('st.expander("Advanced: Drift Event Details", expanded=False)',drift)
        self.assertIn("dd.prepare_drift_chart_data",drift)
        self.assertIn("dd.prepare_drift_event_table",drift)
        self.assertNotIn("st.json(",drift)

    def test_model_comparison_main_view_and_advanced_sections(self):
        root=Path(__file__).resolve().parents[1]
        source=(root/"app.py").read_text(encoding="utf-8")
        page=source.split("def comparison():",1)[1].split("def federated():",1)[0]
        self.assertIn('compare_chart(view, ("accuracy", "macro_f1", "weighted_f1"), "Final test metric comparison")',page)
        for metric in ("Accuracy","Macro precision","Macro recall","Macro-F1","Weighted-F1"):
            self.assertIn(f'"{metric}"',page)
        self.assertIn('st.dataframe(showdf, hide_index=True',page)
        self.assertIn("Key finding:",page)
        self.assertIn("has the highest saved test Macro-F1",page)
        self.assertIn("Accuracy and Weighted-F1 can obscure poor performance on rare classes.",page)
        self.assertIn('st.expander("Advanced: Validation Selection")',page)
        self.assertIn('st.expander("Advanced: Rare-Class Results")',page)
        self.assertNotIn("expanded=True",page)
        self.assertNotIn('st.expander("Validation-based checkpoint selection")',page)
        self.assertNotIn('st.expander("Rare-class comparison")',page)
        for column in ("Test support","Baseline F1","Improved centralized F1","FedAvg F1"):
            self.assertIn(f'"{column}"',page)
        self.assertIn("Validation metrics selected checkpoints",page)
        self.assertIn("Test metrics were evaluated only after model selection",page)
        metrics=dd.load_comparison_metrics()
        self.assertEqual(metrics.Model.tolist(),list(dd.MODEL_PATHS))
        self.assertEqual(metrics.Model.tolist(),["Original centralized baseline","Frozen improved centralized","FedAvg"])
        self.assertEqual(len(metrics),3)

    def test_federated_chart_and_configuration_formatting(self):
        f=dd.load_federated_summary()
        selected=int(f["validation"]["selected_round"])
        history=dd.prepare_federated_chart_data(f["rounds"].sample(frac=1,random_state=7),selected)
        self.assertEqual(history["round"].tolist(),sorted(f["rounds"]["round"].tolist()))
        self.assertEqual(history.loc[history.selected_checkpoint,"round"].tolist(),[selected])
        expected=f["rounds"].set_index("round").loc[selected,"validation_macro_f1"]
        actual=history.loc[history["round"]==selected,"validation_macro_f1"].iloc[0]
        self.assertAlmostEqual(float(actual),float(expected),places=12)
        config=dd.format_metadata_table(f["config"])
        self.assertEqual(config.columns.tolist(),["Setting","Value"])
        self.assertIn("selection_metric",config.Setting.tolist())
        self.assertEqual(float(f["test_metrics"]["macro_f1"]),float(dd.read_json(dd.EXPERIMENTS/"federated"/"experiment"/"test_metrics.json")["macro_f1"]))

    def test_federated_page_summary_client_cards_chart_and_advanced_details(self):
        root=Path(__file__).resolve().parents[1]
        source=(root/"app.py").read_text(encoding="utf-8")
        page=source.split("def federated():",1)[1].split("def drift():",1)[0]
        for label in ("Simulated clients","Communication rounds","Validation Macro-F1","FedAvg test Macro-F1"):
            self.assertIn(f'("{label}"',page)
        self.assertIn('num(client.get("sample_count"))',page)
        self.assertIn('num(client.get("represented_classes"))',page)
        self.assertIn('label=f"Selected checkpoint: round {selected_round}"',page)
        self.assertIn('st.markdown("Each client trains locally, and the server combines model parameters using sample-count-weighted averaging.")',page)
        self.assertIn('st.caption("The three clients are simulated on one computer. This experiment does not provide a formal privacy guarantee.")',page)
        self.assertIn('st.expander("Advanced: Training Configuration")',page)
        self.assertIn('st.tabs(["Configuration", "Client Distributions", "Round History"])',page)
        self.assertNotIn("expanded=True",page)
        self.assertNotIn("How FedAvg works",page)

        f=dd.load_federated_summary()
        selected=int(f["validation"]["selected_round"])
        self.assertEqual(selected,12)
        self.assertEqual(len(f["clients"]),3)
        training_rows=int(dd.load_dataset_summary()["class_counts"]["train"].sum())
        self.assertEqual(sum(int(client["sample_count"]) for client in f["clients"]),training_rows)
        history=dd.prepare_federated_chart_data(f["rounds"],selected)
        self.assertEqual(history.loc[history.selected_checkpoint,"round"].tolist(),[12])

    def test_drift_chart_and_event_table_formatting(self):
        x=dd.load_drift_summary()
        original=x["scores"].copy(deep=True)
        clean=dd.prepare_drift_chart_data(x["scores"],"no_drift")
        shifted=dd.prepare_drift_chart_data(x["scores"],"synthetic_shift")
        events=dd.prepare_drift_event_table(x["scores"])
        self.assertEqual(len(clean),24)
        self.assertEqual(len(shifted),24)
        self.assertEqual(clean.groupby("client_id").size().tolist(),[8,8,8])
        self.assertAlmostEqual(float(shifted.threshold.iloc[0]),float(x["config"]["threshold"]))
        self.assertEqual(events.columns.tolist(),["Scenario","Client","Window","Maximum PSI","Top feature","Alert status"])
        self.assertEqual(events.Scenario.value_counts().to_dict(),{"No-drift control":24,"Synthetic shift":24})
        self.assertEqual((events["Alert status"]=="Alert").sum(),int(x["summary"].alerts.sum()))
        self.assertTrue(events["Maximum PSI"].str.match(r"^-?\d+\.\d{3}$").all())
        pd.testing.assert_frame_equal(x["scores"],original)
        config=dd.format_metadata_table(x["config"])
        self.assertIn("threshold",config.Setting.tolist())

    def test_drift_page_has_scenario_specific_zero_safe_chart_and_preserves_evidence(self):
        root=Path(__file__).resolve().parents[1]
        source=(root/"app.py").read_text(encoding="utf-8")
        page=source.split("def drift():",1)[1].split("def selective():",1)[0]
        for label in ("PSI threshold","No-drift alerts","Synthetic-shift alerts","Shift windows detected"):
            self.assertIn(f'("{label}"',page)
        self.assertIn('["no_drift", "synthetic_shift"]',page)
        self.assertIn('ax.set_yscale("symlog", linthresh=0.01)',page)
        self.assertIn('"Maximum PSI (symlog; linear near zero)"',page)
        self.assertIn('"Maximum PSI (linear scale)"',page)
        self.assertIn('ax.axhline(float(thresholds[0])',page)
        self.assertIn('st.dataframe(event_table, hide_index=True',page)
        self.assertIn('st.warning("Synthetic distribution shifts are experimental scenarios. Detected drift indicates a feature-distribution change, not proof of a cyberattack.")',page)
        self.assertIn('st.expander("Advanced: Drift Event Details", expanded=False)',page)
        self.assertIn('st.tabs(["Detector configuration", "Event records", "Feature PSI audit"])',page)

        saved=dd.load_drift_summary()
        self.assertAlmostEqual(float(saved["config"]["threshold"]),0.25)
        original=saved["scores"].copy(deep=True)
        for scenario in ("no_drift","synthetic_shift"):
            chart=dd.prepare_drift_chart_data(saved["scores"],scenario)
            self.assertEqual(set(chart.client_id.astype(int)),{1,2,3})
            self.assertEqual(len(chart),24)
            self.assertEqual(chart.threshold.unique().tolist(),[0.25])
        self.assertGreater(float(saved["scores"].loc[saved["scores"].scenario=="synthetic_shift","max_feature_psi"].max()),12.0)
        pd.testing.assert_frame_equal(saved["scores"],original)

class TrafficInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.p=ti.load_predictor()
    def demo(self,n=1):return pd.DataFrame([r.to_dict() for r in self.p.demo_records(count=n)])
    def digest(self):
        h=hashlib.sha256()
        for a in list(self.p.model.coefs_)+list(self.p.model.intercepts_):h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()
    def test_hash_and_expected_contract(self):
        v=ti.verify_frozen_artifacts()
        self.assertEqual(v["hashes"]["checkpoint"],"47984db438d22952a8624c675cfc31cae67885f1e0e016f2186679fffd52b88a")
        self.assertEqual((len(self.p.input_features),len(self.p.retained_features)),(46,44))
    def test_demo_transform_matches_train_array(self):
        seed=20261026;rng=np.random.default_rng(seed)
        index=int(np.sort(rng.choice(5492542,size=1,replace=False))[0])
        train=np.load(ti.TRAIN_FEATURES,mmap_mode="r",allow_pickle=False)
        x=self.p.transform(self.demo())
        self.assertEqual(x.shape,(1,44));self.assertEqual(x.dtype,np.float32)
        np.testing.assert_allclose(x[0],train[index],rtol=2e-5,atol=2e-4)
    def test_demo_opens_only_training_features(self):
        original=np.load;opened=[]
        def track(path,*a,**kw):opened.append(Path(path).name);return original(path,*a,**kw)
        with mock.patch("traffic_inference.np.load",side_effect=track):self.p.demo_records(seed=3,count=2)
        self.assertEqual(opened,["X_train.npy"])
    def test_demo_examples_still_classify(self):
        examples=self.p.demo_records()
        self.assertTrue(all(len(example)==46 for example in examples))
        input_frame=pd.DataFrame([example.to_dict() for example in examples])
        self.assertEqual(self.p.transform(input_frame).shape,(len(examples),44))
        result=self.p.predict(input_frame)
        self.assertEqual(len(result["labels"]),len(examples))
        self.assertTrue(all(status==ti.traffic_status(label) for label,status in zip(result["labels"],result["status"])))
        self.assertTrue(all(len(top)==3 for top in result["top3"]))

    def test_traffic_page_exposes_demo_only_and_hides_feature_values_by_default(self):
        root=Path(__file__).resolve().parents[1]
        source=(root/"app.py").read_text(encoding="utf-8")
        page=source.split("def traffic():",1)[1].split("def dataset():",1)[0]
        self.assertIn("Illustrative example",page)
        self.assertIn('st.button("Classify example"',page)
        self.assertNotIn("Advanced: Example Feature Values",page)
        self.assertNotIn("frame.T",page)
        self.assertNotIn("CSV upload",page)
        self.assertNotIn("Manual input",page)
        self.assertNotIn("file_uploader",page)
        self.assertNotIn("data_editor",page)
        self.assertNotIn("parse_csv_bytes",page)
        self.assertLess(page.index("Illustrative example"),page.index('st.button("Classify example"'))
        self.assertIn("not a guarantee of malicious intent",page)
        self.assertIn("No live packet capture",page)
    def test_prediction_labels_probabilities_and_binary_summary(self):
        out=self.p.predict(self.demo(2))
        self.assertEqual(out["probabilities"].shape,(2,34));self.assertTrue(np.allclose(out["probabilities"].sum(axis=1),1,atol=1e-4))
        self.assertTrue(set(out["labels"]).issubset(set(self.p.id_to_class.values())))
        self.assertEqual(ti.traffic_status("BenignTraffic"),"NORMAL TRAFFIC")
        for label in self.p.id_to_class.values():
            if label!="BenignTraffic":self.assertEqual(ti.traffic_status(label),"MALICIOUS / ATTACK-LIKE TRAFFIC")
    def test_schema_order_missing_extra_label_and_invalid_values(self):
        df=self.demo();self.assertEqual(self.p.transform(df[list(reversed(self.p.input_features))]).shape,(1,44))
        for bad in [df.drop(columns=[self.p.input_features[0]]),df.assign(extra=1),df.assign(label="x")]:
            with self.assertRaises(ti.InferenceError):self.p.transform(bad)
        for value in [np.nan,pd.NA,np.inf,"not numeric"]:
            bad=df.copy();first=self.p.input_features[0];bad[first]=bad[first].astype(object);bad.loc[0,first]=value
            with self.assertRaises(ti.InferenceError):self.p.transform(bad)
    def test_csv_batch_and_rejections(self):
        df=self.demo(2);raw=df.to_csv(index=False).encode()
        parsed=ti.parse_csv_bytes(raw,self.p)
        self.assertEqual(parsed.columns.tolist(),self.p.input_features)
        self.assertEqual(len(self.p.predict(parsed)["labels"]),2)
        for frame in [df.drop(columns=[self.p.input_features[-1]]),df.assign(label="x"),df.assign(extra=1)]:
            with self.assertRaises(ti.InferenceError):ti.parse_csv_bytes(frame.to_csv(index=False).encode(),self.p)
        with self.assertRaises(ti.InferenceError):ti.parse_csv_bytes(b"x"*20,self.p,max_bytes=10)
    def test_synthetic_drift_is_distinguished_from_traffic_prediction(self):
        root=Path(__file__).resolve().parents[1]
        app=(root/"app.py").read_text(encoding="utf-8")
        self.assertIn("Synthetic distribution shifts are experimental scenarios.",app)
        self.assertIn("Detected drift indicates a feature-distribution change, not proof of a cyberattack.",app)
        self.assertIn("not a guarantee of malicious intent.",app)
        self.assertIn("No live packet capture.",app)
        self.assertIn("BenignTraffic maps to NORMAL.",app)

    def test_prediction_does_not_modify_checkpoint_and_dashboard_never_loads_test_arrays(self):
        before=self.digest();self.p.predict(self.demo());self.assertEqual(before,self.digest())
        root=Path(__file__).resolve().parents[1]
        source="\n".join((root/n).read_text(encoding="utf-8") for n in ("app.py","dashboard_data.py","traffic_inference.py"))
        self.assertNotIn("X_test.npy",source);self.assertNotIn("y_test.npy",source)

if __name__=="__main__":unittest.main()

