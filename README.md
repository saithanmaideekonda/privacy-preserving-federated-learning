# Federated IoT IDS Dashboard

A read-only Streamlit dashboard for the full-dataset CICIoT2023 experiments, with on-demand classification of user-supplied engineered traffic-flow features.

## Run locally

Open PowerShell in this workspace:

    python -m pip install -r requirements.txt
    python -m streamlit run app.py

The dashboard does not train models when it starts. Pages read saved metrics and experiment artifacts; Traffic Classification loads the frozen candidate and saved preprocessing. Illustrative rows are reconstructed from the prepared training feature array only.

## Traffic Classification

Traffic Classification provides one **Illustrative Demo** mode. Examples are reconstructed from prepared training features only; their labels are not loaded. Select an example and choose **Classify example**. The frozen Stage 3 model uses the saved training-fitted preprocessing and label mapping; feature values are kept out of the demo view.

Predictions are on-demand model classifications, not packet capture, live network monitoring, or guarantees of malicious intent. BenignTraffic maps to NORMAL; all other labels map to ATTACK-LIKE. Displayed predict_proba values are uncalibrated model scores.

## Dashboard sections

Overview, Traffic Classification, Dataset, Model Comparison, Federated Learning, Drift Detection, Selective Retraining, and Model Evaluation.

FedAvg clients are simulated on one computer. The experiment does not implement secure aggregation, differential privacy, or formal privacy guarantees. PSI detects distribution changes, not confirmed attacks; the injected drift scenario is synthetic.

## Reproduction and testing

Run the existing project suite plus dashboard checks:

    python -m unittest discover -s tests -v
    python -m py_compile app.py dashboard_data.py traffic_inference.py

Saved final comparisons read metrics, confusion matrices, and per-class tables only. The dashboard never loads prepared test feature or label arrays. Page refresh does not rerun training.

## Data attribution

The project uses the locally prepared CICIoT2023 dataset. Kaggle source: https://www.kaggle.com/datasets/himadri07/ciciot2023. See the inspection and phase reports for local source inspection, split methodology, and caveats.

