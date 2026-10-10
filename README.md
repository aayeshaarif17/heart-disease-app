# Heart Disease Prediction System (AIC354 – Assignment 2, Task 2)

Deployment URL: **(paste your own https://<your-app>.streamlit.app link here after deploying)**

## Files
| File | Purpose |
|---|---|
| `heart_disease_prediction.ipynb` | Full notebook (Titanic workflow adapted to heart disease), already executed |
| `processed.cleveland.data`, `heart-disease.names` | UCI dataset file used + its documentation (https://archive.ics.uci.edu/dataset/45/heart+disease) |
| `heart_svc_trained_model.pkl` | Trained SVC model (output of notebook Step 6.4) |
| `heart_label_encoders.pkl` | The 5 fitted LabelEncoders (4 inputs + target) |
| `app.py`, `requirements.txt`, `runtime.txt` | Streamlit web app + deployment config |
| `*.csv` | Intermediate files written by the notebook (as in the Titanic notebook) |

## Run locally
```
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install jupyter prettytable seaborn matplotlib     # only needed for the notebook
jupyter notebook heart_disease_prediction.ipynb        # Restart & Run All
streamlit run app.py
```
Inputs (exactly four): Sex, ChestPain, ExerciseAngina, Thal. Output: heart disease likely / not likely.

## Deploy (Streamlit Community Cloud)
1. New **public** GitHub repo; upload `app.py`, `requirements.txt`, `runtime.txt`, `heart_svc_trained_model.pkl`, `heart_label_encoders.pkl` to the repo root.
2. https://share.streamlit.io → sign in with GitHub → Create app → pick repo, branch `main`, file `app.py` → Deploy.
3. If the build fails on packages, Settings → General → Python version → 3.11.
4. Open the URL, test a prediction, then paste the URL into the first cell of the notebook (and this README).

The pickles were created with scikit-learn 1.7.2 / pandas 2.3.3 / numpy 2.2.6 (pinned in `requirements.txt`); keep these versions so the model loads.
