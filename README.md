# InternPe Machine Learning Projects

Four machine learning projects from my InternPe internship (February to March 2024):
two classification problems, one regression problem, and a small neural network.
Each notebook runs from the raw data through to an evaluated model, and the IPL
project also ships a small Streamlit app.

The thing I paid most attention to here is honest evaluation. Every score below
comes from held-out data or cross-validation, never the training set, and the IPL
notebook shows how an easy mistake in the train/test split can make a model look
far better than it is.

## Results

| Project | Task | Data | Headline result |
| --- | --- | --- | --- |
| Diabetes prediction | Classification | Pima, 768 patients | ROC-AUC 0.81; recall raised 0.50 to 0.69 by lowering the threshold |
| Used car price | Regression | Quikr scrape, ~800 cars | R2 0.64 (5-fold CV), MAE about Rs 1.5 lakh |
| IPL win predictor | Classification | Ball-by-ball, 2008-2019 | Accuracy 0.73 on unseen matches after fixing data leakage |
| Breast cancer | Neural network | Wisconsin, 569 samples | ROC-AUC 0.995, recall 0.95 on malignant; level with a logistic-regression baseline (AUC 0.996) |

## Projects

### 1. Diabetes prediction
Predicting whether a patient is diabetic from 8 clinical measurements. The dataset
records several impossible zeros (glucose, blood pressure, BMI cannot be 0), so I
treat those as missing and impute them inside the pipeline, fit on the training
split only. Because this is a screening problem I care about recall, not just
accuracy, so I compare models on ROC-AUC and show how lowering the decision
threshold catches more diabetic patients at the cost of some precision.



### 2. Used car price prediction
Predicting resale price from a messy Quikr scrape, so most of the work is cleaning:
non-numeric years, "Ask For Price" entries, kilometres mixed with text. A common
shortcut on this dataset is to loop over hundreds of random splits and keep the one
with the best score. That just picks a lucky test set, so instead I report 5-fold
cross-validated R2 and mean absolute error in rupees, which is easier to read than
R2 on its own.

`notebooks/02_car_price_prediction.ipynb`, saved model in `models/`

### 3. IPL win predictor
Estimating the chasing team's win probability from the live match state (runs left,
balls left, wickets in hand, run rates). The main point of this project is the
train/test split. Every ball of a match shares the same result, so a plain random
split leaks the outcome and the model scores far higher than it should. I split by
match instead, so no game appears in both train and test. The notebook trains both
ways on purpose to show the gap. The logistic regression is saved and served by the
Streamlit app in `ipl-app/`.

`notebooks/03_ipl_win_prediction.ipynb`

### 4. Breast cancer classification
Classifying tumours as malignant or benign with a small Keras neural network, using
logistic regression as a baseline. Malignant is the positive class because a missed
malignant case is the costly mistake. The honest takeaway is that on a dataset this
small and this clean, the neural network does not beat the linear baseline by much,
which is worth knowing before reaching for a heavier model.

`notebooks/04_breast_cancer_classification.ipynb`

## Running it

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# open the notebooks
jupyter notebook

# run the IPL app
cd ipl-app
streamlit run app.py
```

The IPL app loads the trained model from `pipe.pkl`, so it runs without retraining
and is ready to deploy on Streamlit Community Cloud (`ipl-app/requirements.txt` lists
just what the app needs).

## Repository layout

```
data/                 datasets (IPL ball-by-ball data, Pima, Quikr cars)
notebooks/            one notebook per project
models/               saved car price model
ipl-app/              Streamlit app + its trained model
requirements.txt      pinned versions for the notebooks
```

## Data sources

- Pima Indians Diabetes dataset
- Used car listings scraped from Quikr
- IPL ball-by-ball data, seasons 2008 to 2019
- Breast Cancer Wisconsin dataset (loaded from scikit-learn)
