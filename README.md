# InternPe Machine Learning Projects

Four machine learning projects from my InternPe internship (February to March 2024):
two classification problems, one regression problem, and a small neural network.
Each notebook runs from the raw data through to an evaluated model, and the IPL
project also ships a small Streamlit app.

The thing I paid most attention to here is honest evaluation. Every score below
comes from held-out data or cross-validation, never the training set, and the IPL
notebook shows how an easy mistake in the train/test split can make a model look
far better than it is.

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



### 3. IPL win predictor
Estimating the chasing team's win probability from the live match state (runs left,
balls left, wickets in hand, run rates). The main point of this project is the
train/test split. Every ball of a match shares the same result, so a plain random
split leaks the outcome and the model scores far higher than it should. I split by
match instead, so no game appears in both train and test. The notebook trains both
ways on purpose to show the gap. The logistic regression is saved and served by the
Streamlit app in `ipl-app/`.



### 4. Breast cancer classification
Classifying tumours as malignant or benign with a small Keras neural network, using
logistic regression as a baseline. Malignant is the positive class because a missed
malignant case is the costly mistake. The honest takeaway is that on a dataset this
small and this clean, the neural network does not beat the linear baseline by much,
which is worth knowing before reaching for a heavier model.

## Data sources

Pima Indians Diabetes dataset
Used car listings scraped from Quikr
IPL ball-by-ball data, seasons from 2008 to 2019
Breast Cancer Wisconsin dataset 
