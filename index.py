from sklearn import datasets
import h2o
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators.xgboost import H2OXGBoostEstimator
import pandas as pd
def load_h2o():
    h2o.init()

def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    return df

def load_data():
    df=h2o.H2OFrame(sklearn_to_df(datasets.load_breast_cancer()))
    df["target"]=df["target"].asfactor()
    return df

def split_data(data):
    return data.split_frame(ratios=[0.75])

def train_deep_learning(train, test):
    model = H2ODeepLearningEstimator(hidden=[50, 50, 50], epochs=100)
    model.train(y="target", training_frame=train, validation_frame=test)
    return model

def train_xgboost(train, test):
    model = H2OXGBoostEstimator()
    model.train(y="target", training_frame=train, validation_frame=test)
    return model

def save_model(model):
    return h2o.save_model(model=model, path="./", force=True)

load_h2o()
data=load_data()
train, test=split_data(data)
model_deep_learning=train_deep_learning(train, test)
model_xgboost=train_xgboost(train, test)
save_model(model_deep_learning)
save_model(model_xgboost)