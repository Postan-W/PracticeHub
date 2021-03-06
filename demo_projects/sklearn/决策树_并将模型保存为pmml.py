import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml import sklearn2pmml
import joblib
import os
"""
sklearn: 0.20.4
sklearn2pmml: 0.48.0
"""
#引入Java8
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Java/jdk1.8.0_151/bin'

X = [[1, 2, 3, 1], [2, 4, 1, 5], [7, 8, 3, 6], [4, 8, 4, 7], [2, 5, 6, 9]]
y = [0, 1, 0, 2, 1]
pipeline = PMMLPipeline([("classifier", tree.DecisionTreeClassifier(random_state=9))])
pipeline.fit(X, y)
joblib.dump(pipeline,"./models/decision_tree1.pkl")
sklearn2pmml(pipeline, "../../pmml_files/decision_tree1.pmml", with_repr=True)