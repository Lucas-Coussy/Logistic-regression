#import dataset needed packages
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import numpy as np

#import our class
from my_classes.logisticregression_class import Logisticregression

data = load_digits()

X = data['data']
y = data['target']

index = np.isin(y, [0, 1])

X = X[index]
y = y[index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

model = Logisticregression(learning_rate=0.001, epochs=50, threshold=0.5)
model = model.fit(X_train, y_train)

model.Logistic_graph(X_train,xylim=100,name="logistic_graph_1")

model.Logistic_heatmap(name="heatmap_1")

model.Logistic_heatmap_with_costs(name="heatmap_with_cost_1")