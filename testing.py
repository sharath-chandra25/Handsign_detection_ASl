import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

data_file = pickle.load(open('./data.pkl', 'rb'))
features = np.asarray(data_file['data'])
targets = np.asarray(data_file['labels'])

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

x_train, x_test, y_train, y_test = train_test_split(scaled_features, targets, test_size=0.2, shuffle=True, stratify=targets)

rf_model = RandomForestClassifier()
knn_model = KNeighborsClassifier()
lr_model = LogisticRegression(max_iter=5000)

rf_model.fit(x_train, y_train)
knn_model.fit(x_train, y_train)
lr_model.fit(x_train, y_train)

model_list = [rf_model, knn_model, lr_model]
accuracy_scores = []

for model in model_list:
    predictions = model.predict(x_test)
    accuracy_scores.append(accuracy_score(y_test, predictions))

optimal_model = model_list[np.argmax(accuracy_scores)]
print(f'Best model: {optimal_model} with score: {np.max(accuracy_scores)}')

optimal_model.fit(scaled_features, targets)

with open('model.pkl', 'wb') as output_file:
    pickle.dump({'model': optimal_model}, output_file)

print("Model saved successfully.")
