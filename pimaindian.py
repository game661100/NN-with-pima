import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('diabetes.csv')
X = df.iloc[:, 0:8].values
Y = df.iloc[:, 8].values
seed = 42
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=10)

model = keras.models.Sequential([
    keras.layers.Dense(24, input_dim=8, activation='relu'),
    keras.layers.Dense(12, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=1)
model.evaluate(X_test, y_test)

# Get binary predictions
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='g')
plt.text(0.5, -0.15, f'Accuracy: {accuracy:.2f}', fontsize=12, ha='center', transform=plt.gca().transAxes)
plt.show()
