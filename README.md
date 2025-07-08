# 🐱🐶 Cats and Dogs Classification using K-Nearest Neighbors (KNN)

This project demonstrates a **machine learning classification task** using the **K-Nearest Neighbors (KNN) algorithm** to distinguish between images of cats and dogs based on extracted features. The dataset used is `CatsAndDogs_v2.csv`.

---

## 📂 Project Structure

```
├── CatsAndDogs_v2.csv         # Dataset containing feature vectors and labels
├── ml_project2.ipynb          # Jupyter Notebook with complete code and analysis
├── README.md                  # Project overview and instructions
```

---

## 📝 Dataset Description

**File:** `CatsAndDogs_v2.csv`

This dataset contains **tabular data** representing preprocessed features extracted from images of cats and dogs.  
Each row corresponds to a single image, with:

- **Feature Columns** – numeric values (e.g., pixel intensities, color histograms, texture features).
- **Label Column** – the class label (`Cat` or `Dog`).

Example (illustrative):

| Feature1 | Feature2 | ... | FeatureN | Label |
|----------|----------|-----|----------|-------|
| 0.35     | 0.76     | ... | 0.12     | Cat   |
| 0.58     | 0.43     | ... | 0.89     | Dog   |

---

## 🧠 Algorithm Overview

**K-Nearest Neighbors (KNN)** is a simple, non-parametric classification algorithm:

1. **Compute Distances:** For each test point, calculate the distance (e.g., Euclidean) to all training points.
2. **Find Neighbors:** Identify the *k* nearest neighbors.
3. **Vote:** Assign the most common class label among those neighbors.

Advantages:
- Easy to implement
- No assumptions about data distribution

Considerations:
- Sensitive to feature scaling
- Computationally expensive for large datasets

---

## 🚀 Project Workflow

The project follows these main steps:

1. **Import Libraries**
   - `pandas`, `numpy`: Data manipulation
   - `scikit-learn`: Model building and evaluation
   - `matplotlib`: Visualization

2. **Load and Explore Data**
   - Read `CatsAndDogs_v2.csv`
   - Inspect shape, data types, and label distribution

3. **Preprocess Data**
   - Encode labels (`Cat` = 0, `Dog` = 1)
   - Scale features (e.g., StandardScaler)

4. **Split Data**
   - Training set and test set (e.g., 80/20 split)

5. **Train KNN Classifier**
   - Choose `k` (number of neighbors)
   - Fit model to training data

6. **Evaluate Model**
   - Predict labels on test data
   - Compute metrics:
     - Accuracy
     - Confusion matrix
     - Classification report (precision, recall, F1-score)

7. **Visualize Results**
   - Plot confusion matrix
   - Optional: Decision boundaries (if using 2 features)

---

## ⚙️ How to Run

Follow these steps to run the project:

1. **Install Requirements**

   Make sure you have Python >=3.7 and install dependencies:

   ```bash
   pip install pandas numpy scikit-learn matplotlib
   ```

2. **Open Notebook**

   Launch Jupyter Notebook:

   ```bash
   jupyter notebook ml_project2.ipynb
   ```

3. **Run Cells**

   Execute each cell sequentially to reproduce preprocessing, training, and evaluation.

---

## 🎯 Results

Your model performance will depend on:

- Number of neighbors (`k`)
- Choice of distance metric
- Data scaling

Example metrics (hypothetical):

- **Accuracy:** 92%
- **Precision (Dog):** 90%
- **Recall (Dog):** 95%
- **F1-score:** 92%

---

## 🔧 Tuning & Improvements

To improve your classifier:

- Experiment with different `k` values (e.g., 3–15)
- Try distance metrics (Manhattan, Minkowski)
- Use cross-validation to select optimal hyperparameters
- Incorporate feature selection techniques

---

## 📈 Sample Code Snippet

Example code to train the classifier:

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("CatsAndDogs_v2.csv")

# Split features and labels
X = data.drop("Label", axis=1)
y = data["Label"].map({"Cat": 0, "Dog": 1})

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Evaluate
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)
```

---

## 📚 References

- [Scikit-learn Documentation – KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [Introduction to KNN](https://towardsdatascience.com/k-nearest-neighbors-knn-algorithm-for-machine-learning-e883219c8f26)

---

## 🙌 Acknowledgements

This project was created as part of a machine learning assignment to practice classification using KNN.

---
## 📸 Project Screenshots

- [Confusion Matrix Screenshot](Screenshot%2030.png)
- [Recall ,Precision and F1 Score Screenshot](Screenshot%2017.png)

---

## 📩 Contact

**Shilpa K C**  
[LinkedIn](https://www.linkedin.com/in/shilpa-kc) | [Email](shilpakcc@gmail.com)

For questions or suggestions, feel free to reach out.

