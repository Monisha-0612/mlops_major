from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

def main():
    # Load dataset
    data = fetch_olivetti_faces()
    X = data.images.reshape(len(data.images), -1)  # flatten images
    y = data.target

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Train model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Training completed. Test accuracy: {acc:.4f}")

    # Save model + test data
    joblib.dump(model, "savedmodel.pth")
    joblib.dump((X_test, y_test), "test_data.pkl")

if __name__ == "__main__":
    main()
