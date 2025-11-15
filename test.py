import joblib
from sklearn.metrics import accuracy_score

def main():
    # Load trained model
    model = joblib.load("savedmodel.pth")
    X_test, y_test = joblib.load("test_data.pkl")

    # Predict
    y_pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy from test.py: {acc:.4f}")

if __name__ == "__main__":
    main()
