# app/train.py
import logging
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model():
    # Load data
    logger.info("Loading iris dataset...")
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    logger.info("Training model...")
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    score = model.score(X_test, y_test)
    logger.info(f"Model accuracy: {score:.4f}")

    # Save model
    logger.info("Saving model...")
    joblib.dump(model, "app/model/iris_model.pkl")

#executed train.py
if __name__ == "__main__":
    train_model()
