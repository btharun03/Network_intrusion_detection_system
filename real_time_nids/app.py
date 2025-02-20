import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib
from flask import Flask, render_template, jsonify

app = Flask(__name__)

base_models = [
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=10, random_state=42))
]
stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression()
)

try:
    stacking_model = joblib.load("stacking_model.pkl")
except Exception as e:
    print(f"Error loading model: {e}. Training a new model.")
    
    column_names = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
        "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
        "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
        "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
        "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
        "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
    ]
    
    X_train = pd.DataFrame(np.random.rand(100, len(column_names)), columns=column_names)
    
    y_train = np.random.choice([0, 1], size=100, p=[0.7, 0.3])  
    stacking_model.fit(X_train, y_train)
    joblib.dump(stacking_model, "stacking_model.pkl")

def generate_test_data(num_samples=5):
    
    column_names = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
        "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
        "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
        "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
        "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
        "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
    ]

    test_data = pd.DataFrame(np.random.rand(num_samples, len(column_names)), columns=column_names)
    return test_data

@app.route('/predict')
def predict():
    test_data = generate_test_data()
    print(f"Generated test data shape: {test_data.shape}")  
    test_features = test_data.copy()  
    print(f"Test features shape: {test_features.shape}") 

    predictions = stacking_model.predict(test_features)
    test_data['prediction'] = predictions  
    return jsonify(test_data.to_dict(orient="records"))

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
