from google.ads.google_ads.client import GoogleAdsClient
from google.cloud import bigquery
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import pickle
import os

# Initialize Flask App
app = Flask(__name__)

# Step 1: Fetch Google Ads Data
def fetch_google_ads_data(client, customer_id):
    query = """
    SELECT campaign.id, campaign.name, metrics.clicks, 
           metrics.impressions, metrics.average_cpc, 
           metrics.conversions, metrics.cost_micros
    FROM campaign
    WHERE segments.date DURING LAST_30_DAYS
    """
    response = client.service.google_ads.search(customer_id=customer_id, query=query)
    return [
        {"campaign_id": row.campaign.id, "clicks": row.metrics.clicks, "cost": row.metrics.cost_micros / 1_000_000}
        for row in response
    ]

# Step 2: Store Data in BigQuery
def store_data_bigquery(dataset_id, table_id, data):
    client = bigquery.Client()
    table_ref = client.dataset(dataset_id).table(table_id)
    client.insert_rows_json(table_ref, data)

# Step 3: Train Machine Learning Model
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Step 4: Load & Train Model
if os.path.exists("model.pkl"):
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
else:
    # Dummy dataset for training (Replace with actual dataset)
    data = pd.DataFrame({
        "clicks": [100, 150, 200, 250, 300],
        "cost": [10, 15, 20, 25, 30],
        "conversions": [5, 7, 9, 12, 14]
    })
    X_train = data[["clicks", "cost"]]
    y_train = data["conversions"]
    model = train_model(X_train, y_train)
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

# Step 5: API for Real-time Recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    prediction = model.predict([[data["clicks"], data["cost"]]])[0]
    return jsonify({"recommended_conversions": prediction})

if __name__ == "__main__":
    app.run(debug=True)
