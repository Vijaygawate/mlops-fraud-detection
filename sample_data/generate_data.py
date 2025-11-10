import pandas as pd
import numpy as np
import boto3
import os

# Generate synthetic fraud detection dataset
np.random.seed(42)

# Generate 10,000 transactions
n_samples = 10000

# Features
data = {
    'transaction_amount': np.random.exponential(scale=50, size=n_samples),
    'merchant_category': np.random.randint(1, 20, size=n_samples),
    'transaction_hour': np.random.randint(0, 24, size=n_samples),
    'days_since_last_transaction': np.random.exponential(scale=5, size=n_samples),
    'account_age_days': np.random.exponential(scale=365, size=n_samples),
    'num_transactions_last_30days': np.random.poisson(lam=15, size=n_samples),
    'avg_transaction_amount': np.random.exponential(scale=40, size=n_samples),
    'is_international': np.random.binomial(1, 0.1, size=n_samples),
    'device_type': np.random.randint(1, 4, size=n_samples),
}

# Generate fraud labels (5% fraud rate)
fraud_probability = 0.05
data['is_fraud'] = np.random.binomial(1, fraud_probability, size=n_samples)

# Make fraud cases more extreme
fraud_mask = data['is_fraud'] == 1
data['transaction_amount'][fraud_mask] *= 3  # Fraudulent transactions are larger

df = pd.DataFrame(data)

# Save to CSV
output_file = 'transactions.csv'
df.to_csv(output_file, index=False)

print(f"Generated {n_samples} transactions with {df['is_fraud'].sum()} fraudulent transactions")
print(f"Data saved to {output_file}")
print(f"\nSample data:")
print(df.head(10))

# Upload to S3 (update bucket name from your Terraform output)
s3_bucket = os.environ.get('ML_DATA_BUCKET', 'mlops-project-dev-ml-data-8c2241a2')
s3_key = 'raw-data/transactions.csv'

try:
    s3 = boto3.client('s3')
    s3.upload_file(output_file, s3_bucket, s3_key)
    print(f"\nData uploaded to s3://{s3_bucket}/{s3_key}")
except Exception as e:
    print(f"\nNote: Upload to S3 manually or set ML_DATA_BUCKET environment variable")
    print(f"Error: {e}")
