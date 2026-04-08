# Customer Segmentation using K-Means Clustering

## Project Goal
This project applies unsupervised machine learning to segment a customer base based on demographic and socioeconomic factors. This allows for targeted marketing and better understanding of consumer behavior.

## Workflow
1. **Data Preprocessing**: Standardized features using `StandardScaler` for distance-based clustering.
2. **Hyperparameter Tuning**: Utilized the **Elbow Method** and **Silhouette Scores** to determine optimal clusters (k=4).
3. **Dimensionality Reduction**: Implemented **PCA** to project 4D data into 2D for visualization.

## Real-World Data Pipeline Integration
This clustering logic is designed to be part of an automated **ETL (Extract, Transform, Load)** workflow:
- **Data Sourcing**: Customer data is pulled from a CRM (like Salesforce) or a Firebase backend via a scheduled Data Engineering pipeline.
- **Processing Layer**: The script standardizes incoming batches of "New Customer" data to ensure they match the scale of the original training set.
- **Batch Loading**: The resulting cluster assignments are written back to a centralized Data Warehouse (e.g., BigQuery or Snowflake), allowing marketing teams to query segments directly via SQL.

## Key Libraries
- `pandas`, `scikit-learn`, `matplotlib`.
