import os
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import plotly.express as px

def run_dbscan_and_plot(
    input_csv='dbscan_ready.csv',  # file is in the same folder as this script
    eps=0.5,
    min_samples=5
):
    """
    Loads the dbscan_ready.csv file (in the same folder as this script), where each row is 
    (Referee, Team, Fouls, YellowCards, RedCards), then:
      1. Runs DBSCAN on the numeric features.
      2. Reduces dimensions to 2 via PCA for visualization.
      3. Creates an interactive Plotly scatter plot with:
         - Color indicating the DBSCAN cluster.
         - Different marker symbols for each Team.
         - Hover tooltips displaying Referee, Team, cluster label, and stats.
    """
    # Construct the absolute path for the CSV file based on the script's location.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv_path = os.path.join(script_dir, input_csv)
    
    # Load the data
    df = pd.read_csv(input_csv_path)

    # Prepare features for clustering by dropping non-numeric columns
    X = df.drop(columns=['Referee', 'Team'])

    # Run DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(X)
    df['Cluster'] = dbscan.labels_

    # Reduce dimensions to 2 using PCA for visualization
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X)
    df['PC1'] = reduced[:, 0]
    df['PC2'] = reduced[:, 1]

    # Create an interactive Plotly scatter plot
    fig = px.scatter(
        df,
        x='PC1',
        y='PC2',
        color='Cluster',               # Color by DBSCAN cluster label
        symbol='Team',                 # Different marker shape per team
        hover_data=['Referee', 'Team', 'Cluster', 'Fouls', 'YellowCards', 'RedCards'],
        title='DBSCAN Clustering on (Referee, Team)'
    )

    # Display the plot
    fig.show()

if __name__ == "__main__":
    # Adjust eps and min_samples as needed.
    run_dbscan_and_plot(input_csv='dbscan_ready.csv', eps=0.5, min_samples=5)
