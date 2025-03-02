import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def create_ref_team_dataframe(
    input_csv='data/processed.csv', 
    output_csv='data/dbscan_ready.csv',
    top6 = ['Arsenal', 'Chelsea', 'Liverpool', 'Manchester City', 'Manchester United', 'Tottenham']
):
    """
    1. Reads the processed CSV.
    2. For each row (match), collects stats for the 'HomeTeam' and 'AwayTeam' if they're in the top-6.
    3. Aggregates stats by (Referee, Team).
    4. Ensures each referee has exactly 6 rows, one for each top-6 team (filling with 0 if no matches).
    5. Scales numeric features.
    6. Saves to CSV.
    """

    # Load your processed data from the data folder
    df = pd.read_csv(input_csv)

    # -------------------------------------------
    # Step 1: Collect match-level data only if team is in top-6
    # -------------------------------------------
    # We will create an "expanded" DataFrame where each row corresponds to:
    #   (Referee, Team, [Fouls], [YellowCards], [RedCards], etc.)
    # for home AND away sides, if that side is in top-6.

    rows = []
    for _, row in df.iterrows():
        referee = row['Referee']
        
        # Check if home team is in top-6
        if row['HomeTeam'] in top6:
            rows.append({
                'Referee': referee,
                'Team': row['HomeTeam'],
                'Fouls': row.get('HomeFouls', 0),
                'YellowCards': row.get('HomeYellowCards', 0),
                'RedCards': row.get('HomeRedCards', 0)
            })

        # Check if away team is in top-6
        if row['AwayTeam'] in top6:
            rows.append({
                'Referee': referee,
                'Team': row['AwayTeam'],
                'Fouls': row.get('AwayFouls', 0),
                'YellowCards': row.get('AwayYellowCards', 0),
                'RedCards': row.get('AwayRedCards', 0)
            })

    df_expanded = pd.DataFrame(rows)

    # -------------------------------------------
    # Step 2: Aggregate stats by (Referee, Team)
    # -------------------------------------------
    # Summing over multiple matches. 
    # If your dataset uses different column names for fouls/cards, adjust accordingly.
    df_grouped = df_expanded.groupby(['Referee','Team'], as_index=False).agg({
        'Fouls': 'sum',
        'YellowCards': 'sum',
        'RedCards': 'sum'
    })

    # -------------------------------------------
    # Step 3: Ensure each referee has 6 rows, one for each top-6 team
    # -------------------------------------------
    # Create a complete set of (Referee, Team) pairs for the referees in df_grouped
    all_refs = df_grouped['Referee'].unique()
    all_teams = top6

    # Create a MultiIndex for every (Referee, Team) combination
    all_combos = pd.MultiIndex.from_product([all_refs, all_teams], names=['Referee','Team'])
    
    # Reindex df_grouped so it has a row for every (Referee, Team) combination, filling missing values with 0
    df_grouped = df_grouped.set_index(['Referee','Team']).reindex(all_combos, fill_value=0).reset_index()

    # -------------------------------------------
    # Step 4: Scale the numeric features
    # -------------------------------------------
    features = ['Fouls', 'YellowCards', 'RedCards']
    scaler = StandardScaler()
    df_grouped_scaled = df_grouped.copy()
    df_grouped_scaled[features] = scaler.fit_transform(df_grouped[features])

    # -------------------------------------------
    # Step 5: Save to CSV for DBSCAN
    # -------------------------------------------
    df_grouped_scaled.to_csv(output_csv, index=False)
    print(f"Data frame organized for DBSCAN (Ref-Team rows) and saved as '{output_csv}'.")

# Run the preprocessing if this script is executed directly
if __name__ == "__main__":
    create_ref_team_dataframe()
