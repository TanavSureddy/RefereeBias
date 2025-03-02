import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def create_ref_team_dataframe(
    input_csv='data/processed.csv', 
    output_csv='data/dbscan_ready.csv',
    top6 = ["Arsenal", "Chelsea", "Liverpool", "Man City", "Man United", "Tottenham"]
):
    """
    1. Reads processed.csv from 'data/'.
    2. Filters matches to only those involving a top-6 team.
    3. Expands each match into (Referee, Team, Fouls, YellowCards, RedCards)
       rows for home and away if they're in top-6.
    4. Aggregates stats by (Referee, Team).
    5. Reindexes so each Referee has exactly 6 rows (one per top-6 team),
       filling missing stats with 0.
    6. Drops referees who never officiated any top-6 match (all stats = 0).
    7. Scales numeric features.
    8. Saves final CSV to 'data/dbscan_ready.csv'.
    """

    print(f"Reading: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Initial shape of df: {df.shape}")

    # 1) Filter: Keep only matches where HomeTeam or AwayTeam is in top-6
    df_top6 = df[(df['HomeTeam'].isin(top6)) | (df['AwayTeam'].isin(top6))]
    print(f"After filtering for top-6 teams, shape: {df_top6.shape}")

    if df_top6.empty:
        print("ERROR: No matches found for the given top-6 team names.")
        return

    # 2) Expand rows: For each match, create a row for the team that is in top-6.
    rows = []
    for _, row in df_top6.iterrows():
        referee = row['Referee']
        # For the home side
        if row['HomeTeam'] in top6:
            rows.append({
                'Referee': referee,
                'Team': row['HomeTeam'],
                'Fouls': row.get('HomeTeamFouls', 0),
                'YellowCards': row.get('HomeTeamYellowCards', 0),
                'RedCards': row.get('HomeTeamRedCards', 0)
            })
        # For the away side
        if row['AwayTeam'] in top6:
            rows.append({
                'Referee': referee,
                'Team': row['AwayTeam'],
                'Fouls': row.get('AwayTeamFouls', 0),
                'YellowCards': row.get('AwayTeamYellowCards', 0),
                'RedCards': row.get('AwayTeamRedCards', 0)
            })

    df_expanded = pd.DataFrame(rows)
    print(f"After expanding into (Referee, Team) rows, shape: {df_expanded.shape}")
    
    if df_expanded.empty:
        print("ERROR: After expansion, there are no rows. Check your column names for fouls/cards.")
        return

    # 3) Aggregate by (Referee, Team)
    df_grouped = df_expanded.groupby(['Referee', 'Team'], as_index=False).agg({
        'Fouls': 'sum',
        'YellowCards': 'sum',
        'RedCards': 'sum'
    })
    print(f"After grouping, shape: {df_grouped.shape}")

    # 4) Reindex so each Referee has 6 rows (one per top-6 team)
    all_refs = df_grouped['Referee'].unique()
    all_combos = pd.MultiIndex.from_product([all_refs, top6], names=['Referee', 'Team'])
    df_grouped = df_grouped.set_index(['Referee', 'Team']).reindex(all_combos, fill_value=0).reset_index()
    print(f"After reindexing, shape: {df_grouped.shape}")

    # 5) Drop referees who never officiated any top-6 match (all stats = 0)
    sums_per_ref = df_grouped.groupby('Referee')[['Fouls', 'YellowCards', 'RedCards']].sum().reset_index()
    sums_per_ref['SumAll'] = sums_per_ref['Fouls'] + sums_per_ref['YellowCards'] + sums_per_ref['RedCards']
    valid_refs = sums_per_ref[sums_per_ref['SumAll'] > 0]['Referee']
    df_grouped = df_grouped[df_grouped['Referee'].isin(valid_refs)].copy()
    print(f"After dropping referees with 0 stats, shape: {df_grouped.shape}")

    if df_grouped.empty:
        print("ERROR: After dropping 0-stat referees, no data remains.")
        return

    # 6) Scale numeric features (Fouls, YellowCards, RedCards)
    features = ['Fouls', 'YellowCards', 'RedCards']
    scaler = StandardScaler()
    df_grouped[features] = scaler.fit_transform(df_grouped[features])

    # 7) Save final CSV
    df_grouped.to_csv(output_csv, index=False)
    print(f"Data frame organized for DBSCAN (Ref-Team rows) and saved as '{output_csv}'.")
    print(f"Final shape: {df_grouped.shape[0]} rows × {df_grouped.shape[1]} columns")

if __name__ == "__main__":
    create_ref_team_dataframe()
