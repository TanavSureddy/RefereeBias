import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def canonicalize_referee(name):
    """
    Converts a referee name to a canonical form.
    For example, all variants that contain 'wilkes' become 'Clive Wilkes'.
    You can expand this function with additional mappings as needed.
    """
    name_clean = name.lower().replace('.', '').strip()
    if 'wilkes' in name_clean:
        return 'Clive Wilkes'
    # Add other mappings here if necessary.
    return name_clean.title()

def create_ref_team_dataframe(
    input_csv='data/processed.csv', 
    output_csv='data/dbscan_ready.csv',
    top6 = ["Arsenal", "Chelsea", "Liverpool", "Man City", "Man United", "Tottenham"]
):
    """
    Process the processed.csv file to generate a cleaned dbscan_ready.csv with 180 rows.
    
    Steps:
      1. Read processed.csv.
      2. Filter for matches involving one of the top-6 teams.
      3. Canonicalize referee names.
      4. Select the top 30 referees by unique MatchID count.
      5. Expand each match into rows for (Referee, Team, Fouls, YellowCards, RedCards) for home and away.
      6. Group by (Referee, Team) and aggregate the fouls/cards.
      7. Drop duplicate rows for the same team where the numeric values are identical.
      8. Reindex so each top-30 referee has exactly one row per top6 team.
      9. Scale numeric features.
      10. Save the final CSV.
    """
    print(f"Reading: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Initial shape: {df.shape}")

    # 1) Filter for matches where HomeTeam or AwayTeam is in top6
    df_top6 = df[(df['HomeTeam'].isin(top6)) | (df['AwayTeam'].isin(top6))].copy()
    print(f"After filtering for top-6 teams: {df_top6.shape}")

    if df_top6.empty:
        print("ERROR: No matches found for the given top-6 team names.")
        return

    # 2) Canonicalize referee names
    df_top6['Referee_Canonical'] = df_top6['Referee'].apply(canonicalize_referee)

    # 3) Determine top 30 referees by number of games (using unique MatchID)
    ref_games = df_top6.groupby('Referee_Canonical')['MatchID'].nunique().reset_index(name='GameCount')
    top_30_refs = ref_games.sort_values(by='GameCount', ascending=False).head(30)['Referee_Canonical'].tolist()
    print(f"Top 30 referees: {top_30_refs}")

    # 4) Keep only rows for the top 30 referees
    df_top6 = df_top6[df_top6['Referee_Canonical'].isin(top_30_refs)]
    print(f"After filtering to top 30 referees: {df_top6.shape}")

    # 5) Expand each match into rows for each top6 team (home and/or away)
    rows = []
    for _, row in df_top6.iterrows():
        referee = row['Referee_Canonical']
        # Home side row if applicable
        if row['HomeTeam'] in top6:
            rows.append({
                'Referee': referee,
                'Team': row['HomeTeam'],
                'Fouls': row.get('HomeTeamFouls', 0),
                'YellowCards': row.get('HomeTeamYellowCards', 0),
                'RedCards': row.get('HomeTeamRedCards', 0)
            })
        # Away side row if applicable
        if row['AwayTeam'] in top6:
            rows.append({
                'Referee': referee,
                'Team': row['AwayTeam'],
                'Fouls': row.get('AwayTeamFouls', 0),
                'YellowCards': row.get('AwayTeamYellowCards', 0),
                'RedCards': row.get('AwayTeamRedCards', 0)
            })
    df_expanded = pd.DataFrame(rows)
    print(f"After expansion: {df_expanded.shape}")

    if df_expanded.empty:
        print("ERROR: No rows after expansion. Check your column names for fouls/cards.")
        return

    # 6) Aggregate stats by (Referee, Team)
    df_grouped = df_expanded.groupby(['Referee', 'Team'], as_index=False).agg({
        'Fouls': 'sum',
        'YellowCards': 'sum',
        'RedCards': 'sum'
    })
    print(f"After grouping: {df_grouped.shape}")

    # 7) Drop duplicates if numeric columns (Fouls, YellowCards, RedCards) are identical for the same team.
    # This step assumes that if the numbers are exactly identical, they likely represent the same referee.
    df_grouped = df_grouped.drop_duplicates(subset=['Team', 'Fouls', 'YellowCards', 'RedCards'])
    print(f"After dropping duplicates based on numeric values: {df_grouped.shape}")

    # 8) Reindex so that each referee gets exactly one row per top6 team.
    # Use the list of top 30 referees from earlier.
    all_refs = top_30_refs
    all_combos = pd.MultiIndex.from_product([all_refs, top6], names=['Referee', 'Team'])
    df_grouped = df_grouped.set_index(['Referee', 'Team']).reindex(all_combos, fill_value=0).reset_index()
    print(f"After reindexing: {df_grouped.shape}")

    # Check: Expected rows = number of top refs * number of teams.
    expected_rows = len(top_30_refs) * len(top6)
    if df_grouped.shape[0] != expected_rows:
        print(f"Warning: Expected {expected_rows} rows, but got {df_grouped.shape[0]} rows.")

    # 9) Scale numeric features
    features = ['Fouls', 'YellowCards', 'RedCards']
    scaler = StandardScaler()
    df_grouped[features] = scaler.fit_transform(df_grouped[features])

    # 10) Save the final CSV
    df_grouped.to_csv(output_csv, index=False)
    print(f"Data frame organized for DBSCAN and saved as '{output_csv}'.")
    print(f"Final shape: {df_grouped.shape[0]} rows × {df_grouped.shape[1]} columns")

if __name__ == "__main__":
    create_ref_team_dataframe()
