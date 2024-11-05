import streamlit as st
import sqlite3
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import plotly.express as px # type: ignore

# Streamlit page config must be the first Streamlit command
st.set_page_config(page_title="LMP Pitching Stats", layout="wide")

# Load data with caching to improve performance
@st.cache_data
def load_players_data():
    players_data_files = glob.glob(os.path.join('stats_data', 'players_data_*.csv'))
    players_df_list = [pd.read_csv(file) for file in players_data_files]
    return pd.concat(players_df_list, ignore_index=True)

@st.cache_data
def load_standard_stats():
    standard_stats_files = glob.glob(os.path.join('stats_data', 'df_standard_stats_*.csv'))
    standard_stats_df_list = [pd.read_csv(file) for file in standard_stats_files]
    return pd.concat(standard_stats_df_list, ignore_index=True)

@st.cache_data
def load_advanced_stats():
    advanced_stats_files = glob.glob(os.path.join('stats_data', 'df_advanced_stats_*.csv'))
    advanced_stats_df_list = [pd.read_csv(file) for file in advanced_stats_files]
    return pd.concat(advanced_stats_df_list, ignore_index=True)

@st.cache_data
def load_hit_trajectory():
    hit_trajectory_files = glob.glob(os.path.join('stats_data', 'hit_trajectory_lmp_*.csv'))
    hit_trajectory_df_list = [pd.read_csv(file) for file in hit_trajectory_files]
    return pd.concat(hit_trajectory_df_list, ignore_index=True)

@st.cache_data
def load_stadium_data():
    return pd.read_csv(os.path.join('stats_data', 'stadium.csv'))

@st.cache_data
def load_headshots():
    conn = sqlite3.connect(os.path.join('stats_data', 'player_headshots_lmp.db'))
    headshots_df = pd.read_sql_query("SELECT playerId, headshot_url FROM player_headshots_lmp", conn)
    conn.close()
    return headshots_df

# Load datasets
players_df = load_players_data()
standard_stats_df = load_standard_stats()
advanced_stats_df = load_advanced_stats()
hit_trajectory_df = load_hit_trajectory()
team_data = load_stadium_data()
headshots_df = load_headshots()
pitchers_df = pd.read_csv('pitchers_df.csv')  # Use the provided file for pitchers_df

@st.cache_data

def load_data():
    league_avg_df = pd.read_csv('league_avg.csv')
    team_data_std_df = pd.read_csv('team_data_std_p.csv')
    team_data_adv_df = pd.read_csv('team_data_adv_p.csv')
    return league_avg_df, team_data_std_df, team_data_adv_df

league_avg_df, team_data_std_df, team_data_adv_df = load_data()

@st.cache_data
def load_team_era_whip():
    return pd.read_csv('teams_era_whip.csv')

# Load the data
team_era_whip_df = load_team_era_whip()

# Toggle for selecting Teams or League view
view_selection = st.radio("Select View:", ["Players", "Teams"], index=0, horizontal=True)
# st.divider()

if view_selection == 'Players':

    # Ensure 'playerId' and 'id' are of the same type
    headshots_df['playerId'] = headshots_df['playerId'].astype(int)
    players_df = pd.merge(players_df, headshots_df, left_on='id', right_on='playerId', how='left')

    # Ensure 'player_id' in stats DataFrames is of type integer
    standard_stats_df['player_id'] = standard_stats_df['player_id'].astype(int)
    advanced_stats_df['player_id'] = advanced_stats_df['player_id'].astype(int)


    # st.set_page_config(page_title="LMP Batting Stats", layout="wide")
    logo_and_title = """
        <div style="display: flex; align-items: center;">
            <img src="https://www.lmp.mx/assets/img/header/logo_80_aniversary.webp" alt="LMP Logo" width="50" height="50">
            <h1 style="margin-left: 10px;">LMP Pitching Stats</h1>
        </div>
    """

    # Display the logo and title using st.markdown
    st.markdown(logo_and_title, unsafe_allow_html=True)
    st.divider()

    # Get team and player info from `standard_stats_df`
    pitchers_with_teams = standard_stats_df[['player_id', 'team', 'Name']].drop_duplicates()

    # Merge the team information with the player dataset
    players_with_teams = pd.merge(players_df, pitchers_with_teams, left_on='id', right_on='player_id', how='inner')

    # Filter to exclude non-pitcher positions
    pos_to_ignore = ['OF', 'IF', 'C', 'SS', '2B', '1B', '3B']
    non_pitchers_df = players_with_teams[~players_with_teams['POS'].isin(pos_to_ignore)]
    pitchers_unique = non_pitchers_df.drop_duplicates(subset=['player_id'])

    pitchers_unique['fullName'] = pitchers_unique['fullName'].astype(str)
    pitchers_unique = pitchers_unique.sort_values('fullName')

    # Set Manny Barreda as the default pitcher
    default_pitcher = 'Manny Barreda'
    default_pitcher_index = next((i for i, name in enumerate(pitchers_unique['fullName']) if name == default_pitcher), 0)

    # Add "ALL" to the list of teams
    teams_unique = ["ALL"] + pitchers_unique['team'].unique().tolist()

    # Layout adjustments for pitcher and team selectboxes
    col1, col2, empty_col1, empty_col2 = st.columns([1, 1, 1, 1])  # Adjust the layout

    # Select a team with the "ALL" option
    with col2:
        selected_team = st.selectbox("Filter by Team", teams_unique, index=0)

    # Filter the pitchers based on the selected team or show all pitchers if "ALL" is selected
    if selected_team == "ALL":
        team_pitchers = pitchers_unique
    else:
        team_pitchers = pitchers_unique[pitchers_unique['team'] == selected_team]

    # Update the pitcher selectbox to show only pitchers from the selected team (or all if "ALL" is selected)
    with col1:
        selected_pitcher = st.selectbox("Select a Pitcher", team_pitchers['fullName'].tolist(), index=default_pitcher_index if selected_team == "ALL" else 0)

    # --- K-BB% Plotting Function ---
    def plot_pitcher_kbb_styled(pitcher_name, pitcher_data, league_avg_kbb):
        if pitcher_data.empty:
            st.write(f"No data available for {pitcher_name}.")
            return
        
        # Check if 'Date' column exists and is formatted correctly
        if 'Date' in pitcher_data.columns:
            try:
                pitcher_data['Date'] = pd.to_datetime(pitcher_data['Date'], errors='coerce')
                pitcher_data = pitcher_data.dropna(subset=['Date'])  # Remove rows with invalid dates
                pitcher_data = pitcher_data.sort_values('Date')  # Sort by date for accurate plotting
            except Exception as e:
                st.write(f"Error in date formatting: {e}")
                return
        
        plt.figure(figsize=(8.5, 4))
        
        # Set the facecolor of the plot to beige
        plt.gca().set_facecolor('beige')   # Set plot background color
        plt.gcf().set_facecolor('beige')   # Set figure background color
        
        # Plot the pitcher's K-BB% over time (if 'Date' exists) or index otherwise
        if 'Date' in pitcher_data.columns:
            plt.plot(
                pitcher_data['Date'], pitcher_data['K-BB%'], 
                color='blue',         # Line color
                linestyle='-',        # Solid line
                marker='o',           # Circle markers
                linewidth=2,          # Line width
                label='_nolegend_'    # Exclude from legend
            )
        else:
            plt.plot(
                pitcher_data.index, pitcher_data['K-BB%'], 
                color='blue',         # Line color
                linestyle='-',        # Solid line
                marker='o',           # Circle markers
                linewidth=2,          # Line width
                label='_nolegend_'    # Exclude from legend
            )
        
        # Add a horizontal line for the league average K-BB% with custom style
        plt.axhline(
            y=league_avg_kbb, 
            color='red',          # Red color
            linestyle='--',       # Dashed line
            linewidth=2,          # Line width
            label=f'League Avg K-BB%: {league_avg_kbb}%'
        )
        
        # Add titles and labels
        plt.title(f'Rolling K-BB% for {pitcher_name}', fontsize=14)
        # plt.xlabel('Date' if 'Date' in pitcher_data.columns else 'Index', fontsize=12)
        # plt.ylabel('K-BB%', fontsize=12)
        
        # Remove the grid
        plt.grid(False)
        
        # Format the x-axis to show only the month and day if 'Date' exists
        if 'Date' in pitcher_data.columns:
            plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))
        
        # Format the y-axis to display 1 decimal place
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}%'))

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Add the legend to display both the pitcher's K-BB% and league average
        plt.legend(loc='best')

        # Show the plot
        plt.tight_layout()
        st.pyplot(plt)


    # Get player data for the selected pitcher
    if not team_pitchers.empty:
        player_data = team_pitchers[team_pitchers['fullName'] == selected_pitcher].iloc[0]
        
        # Display player information in three columns and the plot in the fourth
        st.subheader("Player Information")
        col1, col2, col3, col4 = st.columns([.5, .5, .5, .8])  # Adjust column widths

        # Player information in col1 and col2
        with col1:
            st.write(f"**Full Name:** {player_data['fullName']}")
            st.write(f"**Position:** {player_data['POS']}")
            st.write(f"**B/T:** {player_data['B/T']}")

        with col2:
            st.write(f"**Birthdate:** {player_data['birthDate']}")
            st.write(f"**Birthplace:** {player_data['Birthplace']}")

        # Display headshot in col3
        with col3:
            if pd.notna(player_data['headshot_url']):
                st.image(player_data['headshot_url'], width=150)
            else:
                st.image(os.path.join('stats_data', 'current.jpg'), width=150)

        # Plot K-BB% in col4
        with col4:
            league_avg_kbb = 9.7  # League average K-BB% is 10.5%
            pitcher_data_filtered = pitchers_df[pitchers_df['FullName'] == selected_pitcher]  # Filter data for the selected pitcher
            plot_pitcher_kbb_styled(selected_pitcher, pitcher_data_filtered, league_avg_kbb)



    # # --- Standard Stats ---
    # Filter stats for the selected player
    standard_stats = standard_stats_df[standard_stats_df['player_id'] == player_data['id']]
    standard_stats.loc[:, 'season'] = standard_stats['season'].astype(int)

    standard_columns = ['season', 'Name', 'team', 'POS', 'G', 'GS', 'IP', 'QS','ERA','WHIP', 'W', 'L', 'SV', 'SVOpp','HLD', 'K', 'BB', 'IBB', 'BF', 'H', 'HR', 'ER', 'HBP', 'GIDP', 'WP']
    standard_stats_filtered = standard_stats[standard_columns].copy()

    # Sort by season in descending order and by team
    standard_stats_filtered = standard_stats_filtered.sort_values(by=['season', 'team'], ascending=[False, False])

    # Apply formatting to highlight rows where 'team' is '2 Teams'
    def highlight_two_teams(row):
        return ['background-color: #2E2E2E; color:white' if row['team'] == '2 Teams' else '' for _ in row]

    # Format numeric columns in standard stats to three decimal places
    standard_stats_formatted = standard_stats_filtered.style.format({
        'IP': '{:.1f}',
        'ERA': '{:.2f}',
        'WHIP': '{:.2f}',
        'OPS': '{:.3f}'
    }).apply(highlight_two_teams, axis=1)

    # Display Standard Stats table
    st.subheader("Standard Stats", divider='gray')
    st.dataframe(standard_stats_formatted, hide_index=True, use_container_width=True)

    # --- Advanced Stats ---
    # Load FIP data from the CSV file
    @st.cache_data
    def load_fip_data():
        fip_df = pd.read_csv(os.path.join('stats_data', 'FIP_files.csv'))
        fip_df = fip_df.rename(columns={'x_FIPFB': 'xFIP'})
        # Select only relevant columns to reduce memory usage
        return fip_df[['player_id', 'season', 'FIP', 'xFIP']]

    # Load the FIP data
    fip_df = load_fip_data()
    # Filter stats for the selected player
    advanced_stats = advanced_stats_df[advanced_stats_df['player_id'] == player_data['id']]
    advanced_stats.loc[:, 'season'] = advanced_stats['season'].astype(int)

    advanced_stats_df = advanced_stats_df.merge(fip_df, on=['player_id', 'season'], how='left')

    # Update the Advanced Stats columns to include 'x_FIPFB'
    advanced_columns = [
        'season', 'Name', 'team', 'POS', 'FIP', 'xFIP', 'BABIP', 'AVG', 'OBP', 'SLG', 'OPS',
        'K%', 'BB%', 'K-BB%', 'K/9', 'BB/9', 'K/BB', 'HR/9', 'HR/FB%'
    ]

    # Filter advanced stats for the selected player and ensure data is sorted by season and team
    advanced_stats = advanced_stats_df[advanced_stats_df['player_id'] == player_data['id']]
    advanced_stats.loc[:, 'season'] = advanced_stats['season'].astype(int)
    advanced_stats_filtered = advanced_stats[advanced_columns].sort_values(by=['season', 'team'], ascending=[False, False])

    # Format numeric columns in advanced stats, including 'x_FIPFB'
    advanced_stats_formatted = advanced_stats_filtered.style.format({
        'BABIP': '{:.3f}', 'K%': '{:.1f}', 'BB%': '{:.1f}', 'K-BB%': '{:.1f}',
        'AVG': '{:.3f}', 'OBP': '{:.3f}', 'SLG': '{:.3f}', 'OPS': '{:.3f}',
        'K/9': '{:.2f}', 'BB/9': '{:.2f}', 'K/BB': '{:.2f}', 'HR/9': '{:.2f}',
        'HR/FB%': '{:.1f}', 'xFIP': '{:.2f}', 'FIP': '{:.2f}'
    }).apply(highlight_two_teams, axis=1)

    # Display the updated Advanced Stats table with 'x_FIPFB' included
    st.subheader("Advanced Stats", divider='gray')
    st.dataframe(advanced_stats_formatted, hide_index=True, use_container_width=True)


    # Batted Ball Data
    batted_ball_data = advanced_stats_df[advanced_stats_df['player_id'] == player_data['id']]

    batted_ball_data.loc[:, 'season'] = batted_ball_data['season'].astype(int)

    batted_columns = ['season', 'Name', 'team', 'POS', 'LD%', 'GB%', 'FB%', 'PopUp%', 'P/IP','Str%','SwStr%', 'Whiff%','CSW%', 'CStr%', 'F-Strike%']
    batted_ball_data_filtered = batted_ball_data[batted_columns].copy()

    batted_ball_data_filtered = batted_ball_data_filtered.sort_values(by=['season', 'team'], ascending=[False, False])

    batted_ball_formatted = batted_ball_data_filtered.style.format({
        'LD%': '{:.1f}',
        'GB%': '{:.1f}',
        'FB%': '{:.1f}',
        'PopUp%': '{:.1f}',
        'P/IP': '{:.1f}',
        'SwStr%': '{:.1f}',
        'Whiff%': '{:.1f}',
        'Str%': '{:.1f}',
        'CSW%': '{:.1f}',
        'CStr%': '{:.1f}',
        'F-Strike%': '{:.1f}',
    }).apply(highlight_two_teams, axis=1)

    st.subheader("Batted Ball & Plate Discipline", divider='gray')
    st.dataframe(batted_ball_formatted, hide_index=True, use_container_width=True)

    # Batted Ball Distribution Section
    st.subheader(f"Batted Ball Distribution for {selected_pitcher}")

    # Create season column from date in hit_trajectory_df
    hit_trajectory_df['date'] = pd.to_datetime(hit_trajectory_df['date'])
    hit_trajectory_df['season'] = hit_trajectory_df['date'].dt.year

    # Get available seasons
    available_seasons = sorted(hit_trajectory_df['season'].unique(), reverse=True)

    col1, col2 =st.columns([1,3])
    with col1:
        selected_season = st.selectbox("Select Season", available_seasons)

    # Filter the hit trajectory data based on the selected season and batter
    filtered_hit_trajectory = hit_trajectory_df[
        (hit_trajectory_df['season'] == selected_season) &
        (hit_trajectory_df['pitchername'] == selected_pitcher)
    ]

    # Event types
    event_types = ['single', 'double', 'triple', 'home_run', 'out']
    col1, col2 =st.columns([1,2])
    with col1:
        selected_events = st.multiselect("Select Event Types", event_types, default=event_types)

    # All 'outs'
    out_events = ['field_out', 'double_play', 'force_out', 'sac_bunt', 'grounded_into_double_play', 'sac_fly', 'fielders_choice_out', 'field_error', 'sac_fly_double_play']
    filtered_hit_trajectory.loc[:, 'event'] = filtered_hit_trajectory['event'].apply(lambda x: 'out' if x in out_events else x)


    # Define splits for LHP and RHP
    vs_LHP = filtered_hit_trajectory[filtered_hit_trajectory['split_pitcher'] == 'vs_LHB']
    vs_RHP = filtered_hit_trajectory[filtered_hit_trajectory['split_pitcher'] == 'vs_RHB']

    # Filter the data for the selected events
    vs_LHP = vs_LHP[vs_LHP['event'].isin(selected_events)]
    vs_RHP = vs_RHP[vs_RHP['event'].isin(selected_events)]

    # Create two columns for side-by-side plots
    col1, col2 = st.columns(2)

    # Function to plot the field and hit outcomes
    def plot_field_and_hits(team_data, hit_data, selected_column, palette, plot_title):
        plt.figure(figsize=(8,8))
        y_offset = 275
        excluded_segments = ['outfield_inner']
        
        # Plot the field layout
        for segment_name in team_data['segment'].unique():
            if segment_name not in excluded_segments:
                segment_data = team_data[team_data['segment'] == segment_name]
                plt.plot(segment_data['x'], -segment_data['y'] + y_offset, linewidth=4, zorder=1, color='forestgreen', alpha=0.5)

        # Adjust hit coordinates and plot the hits
        hit_data['adj_coordY'] = -hit_data['coordY'] + y_offset
        sns.scatterplot(data=hit_data, x='coordX', y='adj_coordY', hue=selected_column, palette=palette, edgecolor='black', s=100, alpha=0.7)

        plt.text(295, 23, 'Created by: @iamfrankjuarez', fontsize=8, color='grey', alpha=0.3, ha='right')

        plt.title(plot_title, fontsize=15)
        plt.xlabel("")
        plt.ylabel("")
        plt.legend(title=selected_column, title_fontsize='11', fontsize='11', borderpad=1)
        plt.xticks([])
        plt.yticks([])
        plt.xlim(-50, 300)
        plt.ylim(20, 300)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(False)
        st.pyplot(plt)

    # Plot for vs LHP
    with col1:
        if not vs_LHP.empty:
            plot_title = f"Batted Ball Outcomes vs LHP for {selected_pitcher}"
            plot_field_and_hits(team_data, vs_LHP, 'event', {
                'single': 'darkorange', 'double': 'purple', 'triple': 'yellow', 'home_run': 'red', 'out': 'grey'
            }, plot_title)
        else:
            st.write("No data available for vs LHP.")

    # Plot for vs RHP
    with col2:
        if not vs_RHP.empty:
            plot_title = f"Batted Ball Outcomes vs RHP for {selected_pitcher}"
            plot_field_and_hits(team_data, vs_RHP, 'event', {
                'single': 'darkorange', 'double': 'purple', 'triple': 'yellow', 'home_run': 'red', 'out': 'grey'
            }, plot_title)
        else:
            st.write("No data available for vs RHP.")

else:
    st.markdown("<h1 style='text-align: center;'>League & Teams</h1>", unsafe_allow_html=True)
    # st.divider()

    # Round and format the ERA and WHIP columns
    team_era_whip_df['ERA'] = team_era_whip_df['ERA'].round(2)
    team_era_whip_df['WHIP'] = team_era_whip_df['WHIP'].round(2)

    team_era_whip_df['ERA'] = team_era_whip_df['ERA'].apply(lambda x: f"{x:.2f}")
    team_era_whip_df['WHIP'] = team_era_whip_df['WHIP'].apply(lambda x: f"{x:.2f}")
    team_era_whip_df['IP'] = team_era_whip_df['IP'].apply(lambda x: f"{x:.1f}")
    # Separate SP and RP data
    sp_df = team_era_whip_df[team_era_whip_df['position'] == 'SP']
    rp_df = team_era_whip_df[team_era_whip_df['position'] == 'RP']
    # Create a layout with columns for title and dataframes
    col1, col2, col3, col4 = st.columns([2,2,2,2])

    # Add titles above the dataframes
    with col1:
        st.markdown("<h3 style='text-align: left;'>Starting Pitching</h3>", unsafe_allow_html=True)
        st.dataframe(sp_df[['Team', 'ERA', 'WHIP', 'IP']], width=400, hide_index=True)

    with col2:
        st.markdown("<h3 style='text-align: left;'>Relief Pitching</h3>", unsafe_allow_html=True)
        st.dataframe(rp_df[['Team', 'ERA', 'WHIP', 'IP']], width=400, hide_index=True)


    st.subheader("League Averages", divider='gray')

    league_avg_df.insert(2, 'FIP', 3.47)
    league_avg_df.insert(3, 'xFIP', 3.84)
    league_columns = ['ERA', 'WHIP', 'FIP', 'xFIP','K%', 'BB%', 'K-BB%', 'SwStr%', 'Whiff%', 'Str%', 'CSW%', 'CStr%', 'F-Strike%', 'LD%', 'GB%', 'FB%', 'PopUp%', 'HR/FB%', 'BABIP', 'AVG', 'OBP', 'SLG', 'OPS',
                      'K/9', 'BB/9', 'H/9', 'R/9', 'HR/9', 'K/BB']
    league_avg_formatted = league_avg_df[league_columns].style.format({
        'xFIP': '{:.2f}', 'FIP': '{:.2f}',
        'ERA': '{:.2f}', 'WHIP': '{:.2f}',
        'BABIP': '{:.3f}', 'K%': '{:.1f}', 'BB%': '{:.1f}', 'K-BB%': '{:.1f}',
        'AVG': '{:.3f}', 'OBP': '{:.3f}', 'SLG': '{:.3f}', 'OPS': '{:.3f}',
        'K/9': '{:.2f}', 'BB/9': '{:.2f}', 'K/BB': '{:.2f}', 'HR/9': '{:.2f}',
        'H/9': '{:.2f}', 'R/9': '{:.2f}','HR/FB%': '{:.1f}',
        'LD%': '{:.1f}',
        'GB%': '{:.1f}',
        'FB%': '{:.1f}',
        'PopUp%': '{:.1f}',
        'P/IP': '{:.1f}',
        'SwStr%': '{:.1f}',
        'Whiff%': '{:.1f}',
        'Str%': '{:.1f}',
        'CSW%': '{:.1f}',
        'CStr%': '{:.1f}',
        'F-Strike%': '{:.1f}',
    })
    
    st.dataframe(league_avg_formatted, use_container_width=True, hide_index=True)

    team_abbreviations = {
    'MXC': 'Aguilas de Mexicali',
    'JAL': 'Charros de Jalisco',
    'MOC': 'Caneros de los Mochis',
    'NAV': 'Mayos de Navojoa',
    'HER': 'Naranjeros de Hermosillo',
    'CUL': 'Tomateros de Culiacan',
    'MAZ': 'Venados de Mazatlan',
    'OBR': 'Yaquis de Obregon',
    'GSV': 'Algodoneros de Guasave',
    'MTY': 'Sultanes de Monterrey'
    }
    
    team_data_std_df['team'] = team_data_std_df['team'].replace(team_abbreviations)
    team_data_adv_df['team'] = team_data_adv_df['team'].replace(team_abbreviations)

    st.subheader("Team Standard Stats", divider='gray')
    standard_columns = ['team', 'W', 'L', 'ERA', 'WHIP', 'GS', 'QS', 'SV', 'SVOpp','BS', 'HLD', 'IP', 'BF', 'AB', 'R', 'ER', 'K', '2B', '3B', 'HR', 'TB', 'BB', 'IBB', 'HBP', 'BK', 'WP', 'SF']
    team_standard_formatted = team_data_std_df[standard_columns].style.format({
        'ERA': '{:.2f}',
        'WHIP': '{:.2f}',
        'IP': '{:.1f}'
    })
    st.dataframe(team_standard_formatted, use_container_width=True, hide_index=True)

    st.subheader("Team Advanced Stats", divider='gray')
    advanced_columns = ['team','K%', 'BB%', 'K-BB%', 'BABIP', 'AVG', 'OBP', 'SLG', 'OPS', 'K/9', 'BB/9', 'K/BB', 'H/9', 'R/9', 'HR/9']
    team_advanced_formatted = team_data_adv_df[advanced_columns].style.format({
        'BABIP': '{:.3f}', 'K%': '{:.1f}', 'BB%': '{:.1f}', 'K-BB%': '{:.1f}',
        'AVG': '{:.3f}', 'OBP': '{:.3f}', 'SLG': '{:.3f}', 'OPS': '{:.3f}',
        'K/9': '{:.2f}', 'BB/9': '{:.2f}', 'K/BB': '{:.2f}', 'HR/9': '{:.2f}',
        'H/9': '{:.2f}', 'R/9': '{:.2f}'        
    })
    st.dataframe(team_advanced_formatted, use_container_width=True, hide_index=True)

    st.subheader("Team Batted Ball & Plate Discipline", divider='gray')
    batted_columns = ['team', 'LD%', 'GB%', 'FB%', 'PopUp%', 'HR/FB%', 'Str%', 'SwStr%', 'Whiff%', 'CSW%', 'CStr%', 'F-Strike%']
    team_batted_formatted = team_data_adv_df[batted_columns].style.format({
        'LD%': '{:.1f}',
        'GB%': '{:.1f}',
        'FB%': '{:.1f}',
        'PopUp%': '{:.1f}',
        'P/IP': '{:.1f}',
        'SwStr%': '{:.1f}',
        'Whiff%': '{:.1f}',
        'Str%': '{:.1f}',
        'CSW%': '{:.1f}',
        'CStr%': '{:.1f}',
        'F-Strike%': '{:.1f}',
        'HR/FB%': '{:.1f}'
    })
    st.dataframe(team_batted_formatted, use_container_width=True, hide_index=True)

        # Plotting team-level metrics
    st.subheader("Team Performance Overview")
    selected_metric = st.selectbox("Select Metric for Visualization:", ['K%', 'BB%', 'K-BB%', 'K/9', 'BB/9', 'H/9', 'HR/9', 'HR/FB%', 'SwStr%', 'Whiff%', 'CSW%', 'FB%', 'GB%', 'LD%',])

    fig = px.bar(
    team_data_adv_df.sort_values(by=selected_metric, ascending=False), x='team', y=selected_metric,
        title=f"Team {selected_metric} Comparison", labels={'team': 'Team', selected_metric: selected_metric},
        color='team',
        color_discrete_map={
            'Aguilas de Mexicali': '#19255b',
            'Naranjeros de Hermosillo': '#fc8708',
            'Yaquis de Obregon': '#134489',
            'Mayos de Navojoa': '#fcef04',
            'Tomateros de Culiacan': '#701d45',
            'Venados de Mazatlan': '#ea0a2a',
            'Charros de Jalisco': '#b99823',
            'Sultanes de Monterrey': '#1f2344',
            'Caneros de los Mochis': '#10964c',
            'Algodoneros de Guasave': '#85a8e2'
        }
    )
    fig.update_layout(width=1000, height=500)
    fig.update_traces(marker=dict(line=dict(color='white', width=0.75)))
    st.plotly_chart(fig)