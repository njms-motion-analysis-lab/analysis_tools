import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# Load the spreadsheet
file_path = "/Users/stephenmacneille/acl_soccer_before.xlsx"
spreadsheet = pd.read_excel(file_path)

# Filter out rows where 'Stats Link' ends with 'all_stats_keeper'
spreadsheet = spreadsheet.dropna(subset=['Stats Link'])
spreadsheet = spreadsheet[~spreadsheet['Stats Link'].str.endswith('all_stats_keeper')]

# Function to fetch player data from fbref
def fetch_player_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract relevant stats from Playing Time table
    playing_time_table = soup.select_one('table#stats_playing_time_dom_lg')
    shooting_table = soup.select_one('table#stats_shooting_dom_lg')

    stats = {'Pre': {}, 'Post': {}}
    
    # Parse the tables
    def parse_table(table):
        data = []
        if table:
            rows = table.select('tbody tr')
            for row in rows:
                season = row.select_one('th[data-stat="year_id"]').text if row.select_one('th[data-stat="year_id"]') else ''
                stats = {col['data-stat']: col.text for col in row.select('td')}
                stats['season'] = season
                data.append(stats)
        return data
    
    playing_time_stats = parse_table(playing_time_table)
    shooting_stats = parse_table(shooting_table)
    
    return playing_time_stats, shooting_stats

# List to store player data
players_data = []

# Iterate over each player in the spreadsheet
for index, row in spreadsheet.iterrows():
    try:
        player_name = row['Player']
        stats_link = row['Stats Link']
        date_of_injury = row['Date of Injury']
        year_prior = int(row['Year Prior?']) if not pd.isna(row['Year Prior?']) else None
        year_after = int(row['Year After?']) if not pd.isna(row['Year After?']) else None
        
        if year_prior is None or year_after is None:
            print(f"Skipping {player_name} due to missing Year Prior or Year After values.")
            continue
        
        print(f"Fetching data for {player_name} from {stats_link}")
        
        # Fetch data
        playing_time_stats, shooting_stats = fetch_player_data(stats_link)
        
        # Divide data into pre and post injury
        pre_stats, post_stats = {'playing_time': [], 'shooting': []}, {'playing_time': [], 'shooting': []}
        
        for stats in playing_time_stats:
            season_year = stats['season'][:4]
            if season_year.isdigit():
                if int(season_year) <= year_prior:
                    pre_stats['playing_time'].append(stats)
                else:
                    post_stats['playing_time'].append(stats)
        
        for stats in shooting_stats:
            season_year = stats['season'][:4]
            if season_year.isdigit():
                if int(season_year) <= year_prior:
                    pre_stats['shooting'].append(stats)
                else:
                    post_stats['shooting'].append(stats)

        # Calculate averages
        def calculate_averages(data):
            if not data:
                return {}
            df = pd.DataFrame(data)
            df = df.apply(pd.to_numeric, errors='coerce')
            return df.mean().to_dict()

        pre_averages = {
            'playing_time': calculate_averages(pre_stats['playing_time']),
            'shooting': calculate_averages(pre_stats['shooting'])
        }
        post_averages = {
            'playing_time': calculate_averages(post_stats['playing_time']),
            'shooting': calculate_averages(post_stats['shooting'])
        }
        
        player_stats = {
            'Player': player_name,
            'Pre': pre_averages,
            'Post': post_averages
        }
        
        # Append to the list
        players_data.append(player_stats)
        
        # Sleep to avoid overloading the server
        time.sleep(6.1)  # Adjust the timing as necessary
    
    except Exception as e:
        print(f"Error fetching data for {player_name}: {e}")

# Convert list to DataFrame
players_df = pd.DataFrame(players_data)

# Save the DataFrame to a new Excel file
output_file_path = "/Users/stephenmacneille/acl_soccer_after.xlsx"
players_df.to_excel(output_file_path, index=False)

print(f"Data saved to {output_file_path}")

# Display the first few rows of the result
print(players_df.head())