
import requests
URL = 'https://fbref.com/en/players/4e99492b/Becky-Edwards#all_stats_standard'
r = requests.get(URL)
print(r.content)