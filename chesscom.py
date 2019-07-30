import pandas as pd
import urllib.request, json
from datetime import datetime

API_MATCH_URL = 'https://api.chess.com/pub/match/'
API_PLAYER_URL = 'https://api.chess.com/pub/player/'

def get_match_data(match):
    response = urllib.request.urlopen(API_MATCH_URL+str(match))
    data = json.loads(response.read())
    return data

def get_match_name(match):
    data = get_match_data(match)
    return data['name']

def get_num_boards(match):
    data = get_match_data(match)
    return data['boards']

def get_match_status(match):
    data = get_match_data(match)
    return data['status']

def get_teams(match, team_id=None):
    data = get_match_data(match)
    if team_id == 1:
        return data['teams']['team1']
    elif team_id == 2:
        return data['teams']['team2']
    else:
        return data['teams']

def get_team_id(match, team_name):
    data = get_match_data(match)
    if team_name == data['teams']['team1']['name']:
        return 1
    elif team_name == data['teams']['team2']['name']:
        return 2
    else:
        return 0

def get_team_players(match, team):
    if isinstance(team, int):
        team_id = 'team'+str(team)
    elif isinstance(team, str):
        if team == 'team1' or team == 'team2':
            team_id = team
        else:
            # team = team name
            t_id = get_team_id(match, team)
            if t_id == 0:
                return []
            else:
                team_id = 'team'+str(t_id)
    else:
        return []
    
    data = get_match_data(match)
    return data['teams'][team_id]['players']

def get_player_stats(username):
    response = urllib.request.urlopen(API_PLAYER_URL+username+'/stats')
    stats = json.loads(response.read())
    return stats

def get_players_stats(match, team):
    players = get_team_players(match, team)
    stats = []
    for p in players:
        stats.append(get_player_stats(p['username']))
    return stats

def get_match_boards(match):
    num_boards = get_num_boards(match)
    match_status = get_match_status(match)
    boards = []
    
    if match_status == 'registration':
        pass
    else:
        for b_id in range(1, num_boards+1):
            response = urllib.request.urlopen(API_MATCH_URL+str(match)+'/'+str(b_id))
            data = json.loads(response.read())
            boards.append(data)
    
    return boards
        