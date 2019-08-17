import pandas as pd
import urllib.request, json
from datetime import datetime

API_MATCH_URL = 'https://api.chess.com/pub/match/'
API_PLAYER_URL = 'https://api.chess.com/pub/player/'

def get_match_data(match):
    if isinstance(match, (int, float, complex)) and not isinstance(match, bool):
        try:
            response = urllib.request.urlopen(API_MATCH_URL+str(match))
            data = json.loads(response.read())
        except:
            data = {}
    elif isinstance(match, str):
        try:
            response = urllib.request.urlopen(match)
            data = json.loads(response.read())
        except:
            data = {}
    elif isinstance(match, dict):
        if 'data' in match.keys():
            data = match['data']
        else:
            data = match
    else:
        data = {}
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

def get_teams_names(match, team_id=None):
    data = get_match_data(match)
    if team_id == 1:
        return data['teams']['team1']['name']
    elif team_id == 2:
        return data['teams']['team2']['name']
    else:
        return [data['teams']['team1']['name'], data['teams']['team2']['name']]

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
    match_data = get_match_data(match)
    num_boards = get_num_boards(match)
    match_status = get_match_status(match)
    boards = []

    if match_status == 'registration':
        pass
    else:
        for b_id in range(1, num_boards+1):
            response = urllib.request.urlopen(match_data['@id']+'/'+str(b_id))
            data = json.loads(response.read())
            boards.append(data)

    return boards

def get_match_elos_list(match, format='dict'):
    data = get_match_data(match)
    match_type = data['settings']['rules'] + '_' + data['settings']['time_class']

    teams_data = get_teams(match)
    teams = [teams_data['team1']['name'], teams_data['team2']['name']]
    players_team1 = [p['username'] for p in get_team_players(match, 1)]
    players_team2 = [p['username'] for p in get_team_players(match, 2)]
    boards_stats = []

    boards = get_match_boards(match)
    games_per_board = len(boards[0]['games'])
    for b in boards:
        players = list(b['board_scores'].keys())
        if players[0] in players_team2:
            players = players[::-1]
        player_1 = get_player_stats(players[0])
        player_2 = get_player_stats(players[1])
        elos = [player_1[match_type]['last']['rating'], player_2[match_type]['last']['rating']]
        rds = [player_1[match_type]['last']['rd'], player_2[match_type]['last']['rd']]

        if format == 'dict':
            boards_stats.append({'players': players, 'elos': elos, 'rds': rds})
        elif format == 'list':
            boards_stats.append([players[0], players[1], elos[0], elos[1], rds[0], rds[1]])
        #print(boards_stats[-1])

    return {'teams': teams, 'games_per_board': games_per_board, 'boards_stats': boards_stats}
