import chesscom as chess
import chess_match as cm
import pandas as pd
import json
from datetime import datetime
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

DATA_FOLDER = './data'
OUTPUTS_FOLDER = './outputs'
RESULTS_FOLDER = './results'

import re
_slugify_strip_re = re.compile(r'[^\w\s-]')
_slugify_hyphenate_re = re.compile(r'[-\s]+')
def _slugify(value):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.

    From Django's "django/template/defaultfilters.py".
    """
    import unicodedata
    if not isinstance(value, str):
        value = str(value)
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore')
    value = value.decode('utf-8')
    value = str(_slugify_strip_re.sub('', value).strip().lower())
    return _slugify_hyphenate_re.sub('-', value)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-id", dest='match_id', type=str, help="Number of match id")
    parser.add_argument("-url", dest='match_url', type=str, help="URL of the match")
    parser.add_argument("-N", dest='N_predict', type=int, default=1000, help="Number of trials to predict match resutl")
    parser.add_argument("-Nb", dest='N_bias', type=int, default=1000, help="Number of trials to predict match biased resutl")
    parser.add_argument("-bias", dest='bias', type=float, default=0.0, help="ELOs bias for Team A")
    parser.add_argument("-u", action='store_true', help="Sorce update data from web")
    parser.add_argument("-plot", action='store_true', default=False, help="Show plots")
    args = parser.parse_args()

    # Turn interactive plotting off
    if not args.plot:
        plt.ioff()

    # check folders
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
    if not os.path.exists(OUTPUTS_FOLDER):
        os.makedirs(OUTPUTS_FOLDER)
    if not os.path.exists(OUTPUTS_FOLDER):
        os.makedirs(OUTPUTS_FOLDER)

    # resolve match_id
    if args.match_id:
        match_id = int(args.match_id)
    elif args.match_url:
        match_id = args.match_url

    # get data about match
    data = chess.get_match_data(match_id)

    #print('\n')
    match_name = _slugify(chess.get_match_name(match_id))
    print('\nMatch info:')
    print('\tName:\t{}'.format(chess.get_match_name(match_id)))

    teams_names = chess.get_teams_names(match_id)
    print('\tTeam A:\t{}'.format(teams_names[0]))
    print('\tTeam B:\t{}'.format(teams_names[1]))

    # Get ELOs list into matrix (M)
    print('\nReading ELOs list')
    match_stats_filename = DATA_FOLDER+'/'+match_name+'_match_stats.xlsx'

    if os.path.exists(match_stats_filename) and not args.u:
        print('\tA backup file was found!')
        print('\tLoading from file {} ...'.format(match_stats_filename))
        match_stats_list_df = pd.read_excel(match_stats_filename)
        match_stats_list_np = match_stats_list_df.to_numpy()
        M = np.array([s[2:] for s in match_stats_list_np]).astype(int)
        print('\tNote: if you want update backup file, use -u argument.')
        print('\tDone!')
    else:
        print('\tLoading from web ...')
        match_stats_list = chess.get_match_elos_list(data, format='list')
        print('\tDone!')

        print('\tSaving backup file{}'.format(match_stats_filename))
        match_stats_list_df = pd.DataFrame.from_dict(match_stats_list['boards_stats'])
        match_stats_list_df.to_excel(match_stats_filename, index=False)

        M = np.array([s[2:] for s in match_stats_list['boards_stats']])

    ## Match statistics
    rd_vals = []
    for g in M:
        rd_vals.append(g[0]-g[1])

    # ploting ELOs info
    cm.plot_match_elo_diff_hist(M, match_name=match_name, teams=teams_names, show_plot=args.plot)
    cm.plot_match_elos(M, match_name=match_name, teams=teams_names, show_plot=args.plot)
    cm.plot_match_elo_diff(M, match_name=match_name, teams=teams_names, show_plot=args.plot)

    ## N matches simulation
    print('\nSimulation of match - Result prediction:')
    N = args.N_predict

    M_scores = np.empty([N,4])

    for n in range(N):
        M_results = cm.match_results(M)

        A_score = np.sum(M_results[:,0])
        B_score = np.sum(M_results[:,1])
        Dif_score = A_score - B_score
        w_d_l = 1 if Dif_score > 0 else -1 if Dif_score < 0 else 0

        M_scores[n] = [A_score, B_score, Dif_score, w_d_l]

    #print(M_scores)

    A_score_mean = np.mean(M_scores[:,0])
    B_score_mean = np.mean(M_scores[:,1])
    A_score_std = np.std(M_scores[:,0])
    B_score_std = np.std(M_scores[:,1])

    fig = plt.figure(figsize=[15,5])
    plt.hist(M_scores[:,2], bins=np.arange(np.min(M_scores[:,2]),np.max(M_scores[:,2])+1), density=False)
    plt.title('Final score (difference) histogram for {}'.format(teams_names[0]))
    plt.xlabel('Match score difference')
    plt.ylabel('Frequency')
    plt.savefig(OUTPUTS_FOLDER+'/'+match_name+'_final_score_hist.png')
    if not args.plot:
        plt.close(fig)
    plt.show()


    fig = plt.figure(figsize=[15,5])
    p_res, bins, patches = plt.hist(M_scores[:,3], bins=[-1,0,1,2], density=True, rwidth=0.9, color='#607c8e')
    plt.title('Lose/Draw/Win Probabilities for {}'.format(teams_names[0]))
    plt.xticks(np.arange(3)-0.5, ('Lose', 'Draw', 'Win',))
    plt.ylabel('Probability')
    plt.savefig(OUTPUTS_FOLDER+'/'+match_name+'_LDW_probs.png')
    if not args.plot:
        plt.close(fig)
    plt.show()
    #print(p_res)

    print('Team A ({}):'.format(teams_names[0]))
    print('\tWin chances = {:0.2f} %'.format(p_res[2]*100))
    print('\tDraw chances = {:0.2f} %'.format(p_res[1]*100))
    print('\tLose chances = {:0.2f} %'.format(p_res[0]*100))


    cmap = plt.get_cmap('Set1')
    cmap_A = cmap(1)
    cmap_B = cmap(2)

    fig = plt.figure(figsize=[15,5])
    plt.hist(M_scores[:,0], bins=np.arange(np.min(M_scores[:,0]),np.max(M_scores[:,0])+1),
             density=False, label='{}'.format(teams_names[0]), color=cmap_A, alpha=0.8)
    plt.hist(M_scores[:,1], bins=np.arange(np.min(M_scores[:,1]),np.max(M_scores[:,1])+1),
             density=False, label='{}'.format(teams_names[1]), color=cmap_B, alpha=0.8)

    plt.axvline(x=np.mean(M_scores[:,0]), color=cmap_A)
    plt.axvline(x=np.mean(M_scores[:,1]), color=cmap_B)
    plt.axvline(x=M.shape[0], color='grey')

    plt.legend()
    plt.title('Final score probability')
    plt.xlabel('Match score')
    plt.ylabel('Frequency')
    plt.savefig(OUTPUTS_FOLDER+'/'+match_name+'_final_score_prob.png')
    if not args.plot:
        plt.close(fig)
    plt.show()

    print(r'Expected final score = {:0.2f} (±{:0.2f}) - {:0.2f} (±{:0.2f})'.format(A_score_mean, A_score_std, B_score_mean, B_score_std))
    print(r'Expected effectiveness = {:0.2f} % - {:0.2f} %'.format(100*A_score_mean/(A_score_mean + B_score_mean), 100*B_score_mean/(A_score_mean + B_score_mean)))

    ## Variance over Team A ELOs
    print('\nCalculation of Variance over Team A ELOs')
    N = 400 # trials per match simulation

    S = 21 # elo_var_steps
    elo_var_min = -100.0
    elo_var_max = 100.0

    elo_var_vec = np.linspace(elo_var_min, elo_var_max, num=S)

    M_scores = np.empty([S,N,4])

    for s in range(S):
        M_aux = np.copy(M)
        M_aux[:,0] = M[:,0] + elo_var_vec[s]

        for n in range(N):
            M_results = cm.match_results(M_aux)

            A_score = np.sum(M_results[:,0])
            B_score = np.sum(M_results[:,1])
            Dif_score = A_score - B_score
            w_d_l = 1 if Dif_score > 0 else -1 if Dif_score < 0 else 0

            M_scores[s,n] = [A_score, B_score, Dif_score, w_d_l]

    A_score_means = np.mean(M_scores[:,:,0], axis=1)
    B_score_means = np.mean(M_scores[:,:,1], axis=1)
    A_score_stds = np.std(M_scores[:,:,0], axis=1)
    B_score_stds = np.std(M_scores[:,:,1], axis=1)

    cmap = plt.get_cmap('Set1')
    cmap_A = cmap(1)
    cmap_B = cmap(2)

    fig = plt.figure(figsize=(10,5))
    plt.plot(elo_var_vec, A_score_means, color=cmap_A, label='{}'.format(teams_names[0]))
    plt.plot(elo_var_vec, B_score_means, color=cmap_B, label='{}'.format(teams_names[1]))
    plt.fill_between(elo_var_vec, A_score_means - A_score_stds, A_score_means + A_score_stds, color=cmap_A, alpha=0.2)
    plt.fill_between(elo_var_vec, B_score_means - B_score_stds, B_score_means + B_score_stds, color=cmap_B, alpha=0.2)
    plt.title('Predicted Score vs ELO bias')
    plt.xlabel('{} ELO bias'.format(teams_names[0]))
    plt.ylabel('Score')
    plt.xlim([elo_var_min, elo_var_max])
    plt.legend()
    plt.grid()
    plt.savefig(OUTPUTS_FOLDER+'/'+match_name+'_score_elo_bias.png')
    if not args.plot:
        plt.close(fig)
    plt.show()

    cmap = plt.get_cmap('Set1')
    cmap_A = cmap(1)
    cmap_B = cmap(2)

    fig = plt.figure(figsize=(10,5))
    plt.plot(elo_var_vec, A_score_means/(A_score_means+B_score_means), color=cmap_A, label='{}'.format(teams_names[0]))
    plt.plot(elo_var_vec, B_score_means/(A_score_means+B_score_means), color=cmap_B, label='{}'.format(teams_names[1]))

    plt.title('Predicted Effectivness vs ELO bias')
    plt.xlabel('{} ELO bias'.format(teams_names[0]))
    plt.ylabel('Effectivness')
    plt.xlim([elo_var_min, elo_var_max])
    plt.legend()
    plt.grid()
    plt.savefig(OUTPUTS_FOLDER+'/'+match_name+'_effec_elo_bias.png')
    if not args.plot:
        plt.close(fig)
    plt.show()

    M_scores[7,:,3]
    lose_vec = np.sum(M_scores[:,:,3]<0, axis=1)
    draw_vec = np.sum(M_scores[:,:,3]==0, axis=1)
    win_vec = np.sum(M_scores[:,:,3]>0, axis=1)
    N_boards = M_scores.shape[1]

    fig = plt.figure(figsize=(10,5))
    plt.plot(elo_var_vec, win_vec/N_boards, label='win')
    plt.plot(elo_var_vec, draw_vec/N_boards, label='draw')
    plt.plot(elo_var_vec, lose_vec/N_boards, label='lose')
    plt.title('{} Probabilities'.format(teams_names[0]))
    plt.xlabel('{} ELO bias'.format(teams_names[0]))
    plt.ylabel('probability')
    plt.xlim([elo_var_min, elo_var_max])
    plt.legend()
    plt.grid()
    plt.savefig(OUTPUTS_FOLDER+'/'+match_name+'_prob_elo_bias.png')
    if not args.plot:
        plt.close(fig)
    plt.show()

    ## Show Biased Analysis
    Team_A_bias = args.bias

    if Team_A_bias != 0:
        print('\nSimulation of BIASED match - Result prediction:')
        print('\tTeam A bias = {}'.format(Team_A_bias))
        M_biased = np.copy(M)
        M_biased[:,0] = M[:,0] + Team_A_bias

        cm.plot_match_elo_diff_hist(M_biased, match_name=match_name+'_biased', teams=teams_names, show_plot=args.plot)
        cm.plot_match_elos(M_biased, match_name=match_name+'_biased', teams=teams_names, show_plot=args.plot)
        cm.plot_match_elo_diff(M_biased, match_name=match_name+'_biased', teams=teams_names, show_plot=args.plot)

        ## Biased Match simulation
        N = args.N_bias

        M_scores = np.empty([N,4])

        for n in range(N):
            M_results = cm.match_results(M_biased)

            A_score = np.sum(M_results[:,0])
            B_score = np.sum(M_results[:,1])
            Dif_score = A_score - B_score
            w_d_l = 1 if Dif_score > 0 else -1 if Dif_score < 0 else 0


            M_scores[n] = [A_score, B_score, Dif_score, w_d_l]

        #print(M_scores)

        A_score_mean = np.mean(M_scores[:,0])
        B_score_mean = np.mean(M_scores[:,1])
        A_score_std = np.std(M_scores[:,0])
        B_score_std = np.std(M_scores[:,1])

        ## Ploting results
        fig = plt.figure(figsize=[15,5])
        plt.hist(M_scores[:,2], bins=np.arange(np.min(M_scores[:,2]),np.max(M_scores[:,2])+1), density=False)
        plt.title('Final score (difference) histogram for {} biased'.format(teams_names[0]))
        plt.xlabel('Match score difference')
        plt.ylabel('Frequency')
        plt.savefig(OUTPUTS_FOLDER+'/'+match_name+'_final_score_hist_biased.png')
        if not args.plot:
            plt.close(fig)
        plt.show()

        fig = plt.figure(figsize=[15,5])
        p_res, bins, patches = plt.hist(M_scores[:,3], bins=[-1,0,1,2], density=True, rwidth=0.9, color='#607c8e')
        plt.title('Lose/Draw/Win Probabilities for {} biased'.format(teams_names[0]))
        plt.xticks(np.arange(3)-0.5, ('Lose', 'Draw', 'Win',))
        plt.ylabel('Probability')
        plt.savefig(OUTPUTS_FOLDER+'/'+match_name+'_LDW_probs_biased.png')
        if not args.plot:
            plt.close(fig)
        plt.show()
        #print(p_res)

        print('Team A ({}):'.format(teams_names[0]))
        print('\tWin chances = {:0.2f} %'.format(p_res[2]*100))
        print('\tDraw chances = {:0.2f} %'.format(p_res[1]*100))
        print('\tLose chances = {:0.2f} %'.format(p_res[0]*100))

        cmap = plt.get_cmap('Set1')
        cmap_A = cmap(1)
        cmap_B = cmap(2)

        fig = plt.figure(figsize=[15,5])
        plt.hist(M_scores[:,0], bins=np.arange(np.min(M_scores[:,0]),np.max(M_scores[:,0])+1),
                 density=False, label='{}'.format(teams_names[0]), color=cmap_A, alpha=0.8)
        plt.hist(M_scores[:,1], bins=np.arange(np.min(M_scores[:,1]),np.max(M_scores[:,1])+1),
                 density=False, label='{}'.format(teams_names[1]), color=cmap_B, alpha=0.8)

        plt.axvline(x=np.mean(M_scores[:,0]), color=cmap_A)
        plt.axvline(x=np.mean(M_scores[:,1]), color=cmap_B)
        plt.axvline(x=M.shape[0], color='grey')

        plt.legend()
        plt.title('Final score probability')
        plt.xlabel('Match score')
        plt.ylabel('Frequency')
        plt.savefig(OUTPUTS_FOLDER+'/'+match_name+'_final_score_prob_biased.png')
        if not args.plot:
            plt.close(fig)
        plt.show()

        print(r'Expected final score = {:0.2f} (±{:0.2f}) - {:0.2f} (±{:0.2f})'.format(A_score_mean, A_score_std, B_score_mean, B_score_std))
        print(r'Expected effectiveness = {:0.2f} % - {:0.2f} %'.format(100*A_score_mean/(A_score_mean + B_score_mean), 100*B_score_mean/(A_score_mean + B_score_mean)))

    print('\nDone!\n')

if __name__ == '__main__':
    main()
