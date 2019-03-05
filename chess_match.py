import numpy as np
import matplotlib.pyplot as plt

def get_draw_rate(R1, R2):
    rd = np.abs(R1 - R2)
    ra = (R1 + R2)/2
    
    dr = -rd/32.49 + np.exp((ra-2254.7)/208.49)+23.87
    return dr/100

def get_result_expectation(R1, R2):
    rd = R1 - R2
    dr = get_draw_rate(R1, R2)
    
    Pw = 1 / (1 + np.power(10, -1*rd/400)) - 0.5*dr
    Pl = 1 / (1 + np.power(10, rd/400)) - 0.5*dr
    Pd = dr
    return Pw, Pl, Pd

def plot_match_elo_diff_hist(M, match_name='match', savefig=True):
    rd_vals = []
    for g in M:
        rd_vals.append(g[0]-g[1])

    fig = plt.figure(figsize=[15,5])
    plt.hist(rd_vals)
    plt.title('Histogram of ELO differences')
    plt.xlabel('ELO difference')
    plt.ylabel('frequency')
    if savefig:
        plt.savefig('./outputs/'+match_name+'_elo_diff_hist.png')
    plt.show()
    
    return

def plot_match_elos(M, match_name='match', savefig=True):
    
    n = np.arange(1,M.shape[0]+1)
    cmap = plt.get_cmap('Set1')
    cmap_A = cmap(1)
    cmap_B = cmap(2)

    plt.figure(figsize=(15,5))
    
    plt.plot(n, M[:,0], color=cmap_A, label='Team A')
    plt.plot(n, M[:,1], color=cmap_B, label='Team B')
    
    if M.shape[1] > 2:
        plt.fill_between(n, M[:,0] - M[:,2], M[:,0] + M[:,2], color=cmap_A, alpha=0.2)
        plt.fill_between(n, M[:,1] - M[:,3], M[:,1] + M[:,3], color=cmap_B, alpha=0.2)
    
    plt.xlim([1, len(n)])
    plt.xlabel('# board')
    plt.ylabel('ELO')
    plt.title('ELO per board')
    plt.legend()
    if savefig:
        plt.savefig('./outputs/'+match_name+'_elos.png')
    plt.show()
    
    return

def plot_match_elo_diff(M, match_name='match', savefig=True):
    n = np.arange(1,M.shape[0]+1)
    cmap = plt.get_cmap('Set1')
    cmap_diff = cmap(3)
    
    diff = M[:,0] - M[:,1]
    
    plt.figure(figsize=(15,5))
    plt.plot(n, diff, color=cmap_diff, label='diff')
    
    if M.shape[1] > 2:
        diff_var = np.sqrt(np.power(M[:,2],2) + np.power(M[:,3],2))
        plt.fill_between(n, diff - diff_var, diff + diff_var, color=cmap_diff, alpha=0.2)
    
    plt.xlim([1, len(n)])
    plt.xlabel('# board')
    plt.ylabel('ELO difference')
    plt.title('ELO difference per board')
    plt.legend()
    if savefig:
        plt.savefig('./outputs/'+match_name+'_elo_diff.png')
    plt.show()
    
    return

def match_results(M, seed=None, games_per_board=2):
    np.random.seed(seed)
    
    m_trial = np.random.uniform(0.0, 1.0, size=M.shape[0]*games_per_board)
    
    M_results = np.empty([M.shape[0], 2])
    for idx, g in enumerate(M):
        
        ELO_A = g[0]
        ELO_B = g[1]
        
        if len(g) > 2:
            ELO_A = np.random.normal(loc=ELO_A, scale=g[2])
            ELO_B = np.random.normal(loc=ELO_B, scale=g[3])
        
        Pw, Pl, Pd = get_result_expectation(ELO_A, ELO_B)
        
        g_point = 0
        
        for bg in range(games_per_board):
            res = m_trial[idx*games_per_board + bg]

            if res < Pl:
                # lose
                g_point = g_point
            elif res < (Pl + Pd):
                g_point = g_point + 0.5
            else:
                g_point = g_point + 1

        #M_results.append(np.asarray([g_point, games_per_board-g_point]))
        M_results[idx] = np.asarray([g_point, games_per_board-g_point])
        
    return np.asarray(M_results)
    