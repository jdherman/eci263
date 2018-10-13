from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt

# NFL Scheduling: Works now!! 11/23/16

# minimize the number of duplicates
# a duplicate is when a team is scheduled for two games in one week
def objective_function(S):
  duplicates = 0

  for w in range(17):
    teams = games[S==w, 1:].reshape(-1)
    duplicates += (teams.size - np.unique(teams).size)

  return duplicates

# print the final schedule to console
def prettyprint(S):
  names = np.genfromtxt('data/nfl-games-2012-string.csv', delimiter=',', dtype='str', usecols=[1,2], skip_header=1)
  print('\n ---SCHEDULE--- \n')

  for w in range(n_weeks):
    print('\n~ Week %d ~' % (w+1))
    game_id = games[S==w, 0]-1 # indexed from zero
  
    for g in names[game_id]:
      print('%s\tvs.\t%s' % (g[0],g[1]))

# save final schedule to CSV
def make_me_a_csv(S, filename):
  names = np.genfromtxt('data/nfl-games-2012-string.csv', delimiter=',', dtype='str', skip_header=1)
  names = np.hstack((names[:256], S.reshape(-1,1)))
  np.savetxt(filename, names, delimiter=',', fmt = '%s')


seed = 10 # seed 10: ~456000 NFE to converge
np.random.seed(seed)

games = np.loadtxt('data/nfl-games-2012-nobyes.csv', delimiter=',', skiprows=1, dtype='int32')
n_games = games.shape[0]
n_weeks = 17

# this is the requirement from the real schedule
games_per_week = [16,16,16,15,14,14,13,14,14,14,14,16,16,16,16,16,16]

# random initial schedule
S = [i for i in range(n_weeks) for _ in range(games_per_week[i])]
S = np.array(S)
S = np.random.permutation(S)
bestf = objective_function(S)
nfe = 0

# enumeration would require: 256! / (16!)^16
# (divide out all the swapping 1's,2's) - roughly 10^400

while bestf > 0:

  S_trial = np.copy(S) # do not operate on original list
  a = np.random.randint(n_games)
  b = np.random.randint(n_games)

  # swap a random pair
  S_trial[a],S_trial[b] = S_trial[b],S_trial[a]
  trial_f = objective_function(S_trial)

  if trial_f <= bestf: # minimizing. <= important!!!
    S = S_trial
    bestf = trial_f

  nfe += 1

  if nfe % 1000 == 0:
    print('NFE: %d, duplicates: %d' % (nfe, bestf))

print('%d NFE to converge.' % nfe)
prettyprint(S)
make_me_a_csv(S, 'nfl_schedule_seed_%d.csv' % seed)