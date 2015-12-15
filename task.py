__author__ = 'Mingaleev Almaz'

import sys
import time
import viterbi
import numpy as np

def read_data(filename):
    f = open(filename, 'r')
    counts = f.readline()
    states_str, obs_str = counts.split()

    states_count = int(states_str)
    obs_count = int(obs_str)

    states = [int(x) for x in f.readline().split()]
    probs = [float(x) for x in f.readline().split()]
    observations = [int(x) for x in f.readline().split()]
    transition_probs = np.array([float(x) for x in np.array(f.readline().split())]).reshape((states_count, states_count)).tolist()
    emission_probs = np.array([float(x) for x in np.array(f.readline().split())]).reshape((states_count, states_count)).tolist()
    print('Most probable hidden states are: ')
    print(viterbi.FindHiddenStates(states, probs, observations, transition_probs, emission_probs))


def main(argv):
    c_start_time = time.time()
    read_data(argv[0])
    c_end_time = time.time()
    frac = c_end_time - c_start_time
    print('total time {0:.4f}'.format(frac))


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1:])
    else:        
        print("Usage: task.py <input_file>")
