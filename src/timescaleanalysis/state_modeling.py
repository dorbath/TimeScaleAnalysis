"State modeling"

import numpy as np


def generate_state_trajectory(trajectory, states, state_boundaries, fill_undef_states=True, finalFrameFinalState=True):
    """Generate state trajectory from simulated trajectory
    HERE: the trajectory must be of a single parameter
    Parameters
    ----------
    trajectory: array, single trajectory of a single parameter
    states: array, all possible states
    state_boundaries: array, lower and upper boundaries of each state in the same order as 'states'
    fill_undef_states: bool, set frames between two states to the last populated one

    Return
    ------
    temp_state_traj: array, generated state trajectory
    """
    temp_state_traj = np.full(len(trajectory), -1, dtype=np.int8)
    for idxS, sb in enumerate(state_boundaries):
        if sb[0] > sb[1]:
            lb, ub = sb[1], sb[0]
        else:
            lb, ub = sb[0], sb[1]
        temp_state_traj[(trajectory > lb) & (trajectory < ub)] = states[idxS]

    # Select frames in intermediate states
    temp_idx_transition = np.where(temp_state_traj == -1)
    # If first frame is not in a state, set it to the first populated one
    if np.size(temp_idx_transition[0]) > 0 and temp_idx_transition[0][0] == 0:
        temp_state_traj[0] = np.delete(temp_state_traj, temp_idx_transition)[0]
    if finalFrameFinalState:
        # NOTE: This is only needed if the systems ends always in the final state
        temp_state_traj[-1] = states[-1]

    # Select frames in intermediate states again due to overwriting of first and last frame
    temp_idx_transition = np.where(temp_state_traj == -1)
    # Smooth out intermediate positions between states
    if fill_undef_states:
        for i in temp_idx_transition[0]:
            temp_state_traj[i] = temp_state_traj[i-1]

    return temp_state_traj


def split_state_traj(state_traj, states, start=None, stop=None, coringTime=100):
    """Slice state trajectories into transitions from 'start' to 'stop'."""
    if start is None:
        start = states[0]
    if stop is None:
        stop = states[-1]

    ## NOTE: If final frame before the trajectory leaves
    ## should be used, add flag 'first=False' 

    # store indices of slices
    ret_first_idx, ret_final_idx = [], []
    first_state_idx = np.where(state_traj[0]==start)[0]
    temp_start = state_coring(state_traj[0], first_state_idx, coringTime=coringTime)
    temp_state_traj = state_traj[0][temp_start:]

    final_state_idx = np.where(temp_state_traj==stop)[0]
    temp_final = state_coring(temp_state_traj, final_state_idx, coringTime=coringTime)
    temp_next_final = temp_final
    temp_final += temp_start

    temp_state_arr = []
    while len(final_state_idx) != 0:
        ret_first_idx.append(temp_start)
        ret_final_idx.append(temp_final)
        temp_state_arr.append(temp_state_traj[0:temp_next_final+1]) # save sliced traj
        temp_state_traj = temp_state_traj[temp_next_final+1:] # eliminate slice from rest of traj
        first_state_idx = np.where(temp_state_traj==start)[0] # find new start position
        if len(first_state_idx) == 0: # break if none found
            break
        temp_next_start = state_coring(temp_state_traj, first_state_idx, coringTime=coringTime)
        temp_start = temp_final+temp_next_start+1
        temp_state_traj = temp_state_traj[temp_next_start:] # shift traj to start position
        final_state_idx = np.where(temp_state_traj==stop)[0] #find new finish point
        if len(final_state_idx) == 0: # break if none found
            break
        temp_next_final = state_coring(temp_state_traj, final_state_idx, coringTime=coringTime)
        temp_final += temp_next_start+temp_next_final+1
    
    return temp_state_arr, ret_first_idx, ret_final_idx


def state_coring(state_traj, state_idx, coringTime=100, first=True):
    """Check that the state traj remains a significant time in the cored state

    Parameters
    ----------
    state_traj: state trajectory
    state_idx: index array for state to verify coring condition
    coringTime: number of frames in state
    first: boolean  (True)  return first frame in state + 1/2 coringTime 
                    (False) return final frame in state
    """
    state_pos = state_traj[state_idx[0]]
    if np.all(state_traj[state_idx[0]:state_idx[0]+coringTime] == state_pos):
        ret_idx = state_idx[0]
    else:
        for sIdx in state_idx:
            if np.all(state_traj[sIdx:sIdx+coringTime] == state_pos):
                ret_idx = sIdx
                break
    ## If first frame in transition is searched, idx is found
    if 'ret_idx' not in locals():
        print(f'No frame fullfills coring condition of {coringTime}frames. Choose smaller coring time')   
        sys.exit()

    if first:
        return ret_idx + int(coringTime/2)
    else:
        for s in state_idx[1:]:
            if s-ret_idx == 1:
                ret_idx = s
            else:
                break
        return ret_idx