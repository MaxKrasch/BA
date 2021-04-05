pb1_ground_only_simple

Reward Function: penalty = 0
            if not next_state[24] and not next_state[27] and next_state[25] and next_state[26]:
                penalty = -1

            if not next_state[25] and not next_state[26] and next_state[27] and next_state[24]:
                penalty = -1

            reward = reward - 0.2 * penalty


Funktioniert!!! hoppelt vorwaerts


