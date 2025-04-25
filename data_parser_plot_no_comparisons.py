import json
import numpy as np
import matplotlib.pyplot as plt

# Load data
file_path = "solution_concept_nash_results.json"
with open(file_path, 'r') as f:
    data = json.load(f)

def markowitz_utility(w, Sigma, lambda_mu):
    w = np.array(w)
    Sigma = np.array(Sigma)
    lambda_mu = np.array(lambda_mu)
    return np.dot(w, np.dot(Sigma, w)) - np.dot(lambda_mu, w)


def compute_spread(final_points):
    """
    final_points: List of n-dimensional points (each point is a list or array)
    Returns the average Euclidean distance to the centroid
    """
    arr = np.array(final_points)
    if len(arr) == 0:
        return 0.0
    centroid = np.mean(arr, axis=0)
    distances = np.linalg.norm(arr - centroid, axis=1)
    return np.mean(distances)


def process_multiple_progressions(progression_list, sigma_list, lambda_list, max_len=1000):
    """
    progression_list: list of progressions from all tests
    sigma_list: list of Sigma_set for each test
    lambda_list: list of lambda_mu for each test
    """
    utility_curves = []
    counter = 0
    for progression, Sigma_set, lambda_mu_set in zip(progression_list, sigma_list, lambda_list):
        if not progression:
            continue
        # print("Doing Test: ", counter)
        counter += 1
        padded_progression = progression[:]
        if len(padded_progression) < max_len:
            padded_progression += [padded_progression[-1]] * (max_len - len(padded_progression))
        else:
            padded_progression = padded_progression[:max_len]

        step_utilities = []
        for state in padded_progression:
            agent_utils = [
                markowitz_utility(state, Sigma_set[i], lambda_mu_set[i])
                for i in range(len(Sigma_set))
            ]
            step_utilities.append(agent_utils)  # average over agents
        utility_curves.append(step_utilities)  # one test's curve

    return np.mean(np.array(utility_curves), axis=0) if utility_curves else np.zeros(max_len)

def process_progressions(state_progressions, Sigma_set, lambda_set, max_len=1000):
    num_agents = len(Sigma_set)
    utilities = []

    for progression in state_progressions:
        padded_progression = progression + [progression[-1]] * (max_len - len(progression))
        step_utilities = []
        for state in padded_progression:
            agent_utils = [
                markowitz_utility(state, Sigma_set[i], lambda_set[i])
                for i in range(num_agents)
            ]
            step_utilities.append(np.mean(agent_utils))  # avg over agents
        utilities.append(step_utilities)  # collect this test's avg utilities

    return np.mean(utilities, axis=0)  # avg across tests

# Agent-stock combinations
num_agents_list = ["2", "3", "5", "10", "20"]
num_stocks_list = ["5", "10", "20", "50"]

# Iterate and plot
for a in num_agents_list:
    for s in num_stocks_list:
        if a not in data or s not in data[a]:
            continue

        progression_ours = []
        progression_nash = []
        Sigma_list_all = []
        Lambda_list_all = []
        ours_final_points = []  # list of final_simplex points for "ours"
        nash_final_points = []  # list of nbs_simplex points for "nash"

        for entry in data[a][s]:
            # print("Entry Size: ", len(entry))
            (
                Sigma_set_list, lambda_mu_set_list, final_point, nbs_point, starting_state_w, solution_set_np, state_progression_ours, state_progression_nash
            ) = entry
            ours_final_points.append(state_progression_ours[-1])
            nash_final_points.append(state_progression_nash[-1])

            progression_ours.append(state_progression_ours)
            progression_nash.append(state_progression_nash)
            
            Sigma_list_all.append(Sigma_set_list)
            Lambda_list_all.append(lambda_mu_set_list)
        our_spread = compute_spread(ours_final_points)
        nash_spread = compute_spread(nash_final_points)

        print("Collecting data for: ", a, s)
        print(f"Ours Spread: {our_spread:.6f}")
        print(f"Nash Spread: {nash_spread:.6f}")
        # print("Collecting data for: ", a, s)
        avg_ours = process_multiple_progressions(progression_ours, Sigma_list_all, Lambda_list_all)
        avg_nash = process_multiple_progressions(progression_nash, Sigma_list_all, Lambda_list_all)
        agent_1_curve_ours = avg_ours[:, 0]  # shape: (time_steps,)
        agent_2_curve_ours = avg_ours[:, 1]  # shape: (time_steps,)

        agent_1_curve_nash = avg_nash[:, 0]  # shape: (time_steps,)
        agent_2_curve_nash = avg_nash[:, 1]  # shape: (time_steps,)
        # Plot
        plt.figure()
        x = np.arange(1000)
        plt.plot(x, agent_1_curve_ours, label="Ours Agent 1")
        plt.plot(x, agent_2_curve_ours, label="Ours Agent 2")

        plt.plot(x, agent_1_curve_nash, label="Nash Agent 1")
        plt.plot(x, agent_2_curve_nash, label="Nash Agent 2")

        plt.title(f"Average Utility Progression ({a} Agents, {s} Stocks)")
        plt.xlabel("Iteration")
        plt.ylabel("Average Agent Utility")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"utility_progression_agents_{a}_stocks_{s}.png", dpi=300)
        plt.close()