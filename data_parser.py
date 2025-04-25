import json
import numpy as np
from itertools import combinations
import math

def markowitz_utility(w, Sigma, lambda_mu):
    w = np.array(w)
    Sigma = np.array(Sigma)
    lambda_mu = np.array(lambda_mu)
    return np.dot(w, np.dot(Sigma, w)) - np.dot(lambda_mu, w)

def calculate_nash_product(w, Sigma, lambda_mu, disagreement = -1.0):
    return np.dot(lambda_mu, w) - np.dot(w, np.dot(Sigma, w)) - disagreement

def average_pairwise_distance(solution_set):
    arr = np.array(solution_set)
    num_agents = arr.shape[0]
    if num_agents < 2:
        return 0.0
    distances = [
        np.linalg.norm(arr[i] - arr[j])
        for i, j in combinations(range(num_agents), 2)
    ]
    return np.mean(distances)

def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

with open('solution_concept_nash_results.json', 'r') as f:
    data = json.load(f)

num_agents = ["2", "3", "5", "10", "20"]
num_stocks = ["5", "10", "20", "50"]

for a in num_agents:
    for s in num_stocks:
        print(f"Parsing Data for {a} agents and {s} stocks")
        avg_pairwise_dists = []
        dist_to_nbs_list = []
        dist_from_starting_list = []
        dist_from_starting_nash = []
        nash_solutions_list = []
        our_solutions_list = []

        # NEW: Utility-based metrics
        utility_pairwise_ours = []
        utility_pairwise_nash = []
        utility_dist_ours_vs_nash = []
        utility_rel_dist_ours_vs_nash = []
        relative_l1_errors = []
        for entry in data[a][s]:
            (
                Sigma_set_list, lambda_mu_set_list, final_point, nbs_point, starting_state_w, solution_set_np, state_progression_ours, state_progression_nash
            ) = entry
            
            # Existing solution space metrics
            epsilon = 1e-8  # for stability

            nash_solutions_list.append(nbs_point)
            our_solutions_list.append(final_point)
            utility_values_ours = np.array([markowitz_utility(final_point, Sigma_set_list[i], lambda_mu_set_list[i]) for i in range(len(Sigma_set_list))])
            utility_values_nash = np.array([markowitz_utility(nbs_point, Sigma_set_list[i], lambda_mu_set_list[i]) for i in range(len(Sigma_set_list))])
            utility_values_start = [markowitz_utility(starting_state_w, Sigma_set_list[i], lambda_mu_set_list[i]) for i in range(len(Sigma_set_list))]
            utility_values_opt = [markowitz_utility(starting_state_w, Sigma_set_list[i], lambda_mu_set_list[i]) for i in range(len(Sigma_set_list))]
            utility_difference_us_nash = utility_values_nash -  utility_values_ours
            utility_difference_start_nash = utility_values_nash - utility_values_start
            
            product_components_ours = [calculate_nash_product(final_point, Sigma_set_list[i], lambda_mu_set_list[i]) for i in range(len(Sigma_set_list))]
            product_components_nash = [calculate_nash_product(nbs_point, Sigma_set_list[i], lambda_mu_set_list[i]) for i in range(len(Sigma_set_list))]
            
            percentage_utilty = math.prod(product_components_ours) / math.prod(product_components_nash)
            numerator = np.sum(np.abs(utility_difference_us_nash))
            denominator = np.sum(np.abs(utility_difference_start_nash))
            relative_l1_error = numerator / denominator
            if relative_l1_error > 32:
                print("edge case: ", utility_difference_us_nash, utility_difference_start_nash)
            utility_diff = utility_values_ours - utility_values_nash
            # if denominator < 0:
            #     print("Checking Denominator: ", denominator, utility_values_nash)
            l1_norm = np.sum(np.abs(utility_diff))

            utility_values_worst = [markowitz_utility(solution_set_np[i], Sigma_set_list[i], lambda_mu_set_list[i]) for i in range(len(Sigma_set_list))]
            utility_values_best = [markowitz_utility(starting_state_w, Sigma_set_list[i], lambda_mu_set_list[i]) for i in range(len(Sigma_set_list))]

            # Convert to arrays
            u_ours = np.array(utility_values_ours)
            u_nash = np.array(utility_values_nash)
            u_worst = np.array(utility_values_worst)
            u_best = np.array(utility_values_best)
            

            # Normalize
            norm_ours = (u_ours - u_worst) / (u_best - u_worst + epsilon)
            norm_nash = (u_nash - u_worst) / (u_best - u_worst + epsilon)

            # Distance in normalized utility space
            normed_util_dist = np.linalg.norm(norm_ours - norm_nash)
            utility_rel_dist_ours_vs_nash.append(normed_util_dist)

            avg_pairwise_dists.append(average_pairwise_distance(solution_set_np))
            dist_to_nbs_list.append(euclidean_distance(final_point, nbs_point))
            dist_from_starting_list.append(euclidean_distance(final_point, starting_state_w))
            dist_from_starting_nash.append(euclidean_distance(nbs_point, starting_state_w))
            relative_l1_errors.append(percentage_utilty)
            # New utility space metrics
            utility_pairwise_ours.append(average_pairwise_distance(utility_values_ours))
            utility_pairwise_nash.append(average_pairwise_distance(utility_values_nash))
            utility_dist_ours_vs_nash.append(euclidean_distance(utility_values_ours, utility_values_nash))

        # Print results
        # Convert to arrays for distance calculation
        nash_array = np.array(nash_solutions_list)
        our_array = np.array(our_solutions_list)

        # Optional: only compute if more than 1 point
        if len(nash_array) > 1:
            nash_centroid = np.mean(nash_array, axis=0)
            nash_spread = np.mean(np.linalg.norm(nash_array - nash_centroid, axis=1))
            print(f"Nash Spread: {nash_spread:.4f}")

        if len(our_array) > 1:
            our_centroid = np.mean(our_array, axis=0)
            our_spread = np.mean(np.linalg.norm(our_array - our_centroid, axis=1))
            print(f"Our Spread: {our_spread:.4f}")

        print(f"→ Avg pairwise agent distance (solution space): {np.mean(avg_pairwise_dists):.10f}")
        print(f"→ Avg dist to NBS (solution space): {np.mean(dist_to_nbs_list):.10f}")
        print(f"→ Std dev. dist to NBS (solution space): {np.std(dist_to_nbs_list):.10f}")

        # print(f"→ Avg dist from Starting Ours: {np.mean(dist_from_starting_list):.4f}")
        # print(f"→ Avg dist from Starting Nash: {np.mean(dist_from_starting_nash):.4f}")

        # print(f"→ Avg pairwise agent utility (ours): {np.mean(utility_pairwise_ours):.4f}")
        # print(f"→ Avg pairwise agent utility (nash): {np.mean(utility_pairwise_nash):.4f}")
        print("Values: ", max(relative_l1_errors), min(relative_l1_errors))
        print(f"→ Average Relative L1 Error (utility space): {np.mean(relative_l1_errors):.10f}")
        print(f"→ Std dev. Relative L1 Error (utility space): {np.std(relative_l1_errors):.10f}")

        # print(f"→ Avg dist between Ours and Nash Normalized (utility space): {np.mean(utility_rel_dist_ours_vs_nash):.10f}")
        # print(f"→ Std dev. dist to NBS Normalized (utility space): {np.std(utility_rel_dist_ours_vs_nash):.4f}")

        print("-" * 60)