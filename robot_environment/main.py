import os
import time
import math
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from agents import AgentUtilityFunction


def compute_nash_objective(agent_utilities, positions):
    """
    Compute Nash bargaining objective (sum of log utilities) at a given positions dict.
    
    Args:
        agent_utilities (dict): agent_id -> AgentUtilityFunction
        positions (dict): agent_id -> position (torch tensor)
        
    Returns:
        scalar Nash objective (torch tensor)
    """
    utilities = []
    for i in sorted(agent_utilities.keys()):
        u_i = agent_utilities[i].compute_utility(positions)
        if u_i.item() <= 0:
            raise ValueError(f"Non-positive utility encountered for agent {i}.")
        utilities.append(torch.log(u_i))
    return torch.sum(torch.stack(utilities))

def save_initial_final_side_by_side(history, space_size=10.0, save_dir="plots", label="run"):
    """
    Save a side-by-side static plot of initial and final agent positions,
    coloring odd and even agents differently, and marking the center target (5,5),
    with agent index labels.
    """
    os.makedirs(save_dir, exist_ok=True)

    initial_positions = history[0].reshape(-1, 2)
    final_positions = history[-1].reshape(-1, 2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    num_agents = initial_positions.shape[0]
    agent_ids = np.arange(num_agents)

    # Define masks for odds and evens
    even_mask = (agent_ids % 2 == 0)
    odd_mask = (agent_ids % 2 == 1)

    # Plot initial positions
    axes[0].scatter(initial_positions[even_mask, 0], initial_positions[even_mask, 1],
                    s=100, c="blue", label="Even Agents")
    axes[0].scatter(initial_positions[odd_mask, 0], initial_positions[odd_mask, 1],
                    s=100, c="green", label="Odd Agents")
    axes[0].scatter(5.0, 5.0, s=150, c="red", marker="*", label="Target (5,5)")

    # Add agent index labels for initial positions
    for idx, (x, y) in enumerate(initial_positions):
        axes[0].text(x + 0.2, y + 0.2, str(idx), fontsize=9, ha='center', va='center')

    axes[0].set_xlim(0, space_size)
    axes[0].set_ylim(0, space_size)
    axes[0].set_title(f"Initial Agent Positions {label} Solution")
    axes[0].grid(True)
    axes[0].legend()

    # Plot final positions
    axes[1].scatter(final_positions[even_mask, 0], final_positions[even_mask, 1],
                    s=100, c="blue", label="Even Agents")
    axes[1].scatter(final_positions[odd_mask, 0], final_positions[odd_mask, 1],
                    s=100, c="green", label="Odd Agents")
    axes[1].scatter(5.0, 5.0, s=150, c="red", marker="*", label="Target (5,5)")

    # Add agent index labels for final positions
    for idx, (x, y) in enumerate(final_positions):
        axes[1].text(x + 0.2, y + 0.2, str(idx), fontsize=9, ha='center', va='center')

    axes[1].set_xlim(0, space_size)
    axes[1].set_ylim(0, space_size)
    axes[1].set_title(f"Final Agent Positions {label} Solution")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{label}_side_by_side.png")
    plt.savefig(save_path)
    plt.close()

def plot_agent_trajectories(history, space_size=10.0, save_dir="plots", label="run"):
    """
    Plot the full trajectory of each agent over time using the state history.
    Initial and final points are highlighted; intermediate paths are faint.
    """
    os.makedirs(save_dir, exist_ok=True)

    history = np.array(history)  # shape: [T, num_agents * 2]
    num_steps, flat_dim = history.shape
    num_agents = flat_dim // 2

    agent_ids = np.arange(num_agents)
    even_mask = (agent_ids % 2 == 0)
    odd_mask = (agent_ids % 2 == 1)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot each agent's trajectory
    for i in range(num_agents):
        agent_traj = history[:, 2 * i: 2 * i + 2]  # shape: [T, 2]
        color = "blue" if even_mask[i] else "green"

        # Line plot for trajectory
        ax.plot(agent_traj[:, 0], agent_traj[:, 1],
                linestyle="-", linewidth=1.5, alpha=0.3, color=color)

        # Initial point
        ax.scatter(agent_traj[0, 0], agent_traj[0, 1],
                   color=color, s=60, marker='o', label=f"Agent {i} Start" if i < 2 else None)

        # Final point
        ax.scatter(agent_traj[-1, 0], agent_traj[-1, 1],
                   color=color, s=80, marker='^', label=f"Agent {i} End" if i < 2 else None)

        # Label final point
        ax.text(agent_traj[-1, 0] + 0.2, agent_traj[-1, 1] + 0.2, str(i),
                fontsize=8, ha='center', va='center')

    # Optional: Plot center target
    ax.scatter(5.0, 5.0, s=120, color="red", marker="*", label="Target (5,5)")

    ax.set_xlim(0, space_size)
    ax.set_ylim(0, space_size)
    ax.set_title(f"Agent Trajectories ({label})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{label}_trajectories.png")
    plt.savefig(save_path)
    plt.close()

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def compute_optimal_distance(alpha, beta):
    return math.log(beta / alpha) / (beta - alpha)

def initialize_agents_in_circle(num_agents=10, center=(5.0, 5.0), radius=2.0):
    """
    Evenly space agents in a circle around a center point.
    """
    initial_positions = {}
    angles = np.linspace(0, 2 * np.pi, num_agents, endpoint=False)

    for idx in range(num_agents):
        theta = angles[idx]
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        initial_positions[idx] = (x, y)

    return initial_positions

def generate_grouped_agents_and_distances(num_agents=10, space_size=10.0, scaling_factors=None, circle_init=True):
    """
    Generate agent utilities, desired distances, and initial positions.
    If circle_init=True, initialize agents evenly spaced around a circle.
    """
    max_distance = math.sqrt(space_size**2 + space_size**2)

    agent_utilities = {}
    desired_optimal_distances = {}

    alphas = {}
    betas = {}

    if scaling_factors is None:
        scaling_factors = torch.ones(num_agents)

    # Define groups
    group_assignments = {i: ("even" if i % 2 == 0 else "odd") for i in range(num_agents)}

    for i in range(num_agents):
        alphas[i] = {}
        betas[i] = {}
        desired_optimal_distances[i] = []

        for j in range(num_agents):
            if i == j:
                continue

            # Same group → strong attraction
            if group_assignments[i] == group_assignments[j]:
                alpha_ij = 1.0
                beta_ij = 3.0
            else:  # Different groups → repulsion
                alpha_ij = 0.1
                beta_ij = 0.9

            d_star = compute_optimal_distance(alpha_ij, beta_ij)

            # Extra sanity check: distances must not exceed max
            if d_star > max_distance:
                d_star = torch.tensor(max_distance * 0.9)  # Shrink slightly if needed

            alphas[i][j] = alpha_ij
            betas[i][j] = beta_ij
            d_star = torch.tensor(d_star, dtype=torch.float32)
            desired_optimal_distances[i].append(d_star)
        desired_optimal_distances[i].append(torch.tensor(0, dtype=torch.float32))
        desired_optimal_distances[i] = torch.stack(desired_optimal_distances[i])

    # Set up agent utility functions
    for i in range(num_agents):
        if group_assignments[i] == 'even':
            agent_utilities[i] = AgentUtilityFunction(
                agent_id=i,
                other_agent_ids=[j for j in range(num_agents) if j != i],
                alphas=alphas[i],
                betas=betas[i],
                group_assignments=group_assignments,
                same_group_weight=1.0,
                scaling_factor=1.0,
                center_gamma=0.01,
                device="cpu"
            )
        else:
            agent_utilities[i] = AgentUtilityFunction(
                agent_id=i,
                other_agent_ids=[j for j in range(num_agents) if j != i],
                alphas=alphas[i],
                betas=betas[i],
                group_assignments=group_assignments,
                same_group_weight=1.0,
                scaling_factor=1.0,
                nonlinear_transform=None,
                center_gamma=0.01,
                device="cpu"
            )

    # Initialize positions
    if circle_init:
        initial_positions = initialize_agents_in_circle(
            num_agents=num_agents,
            center=(space_size/2, space_size/2),
            radius=space_size * 0.25  # Example: 25% of the full space size
        )
    else:
        initial_positions = {
            i: (random.uniform(0, space_size), random.uniform(0, space_size))
            for i in range(num_agents)
        }

    return agent_utilities, desired_optimal_distances, initial_positions

def optimize_agents_nash(agent_utilities,
                         initial_positions,
                         steps=500,
                         lr=0.05,
                         space_size=10.0,
                         disagreement_values = None,
                         save_gif_path=None):
    """
    Optimize agent positions using Nash Bargaining Solution (NBS),
    using each agent's full gradient w.r.t. all positions.
    """
    device = "cpu"
    agent_ids = sorted(initial_positions.keys())
    num_agents = len(agent_ids)

    # Initialize positions
    positions = {
        i: torch.tensor(initial_positions[i], dtype=torch.float32, device=device, requires_grad=True)
        for i in agent_ids
    }

    history = []

    for step in range(steps):
        # Inside your step loop
        utilities = []
        grad_logs = []

        for i in agent_ids:
            u_i = agent_utilities[i].compute_utility(positions)
            if disagreement_values == None:
                disagreement_i = 0
            else:
                disagreement_i = disagreement_values[i]
            if u_i.item() <= disagreement_i:
                raise ValueError(f"Utility below disagreement at step {step}: agent {i}")

            grad_u_i = torch.autograd.grad(
                outputs=u_i,
                inputs=[positions[j] for j in agent_ids],
                retain_graph=True,
                create_graph=False
            )

            grad_u_i = torch.cat(grad_u_i)

            utilities.append(u_i)
            grad_logs.append((1.0 / (u_i - disagreement_i)) * grad_u_i)

        # Compute total gradient
        total_grad = torch.stack(grad_logs).sum(dim=0)

        # Normalize gradient
        grad_norm = total_grad.norm()
        if grad_norm > 1e-8:
            total_grad /= grad_norm
        if step == steps - 1 :
            print("Final Gradients: ", total_grad)
        # Update positions
        with torch.no_grad():
            full_state = torch.cat([positions[i] for i in agent_ids])
            full_state += lr * total_grad
            full_state = full_state.clamp(0.0, space_size)

            for idx, i in enumerate(agent_ids):
                positions[i] = full_state[2*idx:2*(idx+1)].detach().requires_grad_(True)

        # Save history
        history.append(full_state.cpu().numpy())

    print("Optimization complete!")
    final_distance = torch.norm(positions[0] - positions[1])
    print("Nash Final Distance:", positions[0], positions[1], final_distance.item(),
          "Expected Optimal:", agent_utilities[0].compute_optimal_distance(1),
          agent_utilities[1].compute_optimal_distance(0))

    return history

def compute_distance_vector(positions, agent_id):
    """
    Compute vector of distances from agent_id to all other agents.
    """
    pos_i = positions[agent_id]
    distance_vector = []
    for j, pos_j in positions.items():
        if j == agent_id:
            continue
        d_ij = torch.norm(pos_i - pos_j)
        distance_vector.append(d_ij)
    pos_center = torch.tensor([5, 5], dtype=torch.float32)
    dist_to_center = torch.norm(pos_i - pos_center)
    distance_vector.append(dist_to_center)
    return torch.stack(distance_vector)

def optimize_agents_our_solution(agent_utilities,
                                           initial_positions,
                                           desired_optimal_distances,
                                           steps=500,
                                           lr=0.05,
                                           space_size=10.0,
                                           save_gif_path=None):
    """
    Optimize agent positions using our solution concept (mediator balancing preferences).
    
    Args:
        agent_utilities (dict): Mapping agent_id -> AgentUtilityFunction
        initial_positions (dict): Mapping agent_id -> (x, y) tuples
        desired_optimal_distances (dict): Mapping agent_id -> torch.Tensor (desired distances to others)
        steps (int): Number of optimization steps
        lr (float): Learning rate
        space_size (float): 2D space bounds
        save_gif_path (str or None): Save animation if path provided

    Returns:
        history (list): Full states over time
    """
    device = "cpu"
    agent_ids = sorted(initial_positions.keys())

    # Initialize positions
    positions = {
        i: torch.tensor(initial_positions[i], dtype=torch.float32, device=device, requires_grad=True)
        for i in agent_ids
    }

    history = []

    for step in range(steps):
        # Zero gradients
        for pos in positions.values():
            if pos.grad is not None:
                pos.grad.zero_()

        # Set up gradient sum for each agent
        grad_sum = {i: torch.zeros_like(positions[i]) for i in agent_ids}
        norm_sum = {i: 0.0 for i in agent_ids}

        # Current positions snapshot
        current_positions = {j: positions[j] for j in agent_ids}

        # For each agent, compute its utility gradient w.r.t. all positions
        for i in agent_ids:
            grads = torch.autograd.grad(
                outputs=agent_utilities[i].compute_utility(current_positions),
                inputs=[positions[j] for j in agent_ids],
                retain_graph=True,
                create_graph=False
            )
            # Compute distance vector mismatch for agent i
            d_vec = compute_distance_vector(positions, i)
            d_opt = desired_optimal_distances[i]
            distance_diff_norm = torch.norm(d_vec - d_opt)

            if distance_diff_norm.item() == 0.0:
                continue

            # Distribute agent i's influence over all positions
            for j, grad_ij in zip(agent_ids, grads):
                grad_norm = torch.norm(grad_ij)
                if grad_norm.item() > 0:
                    grad_sum[j] += (grad_ij / grad_norm) * distance_diff_norm
                    norm_sum[j] += distance_diff_norm

        # Gradient descent step
        with torch.no_grad():
            for i in agent_ids:
                if norm_sum[i] > 0:
                    positions[i] += lr * (grad_sum[i] / norm_sum[i])
                    positions[i].clamp_(0.0, space_size)
        if step == steps - 1 :
            print("Final Gradients: ", grad_sum)
        # Re-enable gradient tracking
        for i in agent_ids:
            positions[i] = positions[i].detach().requires_grad_(True)

        # Save history
        full_state = torch.cat([positions[i] for i in agent_ids])
        history.append(full_state.detach().cpu().numpy())

    print("Optimization complete!")
    final_distance = torch.norm(positions[0] - positions[1])
    print("Ours Final Distance:", positions[0], positions[1], final_distance.item(), "Expected Optimal:", agent_utilities[0].compute_optimal_distance(1), agent_utilities[1].compute_optimal_distance(0))

    return history


def main():
    set_random_seed(50)

    # Settings
    steps = 5000
    lr = 0.1
    space_size = 10.0
    save_gif_path = "multi_agents_trajectory.mp4"
    num_agents = 10  # New setting: 10 agents

    # Generate random agents, desired distances, and initial positions
    print("Generating agents...")
    # scaling_factors = torch.tensor([1, 1, 10], dtype=torch.float32)
    
    agent_utilities, desired_optimal_distances, initial_positions = generate_grouped_agents_and_distances(
        num_agents=num_agents,
        space_size=space_size
    )
    print("Running Our Solution Concept...")
    # Run our solution concept (mediator-based)
    history_ours = optimize_agents_our_solution(
        agent_utilities,
        initial_positions,
        desired_optimal_distances,
        steps= steps,
        lr=lr,
        space_size=space_size
    )
    save_initial_final_side_by_side(history_ours, save_dir="plots", label="Our")
    plot_agent_trajectories(history_ours, save_dir="plots", label="Our")
    print("Running NBS...")
    history_nash = optimize_agents_nash(
        agent_utilities,
        initial_positions,
        steps= steps,
        lr=lr,
        space_size=space_size,
    )
    save_initial_final_side_by_side(history_nash, save_dir="plots", label="Nash")
    plot_agent_trajectories(history_nash, save_dir="plots", label="Nash")


    # After optimizing ours
    final_positions_ours = {i: torch.tensor(history_ours[-1][2*i:2*(i+1)]) for i in range(num_agents)}

    # After optimizing nash
    final_positions_nash = {i: torch.tensor(history_nash[-1][2*i:2*(i+1)]) for i in range(num_agents)}

    # Evaluate Nash objectives
    nash_obj_ours = compute_nash_objective(agent_utilities, final_positions_ours)
    nash_obj_nash = compute_nash_objective(agent_utilities, final_positions_nash)

    print(f"Nash Objective (Our Solution): {nash_obj_ours.item():.6f}")
    print(f"Nash Objective (NBS Optimization): {nash_obj_nash.item():.6f}")


if __name__ == "__main__":
    main()