import torch

def solve_nash_bargaining_formation(agent_utility_functions, initial_positions, 
                                     disagreement=-1.0, steps=1000, lr=0.1, space_size=10.0):
    """
    Solves for the Nash Bargaining Solution for formation control with bounded space.
    
    Args:
        agent_utility_functions (dict): Mapping agent_id -> AgentUtilityFunction object.
        initial_positions (dict): Mapping agent_id -> torch.tensor([x, y]) (requires_grad=True).
        disagreement (float): Disagreement utility (same for all agents).
        steps (int): Number of optimization steps.
        lr (float): Learning rate (initial step size).
        space_size (float): Size of the 2D space (assumes [0, space_size] × [0, space_size]).

    Returns:
        dict: Final positions.
    """
    positions = {k: v.clone().detach().requires_grad_(True) for k, v in initial_positions.items()}
    agent_ids = list(agent_utility_functions.keys())
    
    for step in range(steps):
        # Zero all gradients
        for pos in positions.values():
            if pos.grad is not None:
                pos.grad.zero_()

        # Compute utilities
        utilities = {}
        for agent_id, utility_fn in agent_utility_functions.items():
            utilities[agent_id] = utility_fn.compute_utility(positions)

        # Compute Nash Bargaining Objective: sum log(u_i - d_i)
        objective = 0.0
        for agent_id in agent_ids:
            diff = utilities[agent_id] - disagreement
            if diff.item() <= 0:
                # If utility is too low, treat log(very small positive number)
                diff = torch.tensor(1e-8, device=diff.device)
            objective += torch.log(diff)
        
        # Gradient ascent: maximize objective
        objective.backward()

        # Take a step in the gradient direction
        max_step_size = lr
        success = False
        
        while not success:
            new_positions = {}
            for agent_id in agent_ids:
                new_positions[agent_id] = positions[agent_id] + max_step_size * positions[agent_id].grad

            # Check feasibility
            feasible = True
            for p in new_positions.values():
                if not ((p >= 0.0).all() and (p <= space_size).all()):
                    feasible = False
                    break

            if feasible:
                success = True
            else:
                max_step_size *= 0.1
                if max_step_size < 1e-6:
                    success = True  # stop shrinking if step size too small

        # Update positions
        for agent_id in agent_ids:
            positions[agent_id] = new_positions[agent_id].detach().requires_grad_(True)

    return positions


def estimate_disagreement_utilities(agent_utility_functions, space_size=10.0, 
                                     n_samples=1000, epsilon=0.01, device="cpu"):
    """
    Estimate a disagreement utility for each agent based on random sampling.

    Args:
        agent_utility_functions (dict): Mapping agent_id -> AgentUtilityFunction.
        space_size (float): Size of the 2D space.
        n_samples (int): Number of random samples.
        epsilon (float): Small margin to subtract.
        device (str): PyTorch device.

    Returns:
        dict: Mapping agent_id -> estimated disagreement utility.
    """
    agent_ids = list(agent_utility_functions.keys())
    disagreement_utilities = {agent_id: float('inf') for agent_id in agent_ids}

    for _ in range(n_samples):
        # Random sample
        positions = {agent_id: torch.rand(2, device=device) * space_size for agent_id in agent_ids}
        
        for agent_id, utility_fn in agent_utility_functions.items():
            u_i = utility_fn.compute_utility(positions).item()
            if u_i < disagreement_utilities[agent_id]:
                disagreement_utilities[agent_id] = u_i

    # Subtract a small epsilon to ensure strict inequality
    for agent_id in agent_ids:
        disagreement_utilities[agent_id] -= epsilon

    return disagreement_utilities


