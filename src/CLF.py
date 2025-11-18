import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


def clf_function(x, y):
    return np.linalg.norm(x - y)


def solve_clf_qp(state, target, alpha,u_max):
    """
    Solve the control Lyapunov function (CLF) optimization problem.
    
    Args:
        state (np.ndarray): Current state of the agent.
        target (np.ndarray): Target state to reach.
        u_max (float): Maximum control input magnitude.
    
    Returns:
        np.ndarray: Optimal control input.
    """
    
    x_agent = np.array(state)
    x_target = np.array(target)

    V = np.linalg.norm(x_agent - x_target)
    grad_V = (x_agent - x_target) / V


    u = cp.Variable(2)  # Control input variable
    delta = cp.Variable(nonneg=True)  # Non-negative slack variable for the CBF constraint
    slack_weight = 1e4

    constraints = [grad_V @ u <= -alpha * V + delta,
                   u >= -u_max,
                   u <= u_max]

    objective = cp.Minimize(cp.norm(u, 2) + slack_weight * delta)  # Minimize the control input magnitude

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS)

    return u.value




if __name__ == "__main__":
    from simulator import Continuous2DEnv

    target_region_center = (0, 0)
    target_region_radius = 5
    u_agent_max = 2.0

    targets = {0: {'center': target_region_center, 'radius': target_region_radius, 'u_max': 0, 'movement':{'type': 'static'}, 'color': 'blue'}}
    goals = {
	0: {'center': (20, 13), 'radius': 0, 'movement':{'type':'static'}}, #goal region for the agent
	}

    config = {
        'init_loc': [50, 50], #initial location of the agent (x, y)
        "width": 100.0,
        "height": 100.0,
        "dt": 1,
        "render": True,
        'dt_render': 0.03,
        "goals": goals,  # goal regions for the agent
        "obstacle_location": [100.0, 100.0],
        "obstacle_size": 0.0,
        "targets": targets,  # dictionary of targets for the CBF
        "dynamics": "single integrator", #dynamics model to use
        'u_agent_max': u_agent_max, #max agent speed
        "randomize_loc": False #whether to randomize the agent location at the end of each episode
    }

    env = Continuous2DEnv(config)
    state = env.reset()
    alpha = 1.5 # CLF weight

    for _ in range(100):
        action = solve_clf_qp(state, target_region_center, alpha, u_agent_max)
        print("Action:", action)
        state, reward, done = env.step(action)
        #prev_v = action[0]
        
        #print(state)
        if done:
            break
