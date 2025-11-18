# Obstacle-aware Sequential CBF (created on 10/16/2025)
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from cvxpylayers.torch import CvxpyLayer
import torch

def cbf_with_grads(state, target_state, obstacle_state, target_radius, u_target_max, remaining_t):
    state = torch.tensor(state, dtype=torch.float32, requires_grad=True)
    target_state = torch.tensor(target_state, dtype=torch.float32, requires_grad=True)
    obstacle_state = torch.tensor(obstacle_state, dtype=torch.float32, requires_grad=True)

    x, y = state
    xt, yt = target_state
    xo, yo = obstacle_state
    r_hat = target_radius + u_target_max #dt=1

    dx = xt - x
    dy = yt - y

    # Perpendicular distance from center to line
    h = dx*(yo - y) - dy*(xo - x) / torch.sqrt(dx**2 + dy**2 + 1e-8)

    chord_len = 2 * torch.sqrt(torch.clamp(r_hat**2 - h**2, min=0.0))

    theta = 2 * torch.acos(torch.clamp(h / r_hat, -1.0, 1.0))
    arc_length = r_hat * theta


    dist = torch.sqrt((x - xt)**2 + (y - yt)**2) - target_radius + u_target_max * remaining_t - chord_len + arc_length

    cbf = remaining_t - dist

    grad_state, grad_target_state = torch.autograd.grad(cbf, [state, target_state], create_graph=True)

    return cbf.detach().numpy(), grad_state.detach().numpy(), grad_target_state.detach().numpy()



def build_alpha_layer_2d():
    """
    Differentiable layer that solves:
      minimize_{p in R^2, alpha in R} alpha
      s.t. ||p - rA||_2 <= alpha * RA
           ||p - rB||_2 <= alpha * RB
           alpha >= 0
    Parameters (torch): rA(2,), rB(2,), RA(1,), RB(1,)
    Variables (torch):  p(2,), alpha(1,)
    Returns: CvxpyLayer(problem, parameters=[rA, rB, RA, RB], variables=[p, alpha])
    """
    # Variables
    p = cp.Variable(2)
    alpha = cp.Variable(1)

    # Parameters
    rA = cp.Parameter(2)  # robot center
    rB = cp.Parameter(2)  # obstacle center
    RA = cp.Parameter(1, nonneg=True)
    RB = cp.Parameter(1, nonneg=True)

    constraints = [
        cp.norm(p - rA, 2) <= alpha * RA,
        cp.norm(p - rB, 2) <= alpha * RB,
        alpha >= 0.0
    ]
    prob = cp.Problem(cp.Minimize(alpha), constraints)
    layer = CvxpyLayer(prob, parameters=[rA, rB, RA, RB], variables=[p, alpha])
    return layer

def tvcbf_circle_diffopt(pr, po, vo, Rr, Ro, alpha_layer, beta=1.0):
    """
    Time-varying CBF using differentiable optimization to compute alpha* and its gradients.

    Args:
      pr: np.array(2,)   robot position
      po: np.array(2,)   obstacle position
      vo: np.array(2,)   obstacle velocity (explicit time dependence)
      Rr: float          robot radius
      Ro: float          obstacle radius
      alpha_layer: CvxpyLayer built by build_alpha_layer_2d()
      beta: float        safety offset (>=1 typically)

    Returns:
      h: float
      grad_h_x: np.array(2,)      (∂h/∂pr)
      dh_dt: float                (∂h/∂po)·vo
      alpha_star: float
    """
    # Torch tensors (double precision recommended for cvxpylayers)
    pr_t = torch.tensor(pr, dtype=torch.double, requires_grad=True)
    po_t = torch.tensor(po, dtype=torch.double, requires_grad=True)
    Rr_t = torch.tensor([Rr], dtype=torch.double)
    Ro_t = torch.tensor([Ro], dtype=torch.double)

    # Solve the conic problem: returns p*, alpha*
    # Note: cvxpylayers expects parameters in the same order used when building the layer.
    p_star_t, alpha_star_t = alpha_layer(pr_t, po_t, Rr_t, Ro_t, solver_args={"eps": 1e-6})

    # Define h = alpha* - beta
    h_t = alpha_star_t - beta

    # Gradients via autograd
    grad_pr = torch.autograd.grad(h_t, pr_t, retain_graph=True, allow_unused=False)[0]  # ∂h/∂pr
    grad_po = torch.autograd.grad(h_t, po_t, retain_graph=False, allow_unused=False)[0] # ∂h/∂po

    # Explicit time derivative dh/dt = (∂h/∂po) · vo
    vo_t = torch.tensor(vo, dtype=torch.double)
    dh_dt_t = (grad_po * vo_t).sum()

    # To numpy
    h = float(h_t.detach().cpu().numpy())
    grad_h_x = grad_pr.detach().cpu().numpy()                  # w.r.t. robot state (position)
    dh_dt = float(dh_dt_t.detach().cpu().numpy())
    alpha_star = float(alpha_star_t.detach().cpu().numpy())

    return h, grad_h_x, dh_dt, alpha_star


def solve_cbf_qp(agent_state, u_agent_max, disturbance_interval, target_index, current_t, targets, u_rl):
    """
    Solve QP to get control u satisfying the CBF condition.
    """
    if disturbance_interval is not None:
        w_max = max(abs(disturbance_interval[0]), abs(disturbance_interval[1]))
        u_agent_max_cbf = u_agent_max - w_max #reduce the max agent speed by the disturbance bound (worst-case)
    else:
        u_agent_max_cbf = u_agent_max

    x = agent_state
    u = cp.Variable(2)
    delta = cp.Variable(nonneg=True) #slack variable for the CBF condition

    #get the parameters of the first target in the sequence for the first term:
    first_key = next(iter(targets))
    first_target_center, first_target_radius, u_target_max_first, first_remaining_t = targets[first_key]['center'], targets[first_key]['radius'], targets[first_key]['u_max'], targets[first_key]['remaining_time']

    dx = x[0] - first_target_center[0]
    dy = x[1] - first_target_center[1]
    dist = max(np.sqrt(dx**2 + dy**2), 1e-6)

    target_center, target_radius, u_target_max, remaining_t, _ = targets[target_index]['center'], targets[target_index]["radius"], targets[target_index]['u_max'], targets[target_index]['remaining_time'], targets[target_index]['movement']['type']

    #db_dr = 1 - (u_target_max / u_agent_max) #derivative w.r. to  the remaining time

    #create an array with size len(targets) x 2 for the derivative w.r.t. the target's state
    db_dr = np.zeros(len(targets))  # Initialize derivative w.r.t. remaining time for all targets

    db_dr[0] = 1 - (u_target_max_first / u_agent_max_cbf)  # derivative w.r.t. the remaining time for the first target

    alpha = 1.5

    u_min = np.array([-u_agent_max, -u_agent_max])
    u_max = np.array([u_agent_max, u_agent_max])  # might need to change later!

    u_target = np.zeros((len(targets), 2))  # Initialize target control input
    target_list = list(targets.keys())

    for i in range(len(targets)):
        key = target_list[i]
        u_target_max = targets[key]['u_max']
        target_movement_type = targets[key]['movement']['type']

        if target_movement_type == 'circular':
            xc, yc = targets[key]['movement']['center_of_rotation']
            x0, y0 = targets[key]['center']
            turning_radius = np.linalg.norm(np.array([x0 - xc, y0 - yc]))
            omega = u_target_max / turning_radius #angular velocity
            # Calculate the initial angle from the center of rotation to the initial position
            theta0 = np.arctan2(y0 - yc, x0 - xc)
            theta = theta0 + omega * current_t
            u_target[i] = np.array([np.cos(theta) * u_target_max, np.sin(theta) * u_target_max])

        elif target_movement_type == 'straight' or target_movement_type == 'periodic':
            heading_angle = targets[key]['movement']['heading_angle']
            u_target[i] = np.array([np.cos(heading_angle) * u_target_max, np.sin(heading_angle) * u_target_max])

    #print(db_dx @ (u + u_rl) + db_dx_target @ u_target - db_dr + alpha * b_func(agent_state, u_agent_max, targets, target_index))


   
    pos_obs = np.array(obstacles[0]['center'])   # live obstacle center from simulator (2,) (for 1 obstacle only!!!!) change later!

    cbf_value, db_dx, db_dx_target = cbf_with_grads(agent_state, target_center, np.array([0.0, 0.0]), target_radius, u_target_max, remaining_t)

    cbf_constraint = [db_dx @ (u + u_rl) + db_dx_target.reshape(2,1) @ u_target - np.transpose(db_dr) @ np.ones(len(targets)) + alpha * cbf_value + delta >= 0,
                        u + u_rl >= u_min,
                        u + u_rl <= u_max] #CBF condition (change the reshape later!)

    # #add obstacle avoidance constraints:
    vo = [np.array([obstacles[i]['u_max']*np.cos(obstacles[i]['movement']['heading_angle']), obstacles[i]['u_max']*np.sin(obstacles[i]['movement']['heading_angle'])]) for i in obstacles]  # live obstacle velocity from simulator (2,) (for periodic movement only)
    Rr, Ro = agent_radius, obstacles[0]['radius']

    h, grad_h_x, dh_dt, alpha_star = tvcbf_circle_diffopt(agent_state, pos_obs, vo, Rr, Ro, alpha_layer, beta=1.5)
    cbf_constraint.append(grad_h_x @ (u + u_rl) + dh_dt >= -alpha * h)
    

    Q = np.eye(2)
    slack_weight = 1e4  # Weight for the slack variable

    objective = cp.Minimize(cp.quad_form(u, Q) + slack_weight * delta)  # Minimize control effort and slack variable

    prob = cp.Problem(objective, cbf_constraint)
    prob.solve(solver=cp.ECOS, verbose = False)
    #prob.solve(solver=cp.OSQP, verbose = True)

    if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
        return u.value, delta.value
    else:
        print("QP failed:", prob.status)
        return None, None


if __name__ == "__main__":
    from simulator import Continuous2DEnv

    agent_init_loc = (-30, -0.5)  # initial location of the agent
    u_agent_max = 3 # Max vel. of the system


     #scenario-4: #periodic alternative
    t_windows=[[[0,100],[0,25]]] # STL time windows

    point1_1 = (25, 0)
    point1_2 = (50, 0)

    target_1 = {0: {'id': 1, 'type': 'GF', 'time_window': t_windows[0], 'label': 'target 1', 'center': (30,0) ,'radius': 3, 'u_max': 0.5, 'remaining_time': t_windows[0][1][1], 'movement':{'type': 'periodic', 'point1': point1_1, 'point2': point1_2, 'heading_angle': np.arctan2(point1_2[1] - point1_1[1], point1_2[0] - point1_1[0])}, 'color': 'blue'}}


    point1 = (20, 0)
    point2 = (-15, 0)

    # point1 = (0, -15)
    # point2 = (0, 20)

    obstacle_region_radius = 3
    agent_radius = 1


    obstacles = {
        0: {'label': 1,'center': (30, 0), 'radius': obstacle_region_radius, 'u_max': 1, 'remaining_time': 100, 'movement':{'type': 'periodic', 'point1': point1, 'point2': point2, 'heading_angle': np.arctan2(point2[1] - point1[1], point2[0] - point1[0])}, 'color': 'blue'}
        #0: {'id': 0, 'center': (-30, 30), 'radius': target_region_radius, 'u_max': u_target_max0, 'remaining_time': 100, 'movement':{'type': 'circular', 'omega': 0.1, 'center_of_rotation':(-25,30)}, 'color': 'blue'}, #heading angle is in rad
        #1: {'id': 1, 'center': (-30, -30), 'radius': target_region_radius, 'u_max': u_target_max1, 'remaining_time': 100, 'movement':{'type': 'circular', 'omega': 0.1, 'center_of_rotation':(-25,-30)}, 'color': 'red'}, #heading angle is in rad
        #2: {'center': (35, -30), 'radius': target_region_radius, 'u_max': u_target_max1, 'remaining_time': 100, 'movement':{'type': 'circular', 'omega': 0.1, 'center_of_rotation':(35,-30)}, 'color': 'yellow'}, #heading angle is in rad
        #2: {'center': (-20, -20), 'radius': target_region_radius, 'u_max': u_target_max1, 'remaining_time': 200, 'movement':{'type': 'straight', 'heading_angle': 5*np.pi/4}}
    }

    cbf_targets = [target_1]
    simulation_targets = target_1

    goals = {
	0: {'center': (100, 100), 'radius': 0, 'movement':{'type':'static'}}, #goal region for the agent
	}

        #config dictionary for the environment
    config = {
        'init_loc': agent_init_loc, #initial location of the agent (x, y)
        "width": 50.0,
        "height": 50.0,
        "dt": 1,
        "render": True,
        'dt_render': 0.03, #time interval for rendering
        "goals": goals,  # goal regions for the agent
        "obstacles": obstacles,  # dictionary of obstacles
        "targets": simulation_targets,  # dictionary of targets for the CBF
        "dynamics": "single integrator", #dynamics model to use
        'u_agent_max': u_agent_max, #max agent speed
        "randomize_loc": False,  #whether to randomize the agent location at the end of each episode
        "disturbance": None #disturbance range in both x and y directions [w_min, w_max]
    }

    env = Continuous2DEnv(config)
    state = env.reset()

    action_rl = np.array([0.0, 0])

    action = action_rl

    episode_length = 100
    
    alpha_layer = build_alpha_layer_2d()
    t = 0
    
    min_dist_to_obstacle = float('inf')

    while t <= episode_length:

        selected_target = cbf_targets[0]
        #print("Selected target:", selected_target)
        first_key = next(iter(selected_target))

        #Now solve the QP to get the control input for the target region with the minimum CBF value:
        u_cbf, slack_variable = solve_cbf_qp(state, u_agent_max, None, first_key, t, selected_target, action_rl)
        #u_cbf, info = solve_tvcbf_qp(agent_state=state, obstacles=obstacles, u_ref=action_rl, agent_radius=agent_radius , alpha_layer=alpha_layer)

        action = (u_cbf[0] + action_rl[0], u_cbf[1] + action_rl[1])  # Combine CBF and RL actions

        state, reward, done = env.step(action)

        for i in range(len(cbf_targets)):
            first_key = next(iter(cbf_targets[i]))
            cbf_targets[i][first_key]['remaining_time'] -= 1 #decrease the remaining time for each target region

        #update the minimum distance to obstacle:
        dist_to_obstacle = np.linalg.norm(np.array(state[:2]) - np.array(obstacles[0]['center'])) - (agent_radius + obstacles[0]['radius'])
        if dist_to_obstacle < min_dist_to_obstacle:
            min_dist_to_obstacle = dist_to_obstacle
        
        first_key = next(iter(selected_target))
        #cbf_targets[target_index]['center'] = moving_target(t, center_of_rotation, cbf_targets[target_index]['u_max'])
        task_type = selected_target[first_key]['type']
        time_window = selected_target[first_key]['time_window']

        #calculate the signed distance to each target region:
        target_center = selected_target[first_key]['center']
        target_radius = selected_target[first_key]['radius']
        dist = np.linalg.norm(state[:2] - target_center)
        signed_distance = dist - target_radius


        remove_target = False #flag to indicate whether to remove the visited target region

        if task_type == "F":
            # Handle F type tasks
            a = time_window[0][0]
            b = time_window[0][1]
            if t >= a and t <= b and signed_distance <= 0:
                #within the time window
                remove_target = True

        elif task_type == "G":
            # Handle G type tasks
            a = time_window[0][0]
            b = time_window[0][1]
            if t == a and signed_distance <= 0:
                remove_target = True #remove the target region if the agent is inside it at the start of the time window
                #Hold inside the target region until the end of the time window:
                selected_target[0]['remaining_time'] = 0 #set the remaining time to the length of the time window
                for j in range(b-a):
                    print("Holding inside target region", selected_target[first_key]['id'], "at time", t)
                    t += 1
                    u_cbf = solve_cbf_qp(sequential_CBF, state, u_agent_max, first_key, t, cbf_targets, action_rl)

                    action = (u_cbf[0] + action_rl[0], u_cbf[1] + action_rl[1])  # Combine CBF and RL actions

                    state, reward, done = env.step(action)            

        elif task_type == "FG":
            # Handle FG type tasks
            a = time_window[0][0]
            b = time_window[0][1]
            c = time_window[1][0]
            d = time_window[1][1]
            if t <= (a+c) and t >= (b+c) and signed_distance <= 0:
                #within the time window
                remove_target = True
                selected_target[0]['remaining_time'] = 0 #set the remaining time to the length of the time window
                for j in range(d-c):
                    print("Holding inside target region", selected_target[first_key]['id'], "at time", t)
                    t += 1
                    u_cbf = solve_cbf_qp(sequential_CBF, state, u_agent_max, first_key, t, cbf_targets, action_rl)

                    action = (u_cbf[0] + action_rl[0], u_cbf[1] + action_rl[1])  # Combine CBF and RL actions

                    state, reward, done = env.step(action)    

        elif task_type == "GF":
            # Handle GF type tasks
            a = time_window[0][0]
            b = time_window[0][1]
            c = time_window[1][0]
            d = time_window[1][1]
            if t >= a + c and t <= b + d and signed_distance <= 0: #only remove the target region if it is the first in the sequence
                #within the time window
                print("Agent is inside target region", selected_target[first_key]['id'], "at time", t)
                remove_target = True
                ##################################################################################
                #UPDATE the remaining time for the next target region with the same id (if any):
                ##################################################################################
                for i in range(len(cbf_targets)):
                    first_key = next(iter(cbf_targets[i]))
                    cbf_targets[i][first_key]['remaining_time'] = d - c

        
        t += 1 #increment time step

    print('Minimum distance to obstacle:', min_dist_to_obstacle)

        # if done:
        #     break

    if cbf_targets == {}:
        print("Task completed!")
