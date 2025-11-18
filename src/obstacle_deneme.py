#code for diffOpt time-varying control barrier functions (TCBF) for obstacle avoidance
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from cvxpylayers.torch import CvxpyLayer
import torch


def agent_to_target_dist(state, target_center, target_radius, u_target_max, remaining_t):
    x, y = state
    xc, yc = target_center

    dist = np.sqrt((x - xc)**2 + (y - yc)**2) - target_radius + u_target_max * remaining_t
    #dist = np.sqrt((x - xc)**2 + (y - yc)**2) - target_radius
    return dist


def target_to_target_dist(target_center1, target_center2, target_radius_1, target_radius_2, u_target1_max, u_target2_max, remaining_t1, remaining_t2):
    xc1, yc1 = target_center1
    xc2, yc2 = target_center2

    dist = np.sqrt((xc1 - xc2)**2 + (yc1 - yc2)**2) - target_radius_2 + target_radius_1 + u_target1_max * remaining_t1 + u_target2_max * remaining_t2

    return dist


def sequential_CBF(agent_state, u_agent_max, targets, target_index): #returns the CBF function value
    """Calculate the CBF value for the agent with respect to the target region.
    :param agent_state: Current state of the agent (x, y).
    :param u_agent_max: Maximum speed of the agent.
    :param targets: Dictionary of target regions, each defined by a dictionary (center coordinates, radius, max speed, remaining time).
    :param target_index: Index of the target region to consider.
    :return: CBF value for the agent with respect to the target region."""

    first_key = next(iter(targets))
    first_target_center, first_target_radius, u_target_max_first, first_remaining_t = targets[first_key]['center'], targets[first_key]['radius'], targets[first_key]['u_max'], targets[first_key]['remaining_time']

    #target_center, target_radius, u_target_max, remaining_t = targets[target_index]['center'], targets[target_index]['radius'], targets[target_index]['u_max'], targets[target_index]['remaining_time']

    agent_to_target = agent_to_target_dist(agent_state, first_target_center, first_target_radius, u_target_max_first, first_remaining_t) * (1/ u_agent_max) #distance to first target region scaled by agent's max speed

    target_to_target = 0 # Initialize to 0, will be updated if there are multiple targets (relative distances between targets)

    if len(targets) > 1:
        target_indexes = [index for index in targets.keys()]
        i = target_indexes.index(target_index) 
        l = 1
        while l <= i:
            target_1 = targets[target_indexes[l - 1]]
            target_2 = targets[target_indexes[l]]
            target_to_target += target_to_target_dist(target_1['center'], target_2['center'], target_1['radius'], target_2['radius'], target_1['u_max'], target_2['u_max'], target_1["remaining_time"], target_2['remaining_time']) * (1 / u_agent_max) #distance between targets scaled by target's max speed

            l += 1

    remaining_t = targets[target_index]['remaining_time']
        
    cbf_value = remaining_t - agent_to_target - target_to_target

    #print("CBF value:", cbf_value)
  
    return cbf_value



def solve_cbf_qp(b_func, agent_state, u_agent_max, disturbance_interval, target_index, current_t, targets, u_rl):
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

    db_dx = -1 * np.array([dx / dist, dy / dist]) * (1 / u_agent_max_cbf) #derivative w.r.t. the agent's state

    target_center, target_radius, u_target_max, remaining_t, _ = targets[target_index]['center'], targets[target_index]["radius"], targets[target_index]['u_max'], targets[target_index]['remaining_time'], targets[target_index]['movement']['type']

    #db_dr = 1 - (u_target_max / u_agent_max) #derivative w.r. to  the remaining time

    #create an array with size len(targets) x 2 for the derivative w.r.t. the target's state
    db_dx_target = np.zeros((len(targets), 2))
    db_dr = np.zeros(len(targets))  # Initialize derivative w.r.t. remaining time for all targets

    if len(targets) > 1 and target_index != first_key:  # If there are multiple targets and we are not at the first target
        target_indexes = [index for index in targets.keys()]
        i = target_indexes.index(target_index) #find the index of the target in the list
        l = 1
        while l <= i: #the derivative will be nonzero for the distance function for the targets before and after the current target
            target_1 = targets[target_indexes[l - 1]]
            target_2 = targets[target_indexes[l]]
            target_3 = targets[target_indexes[l + 1]] if l + 1 <= i else None
            if target_3 is not None:
                dx_prev = target_1['center'][0] - target_2['center'][0]
                dy_prev = target_1['center'][1] - target_2['center'][1]
                dist_prev = np.sqrt(dx_prev**2 + dy_prev**2)
                temp = np.array([dx_prev / dist_prev, dy_prev / dist_prev]) * (1 / u_agent_max_cbf)

                dx_next = target_2['center'][0] - target_3['center'][0]
                dy_next = target_2['center'][1] - target_3['center'][1]
                dist_next = np.sqrt(dx_next**2 + dy_next**2)
                temp += np.array([dx_next / dist_next, dy_next / dist_next]) * (1 / u_agent_max_cbf)

                db_dr[l] = 1 - 2* (targets[target_indexes[l]]['u_max']/ u_agent_max_cbf)

            else:
                dx_prev = target_1['center'][0] - target_2['center'][0]
                dy_prev = target_1['center'][1] - target_2['center'][1]
                dist_prev = np.sqrt(dx_prev**2 + dy_prev**2)
                temp = np.array([dx_prev / dist_prev, dy_prev / dist_prev]) * (1 / u_agent_max_cbf)
                
                db_dr[l] = 1 - (targets[target_indexes[l]]['u_max'] / u_agent_max_cbf)  # derivative w.r.t. the remaining time for the target

            db_dx_target[l] = temp
            l += 1
        #add the derivative for the first target in the sequence:
        target_1 = targets[target_indexes[0]]
        target_2 = targets[target_indexes[1]]

        dx_next = target_1['center'][0] - target_2['center'][0]
        dy_next = target_1['center'][1] - target_2['center'][1]
        dist_next = np.sqrt(dx_next**2 + dy_next**2)
        db_dx_target[0] = np.array([dx_next / dist_next, dy_next / dist_next]) * (1 / u_agent_max_cbf) + np.array([dx / dist, dy / dist]) * (1 / u_agent_max_cbf)# derivative w.r.t. the first target's state
        db_dr[0] = 1 - 2 * (u_target_max_first / u_agent_max_cbf)  # derivative w.r.t. the remaining time for the first target
    else:
        db_dx_target[0] = np.array([dx / dist, dy / dist]) * (1 / u_agent_max_cbf)
        db_dr[0] = 1 - (u_target_max_first / u_agent_max_cbf)  # derivative w.r.t. the remaining time for the first target

    #print('db_dx_target:', db_dx_target)
    #print('db_dr:', db_dr)

    # alpha_min = 0.6  # never zero
    # alpha_max = 1.5
    # d_max = 20.0  # beyond this distance, alpha is at max value
    # alpha = alpha_min + (alpha_max - alpha_min) * min(dist / d_max, 1.0)

    # print('alpha:', alpha)

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

    cbf_constraint = [db_dx @ (u + u_rl) + np.transpose(db_dx_target) @ u_target - np.transpose(db_dr) @ np.ones(len(targets)) + alpha * b_func(agent_state, u_agent_max_cbf, targets, target_index) + delta >= 0,  
        u + u_rl >= u_min,
        u + u_rl <= u_max] 

    # add obstacle avoidance constraints:
    pos_obs = np.array(obstacles[0]['center'])   # live obstacle center from simulator (2,) (for 1 obstacle only!!!!) change later!
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



def solve_tvcbf_qp(agent_state, obstacles, u_ref, agent_radius, alpha_layer, gamma=2.0, u_min=None, u_max=None, solver=cp.OSQP, verbose=False):
    u = cp.Variable(2)

    po = np.array(obstacles[0]['center'])   # live obstacle center from simulator (2,) (for 1 obstacle only!!!!) change later!
    #TODO: generalize for other movement types
    vo = [np.array([obstacles[i]['u_max']*np.cos(obstacles[i]['movement']['heading_angle']), obstacles[i]['u_max']*np.sin(obstacles[i]['movement']['heading_angle'])]) for i in obstacles]  # live obstacle velocity from simulator (2,) (for periodic movement only)
    pr = np.array(agent_state[:2])                  # robot position
    Rr, Ro = agent_radius, obstacles[0]['radius']

    h, grad_h_x, dh_dt, alpha_star = tvcbf_circle_diffopt(pr, po, vo, Rr, Ro, alpha_layer, beta=1.5)

    cons = [(grad_h_x @ u + dh_dt >= -gamma * h)]

    if u_min is not None: 
        cons.append(u >= u_min)
    if u_max is not None: 
        cons.append(u <= u_max)

    prob = cp.Problem(cp.Minimize(cp.sum_squares(u - u_ref)), cons)
    prob.solve(solver=solver, verbose=verbose)

    u_opt = np.array(u.value).reshape(-1)
    return u_opt, {"status": prob.status}



if __name__ == "__main__":
    from simulator import Continuous2DEnv

    agent_init_loc = (-5, -5)  # initial location of the agent
    u_agent_max = 3 # Max vel. of the system


     #scenario-4: #periodic alternative
    t_windows=[[[0,100],[0,25]]] # STL time windows

    point1_1 = (25, 0)
    point1_2 = (40, 0)

    #target_1 = {0: {'id': 1, 'type': 'GF', 'time_window': t_windows[0], 'label': 'target 1', 'center': (30,0) ,'radius': 3, 'u_max': 0.5, 'remaining_time': t_windows[0][1][1], 'movement':{'type': 'periodic', 'point1': point1_1, 'point2': point1_2, 'heading_angle': np.arctan2(point1_2[1] - point1_1[1], point1_2[0] - point1_1[0])}, 'color': 'blue'}}


    point1 = (20, 0)
    point2 = (-15, 0)

    # point1 = (0, -15)
    # point2 = (0, 20)


    obstacle_region_radius = 15
    agent_radius = 1


    obstacles = {
        #0: {'label': 1,'center': (30, 0), 'radius': obstacle_region_radius, 'u_max': 1, 'remaining_time': 100, 'movement':{'type': 'periodic', 'point1': point1, 'point2': point2, 'heading_angle': np.arctan2(point2[1] - point1[1], point2[0] - point1[0])}, 'color': 'blue'},
        0: {'label': 1,'center': (3, -3), 'radius': 1.5, 'u_max': 1, 'remaining_time': 100, 'movement':{'type': 'static'}, 'color': 'blue'},
        1: {'label': 1,'center': (-3, 3), 'radius': 1.5, 'u_max': 1, 'remaining_time': 100, 'movement':{'type': 'static'}, 'color': 'blue'}
        #0: {'id': 0, 'center': (-30, 30), 'radius': target_region_radius, 'u_max': u_target_max0, 'remaining_time': 100, 'movement':{'type': 'circular', 'omega': 0.1, 'center_of_rotation':(-25,30)}, 'color': 'blue'}, #heading angle is in rad
        #1: {'id': 1, 'center': (-30, -30), 'radius': target_region_radius, 'u_max': u_target_max1, 'remaining_time': 100, 'movement':{'type': 'circular', 'omega': 0.1, 'center_of_rotation':(-25,-30)}, 'color': 'red'}, #heading angle is in rad
        #2: {'center': (35, -30), 'radius': target_region_radius, 'u_max': u_target_max1, 'remaining_time': 100, 'movement':{'type': 'circular', 'omega': 0.1, 'center_of_rotation':(35,-30)}, 'color': 'yellow'}, #heading angle is in rad
        #2: {'center': (-20, -20), 'radius': target_region_radius, 'u_max': u_target_max1, 'remaining_time': 200, 'movement':{'type': 'straight', 'heading_angle': 5*np.pi/4}}
    }

    #cbf_targets = [target_1]
    target_1 = None
    simulation_targets = target_1

    goals = {
	0: {'center': (6, 5), 'radius': 2, 'movement':{'type':'static'}}
    }
    disturbance_interval = [-0.1, 0.1]

        #config dictionary for the environment
    config = {
        'init_loc': agent_init_loc, #initial location of the agent (x, y)
        "width": 10.0,
        "height": 10.0,
        "dt": 1,
        "render": True,
        'dt_render': 0.03, #time interval for rendering
        "goals": goals,  # goal regions for the agent
        "obstacles": obstacles,  # dictionary of obstacles
        "targets": simulation_targets,  # dictionary of targets for the CBF
        "dynamics": "single integrator", #dynamics model to use
        'u_agent_max': u_agent_max, #max agent speed
        "randomize_loc": False,  #whether to randomize the agent location at the end of each episode
        "disturbance": disturbance_interval #disturbance range in both x and y directions [w_min, w_max]
    }


    env = Continuous2DEnv(config)
    state = env.reset()

    action_rl = np.array([0.0, 0.0])

    action = action_rl

    episode_length = 35
    
    t = 0
    min_dist_to_obstacle = float('inf')
    while t <= episode_length:
        state, reward, done = env.step(action, t)
        t += 1 #increment time step
