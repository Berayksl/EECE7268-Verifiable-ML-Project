#Time-varying CBF for a moving target region
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import copy



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




def solve_cbf_qp(b_func, agent_state, u_agent_max, target_index, current_t, targets, u_rl):
    """
    Solve QP to get control u satisfying the CBF condition.
    """
    x = agent_state
    u = cp.Variable(2)
    delta = cp.Variable(nonneg=True) #slack variable for the CBF condition


    #get the parameters of the first target in the sequence for the first term:
    first_key = next(iter(targets))
    first_target_center, first_target_radius, u_target_max_first, first_remaining_t = targets[first_key]['center'], targets[first_key]['radius'], targets[first_key]['u_max'], targets[first_key]['remaining_time']

    dx = x[0] - first_target_center[0]
    dy = x[1] - first_target_center[1]
    dist = max(np.sqrt(dx**2 + dy**2), 1e-6)

    db_dx = -1 * np.array([dx / dist, dy / dist]) * (1 / u_agent_max) #derivative w.r.t. the agent's state

    target_center, target_radius, u_target_max, remaining_t, target_movement_type = targets[target_index]['center'], targets[target_index]["radius"], targets[target_index]['u_max'], targets[target_index]['remaining_time'], targets[target_index]['movement']['type']

    db_dr = 1 - (u_target_max / u_agent_max) #derivative w.r. to  the remaining time

    if len(targets) > 1 and target_index != first_key:  # If there are multiple targets and we are not at the first target
        target_indexes = [index for index in targets.keys()]
        i = target_indexes.index(target_index) #find the index of the target in the list
        target_1 = targets[target_indexes[i - 1]]
        target_2 = targets[target_indexes[i]]
        dx_target = target_1['center'][0] - target_2['center'][0]
        dy_target = target_1['center'][1] - target_2['center'][1]
        dist_target = max(np.sqrt(dx_target**2 + dy_target**2), 1e-6)
        db_dx_target = np.array([dx_target / dist_target, dy_target / dist_target]) * (1 / u_agent_max)
    else:
        db_dx_target = np.array([dx / dist, dy / dist]) * (1 / u_agent_max)


    alpha_min = 0.6  # never zero
    alpha_max = 1.5
    d_max = 20.0  # beyond this distance, alpha is at max value
    #alpha = alpha_min + (alpha_max - alpha_min) * min(dist / d_max, 1.0)

    # print('alpha:', alpha)

    alpha = 1.5

    u_min = np.array([-u_agent_max, -u_agent_max])
    u_max = np.array([u_agent_max, u_agent_max])  # might need to change later!

    if target_movement_type == 'circular':
        omega = targets[target_index]['movement']['omega']
        u_target = np.array([np.cos(omega*current_t) * u_target_max, np.sin(omega*current_t) * u_target_max])

    elif target_movement_type == 'straight' or target_movement_type == 'periodic':
        heading_angle = targets[target_index]['movement']['heading_angle']
        u_target = np.array([np.cos(heading_angle) * u_target_max, np.sin(heading_angle) * u_target_max])


    #print(db_dx @ (u + u_rl) + db_dx_target @ u_target - db_dr + alpha * b_func(agent_state, u_agent_max, targets, target_index))

    cbf_constraint = [db_dx @ (u + u_rl) + db_dx_target @ u_target - db_dr + alpha * b_func(agent_state, u_agent_max, targets, target_index) + delta >= 0, #CBF condition
        #control input constraints:
        u + u_rl >= u_min,
        u + u_rl <= u_max
        ]

    Q = np.eye(2)
    slack_weight = 1e4  # Weight for the slack variable

    objective = cp.Minimize(cp.quad_form(u, Q) + slack_weight * delta)  # Minimize control effort and slack variable


    prob = cp.Problem(objective, cbf_constraint)
    prob.solve(solver=cp.ECOS, verbose = False)
    #prob.solve(solver=cp.OSQP, verbose = True)

    if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
        return u.value
    else:
        print("QP failed:", prob.status)
        return None



if __name__ == "__main__":
    from simulator import Continuous2DEnv

    target_region_radius = 8.0  # radius of the target region
    u_target_max0 = 0.2 # max speed of the target
    u_target_max1 = 0.2
    u_agent_max = 10.0  #max agent speed

    #sequence = [1, 2, 1, 2, 1, 2]

    point1 = (-20, 10)
    point2 = (20, 10)

    targets = {
        #0: {'center': (-30, 30), 'radius': target_region_radius, 'u_max': u_target_max0, 'remaining_time': 100, 'movement':{'type': 'periodic', 'point1': point1, 'point2': point2, 'heading_angle': np.arctan2(point2[1] - point1[1], point2[0] - point1[0])}, 'color': 'blue'}
        0: {'id': 0, 'center': (-30, 30), 'radius': target_region_radius, 'u_max': u_target_max0, 'remaining_time': 100, 'movement':{'type': 'circular', 'omega': 0.1, 'center_of_rotation':(-25,30)}, 'color': 'blue'}, #heading angle is in rad
        1: {'id': 1, 'center': (-30, -30), 'radius': target_region_radius, 'u_max': u_target_max1, 'remaining_time': 100, 'movement':{'type': 'circular', 'omega': 0.1, 'center_of_rotation':(-25,-30)}, 'color': 'red'}, #heading angle is in rad
        #2: {'center': (35, -30), 'radius': target_region_radius, 'u_max': u_target_max1, 'remaining_time': 100, 'movement':{'type': 'circular', 'omega': 0.1, 'center_of_rotation':(35,-30)}, 'color': 'yellow'}, #heading angle is in rad
        #2: {'center': (-20, -20), 'radius': target_region_radius, 'u_max': u_target_max1, 'remaining_time': 200, 'movement':{'type': 'straight', 'heading_angle': 5*np.pi/4}}
    }

    initial_targets = [(key, copy.deepcopy(value)) for key, value in targets.items()] #copy the targets into a list to iterate over them

    # goals = {
	# 0: {'center': (20, 13), 'radius': 10}, #goal region for the agent
	# 1: {'center': (-40, 13), 'radius': 10}
	# }

    g1_p1 = (50, -30)
    g1_p2 = (50, 30)

    # goals = {
	# 0: {'center': (20, 13), 'radius': 10, 'movement':{'type':'blinking', 'point1': g1_p1, 'point2': g1_p2, 'blink_duration': 50, 'heading_angle': np.arctan2(g1_p2[1] - g1_p1[1], g1_p2[0] - g1_p1[0])}}, #goal region for the agent
	# }

    goals = {
	0: {'center': (100, 100), 'radius': 0, 'movement':{'type':'static'}}, #goal region for the agent
	}


    #config dictionary for the environment
    config = {
        'init_loc': [60, 50], #initial location of the agent (x, y)
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
    
   
    #CBF parameters

    env = Continuous2DEnv(config)
    state = env.reset()

    action_rl = np.array([0.6, 0.6])

    action = action_rl

    episode_length = 300

    distances = []

    t = 0

    while t <= episode_length and len(targets) > 0:
        #print(targets)
        #calculate the CBF values for each target region and take the minimum:
        cbf_values = {}
        for target_index in targets.keys():
            cbf_value = sequential_CBF(state, u_agent_max, targets, target_index)
            cbf_values[target_index] = cbf_value

        #print(cbf_values)

        min_key = min(cbf_values, key=cbf_values.get)  #find the target region with the minimum CBF value

        #Now solve the QP to get the control input for the target region with the minimum CBF value:

        u_cbf = solve_cbf_qp(sequential_CBF, state, u_agent_max, min_key, t, targets, action_rl)


        action = (u_cbf[0] + action_rl[0], u_cbf[1] + action_rl[1])  # Combine CBF and RL actions


        state, reward, done = env.step(action)

        t += 1
        #decrease the remaining time and update the center for each target region
        for target_index in list(targets.keys()):
            targets[target_index]['remaining_time'] -= 1
            #targets[target_index]['center'] = moving_target(t, center_of_rotation, targets[target_index]['u_max'])

            #calculate the signed distance to each target region:
            target_center = targets[target_index]['center']
            target_radius = targets[target_index]['radius']
            dist = np.linalg.norm(state[:2] - target_center)
            signed_distance = dist - target_radius
            
            if signed_distance <= 0: 
                # #hold inside the target region:
                # targets[target_index]['remaining_time'] = 0 #set the remaining time to 0 to hold inside the target region

                # for i in targets.keys():
                #     cbf_value = sequential_CBF(state, u_agent_max, targets, i)
                #     cbf_values[i] = cbf_value

                # min_key = min(cbf_values, key=cbf_values.get)  #find the target region with the minimum CBF value
                
                # while min_key == target_index:
                #     #print(targets)
                #     u_cbf = solve_cbf_qp(sequential_CBF, state, u_agent_max, min_key, t, targets, action_rl)
                #     action = (u_cbf[0] + action_rl[0], u_cbf[1] + action_rl[1])  # Combine CBF and RL actions

                #     state, reward, done = env.step(action)

                #     for i in targets.keys():
                #         cbf_value = sequential_CBF(state, u_agent_max, targets, i)
                #         cbf_values[i] = cbf_value
                #         if i != target_index:
                #             targets[i]['remaining_time'] -= 1

                #     # print(cbf_values)

                #     min_key = min(cbf_values, key=cbf_values.get)  #find the target region with the minimum CBF value

                #     t += 1


                targets.pop(target_index)  # Remove target region if the agent is inside it
                targets[target_index] = copy.deepcopy(initial_targets[target_index][1])  # add the target region back to the dictionary with the initial parameters

        

        # if done:
        #     break
    
    if targets == {}:
        print("Task completed!")

    plt.close()
    plt.plot(distances)
    plt.xlabel("Time step")
    plt.ylabel("Distance to target region")
    plt.title("Distance to Target Region Over Time")
    plt.grid()
    plt.show()
