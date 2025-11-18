#Time-varying CBF for a moving target region (implementation with derivative arrays for targets like in the paper)
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




def solve_cbf_qp(b_func, agent_state, u_agent_max, disturbance_interval, target_index, current_t, targets, u_rl):
    """
    Solve QP to get control u satisfying the CBF condition.
    """
    w_max = max(abs(disturbance_interval[0]), abs(disturbance_interval[1]))
    u_agent_max_cbf = u_agent_max - w_max #reduce the max agent speed by the disturbance bound (worst-case)

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

    cbf_constraint = [db_dx @ (u + u_rl) + np.transpose(db_dx_target) @ u_target - np.transpose(db_dr) @ np.ones(len(targets)) + alpha * b_func(agent_state, u_agent_max_cbf, targets, target_index) + delta >= 0, #CBF condition
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
        return u.value, delta.value
    else:
        print("QP failed:", prob.status)
        return None, None


if __name__ == "__main__":
    from simulator import Continuous2DEnv
    from task_schedule_py3 import task_scheduler
    
    # #scenario-1:
    # t_windows=[[[0,300],[0,100]],[[0,300],[0,100]]] # STL time windows
    # subformula_types = np.array([4,4]) # 1: F, 2: G, 3: FG, 4: GF | Formula Types
    # agent_init_loc = np.array([0,0]) # Initial pos. (x,y) of the agent
    # #target_radii = np.array([8.0, 8.0])  # radii for each target region
    # roi = np.array([[-30, 30, 10],[-30, -30, 10]]) #3rd dimension is the radius
    # u_tar_max = np.array([.5, .5]) #max velocities of the targets
    # u_agent_max = 11 # Max vel. of the system
    # disturbance_interval = [-1, 1]
    # w_max = max(abs(disturbance_interval[0]), abs(disturbance_interval[1]))
    # u_agent_max = u_agent_max - w_max #reduce the max agent speed by the disturbance bound (worst-case)
    # target_movements = {0: {'type': 'circular', 'omega': 0.1, 'center_of_rotation': (-25, 30)}, 1: {'type': 'circular', 'omega': 0.1, 'center_of_rotation': (0, -35)}, 2: {'type': 'static'}, 3: {'type': 'static'}} #movement patterns for each target region

    # roi_disj = [] #np.copy(roi) # Create alternative RoI's (to be modified)  
    # n_tar = len(roi) # of targets
    # disj_map = np.array([np.arange(0,n_tar)]) 
    # rois = [roi]


    # #scenario-2:
    # t_windows=[[[0,20]],[[0,20]],[[33,36]],[[0,20],[0,5]]] # STL time windows
    # subformula_types = np.array([1,1,2,4]) # 1: F, 2: G, 3: FG, 4: GF | Formula Types
    # agent_init_loc = np.array([0,0]) # Initial pos. (x,y) of the agent
    # u_tar_max = .1*np.array([.6, .6, .4, .5, .55])

    # roi=np.array([[-30, 30, 10],[0,-30, 10],[-30, -30, 10],[10, 20, 10]])

    # point1 = (-30, -30)
    # point2 = (-30, 30)

    # target_movements = {0: {'type': 'circular', 'omega': 0.1, 'center_of_rotation': (-25, 30)}, 1: {'type': 'circular', 'omega': 0.1, 'center_of_rotation': (0, -35)}, 2: {'type': 'periodic', 'point1': point1, 'point2': point2, 'heading_angle': np.arctan2(point2[1] - point1[1], point2[0] - point1[0])}, 3: {'type': "static"}} #movement patterns for each target region

    # u_agent_max = 11 # Max vel. of the system
    # disturbance_interval = [-1, 1]
    # w_max = max(abs(disturbance_interval[0]), abs(disturbance_interval[1]))
    # u_agent_max = u_agent_max - w_max #reduce the max agent speed by the disturbance bound (worst-case)

    
    # roi_disj = [] #np.copy(roi) # Create alternative RoI's (to be modified)  
    # n_tar = len(roi) # of targets
    # disj_map = np.array([np.arange(0,n_tar)]) 
    # rois = [roi]

    # #scenario-3: #1 task with an alternative
    # t_windows=[[[0,20]],[[0,20]],[[33,36]],[[0,20],[0,10]]] # STL time windows
    # subformula_types = np.array([1,1,2,4]) # 1: F, 2: G, 3: FG, 4: GF | Formula Types
    # agent_init_loc = np.array([0,0]) # Initial pos. (x,y) of the agent
    # u_tar_max = 0.1*np.array([.6, .6, .4, .5, .55])

    # roi=np.array([[-60, -30, 10],[0,-30, 10],[-30, -30, 10],[10, 20, 10]])
    # target_movements = {0: {'type': 'circular', 'center_of_rotation': (-25, -30)}, 1: {'type': 'circular', 'center_of_rotation': (0, -35)}, 2: {'type': 'static'}, 3: {'type': 'static'}} #movement patterns for each target region
    
    # u_agent_max = 11 # Max vel. of the system
    # disturbance_interval = [-1, 1]
    # w_max = max(abs(disturbance_interval[0]), abs(disturbance_interval[1]))
    # u_agent_max = u_agent_max - w_max #reduce the max agent speed by the disturbance bound (worst-case)

    # roi_disj = np.copy(roi) # Create alternative RoI's (to be modified)
    # n_tar = np.size(roi,0) # of targets
    # alt_inds=np.zeros(n_tar) # Tasks with an alternative will be nonzero only
    # alt_inds[0]=1
    # roi_disj[alt_inds>0]=np.array([[-30,30,10]]) # Alternative target to 1st task
    # alt_movements = {0: {'type': 'circular', 'omega': 0.1, 'center_of_rotation': (-25, 30)}}
    # #disj_map = np.array([np.arange(0,n_tar),np.array([0,0,0,5,0])])
    # rois = [roi,roi_disj] # Create alternative full-RoI lists

    # #scenario-4:(not working)
    # t_windows=[[[0,200],[0,90]],[[0,200],[50,60]],[[0,200],[100,110]]] # STL time windows
    # subformula_types = np.array([4,3,3]) # 1: F, 2: G, 3: FG, 4: GF | Formula Types
    # #t_windows=[[[0,200],[0,90]],[[20,30]],[[120,130]]]
    # #subformula_types = np.array([4,2,2]) # 1: F, 2: G, 3: FG, 4: GF | Formula Types
    # agent_init_loc = np.array([0,0]) # Initial pos. (x,y) of the agent
    # u_tar_max = .1*np.array([0.5, 0.5, 0.5])

    # roi=np.array([[-30, 0, 10],[0,-30, 10],[-30, -30, 10]])

    # point1 = (-30, -30)
    # point2 = (-30, 30)

    # target_movements = {0: {'type': 'circular', 'omega': 0.1, 'center_of_rotation': (-25, 30)}, 1: {'type': 'circular', 'omega': 0.1, 'center_of_rotation': (0, -35)}, 2: {'type': 'periodic', 'point1': point1, 'point2': point2, 'heading_angle': np.arctan2(point2[1] - point1[1], point2[0] - point1[0])}, 3: {'type': "static"}} #movement patterns for each target region

    # u_agent_max = 11 # Max vel. of the system
    # disturbance_interval = [-1, 1]
    # w_max = max(abs(disturbance_interval[0]), abs(disturbance_interval[1]))
    # u_agent_max = u_agent_max - w_max #reduce the max agent speed by the disturbance bound (worst-case)

    
    # roi_disj = [] #np.copy(roi) # Create alternative RoI's (to be modified)  
    # n_tar = len(roi) # of targets
    # disj_map = np.array([np.arange(0,n_tar)]) 
    # rois = [roi]


    #scenario-5
    t_windows=[[[0,60],[0,10]],[[150,180],[0,10]],[[0,190],[0,100]]] # STL time windows
    subformula_types = np.array([3,3,4]) # 1: F, 2: G, 3: FG, 4: GF | Formula Types
    #t_windows=[[[0,200],[0,90]],[[20,30]],[[120,130]]]
    #subformula_types = np.array([4,2,2]) # 1: F, 2: G, 3: FG, 4: GF | Formula Types
    agent_init_loc = np.array([-50,-30]) # Initial pos. (x,y) of the agent
    u_tar_max = np.array([1, 1, 1])

    roi = np.array([[-20, 65, 11],[-60,10, 11],[0, -60, 11]])

    point1 = (-60, -60)
    point2 = (-60, 60)

    #target_movements = {0: {'type': 'random_walk', 'heading_angle': np.random.uniform(0, 2 * np.pi)}, 1: {'type': 'random_walk', 'heading_angle': np.random.uniform(0, 2 * np.pi)}, 2: {'type': 'periodic', 'point1': point1, 'point2': point2, 'heading_angle': np.arctan2(point2[1] - point1[1], point2[0] - point1[0])}, 3: {'type': "static"}} #movement patterns for each target region
    target_movements = {0: {'type': 'random_walk', 'heading_angle': np.random.uniform(0, 2 * np.pi)}, 1: {'type': 'random_walk', 'heading_angle': np.random.uniform(0, 2 * np.pi)}, 2: {'type': 'circular', 'center_of_rotation': (10, 0)}} #movement patterns for each target region


    u_agent_max = 12 # Max vel. of the system
    disturbance_interval = [-1, 1]

    roi_disj = [] #np.copy(roi) # Create alternative RoI's (to be modified)  
    n_tar = len(roi) # of targets
    disj_map = np.array([np.arange(0,n_tar)]) 
    rois = [roi]
    target_colors = ['blue', 'red', 'green', 'black', 'yellow']

    sequence, rem_time, rem_time_realistic, best_roi, gamma, portions, portions0 = task_scheduler(rois,t_windows,subformula_types,agent_init_loc,u_agent_max,u_tar_max)

    print(sequence, rem_time, rem_time_realistic, gamma, best_roi)
    target_labels = ['Target1', 'Target2', 'Charger']
   
    task_types = ["F", "G", "FG", "GF"]
    #create the target dictionary:
    targets = {}
    for i, target_id in enumerate(sequence):
        target_center = best_roi[target_id]
        targets[i] = {
            'id': target_id,
            'type': task_types[subformula_types[target_id]-1],
            'time window': t_windows[target_id],
            'center': best_roi[target_id][:2],
            'radius': best_roi[target_id][2],
            'u_max': u_tar_max[target_id],
            'remaining_time': rem_time_realistic[i],
            'movement': target_movements[target_id],
            'color': target_colors[target_id],
            'label': target_labels[target_id]
        }

    # target_colors = ['blue', 'red', 'green', 'black', 'yellow']
    # simulation_targets = {}
    # for key, value in target_movements.items():
    #     simulation_targets[key] = {
    #         'id': key,
    #         'center': roi[key][:2],
    #         'radius': roi[key][2],
    #         'u_max': u_tar_max[key],
    #         'movement': value,
    #         'color': target_colors[key],
    #         'label': target_labels[key]
    #     }

    initial_remaining_times = {i: targets[i]['remaining_time'] for i in targets}  #dictionary to store the initial remaining times for each target region

    goals = {
	0: {'center': (50, 0), 'radius': 11, 'movement':{'type':'static'}}, #goal region for the agent
	}

    print("Targets:", targets)

    #config dictionary for the environment
    config = {
        'init_loc': agent_init_loc, #initial location of the agent (x, y)
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
        "randomize_loc": False,  #whether to randomize the agent location at the end of each episode
        "disturbance": disturbance_interval #disturbance range in both x and y directions [w_min, w_max]
    }
    
   
    #CBF parameters

    env = Continuous2DEnv(config)
    state = env.reset()

    action_rl = np.array([0.6, 0.6])

    action = action_rl

    episode_length = 400

    distances = []

    t = 0

    while t <= episode_length and len(targets) > 0:
        #print("Current targets:", targets)
        #print("target keys:", targets.keys())
        #print("Current targets:", targets, "\n")
        #calculate the CBF values for each target region and take the minimum:
        cbf_values = {}
        for target_index in targets.keys():
            cbf_value = sequential_CBF(state, u_agent_max, targets, target_index)
            cbf_values[target_index] = cbf_value

        print(cbf_values)

        min_key = min(cbf_values, key=cbf_values.get)  #find the target region with the minimum CBF value
        #print("Minimum CBF value target index:", min_key)

        #Now solve the QP to get the control input for the target region with the minimum CBF value:
        u_cbf, slack_variable = solve_cbf_qp(sequential_CBF, state, u_agent_max, disturbance_interval, min_key, t, targets, action_rl)
        #print("Slack variable:", slack_variable)

        action = (u_cbf[0] + action_rl[0], u_cbf[1] + action_rl[1])  # Combine CBF and RL actions

        state, reward, done = env.step(action)
        #print('Time:',t)
        #print('Targets:', targets)

        for target_index in list(targets.keys()):
            first_key = next(iter(targets))
            targets[target_index]['remaining_time'] -= 1 #decrement the remaining time for all target regions
            #print("Target index:", target_index, "Remaining time:", targets[target_index]['remaining_time'])
            #targets[target_index]['center'] = moving_target(t, center_of_rotation, targets[target_index]['u_max'])
            task_type = targets[target_index]['type']
            time_window = targets[target_index]['time window']

            #calculate the signed distance to each target region:
            target_center = targets[target_index]['center']
            target_radius = targets[target_index]['radius']
            dist = np.linalg.norm(state[:2] - target_center)
            signed_distance = dist - target_radius

            if signed_distance < 0:
                print("inside target region", targets[target_index]['id'], "at time", t)

            remove_target = False #flag to indicate whether to remove the visited target region

            if task_type == "F":
                # Handle F type tasks
                a = time_window[0][0]
                b = time_window[0][1]
                if t >= a and t <= b and signed_distance <= 0:
                    #within the time window
                    remove_target = True
                else:
                    continue #do not remove the target region if the agent is outside the time window

            elif task_type == "G":
                # Handle G type tasks
                a = time_window[0][0]
                b = time_window[0][1]
                if t == a and signed_distance <= 0:
                    remove_target = True #remove the target region if the agent is inside it at the start of the time window
                    #Hold inside the target region until the end of the time window:
                    targets[target_index]['remaining_time'] = 0 #set the remaining time to the length of the time window
                    for j in range(b-a):
                        print("Holding inside target region", targets[target_index]['id'], "at time", t)
                        t += 1

                        for key in targets.keys():
                            if key != target_index: #do not decrement the remaining time for the current target region
                                targets[key]['remaining_time'] -= 1

                        u_cbf, slack_variable = solve_cbf_qp(sequential_CBF, state, u_agent_max, disturbance_interval, min_key, t, targets, action_rl)

                        action = (u_cbf[0] + action_rl[0], u_cbf[1] + action_rl[1])  # Combine CBF and RL actions

                        state, reward, done = env.step(action)
                        print('Targets:', targets)
                else:
                    continue
            elif task_type == "FG":
                # Handle FG type tasks
                a = time_window[0][0]
                b = time_window[0][1]
                c = time_window[1][0]
                d = time_window[1][1]
                if t >= (a+c) and t <= (b+c) and signed_distance <= 0:
                    #within the time window
                    remove_target = True
                    targets[target_index]['remaining_time'] = 0 #set the remaining time to the length of the time window
                    for j in range(d-c):
                        #print("Holding inside target region", targets[target_index]['id'], "at time", t)
                        t += 1

                        for key in targets.keys():
                            if key != target_index: #do not decrement the remaining time for the current target region
                                targets[key]['remaining_time'] -= 1

                        u_cbf, slack_variable = solve_cbf_qp(sequential_CBF, state, u_agent_max, disturbance_interval, min_key, t, targets, action_rl)

                        action = (u_cbf[0] + action_rl[0], u_cbf[1] + action_rl[1])  # Combine CBF and RL actions

                        state, reward, done = env.step(action)


                else: continue

            elif task_type == "GF":
                # Handle GF type tasks
                a = time_window[0][0]
                b = time_window[0][1]
                c = time_window[1][0]
                d = time_window[1][1]
                if t >= a + c and t <= b + d and signed_distance <= 0 and first_key == target_index: #only remove the target region if it is the first in the sequence
                    #within the time window
                    remove_target = True
                    ##################################################################################
                    #UPDATE the remaining time for the next target region with the same id (if any):
                    ##################################################################################
                    keys = list(targets.keys())
                    this_id = targets[target_index]['id']
                    next_key = None
                    i = keys.index(target_index)
                    for kk in keys[i+1:]:              # look ahead
                        if targets[kk]['id'] == this_id:
                            next_key = kk              # found the next duplicate
                            break
                    if next_key is not None:
                        targets[next_key]['remaining_time'] = d - c #set the next iteration's remaining time to the second time window's length
                        print("Updated remaining time for target", targets[next_key]['id'], "to", targets[next_key]['remaining_time'])
                        print(targets)
                else: continue
            
            #remove the target if the conditions are satisfied:
            if remove_target:
                #print("Agent is inside target region:", targets[target_index]['id'])
                targets.pop(target_index)  # Remove target region if the agent is inside it

        t += 1 #increment time step

        # #update the remaining time for all target regions:
        # for target_index in targets.keys():
        #     init_rem_t = initial_remaining_times[target_index]
        #     targets[target_index]['remaining_time'] = init_rem_t - t


        # if done:
        #     break

    if targets == {}:
        print("Task completed!")

    # plt.close()
    # plt.plot(distances)
    # plt.xlabel("Time step")
    # plt.ylabel("Distance to target region")
    # plt.title("Distance to Target Region Over Time")
    # plt.grid()
    # plt.show()
