import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

from dynamics import UnicycleDynamics, SingleIntegratorDynamics
#from Sequential_CBF import sequential_CBF

class Continuous2DEnv:
    def __init__(self, config):
        """
        Initializes the environment with given configuration.
        :param config: Dictionary containing environment parameters.
        """
        self.width = config.get("width", 10.0)
        self.height = config.get("height", 10.0)
        self.dt = config.get("dt", 0.1)
        self.render = config.get("render", False)
        self.show_timer = config.get("show_timer", True)
        self.dt_render = config.get("dt_render", 0.001)
        self.goals = config.get("goals", None)  # dictionary of goals for the agent
        self.obstacles = config.get("obstacles", None)  # dictionary of obstacles
        self.random_loc = config.get("randomize_loc", True)
        self.targets = config.get("targets", None)  # dictionary of targets for the CBF
        self.init_loc = np.array(config.get("init_loc", [5.0, 5.0]))
        self.action_max = config.get("action_max")
        self.action_min = config.get("action_min")
        self.dynamics = config.get("dynamics", "unicycle")
        #self.u_agent_max = config.get("u_agent_max", None)  # max agent speed

        if config.get("disturbance") is not None: #min and max disturbance:
            self.disturbance = True
            self.w_min = config.get("disturbance")[0] 
            self.w_max = config.get("disturbance")[1]
        else:
            self.disturbance = False
        
        if self.targets is not None:
            self.initial_target_centers = {target_index: self.targets[target_index]['center'] for target_index in self.targets.keys()}
        else:
            self.initial_target_centers = {}
        self.simulation_timer = 0


        if self.dynamics == "unicycle":
            self.observation_space = np.zeros((3,))
            self.action_space = np.zeros((2,))
            
        elif self.dynamics == "single integrator":
            self.observation_space = np.zeros((2,))
            self.action_space = np.zeros((2,))

        self.reset()
        #self.precompute_cbf_values(resolution=100)  # Precompute CBF values for visualization

        if self.render:
            self.fig, self.ax = plt.subplots()

            # --- set window location/size (pixels), backend-safe ---
            mng = plt.get_current_fig_manager()
            try:
                # TkAgg backend
                mng.window.wm_geometry("900x650+120+80")  # WxH+X+Y
            except Exception:
                try:
                    # Qt backend
                    # setGeometry(x, y, width, height)
                    mng.window.setGeometry(1200, 80, 1500, 1500)
                except Exception:
                    pass  # other backends may not support moving the window


            self.ax.set_xlim(-self.width, self.width)
            self.ax.set_ylim(-self.height, self.height)
            self.agent_plot, = self.ax.plot([], [], 'ro', label='Agent', markersize=10)

            self.goal_plots = []
            for goal in self.goals.values():
                goal_plot = plt.Circle(goal['center'], goal['radius'], color='g', alpha=0.5, label='Goal')
                self.goal_plots.append(goal_plot)
                self.ax.add_patch(goal_plot)

            if self.show_timer:
                self.timer_text = self.ax.text(
                    0.02, 0.98, f"Time:{self.simulation_timer%301}",
                    transform=self.ax.transAxes,
                    ha='left', va='top', fontsize=25,
                    bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='black', alpha=0.75),
                    zorder=10
                )

            #self.target_region_plot = plt.Circle(self.target_region_center, self.target_region_radius, color='b', fill=False, linestyle='-', label='Target Region')
            #self.ax.add_patch(self.target_region_plot)
            self.safe_region_plots = []
            self.target_region_patches = []
            self.obstacle_region_patches = []
            # self.ax.legend()
            #remove the axis ticks:
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            #self.ani = animation.FuncAnimation(self.fig, self.update_animation, interval=10, cache_frame_data=False)
            self.fig.show()
            self.fig.canvas.draw()






    # def precompute_cbf_values(self, resolution=500):
    #     x_vals = np.linspace(-self.width, self.width, resolution)
    #     y_vals = np.linspace(-self.height, self.height, resolution)
    #     X, Y = np.meshgrid(x_vals, y_vals)

    #     self.CBF_cache = {}
    #     for remaining_t in range(self.task_period + 1):
    #         Z = np.zeros_like(X)
    #         for i in range(X.shape[0]):
    #             for j in range(X.shape[1]):
    #                 state = (X[i, j], Y[i, j])
    #                 Z[i, j] = CBF(state, remaining_t, self.u_agent_max, moving_target(self.task_period - remaining_t, self.target_region_center[0], self.target_region_center[1], self.omega), self.target_region_radius, self.u_target_max)
    #         self.CBF_cache[remaining_t] = Z

    #     self.CBF_grid = (X, Y)



    # def plot_cbf_zero_level(self, remaining_t):
    #     if hasattr(self, 'CBF_cache') and remaining_t in self.CBF_cache:
    #         X, Y = self.CBF_grid
    #         Z = self.CBF_cache[remaining_t]
    #         return X, Y, Z
    #     else:
    #         raise ValueError(f"CBF values for remaining_t = {remaining_t} not found in cache.")



    def cbf_zero_level(self, targets, u_agent_max, resolution=200):

        xlim = (-self.width, self.width)
        ylim = (-self.height, self.height)

        x_vals = np.linspace(xlim[0], xlim[1], resolution)
        y_vals = np.linspace(ylim[0], ylim[1], resolution)

        X, Y = np.meshgrid(x_vals, y_vals)

        cbf_levels = {}

        for target_index in targets.keys():
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    state = (X[i, j], Y[i, j])
                    Z[i, j] = sequential_CBF(state, u_agent_max, targets, target_index)

            cbf_levels[target_index] = (X,Y,Z)

        return cbf_levels
    
    
    def reset(self):
        """Resets the environment."""
        if self.random_loc:
            x = np.random.uniform(-self.width, self.width)
            y = np.random.uniform(-self.height, self.height)
            #resample if inside the goal:
            goal_centers = [goal['center'] for goal in self.goals.values()]
            goal_radii = [goal['radius'] for goal in self.goals.values()]
            for center, radius in zip(goal_centers, goal_radii):
                dist_to_goal = np.linalg.norm(np.array([x, y]) - np.array(center))
                while dist_to_goal <= radius + 2: #add a small buffer
                    x = np.random.uniform(-self.width, self.width)
                    y = np.random.uniform(-self.height, self.height)
                    dist_to_goal = np.linalg.norm(np.array([x, y]) - np.array(center))

            if self.dynamics == 'unicycle':
                theta = self.init_loc[2] + np.random.uniform(-np.pi, np.pi)
                self.agent = UnicycleDynamics(x=x, y=y, theta=theta, dt=self.dt)
                state = np.array([self.agent.x, self.agent.y, self.agent.theta])
            elif self.dynamics == 'single integrator':
                self.agent = SingleIntegratorDynamics(x=x, y=y, dt=self.dt)
                state = np.array([self.agent.x, self.agent.y])
        else:
            if self.dynamics == 'unicycle':
                x = self.init_loc[0]
                y = self.init_loc[1]
                theta = self.init_loc[2]
            elif self.dynamics == 'single integrator':
                x = self.init_loc[0]
                y = self.init_loc[1]

        if self.dynamics == "unicycle":
            self.agent = UnicycleDynamics(x=x, y=y, theta = theta, dt=self.dt)
            state = np.array([self.agent.x, self.agent.y, self.agent.theta])
        elif self.dynamics == "single integrator":
            self.agent = SingleIntegratorDynamics(x=x, y=y, dt=self.dt)
            state = np.array([self.agent.x, self.agent.y])
            #print("Single Integrator Dynamics")
        
        self.initial_state = state.copy()
        return state
    
    # def compute_reward(self):
    #     """Computes the dense reward based on distance to the goal."""
    #     dist_to_goal = np.linalg.norm(np.array([self.agent.x, self.agent.y]) - self.goal_location)

    #     reward = -dist_to_goal
    #     # if dist_to_obstacle <= self.obstacle_size:
    #     #     reward -= 10  # Penalty for hitting an obstacle
        
    #     if dist_to_goal <= self.goal_size:
    #         reward += 100  # Reward for reaching the goal
    #         #print("Goal Reached!")

    #     return reward
    


    def compute_reward(self, current_time): #updated to include multiple goals
        pos = np.array([self.agent.x, self.agent.y])
        # List of distances to all goal centers
        dist = np.linalg.norm(pos - np.array(self.goals[0]['center']))

        rho = self.goals[0]['radius'] - dist  # Dense part: encourages reaching nearest goal
        rho_max = self.goals[0]['radius']

        initial_dist = np.linalg.norm(self.init_loc[0:2] - np.array(self.goals[0]['center']))
        gamma_0 = abs(self.goals[0]['radius'] - initial_dist)
        gamma_inf = 0.99 * min(gamma_0, rho_max) #might need to change later!!!

        t_star = 39 #assuming F[0, 40]

        l = 1/t_star * np.log((gamma_0 - gamma_inf) / (rho_max - gamma_inf))

        gamma = (gamma_0 - gamma_inf) * np.exp(-l * (current_time+1)) + gamma_inf
        
        penalty = 0
        #penalty for hitting obstacles
        for obstacle in self.obstacles.values():
            dist_to_obstacle = np.linalg.norm(pos - np.array(obstacle['center']))
            if dist_to_obstacle <= obstacle['radius']:
                penalty = -100  # Large penalty for hitting an obstacle

        beta = 0.15 #weight for obstacle penalty

        reward = rho + gamma - rho_max + beta * penalty

        if rho >= 0:
            reward += 10  # Reward for reaching the goal

        return reward
    
    def step(self, action, current_time):
        """
        Takes an action (linear and angular velocity) and updates the agent's position.
        :param action: A tuple (v, w) representing linear and angular velocity.
        :return: next_state, reward, done
        """
        #make the updates:
        if self.disturbance:
            w = np.random.uniform(self.w_min, self.w_max, size=2) #sample noise
            action = action + w #add noise to the action
            #print(action)
        
        state = self.agent.update(action) #update the agent state

        #update the target region states:
        if self.targets is not None:
            for target_index in self.targets.keys():
                x_new, y_new = self.dynamic_target(self.simulation_timer, target_index)
                self.targets[target_index]['center'] = (x_new, y_new)

        #update the obstacle states:
        if self.obstacles is not None:
            for obstacle_index in self.obstacles.keys():
                x_new, y_new = self.dynamic_obstacle(self.simulation_timer, obstacle_index)
                self.obstacles[obstacle_index]['center'] = (x_new, y_new)

        #update the goal states:
        for goal_id in self.goals.keys():
            x_new, y_new = self.dynamic_goal(self.simulation_timer, goal_id)
            self.goals[goal_id]['center'] = (x_new, y_new)

        self.simulation_timer += 1

        
        self.agent.x = np.clip(self.agent.x, -self.width, self.width)
        self.agent.y = np.clip(self.agent.y, -self.height, self.height)
        
        # Compute reward
        reward = self.compute_reward(current_time)

        # Calculate distance to the goals
        dist_to_goals = {goal_id: np.linalg.norm(np.array([self.agent.x, self.agent.y]) - np.array(goal['center'])) for goal_id, goal in self.goals.items()}
        # Check if any goal is reached
        done = any(dist <= goal['radius'] for dist, goal in zip(dist_to_goals.values(), self.goals.values()))

        if self.render:
            self.update_plot()
                
        return state, reward, done
    
    
    def update_plot(self):
        time.sleep(self.dt_render)
        # Update agent position
        self.agent_plot.set_data([self.agent.x], [self.agent.y])

        if self.render and self.show_timer:
            self.timer_text.set_text(f"Time:{self.simulation_timer%301}")
        
        # Update targets' positions
        for patch in self.target_region_patches:
            patch.remove()  # remove old
            
        for label in getattr(self, "target_labels", []):
            label.remove()

        self.target_region_patches = []
        self.target_labels = []
        if self.targets is not None:
            for i, target_info in self.targets.items():
                patch = plt.Circle(target_info["center"], target_info["radius"], color=target_info['color'], fill=False, linestyle='-')
                self.ax.add_patch(patch)
                self.target_region_patches.append(patch)
            cx, cy = target_info["center"]
            target_label = target_info['label']
            label = self.ax.text(
                cx, cy, str(target_label),
                ha='center', va='center',
                fontsize=10, color=target_info['color'],
                weight='bold'
            )
            self.target_labels.append(label)

        if self.obstacles is not None:
            # Update obstacles' positions
            for patch in self.obstacle_region_patches:
                patch.remove()

            self.obstacle_region_patches = []
            for obstacle_id, obstacle_info in self.obstacles.items():
                patch = plt.Circle(obstacle_info["center"], obstacle_info["radius"], color='black', alpha=1, label='Obstacle')
                self.ax.add_patch(patch)
                self.obstacle_region_patches.append(patch)


        # Update goals' positions
        for goal_plot in self.goal_plots:
            goal_plot.remove()

        self.goal_plots = []
        for goal_id, goal_info in self.goals.items():
            patch = plt.Circle(goal_info["center"], goal_info["radius"], color='g', alpha=0.5, label='Goal')
            self.ax.add_patch(patch)
            self.goal_plots.append(patch)

        # Redraw
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    # def update_animation(self, frame):
    #     if hasattr(self, 'safe_region_plot'):
    #         for coll in self.safe_region_plot.collections:
    #             try:
    #                 coll.remove()
    #             except ValueError:
    #                 pass  # It's already removed

    #     if hasattr(self, 'target_region_plots'):
    #         for patch in self.target_region_plots:
    #             if patch in self.ax.patches:
    #                 patch.remove()

    #     self.target_region_plots = []

    #     if self.targets:
    #         for target_id, target_info in self.targets.items():
    #             center = target_info["center"]
    #             radius = target_info["radius"]
    #             patch = plt.Circle(center, radius, color=target_info['color'], fill=False, linestyle='-', label='Target Region' if frame == 0 else "")
    #             self.ax.add_patch(patch)
    #             self.target_region_plots.append(patch)


    #     # Update the agent's position in the plot
    #     x = np.array([self.agent.x])  # Convert scalar to NumPy array
    #     y = np.array([self.agent.y])  # Convert scalar to NumPy array

    #     self.agent_plot.set_data(x, y)
    #     self.ax.legend()

    def dynamic_target(self, current_t, target_id):
        if self.targets is not None:
            movement_type = self.targets[target_id]['movement']['type']
            #TODO: add more movement types

            if movement_type == 'circular':
                x0, y0 = self.initial_target_centers[target_id]
                u_target_max = self.targets[target_id]['u_max']
                x, y = self.targets[target_id]['center'] #current position
                xc, yc = self.targets[target_id]['movement']['center_of_rotation']

                if 'theta' not in self.targets[target_id]['movement']:
                    theta = np.arctan2(y0 - yc, x0 - xc) #initial angle
                else:
                    theta = self.targets[target_id]['movement']['theta']

                turning_radius = np.linalg.norm(np.array([x0 - xc, y0 - yc]))
                omega = u_target_max / turning_radius #angular velocity
                
                # Calculate the new angle after time_elapsed
                theta += omega * self.dt
                turning_radius = u_target_max / omega

                x_new = xc + turning_radius * np.cos(theta)
                y_new = yc + turning_radius * np.sin(theta)

                #stop if hit another target:
                for other_id, other_info in self.targets.items():
                    if other_id != target_id:
                        other_center = np.array(other_info['center'])
                        dist = np.linalg.norm(np.array([x_new, y_new]) - other_center)
                        if dist < (self.targets[target_id]['radius'] + other_info['radius']):
                            x_new, y_new = self.targets[target_id]['center'] #stay at the current position
                            theta -= omega * self.dt #stay at the current angle

                # Update theta in targets dictionary
                self.targets[target_id]['movement']['theta'] = theta

            elif movement_type == 'straight':
                x0, y0 = self.initial_target_centers[target_id]
                heading_angle = self.targets[target_id]['movement']['heading_angle']
                u_target_max = self.targets[target_id]['u_max']

                x_new = x0 + np.cos(heading_angle) * u_target_max * current_t
                y_new = x0 + np.sin(heading_angle) * u_target_max * current_t

            elif movement_type == "static":
                #no movement, just return the current position
                x_new, y_new = self.targets[target_id]['center']

            elif movement_type == "periodic":
                # New movement type: periodic back-and-forth between two points
                point1 = np.array(self.targets[target_id]['movement']['point1'])
                point2 = np.array(self.targets[target_id]['movement']['point2'])
                u_target_max = self.targets[target_id]['u_max']

                total_distance = np.linalg.norm(point2 - point1)
                total_time = total_distance / u_target_max

                # Determine current phase within the full cycle (forth + back)
                cycle_time = 2 * total_time
                t_mod = current_t % cycle_time

                if t_mod < total_time:
                    # Moving from point1 to point2
                    alpha = t_mod / total_time
                    position = point1 + alpha * (point2 - point1)
                    direction_vector = point2 - point1
                else:
                    # Moving back from point2 to point1
                    alpha = (t_mod - total_time) / total_time
                    position = point2 - alpha * (point2 - point1)
                    direction_vector = point1 - point2

                x_new, y_new = position[0], position[1]

                heading_angle = np.arctan2(direction_vector[1], direction_vector[0])

                # #stop if hit another target:
                # for other_id, other_info in self.targets.items():
                #     if other_id != target_id:
                #         other_center = np.array(other_info['center'])
                #         dist = np.linalg.norm(np.array([x_new, y_new]) - other_center)
                #         if dist < (self.targets[target_id]['radius'] + other_info['radius']):
                #             x_new, y_new = self.targets[target_id]['center'] #stay at the current position

                # Update heading angle in targets dictionary
                self.targets[target_id]['movement']['heading_angle'] = heading_angle

            elif movement_type == 'random_walk':
                # 0) keep your random re-heading
                if self.simulation_timer % 50 == 0:
                    heading_angle = np.random.uniform(0, 2*np.pi)
                    self.targets[target_id]['movement']['heading_angle'] = heading_angle
                else:
                    heading_angle = self.targets[target_id]['movement']['heading_angle']

                # 1) base velocity
                speed = self.targets[target_id]['u_max'] * 0.5
                v = np.array([np.cos(heading_angle), np.sin(heading_angle)]) * speed

                # 2) super-light separation from neighbors
                AVOID_RADIUS = 2.0   # start with 2–3; tune
                AVOID_GAIN   = 0.8   # 0.5–1.0 is typical
                p = np.array(self.targets[target_id]['center'], dtype=float)
                r_self = self.targets[target_id]['radius']

                rep = np.zeros(2, float)
                for oid, o in self.targets.items():
                    if oid == target_id:
                        continue
                    po = np.array(o['center'], dtype=float)
                    ro = o['radius']
                    delta = p - po
                    d = np.linalg.norm(delta)
                    R = r_self + ro + AVOID_RADIUS
                    if d < R and d > 1e-6:
                        rep += (delta / d) * (R - d) / R   # push straight away

                # apply small push, keep speed ~constant
                v = v + AVOID_GAIN * rep
                n = np.linalg.norm(v)
                #if n > 1e-6:
                    #v = v / n * speed
                heading_angle = float(np.arctan2(v[1], v[0]))

                # 3) integrate
                p_new = p + v * self.dt

                # 4) simple wall bounce
                if p_new[0] < -self.width or p_new[0] > self.width:
                    heading_angle = np.pi - heading_angle
                    v = np.array([np.cos(heading_angle), np.sin(heading_angle)]) * speed
                    p_new[0] = np.clip(p_new[0], -self.width, self.width)

                if p_new[1] < -self.height or p_new[1] > self.height:
                    heading_angle = -heading_angle
                    v = np.array([np.cos(heading_angle), np.sin(heading_angle)]) * speed
                    p_new[1] = np.clip(p_new[1], -self.height, self.height)

                # 5) last-resort bounce if still overlapping
                for oid, o in self.targets.items():
                    if oid == target_id:
                        continue
                    po = np.array(o['center'], float)
                    R = r_self + o['radius']
                    if np.linalg.norm(p_new - po) < R:
                        heading_angle += np.pi  # flip 180°
                        v = np.array([np.cos(heading_angle), np.sin(heading_angle)]) * speed
                        p_new = p + v * self.dt
                        break

                # 6) commit
                x_new, y_new = p_new.tolist()
                self.targets[target_id]['movement']['heading_angle'] = heading_angle

        return (x_new, y_new)
    

    def dynamic_obstacle(self, current_t, target_id):
        if self.obstacles is not None:
            movement_type = self.obstacles[target_id]['movement']['type']
            #TODO: add more movement types

            if movement_type == 'circular':
                x0, y0 = self.initial_target_centers[target_id]
                u_target_max = self.obstacles[target_id]['u_max']
                x, y = self.obstacles[target_id]['center'] #current position
                xc, yc = self.obstacles[target_id]['movement']['center_of_rotation']

                if 'theta' not in self.obstacles[target_id]['movement']:
                    theta = np.arctan2(y0 - yc, x0 - xc) #initial angle
                else:
                    theta = self.obstacles[target_id]['movement']['theta']

                turning_radius = np.linalg.norm(np.array([x0 - xc, y0 - yc]))
                omega = u_target_max / turning_radius #angular velocity
                
                # Calculate the new angle after time_elapsed
                theta += omega * self.dt
                turning_radius = u_target_max / omega

                x_new = xc + turning_radius * np.cos(theta)
                y_new = yc + turning_radius * np.sin(theta)

                #stop if hit another target:
                for other_id, other_info in self.obstacles.items():
                    if other_id != target_id:
                        other_center = np.array(other_info['center'])
                        dist = np.linalg.norm(np.array([x_new, y_new]) - other_center)
                        if dist < (self.obstacles[target_id]['radius'] + other_info['radius']):
                            x_new, y_new = self.obstacles[target_id]['center'] #stay at the current position
                            theta -= omega * self.dt #stay at the current angle

                # Update theta in obstacles dictionary
                self.obstacles[target_id]['movement']['theta'] = theta

            elif movement_type == 'straight':
                x0, y0 = self.initial_target_centers[target_id]
                heading_angle = self.obstacles[target_id]['movement']['heading_angle']
                u_target_max = self.obstacles[target_id]['u_max']

                x_new = x0 + np.cos(heading_angle) * u_target_max * current_t
                y_new = x0 + np.sin(heading_angle) * u_target_max * current_t

            elif movement_type == "static":
                #no movement, just return the current position
                x_new, y_new = self.obstacles[target_id]['center']

            elif movement_type == "periodic":
                # New movement type: periodic back-and-forth between two points
                point1 = np.array(self.obstacles[target_id]['movement']['point1'])
                point2 = np.array(self.obstacles[target_id]['movement']['point2'])
                u_target_max = self.obstacles[target_id]['u_max']

                total_distance = np.linalg.norm(point2 - point1)
                total_time = total_distance / u_target_max

                # Determine current phase within the full cycle (forth + back)
                cycle_time = 2 * total_time
                t_mod = current_t % cycle_time

                if t_mod < total_time:
                    # Moving from point1 to point2
                    alpha = t_mod / total_time
                    position = point1 + alpha * (point2 - point1)
                    direction_vector = point2 - point1
                else:
                    # Moving back from point2 to point1
                    alpha = (t_mod - total_time) / total_time
                    position = point2 - alpha * (point2 - point1)
                    direction_vector = point1 - point2

                x_new, y_new = position[0], position[1]

                heading_angle = np.arctan2(direction_vector[1], direction_vector[0])

                # #stop if hit another target:
                # for other_id, other_info in self.obstacles.items():
                #     if other_id != target_id:
                #         other_center = np.array(other_info['center'])
                #         dist = np.linalg.norm(np.array([x_new, y_new]) - other_center)
                #         if dist < (self.obstacles[target_id]['radius'] + other_info['radius']):
                #             x_new, y_new = self.obstacles[target_id]['center'] #stay at the current position

                # Update heading angle in obstacles dictionary
                self.obstacles[target_id]['movement']['heading_angle'] = heading_angle

            elif movement_type == 'random_walk':
                # 0) keep your random re-heading
                if self.simulation_timer % 50 == 0:
                    heading_angle = np.random.uniform(0, 2*np.pi)
                    self.obstacles[target_id]['movement']['heading_angle'] = heading_angle
                else:
                    heading_angle = self.obstacles[target_id]['movement']['heading_angle']

                # 1) base velocity
                speed = self.obstacles[target_id]['u_max'] * 0.5
                v = np.array([np.cos(heading_angle), np.sin(heading_angle)]) * speed

                # 2) super-light separation from neighbors
                AVOID_RADIUS = 2.0   # start with 2–3; tune
                AVOID_GAIN   = 0.8   # 0.5–1.0 is typical
                p = np.array(self.obstacles[target_id]['center'], dtype=float)
                r_self = self.obstacles[target_id]['radius']

                rep = np.zeros(2, float)
                for oid, o in self.obstacles.items():
                    if oid == target_id:
                        continue
                    po = np.array(o['center'], dtype=float)
                    ro = o['radius']
                    delta = p - po
                    d = np.linalg.norm(delta)
                    R = r_self + ro + AVOID_RADIUS
                    if d < R and d > 1e-6:
                        rep += (delta / d) * (R - d) / R   # push straight away

                # apply small push, keep speed ~constant
                v = v + AVOID_GAIN * rep
                n = np.linalg.norm(v)
                #if n > 1e-6:
                    #v = v / n * speed
                heading_angle = float(np.arctan2(v[1], v[0]))

                # 3) integrate
                p_new = p + v * self.dt

                # 4) simple wall bounce
                if p_new[0] < -self.width or p_new[0] > self.width:
                    heading_angle = np.pi - heading_angle
                    v = np.array([np.cos(heading_angle), np.sin(heading_angle)]) * speed
                    p_new[0] = np.clip(p_new[0], -self.width, self.width)

                if p_new[1] < -self.height or p_new[1] > self.height:
                    heading_angle = -heading_angle
                    v = np.array([np.cos(heading_angle), np.sin(heading_angle)]) * speed
                    p_new[1] = np.clip(p_new[1], -self.height, self.height)

                # 5) last-resort bounce if still overlapping
                for oid, o in self.obstacles.items():
                    if oid == target_id:
                        continue
                    po = np.array(o['center'], float)
                    R = r_self + o['radius']
                    if np.linalg.norm(p_new - po) < R:
                        heading_angle += np.pi  # flip 180°
                        v = np.array([np.cos(heading_angle), np.sin(heading_angle)]) * speed
                        p_new = p + v * self.dt
                        break

                # 6) commit
                x_new, y_new = p_new.tolist()
                self.obstacles[target_id]['movement']['heading_angle'] = heading_angle

        return (x_new, y_new)

    def dynamic_goal(self, current_t, goal_id):
        if self.goals is not None:
            movement_type = self.goals[goal_id]['movement']['type']
            if movement_type == 'static':
                # No movement, just return the current position
                x_new, y_new = self.goals[goal_id]['center']
            elif movement_type == 'periodic':
                # New movement type: periodic back-and-forth between two points
                point1 = np.array(self.goals[goal_id]['movement']['point1'])
                point2 = np.array(self.goals[goal_id]['movement']['point2'])
                u_goal_max = self.goals[goal_id]['u_max']

                total_distance = np.linalg.norm(point2 - point1)
                total_time = total_distance / u_goal_max

                # Determine current phase within the full cycle (forth + back)
                cycle_time = 2 * total_time
                t_mod = current_t % cycle_time

                if t_mod < total_time:
                    # Moving from point1 to point2
                    alpha = t_mod / total_time
                    position = point1 + alpha * (point2 - point1)
                    direction_vector = point2 - point1
                else:
                    # Moving back from point2 to point1
                    alpha = (t_mod - total_time) / total_time
                    position = point2 - alpha * (point2 - point1)
                    direction_vector = point1 - point2

                x_new, y_new = position[0], position[1]

                heading_angle = np.arctan2(direction_vector[1], direction_vector[0])

                # Update heading angle in goals dictionary
                self.goals[goal_id]['movement']['heading_angle'] = heading_angle

            elif movement_type == 'blinking':
                # Blinking movement: switch between two points every blink_duration
                point1 = np.array(self.goals[goal_id]['movement']['point1'])
                point2 = np.array(self.goals[goal_id]['movement']['point2'])
                blink_duration = self.goals[goal_id]['movement']['blink_duration']

                # Determine if we are in the first or second half of the blink cycle
                if (current_t // blink_duration) % 2 == 0:
                    x_new, y_new = point1[0], point1[1]
                else:
                    x_new, y_new = point2[0], point2[1]
                
            else:
                raise NotImplementedError(f"Movement type '{movement_type}' for goal not implemented.")
        else:
            raise ValueError("No goals defined in the environment.")

        return (x_new, y_new)
    

    def set_agent_location(self, x, y, theta=None):
        """
        Sets the agent's location to the specified coordinates manually.
        :param x: x-coordinate of the agent.
        :param y: y-coordinate of the agent.
        :param theta: orientation angle (only for unicycle dynamics).
        """
        if self.dynamics == 'unicycle':
            self.agent = UnicycleDynamics(x=x, y=y, theta=theta, dt=self.dt)
            return np.array([self.agent.x, self.agent.y, self.agent.theta])
        elif self.dynamics == 'single integrator':
            self.agent = SingleIntegratorDynamics(x=x, y=y, dt=self.dt)
            return np.array([self.agent.x, self.agent.y])
        
if __name__ == "__main__":

    target_region_center = [-10, 2]
    target_region_radius = 8
    action_range = [3, 3] #max vx and vy (for single integrator dynamics)

    u_target_max0 = 1.5
    u_target_max1 = 1.5
    u_agent_max = 8 #max agent speed

    goals = {0: {'center': (50, 0), 'radius': 10, 'movement':{'type':'static'}}, #goal region for the agent
	#1: {'center': (-50, 0), 'radius': 10, 'movement':{'type':'static'}}
    }

    targets = {}
    #config dictionary for the environment
    config = {
        'init_loc':[-20, -20], #initial location of the agent (x, y)
        "width": 100.0,
        "height": 100.0,
        "dt": 1,
        "render": True,
		'dt_render': 0.03,
		'goals': goals, #goal regions for the agent
        "obstacle_location": [100.0, 100.0],
        "obstacle_size": 0.0,
        "randomize_loc": False, #whether to randomize the agent location at the end of each episode
		'deterministic': False,
		'auto_entropy':True,
		"dynamics": "single integrator", #dynamics model to use
		"targets": targets,
		"u_agent_max": u_agent_max, #max target speed
		"disturbance": [-0.5, 0.5] #disturbance range in both x and y directions [w_min, w_max]
    }

    env = Continuous2DEnv(config)
    state = env.reset()

    action = (1.0, 1)  #linear vel., angular vel.

    for _ in range(100):
        state, reward, done = env.step(action)

        #print("State:", state)
        # if done:
        #     break

        #state = env.reset()

