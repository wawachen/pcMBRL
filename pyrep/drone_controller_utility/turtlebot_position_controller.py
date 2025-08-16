from pyrep.drone_controller_utility.PID import PID
from pyrep.drone_controller_utility.controller_utility import ControllerUtility
import math
from pyrep.backend import sim
import numpy as np


class SimplePID:
    """
    Simple PID controller based on C++ implementation
    """
    def __init__(self, dt, max_output, min_output, kp, kd, ki):
        self.dt = dt
        self.max_output = max_output
        self.min_output = min_output
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.pre_error = 0.0
        self.integral = 0.0
    
    def calculate(self, setpoint, pv):
        """Calculate PID output given setpoint and process variable"""
        # Error
        error = setpoint - pv
        
        # Proportional term
        p_out = self.kp * error
        
        # Integral term
        self.integral += error * self.dt
        i_out = self.ki * self.integral
        
        # Derivative term
        derivative = (error - self.pre_error) / self.dt if self.dt > 0 else 0.0
        d_out = self.kd * derivative
        
        # Total output
        output = p_out + i_out + d_out
        
        # Limit output
        if output > self.max_output:
            output = self.max_output
        elif output < self.min_output:
            output = self.min_output
        
        # Save error for next iteration
        self.pre_error = error
        
        return output
    
    def reset(self):
        """Reset PID state"""
        self.pre_error = 0.0
        self.integral = 0.0


class TurtlebotPositionController:
    """
    Position controller for turtlebot navigation in 2D space
    Based on C++ control implementation for Turtlebot3
    Includes collision avoidance for multi-robot scenarios
    """
    def __init__(self):
        # Control parameters from C++ implementation
        self.dt = 0.1  # Control loop time step
        self.max_speed = 0.5  # Maximum linear speed
        self.distance_const = 0.5  # Distance threshold
        
        # PID parameters for angular control (theta)
        # dt, max, min, Kp, Kd, Ki
        self.pid_theta = SimplePID(
            dt=self.dt,
            max_output=math.pi,
            min_output=-math.pi,
            kp=0.2,
            kd=0.01,
            ki=0.05
        )
        
        # PID parameters for speed control
        self.pid_velocity = SimplePID(
            dt=self.dt,
            max_output=self.max_speed,
            min_output=0.0,
            kp=0.08,
            kd=0.005,
            ki=0.01
        )
        
        # Control limits
        self.max_linear_vel = self.max_speed
        self.max_angular_vel = 1.0  # rad/s
        self.min_linear_vel = 0.0
        
        # Position tolerance
        self.position_tolerance = 0.1  # meters
        self.heading_tolerance = 0.1   # radians
        
        # Collision avoidance parameters
        self.collision_radius = 0.6  # Safety distance from other robots
        self.detection_radius = 1.5 # Detection range for other robots
        self.avoidance_gain = 2.5   # Strength of avoidance force
        
        # Time tracking
        self.last_time = sim.simGetSimulationTime()
    
    def correct_angle(self, angle):
        """Correct angle to be within [-pi, pi] range"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def calculate_avoidance_force(self, current_pos, other_robots_pos):
        """
        Calculate collision avoidance force based on other robots' positions
        Uses artificial potential field method
        
        Args:
            current_pos: [x, y] current robot position
            other_robots_pos: list of [x, y] positions of other robots
            
        Returns:
            [force_x, force_y] avoidance force vector
        """
        avoidance_force = np.array([0.0, 0.0])
        
        if not other_robots_pos:
            return avoidance_force
        
        for other_pos in other_robots_pos:
            # Calculate distance to other robot
            dx = current_pos[0] - other_pos[0]
            dy = current_pos[1] - other_pos[1]
            distance = np.sqrt(dx**2 + dy**2)
            
            # Skip if too far away or same position
            if distance > self.detection_radius or distance < 0.01:
                continue
            
            # Calculate repulsive force (inversely proportional to distance)
            if distance < self.collision_radius:
                # Strong repulsion when too close
                force_magnitude = self.avoidance_gain * (1.0/(distance+0.0001))
            else:
                # Weaker repulsion at medium distance
                force_magnitude = self.avoidance_gain * 0.5 * (1.0/(distance+0.0001))
            
            # Direction away from other robot
            force_direction_x = dx / distance
            force_direction_y = dy / distance
            
            # Add to total avoidance force
            avoidance_force[0] += force_magnitude * force_direction_x
            avoidance_force[1] += force_magnitude * force_direction_y
        
        # Limit maximum avoidance force
        max_force = 2.0
        force_magnitude = np.linalg.norm(avoidance_force)
        if force_magnitude > max_force:
            avoidance_force = avoidance_force * (max_force / force_magnitude)
        
        return avoidance_force
    
    def check_collision_risk(self, current_pos, current_heading, linear_vel, other_robots_pos):
        """
        Check if current trajectory will lead to collision
        
        Args:
            current_pos: [x, y] current position
            current_heading: current heading angle
            linear_vel: current linear velocity
            other_robots_pos: list of other robot positions
            
        Returns:
            bool: True if collision risk detected
        """
        # Check for robot collision risk
        if (not other_robots_pos or linear_vel < 0.01):
            robot_collision_risk = False
        else:
            # Predict position after short time step
            prediction_time = 0.0  # seconds
            predicted_x = current_pos[0] + linear_vel * math.cos(current_heading) * prediction_time
            predicted_y = current_pos[1] + linear_vel * math.sin(current_heading) * prediction_time
            predicted_pos = [predicted_x, predicted_y]
            
            robot_collision_risk = False
            for other_pos in other_robots_pos:
                distance = np.sqrt((predicted_pos[0] - other_pos[0])**2 + (predicted_pos[1] - other_pos[1])**2)
                if distance < self.collision_radius:
                    robot_collision_risk = True
                    break

        # Check for wall collision risk
        # Assume the field is bounded by x in [-5, 5], y in [-5, 5] (as in env code)
        # Use the same predicted position as above
        if linear_vel < 0.01:
            predicted_x = current_pos[0]
            predicted_y = current_pos[1]
        else:
            prediction_time = 1.0
            predicted_x = current_pos[0] + linear_vel * math.cos(current_heading) * prediction_time
            predicted_y = current_pos[1] + linear_vel * math.sin(current_heading) * prediction_time

        wall_margin = self.collision_radius  # Use robot collision radius as margin
        x_min, x_max = -5 + wall_margin, 5 - wall_margin
        y_min, y_max = -5 + wall_margin, 5 - wall_margin

        wall_collision_risk = (
            predicted_x < x_min or predicted_x > x_max or
            predicted_y < y_min or predicted_y > y_max
        )

        return robot_collision_risk or wall_collision_risk

    def calculate_position_control(self, target_pos, current_pos, current_heading):
        """
        Calculate control commands for turtlebot position control
        Based on C++ control.cpp implementation
        
        Args:
            target_pos: [x, y] target position in world frame
            current_pos: [x, y] current position in world frame
            current_heading: current heading angle in radians
            
        Returns:
            [linear_vel, angular_vel] control commands
        """
        return self.calculate_position_control_with_avoidance(target_pos, current_pos, current_heading, None)
    
    def calculate_position_control_with_avoidance(self, target_pos, current_pos, current_heading, other_robots_pos=None):
        """
        Calculate control commands with collision avoidance
        
        Args:
            target_pos: [x, y] target position in world frame
            current_pos: [x, y] current position in world frame
            current_heading: current heading angle in radians
            other_robots_pos: list of [x, y] positions of other robots (optional)
            
        Returns:
            [linear_vel, angular_vel] control commands
        """
        # Calculate position errors
        pos_error_x = target_pos[0] - current_pos[0]
        pos_error_y = target_pos[1] - current_pos[1]
        
        # Calculate distance to target
        distance_to_target = np.sqrt(pos_error_x**2 + pos_error_y**2)
        
        # Check if we're close enough to target
        if distance_to_target < self.position_tolerance:
            return [0.0, 0.0]
        
        # Calculate basic target direction
        target_direction = np.array([pos_error_x, pos_error_y]) / distance_to_target
        
        # Apply collision avoidance if other robots are present
        if other_robots_pos:
            avoidance_force = self.calculate_avoidance_force(current_pos, other_robots_pos)
            
            # Combine target direction with avoidance force
            combined_direction = target_direction + avoidance_force
            
            # Normalize combined direction
            combined_magnitude = np.linalg.norm(combined_direction)
            if combined_magnitude > 0.01:
                combined_direction = combined_direction / combined_magnitude
            else:
                combined_direction = target_direction
        else:
            combined_direction = target_direction
        
        # Calculate desired heading from combined direction
        target_angle = math.atan2(combined_direction[1], combined_direction[0])
        
        # Calculate angle error
        angle_error = target_angle - current_heading
        angle_error = self.correct_angle(angle_error)
        
        # Set speed based on distance and collision risk
        speed = self.max_speed
        
        # Reduce speed when close to target
        if distance_to_target < 0.7:
            speed_reduction = self.max_speed * math.exp(-abs(distance_to_target))
            speed = self.pid_velocity.calculate(self.max_speed, -speed_reduction)
        
        # Reduce speed if collision risk detected
        if other_robots_pos and self.check_collision_risk(current_pos, current_heading, speed, other_robots_pos):
            speed *= 0.3  # Reduce speed to 30% when collision risk detected
        
        # Reduce speed if heading error is large (need to turn)
        if abs(angle_error) > math.pi/4:  # 45 degrees
            speed *= 0.2
        
        # Calculate angular velocity using PID
        omega = self.pid_theta.calculate(0, -angle_error)
        
        # Limit outputs
        linear_vel = max(self.min_linear_vel, min(speed, self.max_linear_vel))
        angular_vel = max(-self.max_angular_vel, min(omega, self.max_angular_vel))
        
        return [linear_vel, angular_vel]
    
    def reset(self):
        """Reset controller state"""
        self.pid_theta.reset()
        self.pid_velocity.reset()
        self.last_time = sim.simGetSimulationTime()
    
    def set_control_limits(self, max_linear_vel, max_angular_vel, min_linear_vel=0.1):
        """Set control limits"""
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        self.min_linear_vel = min_linear_vel
    
    def set_position_tolerance(self, position_tolerance, heading_tolerance=0.1):
        """Set position and heading tolerance"""
        self.position_tolerance = position_tolerance
        self.heading_tolerance = heading_tolerance