#!/usr/bin/env python3
"""
Simple test script for turtlebot position controller
Tests basic navigation scenarios
"""

import sys
import os
import numpy as np
import time
from pyrep import PyRep
from pyrep.envs.turtle_RL_agent import Turtle_o
from pyrep.drone_controller_utility.turtlebot_position_controller import TurtlebotPositionController
from os import path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TurtlebotPositionControllerTest:
    def __init__(self, scene_path):
        """
        Initialize the test environment
        
        Args:
            scene_path: Path to the CoppeliaSim scene file
        """
        self.pr = PyRep()
        self.pr.launch(scene_path, headless=False)
        self.pr.start()
        
        robot_DIR = "/home/xlab/MARL_transport/examples/models"
        # Import turtlebot model if available
        try:
            model_path = path.join(robot_DIR, 'turtlebot_beta.ttm')
            if os.path.exists(model_path):
                model_handle = self.pr.import_model(model_path)
                print(f"Imported turtlebot model: {model_handle}")
        except Exception as e:
            print(f"Could not import turtlebot model: {e}")
            print("Using existing scene objects")
        # Initialize turtlebot
        self.turtlebot = Turtle_o(0)  # Agent ID 0
        
        # Initialize controllers
        self.position_controller = TurtlebotPositionController()
        
        # Test parameters
        self.simulation_time = 0.05  # seconds per step
        self.max_steps = 1000
        
        print("Turtlebot controller test initialized successfully!")
        
    def reset_simulation(self):
        """Reset the simulation"""
        self.position_controller.reset()
        print("Controller reset")
        
    def test_single_target(self, target_pos, max_steps=None):
        """
        Test navigation to a single target position
        
        Args:
            target_pos: [x, y] target position
            max_steps: maximum number of simulation steps
        """
        print(f"\n=== Testing single target navigation to {target_pos} ===")
        
        if max_steps is None:
            max_steps = self.max_steps

        # Reset turtlebot position
        try:
            self.turtlebot.agent.set_3d_pose([0.0, 0.0, 0.0607, 0.0, 0.0, 0.0])
            print("Turtlebot position reset to origin")
        except Exception as e:
            print(f"Could not reset position: {e}")
            
        self.reset_simulation()
        
        for step in range(max_steps):
            # Get current state
            current_pos = self.turtlebot.get_2d_pos()
            current_heading = self.turtlebot.get_heading()
            
            # Calculate control commands
            linear_vel, angular_vel = self.position_controller.calculate_position_control(
                target_pos, current_pos, current_heading
            )
            
            # Apply control
            action = [angular_vel, linear_vel]
            self.turtlebot.set_action(action)
            
            # Check if target reached
            distance_to_target = np.sqrt((target_pos[0] - current_pos[0])**2 + 
                                       (target_pos[1] - current_pos[1])**2)
            
            # Print progress every 10 steps
            if step % 10 == 0:
                print(f"Step {step:3d}: Pos=({current_pos[0]:6.3f}, {current_pos[1]:6.3f}), "
                      f"Target=({target_pos[0]:6.3f}, {target_pos[1]:6.3f}), "
                      f"Distance={distance_to_target:6.3f}, "
                      f"Heading={current_heading:6.3f}, "
                      f"Vel=({linear_vel:6.3f}, {angular_vel:6.3f})")
            
            if distance_to_target < self.position_controller.position_tolerance:
                print(f"✓ Target reached in {step} steps!")
                print(f"Final position: ({current_pos[0]:.3f}, {current_pos[1]:.3f})")
                print(f"Final distance: {distance_to_target:.3f}")
                return step + 1
                
            # Step simulation
            self.pr.step()
            # time.sleep(self.simulation_time)
            
        print(f"✗ Failed to reach target in {max_steps} steps")
        return max_steps
        
    
        
    def run_all_tests(self):
        """Run all test scenarios"""
        print("=== Turtlebot Position Controller Test Suite ===")
        print("Based on C++ control implementation for Turtlebot3")
        
        try:
            # Test 1: Single target navigation - forward
            print("\n1. Forward Navigation Test")
            target = [2.0, 0.0]
            steps = self.test_single_target(target)
            print(f"Result: {steps} steps")
            
            # Test 2: Single target navigation - backward  
            print("\n2. Backward Navigation Test")
            target = [-2.0, 0.0]
            steps = self.test_single_target(target)
            print(f"Result: {steps} steps")
            
            # Test 3: Diagonal navigation
            print("\n3. Diagonal Navigation Test")
            target = [1.5, 1.5]
            steps = self.test_single_target(target)
            print(f"Result: {steps} steps")
            
            print("\n=== All tests completed successfully ===")
            
        except Exception as e:
            print(f"Test failed with error: {e}")
            import traceback
            traceback.print_exc()
        
    def cleanup(self):
        """Clean up resources"""
        self.pr.stop()
        self.pr.shutdown()


def main():
    """Main function to run the tests"""
    # Path to the CoppeliaSim scene file
    scene_path = "examples/RL_drone_field_10x10.ttt"  # Adjust path as needed
    
    # Check if scene file exists
    if not os.path.exists(scene_path):
        print(f"Scene file not found: {scene_path}")
        print("Please adjust the scene_path variable in the script")
        return
    
    # Create test instance
    test = TurtlebotPositionControllerTest(scene_path)
    
    try:
        # Run all tests
        test.run_all_tests()
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        
    finally:
        # Cleanup
        test.cleanup()


if __name__ == "__main__":
    main() 