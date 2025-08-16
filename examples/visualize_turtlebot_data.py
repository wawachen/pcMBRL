#!/usr/bin/env python3
"""
Turtlebot Data Visualization Script
Visualize position, velocity, and orientation ranges in collected turtlebot data
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
import seaborn as sns
from pathlib import Path
import argparse
import os

class TurtlebotDataVisualizer:
    def __init__(self, data_file, num_agents=None):
        """
        Initialize visualizer
        
        Args:
            data_file: data file path
            num_agents: number of agents, if None will auto-infer
        """
        self.data_file = data_file
        self.data = np.load(data_file)
        self.obs = self.data['obs']
        self.actions = self.data['acs']
        self.obs_next = self.data['obs_next']
        
        # Auto-infer number of agents
        if num_agents is None:
            self.num_agents = self._infer_num_agents()
        else:
            self.num_agents = num_agents
            
        print(f"Data file: {data_file}")
        print(f"Number of agents: {self.num_agents}")
        print(f"Observation data shape: {self.obs.shape}")
        print(f"Action data shape: {self.actions.shape}")
        
        # Calculate observation dimensions per agent
        self.obs_per_agent = self._calculate_obs_per_agent()
        print(f"Observation dimensions per agent: {self.obs_per_agent}")
        
    def _infer_num_agents(self):
        """Infer number of agents from observation data dimensions"""
        obs_dim = self.obs.shape[1]
        
        # For n agents, observation dimension is: 2n + 2n + 3n = 7n
        # Solve equation: 7n = obs_dim
        n = obs_dim // 7
        return n
    
    def _calculate_obs_per_agent(self):
        """Calculate observation dimensions per agent"""
        # Each agent: position(2) + velocity(2) + orientation(3) = 7 dimensions
        return 7
    
    def extract_agent_data(self, agent_id):
        """
        Extract data for specified agent
        
        Args:
            agent_id: agent ID (0 to n-1)
            
        Returns:
            positions: position data (N, 2)
            velocities: velocity data (N, 2) 
            orientations: orientation data (N, 3) - [roll, pitch, yaw]
        """
        if agent_id >= self.num_agents:
            raise ValueError(f"Agent ID {agent_id} out of range [0, {self.num_agents-1}]")
        
        # Calculate start index for this agent's observations
        # Format: [pos1, pos2, ..., posN, vel1, vel2, ..., velN, ori1, ori2, ..., oriN]
        start_idx = agent_id * 2  # Position starts at agent_id * 2
        
        # Extract position (2 dimensions)
        pos_start = start_idx
        pos_end = pos_start + 2
        positions = self.obs[:, pos_start:pos_end]
        
        # Extract velocity (2 dimensions) - starts after all positions
        vel_start = self.num_agents * 2 + agent_id * 2
        vel_end = vel_start + 2
        velocities = self.obs[:, vel_start:vel_end]
        
        # Extract orientation (3 dimensions) - starts after all velocities
        ori_start = self.num_agents * 4 + agent_id * 3
        ori_end = ori_start + 3
        orientations = self.obs[:, ori_start:ori_end]
        
        return positions, velocities, orientations
    
    def extract_all_agents_data(self):
        """Extract data for all agents"""
        all_data = {}
        
        for i in range(self.num_agents):
            positions, velocities, orientations = self.extract_agent_data(i)
            all_data[f'agent_{i}'] = {
                'positions': positions,
                'velocities': velocities,
                'orientations': orientations
            }
            
        return all_data
    
    def visualize_positions(self, save_path=None):
        """Visualize position distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Turtlebot Position Distribution Analysis (Agents: {self.num_agents})', fontsize=16)
        
        all_data = self.extract_all_agents_data()
        
        # 1. Position scatter plot for all agents
        ax = axes[0, 0]
        colors = plt.cm.Set3(np.linspace(0, 1, self.num_agents))
        
        for i, (agent_name, data) in enumerate(all_data.items()):
            positions = data['positions']
            ax.scatter(positions[:, 0], positions[:, 1], 
                      alpha=0.6, s=20, label=f'Agent {i}', color=colors[i])
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('All Agents Position Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # 2. Position range visualization
        ax = axes[0, 1]
        pos_ranges = []
        agent_names = []
        
        for i, (agent_name, data) in enumerate(all_data.items()):
            positions = data['positions']
            x_range = [np.min(positions[:, 0]), np.max(positions[:, 0])]
            y_range = [np.min(positions[:, 1]), np.max(positions[:, 1])]
            pos_ranges.append([x_range, y_range])
            agent_names.append(f'Agent {i}')
        
        # Draw position ranges
        for i, (x_range, y_range) in enumerate(pos_ranges):
            rect = patches.Rectangle((x_range[0], y_range[0]), 
                                   x_range[1] - x_range[0], 
                                   y_range[1] - y_range[0],
                                   linewidth=2, edgecolor=colors[i], 
                                   facecolor='none', alpha=0.7)
            ax.add_patch(rect)
            ax.text((x_range[0] + x_range[1])/2, (y_range[0] + y_range[1])/2, 
                   f'Agent {i}', ha='center', va='center', fontsize=10)
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Position Ranges by Agent')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # 3. X position distribution histogram
        ax = axes[1, 0]
        for i, (agent_name, data) in enumerate(all_data.items()):
            positions = data['positions']
            ax.hist(positions[:, 0], bins=50, alpha=0.6, 
                   label=f'Agent {i}', color=colors[i])
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Frequency')
        ax.set_title('X Position Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Y position distribution histogram
        ax = axes[1, 1]
        for i, (agent_name, data) in enumerate(all_data.items()):
            positions = data['positions']
            ax.hist(positions[:, 1], bins=50, alpha=0.6, 
                   label=f'Agent {i}', color=colors[i])
        
        ax.set_xlabel('Y Position')
        ax.set_ylabel('Frequency')
        ax.set_title('Y Position Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Position analysis saved to: {save_path}")
        
        plt.show()
        
        # Print position statistics
        print("\n=== Position Statistics ===")
        for i, (agent_name, data) in enumerate(all_data.items()):
            positions = data['positions']
            print(f"Agent {i}:")
            print(f"  X range: [{np.min(positions[:, 0]):.3f}, {np.max(positions[:, 0]):.3f}]")
            print(f"  Y range: [{np.min(positions[:, 1]):.3f}, {np.max(positions[:, 1]):.3f}]")
            print(f"  Position std: X={np.std(positions[:, 0]):.3f}, Y={np.std(positions[:, 1]):.3f}")
    
    def visualize_velocities(self, save_path=None):
        """Visualize velocity distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Turtlebot Velocity Distribution Analysis (Agents: {self.num_agents})', fontsize=16)
        
        all_data = self.extract_all_agents_data()
        colors = plt.cm.Set3(np.linspace(0, 1, self.num_agents))
        
        # 3. X velocity distribution
        ax = axes[1, 0]
        for i, (agent_name, data) in enumerate(all_data.items()):
            velocities = data['velocities']
            ax.hist(velocities[:, 0], bins=50, alpha=0.6, 
                   label=f'Agent {i}', color=colors[i])
        
        ax.set_xlabel('X Velocity')
        ax.set_ylabel('Frequency')
        ax.set_title('X Velocity Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Y velocity distribution
        ax = axes[1, 1]
        for i, (agent_name, data) in enumerate(all_data.items()):
            velocities = data['velocities']
            ax.hist(velocities[:, 1], bins=50, alpha=0.6, 
                   label=f'Agent {i}', color=colors[i])
        
        ax.set_xlabel('Y Velocity')
        ax.set_ylabel('Frequency')
        ax.set_title('Y Velocity Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # if save_path:
        #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
        #     print(f"Velocity analysis saved to: {save_path}")
        
        plt.show()
        
        # Print velocity statistics
        print("\n=== Velocity Statistics ===")
        for i, (agent_name, data) in enumerate(all_data.items()):
            velocities = data['velocities']
            print(f"Agent {i}:")
            print(f"  X velocity range: [{np.min(velocities[:, 0]):.3f}, {np.max(velocities[:, 0]):.3f}]")
            print(f"  Y velocity range: [{np.min(velocities[:, 1]):.3f}, {np.max(velocities[:, 1]):.3f}]")
    
    def visualize_orientations(self, save_path=None):
        """Visualize orientation distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Turtlebot Orientation Distribution Analysis (Agents: {self.num_agents})', fontsize=16)
        
        all_data = self.extract_all_agents_data()
        colors = plt.cm.Set3(np.linspace(0, 1, self.num_agents))
        
        # 1. Yaw angle distribution (from roll, pitch, yaw)
        ax = axes[0, 0]
        for i, (agent_name, data) in enumerate(all_data.items()):
            orientations = data['orientations']
            # Use yaw angle (third component)
            yaw_angles = orientations[:, 2]
            # Convert to degrees
            # yaw_degrees = np.degrees(yaw_angles)
            ax.hist(yaw_angles, bins=50, alpha=0.6, 
                   label=f'Agent {i}', color=colors[i])
        
        ax.set_xlabel('Yaw Angle (radians)')
        ax.set_ylabel('Frequency')
        ax.set_title('Yaw Angle Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        # ax.set_xlim(-180, 180)
        
        # 2. Roll angle distribution
        ax = axes[0, 1]
        for i, (agent_name, data) in enumerate(all_data.items()):
            orientations = data['orientations']
            ax.hist(orientations[:,0], bins=50, alpha=0.6, 
                   label=f'Agent {i}', color=colors[i])
        
        ax.set_xlabel('Roll Angle (radians)')
        ax.set_ylabel('Frequency')
        ax.set_title('Roll Angle Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Pitch angle distribution
        ax = axes[1, 0]
        for i, (agent_name, data) in enumerate(all_data.items()):
            orientations = data['orientations']
            pitch_angles = orientations[:, 1]
            ax.hist(pitch_angles, bins=50, alpha=0.6, 
                   label=f'Agent {i}', color=colors[i])
        
        ax.set_xlabel('Pitch Angle (radians)')
        ax.set_ylabel('Frequency')
        ax.set_title('Pitch Angle Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # if save_path:
        #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
        #     print(f"Orientation analysis saved to: {save_path}")
        
        plt.show()
        
        # Print orientation statistics
        print("\n=== Orientation Statistics ===")
        for i, (agent_name, data) in enumerate(all_data.items()):
            orientations = data['orientations']
            roll = orientations[:, 0]
            pitch = orientations[:, 1]
            yaw = orientations[:, 2]
            
            print(f"Agent {i}:")
            print(f"  Roll range: [{np.min(roll):.3f}, {np.max(roll):.3f}] radians")
            print(f"  Pitch range: [{np.min(pitch):.3f}, {np.max(pitch):.3f}] radians")
            print(f"  Yaw range: [{np.min(yaw):.3f}, {np.max(yaw):.3f}] radians")
            print(f"  Average yaw: {np.mean(yaw):.3f} radians")
    
    def visualize_summary(self, save_path=None):
        """Visualize comprehensive summary"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Turtlebot Data Analysis Summary (Agents: {self.num_agents})', fontsize=16)
        
        all_data = self.extract_all_agents_data()
        colors = plt.cm.Set3(np.linspace(0, 1, self.num_agents))
        
        # 1. Position range comparison
        ax = axes[0, 0]
        x_ranges = []
        y_ranges = []
        agent_labels = []
        
        for i, (agent_name, data) in enumerate(all_data.items()):
            positions = data['positions']
            x_ranges.append([np.min(positions[:, 0]), np.max(positions[:, 0])])
            y_ranges.append([np.min(positions[:, 1]), np.max(positions[:, 1])])
            agent_labels.append(f'Agent {i}')
        
        # Plot X ranges
        x_pos = np.arange(len(agent_labels))
        x_widths = [x_range[1] - x_range[0] for x_range in x_ranges]
        x_centers = [(x_range[0] + x_range[1])/2 for x_range in x_ranges]
        
        ax.bar(x_pos, x_widths, bottom=[x_range[0] for x_range in x_ranges], 
               alpha=0.7, color=colors[:len(agent_labels)])
        ax.set_xlabel('Agent')
        ax.set_ylabel('X Position Range')
        ax.set_title('X Position Range Comparison')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(agent_labels)
        ax.grid(True, alpha=0.3)
        
        # 2. Y position range comparison
        ax = axes[0, 1]
        y_widths = [y_range[1] - y_range[0] for y_range in y_ranges]
        ax.bar(x_pos, y_widths, bottom=[y_range[0] for y_range in y_ranges], 
               alpha=0.7, color=colors[:len(agent_labels)])
        ax.set_xlabel('Agent')
        ax.set_ylabel('Y Position Range')
        ax.set_title('Y Position Range Comparison')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(agent_labels)
        ax.grid(True, alpha=0.3)
        
        # 3. Velocity range comparison
        ax = axes[0, 2]
        vel_ranges = []
        for i, (agent_name, data) in enumerate(all_data.items()):
            velocities = data['velocities']
            speed_magnitudes = np.sqrt(velocities[:, 0]**2 + velocities[:, 1]**2)
            vel_ranges.append([np.min(speed_magnitudes), np.max(speed_magnitudes)])
        
        vel_widths = [vel_range[1] - vel_range[0] for vel_range in vel_ranges]
        ax.bar(x_pos, vel_widths, bottom=[vel_range[0] for vel_range in vel_ranges], 
               alpha=0.7, color=colors[:len(agent_labels)])
        ax.set_xlabel('Agent')
        ax.set_ylabel('Velocity Range')
        ax.set_title('Velocity Range Comparison')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(agent_labels)
        ax.grid(True, alpha=0.3)
        
        # 4. Orientation range comparison
        ax = axes[1, 0]
        ori_ranges = []
        for i, (agent_name, data) in enumerate(all_data.items()):
            orientations = data['orientations']
            yaw_angles = np.degrees(orientations[:, 2])
            ori_ranges.append([np.min(yaw_angles), np.max(yaw_angles)])
        
        ori_widths = [ori_range[1] - ori_range[0] for ori_range in ori_ranges]
        ax.bar(x_pos, ori_widths, bottom=[ori_range[0] for ori_range in ori_ranges], 
               alpha=0.7, color=colors[:len(agent_labels)])
        ax.set_xlabel('Agent')
        ax.set_ylabel('Yaw Angle Range (degrees)')
        ax.set_title('Yaw Angle Range Comparison')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(agent_labels)
        ax.grid(True, alpha=0.3)
        
        # 5. Position density heatmap
        ax = axes[1, 1]
        # Combine all agents' position data
        all_positions = np.vstack([data['positions'] for data in all_data.values()])
        
        # Create 2D histogram
        H, xedges, yedges = np.histogram2d(all_positions[:, 0], all_positions[:, 1], bins=50)
        im = ax.imshow(H.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                       aspect='auto', cmap='viridis')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Position Data Density Heatmap')
        plt.colorbar(im, ax=ax)
        
        # 6. Action distribution
        ax = axes[1, 2]
        # Actions are [angular_vel, linear_vel] for each agent
        actions_per_agent = self.actions.shape[1] // self.num_agents
        
        for i in range(self.num_agents):
            start_idx = i * actions_per_agent
            end_idx = start_idx + actions_per_agent
            agent_actions = self.actions[:, start_idx:end_idx]
            
            # Plot angular vs linear velocity
            if actions_per_agent >= 2:
                ax.scatter(agent_actions[:, 0], agent_actions[:, 1], 
                          alpha=0.6, s=20, label=f'Agent {i}', color=colors[i])
        
        ax.set_xlabel('Angular Velocity')
        ax.set_ylabel('Linear Velocity')
        ax.set_title('Action Distribution (Angular vs Linear)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Summary analysis saved to: {save_path}")
        
        plt.show()
    
    def generate_report(self, output_dir="./visualization_results"):
        """Generate complete visualization report"""
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print("Generating visualization report...")
        
        # 1. Position analysis
        print("\n1. Generating position distribution plots...")
        self.visualize_positions(save_path=f"{output_dir}/positions_analysis.png")
        
        # 2. Velocity analysis
        print("\n2. Generating velocity distribution plots...")
        self.visualize_velocities(save_path=f"{output_dir}/velocities_analysis.png")
        
        # 3. Orientation analysis
        print("\n3. Generating orientation distribution plots...")
        self.visualize_orientations(save_path=f"{output_dir}/orientations_analysis.png")
        
        # 4. Summary analysis
        print("\n4. Generating summary analysis plots...")
        self.visualize_summary(save_path=f"{output_dir}/summary_analysis.png")
        
        print(f"\nVisualization report completed! Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Turtlebot Data Visualization Tool')
    parser.add_argument('--data_file', type=str, default='/home/xlab/MARL_transport/turtlebot_demonstrationMBRL_steps50k_3agents_env10.npz',
                       help='Data file path (.npz format)')
    parser.add_argument('--num_agents', type=int, default=None,
                       help='Number of agents (optional, will auto-infer)')
    parser.add_argument('--output_dir', type=str, default='./visualization_results',
                       help='Output directory path')
    parser.add_argument('--mode', type=str, default='full', 
                       choices=['full', 'positions', 'velocities', 'orientations', 'summary'],
                       help='Visualization mode: full(complete), positions, velocities, orientations, summary')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.data_file):
        print(f"Error: Data file {args.data_file} does not exist!")
        return
    
    # Create visualizer
    visualizer = TurtlebotDataVisualizer(args.data_file, args.num_agents)
    
    # Execute visualization based on mode
    if args.mode == 'full':
        visualizer.generate_report(args.output_dir)
    elif args.mode == 'positions':
        visualizer.visualize_positions()
    elif args.mode == 'velocities':
        visualizer.visualize_velocities()
    elif args.mode == 'orientations':
        visualizer.visualize_orientations()
    elif args.mode == 'summary':
        visualizer.visualize_summary()
    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == '__main__':
    main() 