#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Visualization script for OpenCOOD dataset

import os
import argparse
import yaml
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

import sys
sys.path.append('/home/labeling/repos/OpenCOOD')

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.transformation_utils import x_to_world, x1_to_x2

def create_bbx(extent):
    """
    Create bounding box with 8 corners under obstacle vehicle reference.

    Parameters
    ----------
    extent : list
        Width, height, length of the bbx.

    Returns
    -------
    bbx : np.array
        The bounding box with 8 corners, shape: (8, 3)
    """
    bbx = np.array([[extent[0], -extent[1], -extent[2]],
                    [extent[0], extent[1], -extent[2]],
                    [-extent[0], extent[1], -extent[2]],
                    [-extent[0], -extent[1], -extent[2]],
                    [extent[0], -extent[1], extent[2]],
                    [extent[0], extent[1], extent[2]],
                    [-extent[0], extent[1], extent[2]],
                    [-extent[0], -extent[1], extent[2]]])
    return bbx

def lidar_matrix_to_pose(matrix):
    """
    Convert 4x4 transformation matrix to [x, y, z, roll, yaw, pitch]
    """
    if isinstance(matrix, list) and isinstance(matrix[0], list):
        matrix = np.array(matrix)
    if not isinstance(matrix, np.ndarray):
        return matrix
    
    x = matrix[0, 3]
    y = matrix[1, 3]
    z = matrix[2, 3]
    
    # Extract rotation angles from matrix
    pitch = np.arcsin(matrix[2, 0])
    yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    roll = np.arctan2(-matrix[2, 1], matrix[2, 2])
    
    return [x, y, z, np.degrees(roll), np.degrees(yaw), np.degrees(pitch)]

def plot_cav_data(pcd_file, yaml_file, ax, color='blue', label_added=False, vis_range=[-140, -40, 140, 40]):
    """
    Plot point cloud and bounding boxes for a single CAV.
    
    Parameters
    ----------
    pcd_file : str
        Path to the point cloud file
    yaml_file : str
        Path to the YAML file with bounding box information
    ax : matplotlib.axes.Axes
        Axes to plot on
    color : str
        Color to use for point cloud
    label_added : bool
        Whether label has been added to legend
    vis_range : list
        Visualization range [x_min, y_min, x_max, y_max]
    
    Returns
    -------
    bool
        True if label has been added to legend
    """
    # Load point cloud
    pcd = o3d.io.read_point_cloud(pcd_file)
    points = np.asarray(pcd.points)
    
    # Load yaml file
    data = load_yaml(yaml_file)
    
    # Get lidar pose
    lidar_pose = data['lidar_pose']
    if isinstance(lidar_pose, np.ndarray) or (isinstance(lidar_pose, list) and isinstance(lidar_pose[0], list)):
        lidar_pose = lidar_matrix_to_pose(lidar_pose)
    
    # Plot point cloud BEV view (using only x and y coordinates)
    if not label_added:
        ax.scatter(points[:, 0], points[:, 1], s=0.1, c=color, alpha=0.5, label=f"Point Cloud ({os.path.basename(os.path.dirname(pcd_file))})")
        label_added = True
    else:
        ax.scatter(points[:, 0], points[:, 1], s=0.1, c=color, alpha=0.5)
    
    # Plot vehicle bounding boxes
    vehicles = data.get('vehicles', {})
    for vehicle_id, vehicle_info in vehicles.items():
        # Get vehicle position and dimensions
        location = vehicle_info['location']
        extent = vehicle_info['extent']
        angle = vehicle_info['angle']
        
        # Create vehicle pose in global coordinate system
        vehicle_pose = [location[0], location[1], location[2], angle[0], angle[1], angle[2]]
        
        # Transform from object to lidar coordinate system
        object2lidar = x1_to_x2(vehicle_pose, lidar_pose)

        # shape (3, 8)
        bbx = create_bbx(extent).T
        # bounding box under ego coordinate shape (4, 8)
        bbx = np.r_[bbx, [np.ones(bbx.shape[1])]]

        # project the 8 corners to world coordinate
        bbx_lidar = np.dot(object2lidar, bbx).T
        bbx_lidar = np.expand_dims(bbx_lidar[:, :3], 0)[0]

        # Draw the bounding box
        # Bottom four corners (first four points)
        bottom_corners = bbx_lidar[:4, :2]  # Only take x and y coordinates for BEV view
        
        # Connect the bottom four corners to form a closed polygon
        # Connect points in order: 0->1->2->3->0
        for i in range(4):
            j = (i + 1) % 4
            if not label_added:
                ax.plot([bottom_corners[i, 0], bottom_corners[j, 0]],
                        [bottom_corners[i, 1], bottom_corners[j, 1]],
                        'r-', linewidth=2, label="Vehicle Bounding Box")
                label_added = True
            else:
                ax.plot([bottom_corners[i, 0], bottom_corners[j, 0]],
                        [bottom_corners[i, 1], bottom_corners[j, 1]],
                        'r-', linewidth=2)
        
        # Add vehicle ID label
        center_x = np.mean(bottom_corners[:, 0])
        center_y = np.mean(bottom_corners[:, 1])
        ax.text(center_x, center_y, f"{vehicle_id}", 
                color='black', fontsize=8, ha='center', va='center')
    
    # Set axis limits
    ax.set_xlim(vis_range[0], vis_range[2])
    ax.set_ylim(vis_range[1], vis_range[3])
    
    return label_added

def visualize_frame(dataset_root, scenario, frame_id, output_dir=None, vis_range=[-140, -40, 140, 40]):
    """
    Visualize a specific frame from the dataset.
    
    Parameters
    ----------
    dataset_root : str
        Path to the dataset root directory
    scenario : str
        Scenario folder name
    frame_id : str
        Frame ID (timestamp)
    output_dir : str, optional
        Directory to save the visualization, by default None
    vis_range : list, optional
        Visualization range [x_min, y_min, x_max, y_max], by default [-140, -40, 140, 40]
    """
    scenario_path = os.path.join(dataset_root, scenario)
    
    # Check if scenario exists
    if not os.path.exists(scenario_path):
        print(f"Scenario {scenario} not found in {dataset_root}")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Find all CAVs
    cav_list = []
    for item in os.listdir(scenario_path):
        cav_path = os.path.join(scenario_path, item)
        if os.path.isdir(cav_path):
            cav_list.append(item)
    
    cav_list = sorted(cav_list, key=lambda x: int(x))
    
    # plot each CAV with different colors
    colors = plt.cm.tab20(np.linspace(0, 1, len(cav_list)))
    label_added = False
    
    for i, cav in enumerate(cav_list):
        pcd_file = os.path.join(scenario_path, cav, f"{frame_id}.pcd")
        yaml_file = os.path.join(scenario_path, cav, f"{frame_id}.yaml")
        
        if os.path.exists(pcd_file) and os.path.exists(yaml_file):
            label_added = plot_cav_data(pcd_file, yaml_file, ax, color=colors[i], 
                                         label_added=label_added, vis_range=vis_range)
        else:
            print(f"Files not found for CAV {cav}, frame {frame_id}")
    
    # Set axis labels
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Point Cloud BEV View - Scenario: {scenario}, Frame: {frame_id}')
    ax.set_aspect('equal')  # Keep x and y axis scales equal
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # Save or show the plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{scenario}_{frame_id}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.close(fig)
    return fig

def find_frames_in_scenario(dataset_root, scenario):
    """Find all frames in a scenario."""
    scenario_path = os.path.join(dataset_root, scenario)
    
    if not os.path.exists(scenario_path):
        print(f"Scenario {scenario} not found in {dataset_root}")
        return []
    
    # Get first CAV directory
    cavs = [d for d in os.listdir(scenario_path) if os.path.isdir(os.path.join(scenario_path, d))]
    if not cavs:
        print(f"No CAVs found in scenario {scenario}")
        return []
    
    # Get all yaml files in first CAV
    cav_path = os.path.join(scenario_path, cavs[0])
    yaml_files = [f for f in os.listdir(cav_path) if f.endswith('.yaml') and 'additional' not in f]
    
    # Extract frame IDs
    frame_ids = [os.path.splitext(f)[0] for f in yaml_files]
    return sorted(frame_ids)

def main():
    parser = argparse.ArgumentParser(description='Visualize OpenCOOD dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset root directory')
    parser.add_argument('--scenario', type=str, required=True, help='Scenario folder name')
    parser.add_argument('--frame', type=str, help='Frame ID (timestamp). If not provided, all frames will be visualized.')
    parser.add_argument('--output', type=str, default='/home/labeling/repos/OpenCOOD/visualization_output', 
                        help='Directory to save visualizations')
    parser.add_argument('--range', type=float, nargs=4, default=[-140, -40, 140, 40], 
                        help='Visualization range [x_min, y_min, x_max, y_max]')
    
    args = parser.parse_args()
    
    if args.frame:
        # Visualize specific frame
        visualize_frame(args.dataset, args.scenario, args.frame, args.output, args.range)
    else:
        # Visualize all frames
        frame_ids = find_frames_in_scenario(args.dataset, args.scenario)
        if not frame_ids:
            print(f"No frames found in scenario {args.scenario}")
            return
        
        print(f"Found {len(frame_ids)} frames in scenario {args.scenario}")
        for i, frame_id in enumerate(frame_ids):
            print(f"Visualizing frame {i+1}/{len(frame_ids)}: {frame_id}")
            visualize_frame(args.dataset, args.scenario, frame_id, args.output, args.range)

if __name__ == "__main__":
    main() 