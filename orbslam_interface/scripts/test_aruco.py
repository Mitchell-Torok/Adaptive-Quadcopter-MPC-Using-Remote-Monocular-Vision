#!/usr/bin/env python3

"""
Simple script to test ArUco marker detection
This script creates and displays ArUco markers for testing purposes
"""

import cv2
import numpy as np

def generate_aruco_marker(marker_id=0, marker_size=200, save_path=None):
    """Generate an ArUco marker"""
    # Create ArUco dictionary (5x5 to match the C++ code)
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_250)
    
    # Generate marker
    marker_img = cv2.aruco.drawMarker(aruco_dict, marker_id, marker_size)
    
    # Add white border
    border_size = 50
    bordered_img = np.ones((marker_size + 2*border_size, marker_size + 2*border_size), dtype=np.uint8) * 255
    bordered_img[border_size:border_size+marker_size, border_size:border_size+marker_size] = marker_img
    
    if save_path:
        cv2.imwrite(save_path, bordered_img)
        print(f"ArUco marker saved to {save_path}")
    
    return bordered_img

def main():
    # Generate test markers
    for i in range(4):
        marker = generate_aruco_marker(
            marker_id=i, 
            marker_size=200, 
            save_path=f"/tmp/aruco_marker_{i}.png"
        )
        
        # Display marker
        cv2.imshow(f"ArUco Marker {i}", marker)
    
    print("Press any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\nTo test with your SLAM system:")
    print("1. Print one of the generated markers (/tmp/aruco_marker_*.png)")
    print("2. Ensure the printed marker is exactly 150mm (15cm) square")
    print("3. Run your SLAM node with the default aruco_marker_size (0.15m):")
    print("   ros2 run your_package mono_node_cpp")
    print("   Or specify a different size if needed:")
    print("   ros2 run your_package mono_node_cpp --ros-args -p aruco_marker_size:=0.15")

if __name__ == "__main__":
    main()
