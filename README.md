# gesp3D - 3D Gesture Sandbox

gesp3D is an interactive 3D particle engine that allows users to manipulate geometric shapes in a virtual sandbox using real-time hand gestures. The system features a multi-window architecture, separating the visual rendering from the command interface.

## Features

*   **Gesture-Controlled Interaction**: Rotate, zoom, and subdivide 3D shapes using intuitive hand movements.
*   **Multi-Window System**: 
    *   **Sandbox**: High-performance OpenCV window for 3D visualization.
    *   **Console**: Independent Tkinter-based command line for system control.
*   **Dynamic Geometric Shapes**: Switch between Cube, Sphere (Fibonacci distribution), and Pyramid.
*   **Perspective-Invariant Tracking**: 3D gesture recognition that works accurately regardless of hand distance or orientation.

## Technology Stack

*   **Language**: Python 3.12
*   **Hand Tracking**: [MediaPipe](https://mediapipe.dev/) Hand Landmarker (TFLite)
*   **Computer Vision**: OpenCV (cv2)
*   **Numerical Processing**: NumPy (Matrix transformations and vector math)
*   **UI Framework**: Tkinter (Command interface)

## System Architecture & Algorithms

### 1. Multi-Threaded Execution
To ensure a smooth 60+ FPS experience, the system operates on two parallel tracks:
*   **Main Thread**: Manages the Tkinter event loop for the Command Console.
*   **Daemon Thread**: Runs the `run_sandbox` loop, handling camera capture, MediaPipe inference, and particle projection.
*   **Communication**: Windows communicate via thread-safe `queue.Queue` objects for commands and status logs.

### 2. Hand Tracking & Stabilization
The system uses a **Hybrid Interaction Model** to balance stability and intuitiveness:
*   **Positioning**: The object's screen position is anchored to the **Palm Center** (calculated as the centroid of landmarks 0, 5, 9, 13, 17).
*   **Rotation**: Roll, pitch, and yaw are derived from the **Index Finger Tip** (landmark 8), providing a responsive "pointing" feel.
*   **Adaptive Alpha Smoothing**: A non-linear filter $(\alpha)$ is applied to raw coordinates:
    $$P_{smoothed} = \alpha P_{curr} + (1 - \alpha) P_{prev}$$
    The $\alpha$ value scales dynamically based on movement velocity, suppressing jitter during stillness while maintaining zero lag during fast motion.

### 3. 3D Gesture Recognition
Traditional 2D distance checks are prone to perspective errors. gesp3D uses **3D Vector Analysis**:
*   **Normalized Pinch**: The system calculates the 3D Euclidean distance between the thumb and index tips. This distance is then normalized against the current **Hand Scale** (the 3D distance between the wrist and middle MCP).
*   **Result**: Pinching is equally accurate whether your hand is 20cm or 2m from the camera.

### 4. Rendering Engine
The Sandbox uses a custom particle-based projection algorithm:
*   **Perspective Projection**: 3D points $(X, Y, Z)$ are transformed into 2D screen space using a depth-compensation factor based on a virtual `view_distance`.
*   **Fibonacci Sphere**: The sphere utilizes a **Golden Ratio (Phi)** distribution to ensure perfectly uniform point density across its surface, avoiding the "clumping" effect seen in standard latitude/longitude grids.
*   **Z-Layering**: Particles are rendered in depth-sorted bins. Closer particles are larger and more vibrant, while distant particles fade into the background color.

## Command Console
Type these commands into the console to control the sandbox:
*   `/cube`, `/sphere`, `/pyramid`: Switch geometry.
*   `/side <front|back|top|bottom|left|right>`: Snap to a specific camera angle.
*   `/reset`: Reset all transformations to default.
*   `/options`: Show full command list.
*   `/exit`: Gracefully shut down the system.

## Requirements
*   Python 3.12+
*   OpenCV, MediaPipe, NumPy
*   `hand_landmarker.task` (MediaPipe model file)
