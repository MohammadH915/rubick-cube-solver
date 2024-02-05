# rubick-cube-solver
Rubik's Cube Solver
This Python codebase is designed to solve Rubik's Cube puzzles using various search algorithms. It incorporates advanced data structures and algorithms to efficiently find solutions to the puzzles. The code is structured into several sections, each dedicated to a specific aspect of puzzle-solving, including depth-first search (DFS), iterative deepening depth-first search (IDS-DFS), A* search, and bidirectional breadth-first search (BiBFS). Below is an overview of the system's components and their functionalities:

## Features

### System Components

- **Search Algorithms**: Implements several search algorithms to find solutions for Rubik's Cube puzzles.
- **Data Structures**: Utilizes Python's advanced data structures like dictionaries, queues, and heaps for efficient data management and search operations.
- **Heuristic Functions**: Applies heuristic functions to estimate the cost of reaching the goal state from a given state in A* search.
- **State Management**: Manages puzzle states and transitions using numpy arrays for high-performance computations.

### Key Functionalities

#### Depth-First Search (DFS)
- **Purpose**: Explores possible moves in depth to find a solution within a given depth limit.
- **Dependencies**: `numpy`

#### Iterative Deepening Depth-First Search (IDS-DFS)
- **Purpose**: Uses IDS-DFS to progressively deepen the search for a solution, combining the benefits of BFS's completeness and DFS's space efficiency.
- **Dependencies**: `numpy`

#### A* Search
- **Purpose**: Utilizes the A* search algorithm with heuristics to efficiently find the shortest path to the solution.
- **Dependencies**: `numpy`, `heapq`

#### Bidirectional Breadth-First Search (BiBFS)
- **Purpose**: Searches from both the initial and goal states simultaneously to meet in the middle, reducing the search space.
- **Dependencies**: `queue`

## Usage
python main.py --method BiBFS
