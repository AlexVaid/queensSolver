# Queens Solver (LinkedIn)

This project provides an implementation of the Queens problem using image-based board segmentation to define colored regions. 
The solver places queens on a chessboard such that no two queens can attack each other, and each colored region contains at most one queen.

## Features
- Supports custom NxN boards.
- Uses image segmentation for color-based region recognition.
- Implements backtracking to find a valid solution.

## How to Use
1. Run the script, and you will be prompted to snip an image of the board.
2. The solver will find a solution and visualize it.

## Requirements
- Python 3.x
- Libraries: `PIL`, `OpenCV`, `NumPy`, `matplotlib`, `scikit-learn`