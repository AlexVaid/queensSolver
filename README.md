## Queens Solver (ver 2.0)

This project provides an enhanced implementation of the Queens problem solver using image-based board segmentation for defining colored regions. The solver places queens on a chessboard such that no two queens can attack each other, and each colored region contains at most one queen.

## Features

- **Customizable N x N Chessboards**: Easily adjust the size of the chessboard using interactive buttons.
- **Image-based Segmentation**: Detects color regions from an uploaded image to define distinct zones for placing queens.
- **Interactive GUI**: Built with Kivy, providing a smooth and user-friendly interface with hover effects and animations.
- **Backtracking Algorithm**: Utilizes an efficient backtracking algorithm to solve the N-Queens problem.
- **Custom Image Selection**: Upload an image to use for custom visual representation of the board.
- **Solution Visualization**: Displays the solution with a clear and easy-to-understand graphical layout.

## Installation

### Requirements

- **Python 3.6+**
- Required libraries:
  - `kivy`
  - `requests`
  - `plyer`
  - `pillow` (PIL)
  - `opencv-python`
  - `numpy`
  - `matplotlib`
  - `scikit-learn`

### Install dependencies

To install the required dependencies, use:

```bash
pip install kivy requests plyer pillow opencv-python numpy matplotlib scikit-learn
```
