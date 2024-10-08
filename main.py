import subprocess
import time
import ctypes
from PIL import ImageGrab
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random
import matplotlib.patches as patches
from abc import ABC, abstractmethod

# Scissors class handles the clipboard and snipping tool operations
class Scissors:
    # Clears the system clipboard
    @staticmethod
    def clear_clipboard():
        ctypes.windll.user32.OpenClipboard(0)
        ctypes.windll.user32.EmptyClipboard()
        ctypes.windll.user32.CloseClipboard()

    # Runs the Snipping Tool
    @staticmethod
    def run_snipping_tool():
        Scissors.clear_clipboard()
        subprocess.Popen(['SnippingTool.exe'])

    # Closes the Snipping Tool
    @staticmethod
    def close_snipping_tool():
        subprocess.call(['taskkill', '/F', '/IM', 'SnippingTool.exe'])

    # Gets the snipped image from the clipboard
    @staticmethod
    def get_snipped_image():
        while True:
            image = ImageGrab.grabclipboard()
            if isinstance(image, ImageGrab.Image.Image):
                return image
            else:
                time.sleep(1)

# ImageProcessor class processes the snipped image and extracts regions/colors
class ImageProcessor:
    def __init__(self, image, n, n_init=20, random_state=42):
        self.image = self._load_image(image)
        self.image = cv2.GaussianBlur(self.image, (5, 5), 0)  # Applying Gaussian blur to smooth the image
        self.N = n
        self.cell_height = self.image.shape[0] // self.N  # Calculate cell height
        self.cell_width = self.image.shape[1] // self.N   # Calculate cell width
        self.colors = self._extract_colors()  # Extract average colors from cells
        # Apply KMeans clustering to identify distinct regions/colors
        self.kmeans = KMeans(n_clusters=n, n_init=n_init, random_state=random_state).fit(self.colors)
        self.regions = self.kmeans.labels_.reshape(self.N, self.N).tolist()  # Cluster labels for each cell
        self.color_map = [tuple((color * 255).astype(int)) for color in self.kmeans.cluster_centers_]  # Convert to RGB colors
    
    # Load and convert the image
    def _load_image(self, image):
        if isinstance(image, str):
            return cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Extract average colors from each cell of the image
    def _extract_colors(self):
        return [
            np.mean(self.image[i * self.cell_height:(i + 1) * self.cell_height, j * self.cell_width:(j + 1) * self.cell_width], axis=(0, 1)) / 255.0
            for i in range(self.N) for j in range(self.N)
        ]

# Represents a queen on the chessboard
class Queen:
    def __init__(self, row, col):
        self.row = row
        self.col = col

# RegionTracker keeps track of the number of queens in each region/color zone
class RegionTracker:
    def __init__(self, regions, n):
        self.regions = regions
        self.region_queens = [0] * n  # Initialize queen count for each region
    
    # Increment the count for the region
    def increment_region(self, row, col):
        region_id = self.regions[row][col]
        self.region_queens[region_id] += 1

    # Decrement the count for the region
    def decrement_region(self, row, col):
        region_id = self.regions[row][col]
        self.region_queens[region_id] -= 1

    # Check if the region is safe to place a queen
    def is_region_safe(self, row, col):
        region_id = self.regions[row][col]
        return self.region_queens[region_id] == 0

# ChessBoard class represents the board and manages the placement of queens
class ChessBoard:
    def __init__(self, n, image_processor):
        self.N = n
        self.board = [[0] * self.N for _ in range(self.N)]  # Initialize an empty board
        self.regions = image_processor.regions  # Regions/colors from the image
        self.color_map = image_processor.color_map  # Colors associated with regions
        self.queens = []  # List of placed queens
        self.region_tracker = RegionTracker(self.regions, n)  # Track the regions
    
    # Place a queen on the board
    def place_queen(self, queen):
        self.board[queen.row][queen.col] = 1
        self.region_tracker.increment_region(queen.row, queen.col)
        self.queens.append(queen)

    # Remove a queen from the board
    def remove_queen(self, queen):
        if self.board[queen.row][queen.col] == 1:
            self.board[queen.row][queen.col] = 0
            self.region_tracker.decrement_region(queen.row, queen.col)
            self.queens.remove(queen)

    # Check if a position is safe to place a queen
    def is_safe(self, row, col):
        return self._check_row(row) and self._check_column(col) and self._check_neighbors(row, col) and self.region_tracker.is_region_safe(row, col)

    # Check if there is no queen in the given row
    def _check_row(self, row):
        return not any(self.board[row][i] == 1 for i in range(self.N))

    # Check if there is no queen in the given column
    def _check_column(self, col):
        return not any(self.board[i][col] == 1 for i in range(self.N))

    # Check if neighboring cells do not have queens
    def _check_neighbors(self, row, col):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < self.N and 0 <= c < self.N and self.board[r][c] == 1:
                return False
        return True

    # Count the number of conflicts for a given position
    def count_conflicts(self, row, col):
        conflicts = sum(1 for i in range(self.N) if (i != row and self.board[i][col] == 1) or (i != col and self.board[row][i] == 1))
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        conflicts += sum(1 for dr, dc in directions if 0 <= row + dr < self.N and 0 <= col + dc < self.N and self.board[row + dr][col + dc] == 1)
        return conflicts

# CandidateGenerator class generates candidate rows to place a queen
class CandidateGenerator:
    @staticmethod
    def generate_candidate_rows(board, col):
        rows = list(range(board.N))
        random.shuffle(rows)  # Shuffle rows to randomize selection
        rows.sort(key=lambda r: board.count_conflicts(r, col))  # Sort rows by the number of conflicts
        return rows

# Abstract base class for different solver strategies
class SolverStrategy(ABC):
    @abstractmethod
    def solve(self, board):
        pass

# BacktrackingSolver implements the backtracking algorithm to solve the board
class BacktrackingSolver(SolverStrategy):
    def solve(self, board):
        if self._solve_recursive(board, 0):
            print("Coordinates of the queens' placement:")
            for queen in board.queens:
                print(f"Queen {queen.col + 1} at ({queen.col + 1}, {queen.row + 1})")

    # Recursively try to solve the board by placing queens column by column
    def _solve_recursive(self, board, col):
        if col >= board.N:
            return True
        rows = CandidateGenerator.generate_candidate_rows(board, col)
        for row in rows:
            if board.is_safe(row, col):
                queen = Queen(row, col)
                board.place_queen(queen)
                if self._solve_recursive(board, col + 1):
                    return True
                board.remove_queen(queen)  # Backtrack if the solution fails
        return False

# QueensSolver class that uses a given strategy to solve the board
class QueensSolver:
    def __init__(self, board, strategy: SolverStrategy):
        self.board = board
        self.strategy = strategy

    def solve(self):
        self.strategy.solve(self.board)

# BoardRenderer class renders the chessboard and visualizes the solution
class BoardRenderer:
    @staticmethod
    def render_board(board, highlight_conflicts=False):
        plt.figure(1, figsize=(6, 6))
        plt.clf()
        ax = plt.gca()
        # Draw the outer border of the board
        ax.add_patch(plt.Rectangle((-0.5, -0.5), board.N, board.N, facecolor='none', edgecolor='black', linewidth=5, joinstyle='round'))
        conflicted_cells = []

        # Identify conflicted cells if highlighting conflicts is enabled
        if highlight_conflicts:
            conflicted_cells = [(i, j) for i in range(board.N) for j in range(board.N) if board.board[i][j] == 1 and board.count_conflicts(i, j) > 0]

        # Draw the cells of the board with colors from the image
        for i in range(board.N):
            for j in range(board.N):
                facecolor = np.array(board.color_map[board.regions[i][j]]) / 255.0
                rect = patches.Rectangle((j, i), 1, 1, facecolor=facecolor, edgecolor='black', linewidth=1, joinstyle='round')
                ax.add_patch(rect)

        # Draw queens on the board, highlighting conflicts if any
        for i in range(board.N):
            for j in range(board.N):
                if board.board[i][j] == 1:
                    is_conflicted = (i, j) in conflicted_cells
                    color = 'red' if highlight_conflicts and is_conflicted else 'black'
                    ax.text(j + 0.5, i + 0.55, '♛', ha='center', va='center', fontsize=18, color=color, bbox=dict(facecolor='none', edgecolor='none', pad=0.5))

        # Draw borders between different color regions
        for i in range(board.N):
            for j in range(board.N):
                if j < board.N - 1 and board.regions[i][j] != board.regions[i][j + 1]:
                    ax.plot([j + 1, j + 1], [i, i + 1], color='black', linewidth=3)
                if i < board.N - 1 and board.regions[i][j] != board.regions[i + 1][j]:
                    ax.plot([j, j + 1], [i + 1, i + 1], color='black', linewidth=3)
                if j == 0:
                    ax.plot([j, j], [i, i + 1], color='black', linewidth=5)
                if j == board.N - 1:
                    ax.plot([j + 1, j + 1], [i, i + 1], color='black', linewidth=5)
                if i == 0:
                    ax.plot([j, j + 1], [i, i], color='black', linewidth=5)
                if i == board.N - 1:
                    ax.plot([j, j + 1], [i + 1, i + 1], color='black', linewidth=5)

        # Set limits and labels for the plot
        ax.set_xlim(0, board.N)
        ax.set_xticks([i + 0.5 for i in range(board.N)])
        ax.set_xticklabels([str(i + 1) for i in range(board.N)])
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
        ax.set_ylim(0, board.N)
        ax.invert_yaxis()
        ax.set_yticks([i + 0.5 for i in range(board.N)])
        ax.set_yticklabels([str(i + 1) for i in range(board.N)])
        ax.grid(False)
        plt.draw()

# Main function that sets up the board and solves the N-Queens problem
def main():
    n = int(input("Enter the size of the board (N x N): "))  # Input board size
    if n <= 0:
        print("The size of the board must be a positive number.")
        return
    
    Scissors.run_snipping_tool()  # Run the snipping tool to capture the board image
    snipped_image = Scissors.get_snipped_image()  # Get the snipped image from clipboard
    Scissors.close_snipping_tool()  # Close the snipping tool

    # If the snipped image is not obtained, print error message
    if not isinstance(snipped_image, ImageGrab.Image.Image):
        print("Failed to get the image from the clipboard.")
        return

    # Convert image to NumPy array and process
    image_np = np.array(snipped_image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    image_processor = ImageProcessor(image_bgr, n)  # Process image to extract regions
    board = ChessBoard(n, image_processor)  # Create chess board with extracted regions
    strategy = BacktrackingSolver()  # Use backtracking strategy to solve the problem
    solver = QueensSolver(board, strategy)  # Initialize solver
    print("Solving the queens placement using the backtracking algorithm:")
    solver.solve()  # Solve the board
    BoardRenderer.render_board(board, highlight_conflicts=True)  # Render the board with conflicts highlighted
    manager = plt.get_current_fig_manager()
    screen_width = manager.window.winfo_screenwidth()
    screen_height = manager.window.winfo_screenheight()
    manager.window.wm_geometry(f'+{screen_width - 600}+{screen_height - 600}')  # Set window position
    plt.show()  # Display the board
    
# Entry point of the script
if __name__ == "__main__":
    main()