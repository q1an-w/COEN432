import pandas as pd
import numpy as np

# Function to read tiles from the input file


def read_tiles_from_file(file_path, skip_first_line=False):
    # Read the file while skipping the first line if specified
    df = pd.read_csv(file_path, sep='\s+', header=None,
                     skiprows=1 if skip_first_line else 0)
    tiles = []
    for row in df.values:
        for tile in row:
            tile_digits = [int(digit) for digit in f"{tile:04d}"]
            tiles.append(tile_digits)
    return np.array(tiles).reshape(-1, 4)

# Function to rotate a tile


def rotate_tile(tile):
    return [
        tile[3],  # First becomes the last
        tile[0],  # Second becomes the first
        tile[1],  # Third becomes the second
        tile[2]   # Fourth becomes the third
    ]

# Function to generate all rotations of a tile


def generate_rotations(tile):
    rotations = [tile]
    for _ in range(3):
        tile = rotate_tile(tile)
        rotations.append(tile)
    return rotations

# Function to count the number of different tiles (considering rotations)


def count_different_tiles(input_tiles, output_tiles):
    # Generate all unique rotations for each tile in the input
    input_rotations = set()
    for tile in input_tiles:
        for rotation in generate_rotations(tile):
            input_rotations.add(tuple(rotation))

    # Create a set for output tiles with all their rotations
    output_rotations = set()
    for tile in output_tiles:
        for rotation in generate_rotations(tile):
            output_rotations.add(tuple(rotation))

    # Count how many tiles are in the output but not in the input
    different_tiles_count = len(output_rotations - input_rotations)

    return different_tiles_count

# Main function to execute the count


def main(input_file, output_file):
    input_tiles = read_tiles_from_file(input_file)
    output_tiles = read_tiles_from_file(
        output_file, skip_first_line=True)  # Skip the first line of output

    different_count = count_different_tiles(input_tiles, output_tiles)
    print(f"Number of different tiles: {different_count}")


# Example usage
if __name__ == "__main__":
    input_file = "Ass1Input.txt"
    output_file = "Ass1Output.txt"
    main(input_file, output_file)
