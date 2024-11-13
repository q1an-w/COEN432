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
    # For input tiles, generate all rotations and store them in a list as tuples
    input_rotations = []
    for tile in input_tiles:
        input_rotations.extend([tuple(rotation)
                               for rotation in generate_rotations(tile)])

    # For output tiles, generate all rotations and store them in a list as tuples
    output_rotations = []
    for tile in output_tiles:
        output_rotations.extend([tuple(rotation)
                                for rotation in generate_rotations(tile)])

    # Initialize counters for unmatched tiles
    different_tiles_count = 0

    # Create a copy of the input_rotations list to track matched tiles
    unmatched_input = input_rotations.copy()

    # Compare each output tile's rotations against the input tile rotations
    for output_tile in output_rotations:
        if output_tile in unmatched_input:
            # Remove matched tile from input list
            unmatched_input.remove(output_tile)
        else:
            print(output_tile)
            different_tiles_count += 1  # Count the tile as different

    return different_tiles_count

# Main function to execute the count


def main(input_file, output_file):
    input_tiles = read_tiles_from_file(input_file)
    output_tiles = read_tiles_from_file(
        output_file, skip_first_line=True)  # Skip the first line of output

    different_count = count_different_tiles(input_tiles, output_tiles) / 4
    print(f"Number of different tiles: {different_count}")


# Example usage
if __name__ == "__main__":
    input_file = "Ass1Input.txt"
    output_file = "bestoutput.txt"
    main(input_file, output_file)
