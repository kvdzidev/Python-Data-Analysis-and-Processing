from PIL import Image
import numpy as np
import math
import random
import time
from collections import deque


def search_path_with_astar(start, goal, accessible_fn, h, callback_fn):
    open_set = {tuple(start)}
    closed_set = set()
    came_from = {}
    g_score = {tuple(start): 0}
    f_score = {tuple(start): h(start, goal)}

    while open_set:
        callback_fn(closed_set, open_set)

        # Find the node in open_set with the lowest f_score
        current = min(open_set, key=lambda x: f_score.get(x, float('inf')))

        if current == tuple(goal):
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(tuple(start))
            return path[::-1]  # Return reversed path

        open_set.remove(current)
        closed_set.add(current)

        for neighbor in accessible_fn(current):
            if tuple(neighbor) in closed_set:
                continue

            tentative_g_score = g_score.get(current, float('inf')) + h(neighbor, current)

            if tuple(neighbor) not in open_set:
                open_set.add(tuple(neighbor))
            elif tentative_g_score >= g_score.get(tuple(neighbor), float('inf')):
                continue

            # This path is the best so far
            came_from[tuple(neighbor)] = current
            g_score[tuple(neighbor)] = tentative_g_score
            f_score[tuple(neighbor)] = g_score[tuple(neighbor)] + h(neighbor, goal)

    return []

# @funkcja heurytystyczna - metryka manhattan
# @zmienna dx wartosc bezwzgledna pary wspolrzednych jako punkt poczatkowy
# @zmienna dy wartosc bezwzgledna pary wspolrzednych jako punkt poczatkowy
# @return odleglosc pomiedzy punktami a i b wzdluz osi poziomej i pionowej
def h_function_manhattan_metrics(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return dx + dy


# @funkcja heurytystyczna - metryka euklidesa
# @zmienna dx jako punkt poczatkowy pary wspolrzednych
# @zmienna dy jako punkt poczatkowy pary wspolrzednych
# @return zwraca odleglosc prostej linii pomiedzy parami wspolrzednych
def h_function_euclidian_metrics(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx ** 2 + dy ** 2)

# @funkcja heurytystyczna - metryka losowa
# @return zwraca losowe wartosci a i b od 0-50
def h_function_random_metrics(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    print(".")
    return random.random()


def user_choose_option():
    inputting = input("Please choose an option: \nEuclidian \nManhattan \nRandom\n")
    if inputting == "Euclidian":
        return h_function_euclidian_metrics
    elif inputting == "Manhattan":
        return h_function_manhattan_metrics
    elif inputting == "Random":
        return h_function_random_metrics
    else:
        return inputting

def getpixel(image, dims, position):
    if any(p < 0 or p >= dims[i] for i, p in enumerate(position)):
        return None
    return image[position[1], position[0]]  # Pillow uses (y, x) indexing


def setpixel(image, dims, position, value):
    if any(p < 0 or p >= dims[i] for i, p in enumerate(position)):
        return
    image[position[1], position[0]] = value


def accessible(bitmap, dims, point):
    neighbors = []
    height, width = dims  # Get the dimensions (height and width)

    # Loop through each dimension (x, y)
    for i in range(len(point)):
        for delta in [-1, 1]:  # Check both directions (left and right, up and down)
            neighbor = list(point)  # Convert tuple to list to modify
            neighbor[i] += delta  # Modify the coordinate
            neighbor = tuple(neighbor)  # Convert back to tuple after modification

            # Ensure the neighbor is within the image bounds
            x, y = neighbor[0], neighbor[1]
            if 0 <= x < width and 0 <= y < height:
                # Ensure it's walkable (pixel value check)
                if bitmap[y, x][0] == 0:  # Assuming the red channel represents walkability (0 = walkable)
                    neighbors.append(neighbor)
    return neighbors


def load_world_map(fname):
    img = Image.open(fname)
    img = img.convert("RGBA")  # Ensure it's in RGBA format
    pixels = np.array(img)
    dims = pixels.shape[:2]  # (height, width)
    return dims, pixels


def save_world_map(fname, image):
    img = Image.fromarray(image)
    img.save(fname)


def find_pixel_position(image, dims, value):
    for y in range(dims[0]):
        for x in range(dims[1]):
            if tuple(image[y, x]) == value:
                return [x, y]
    raise ValueError("Could not find pixel with the given value!")


if __name__ == "__main__":
    dims, bitmap = load_world_map("img.png")

    start = find_pixel_position(bitmap, dims, (255, 0, 255, 255))  # Cyan pixel
    goal = find_pixel_position(bitmap, dims, (255, 255, 0, 255))  # Yellow pixel

    setpixel(bitmap, dims, start, (0, 0, 0, 255))
    setpixel(bitmap, dims, goal, (0, 0, 0, 255))

    heuristic_function = user_choose_option()

    def on_iteration_nothing(closed_set, open_set):
        pass


    poczatek = time.time()
    path = search_path_with_astar(start, goal, lambda p: accessible(bitmap, dims, p), heuristic_function, on_iteration_nothing)
    print(f"Length: {len(path)} nodes")
    koniec = time.time()
    czas_wykonania = koniec - poczatek
    print(f"Time: {czas_wykonania:} sekund")

    print(path)


    for p in path:
        setpixel(bitmap, dims, p, (255, 0, 0, 255))  # Mark the path with red

    save_world_map("result.png", bitmap)
