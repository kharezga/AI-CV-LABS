import numpy as np

# Global variables
start = [7, 0]
goal = [6, 11]


def validate(h, v, maze):
    """Validate the move.
                          Parameters
                          ----------
                          hori : int
                             horizontal length of the maze
                          vert : int
                            vertical length of the maze
                          maze : numpy array
                              Maze to be solved
                          """
    if not (0 <= h <= 7) or not (0 <= v <= 13):
        return False
    elif maze[h, v] == "#" or maze[h, v] == "o":
        return False
    else:
        return True


def sort(var):
    return var['F_score']


def g_score(path):
    g_scored = len(path) - 1
    return g_scored


def h_score(node):
    hori, vert = node[:2]
    HORI, VERT = goal[:2]
    manh_dist = abs(hori - HORI) + abs(vert - VERT)
    return manh_dist


def printMaze(path, maze):
    for f in range(0, (len(path) - 1)):
        node = path[f]
        h, v = node[:2]
        maze[h, v] = "o"
    print("Solved maze:", maze, sep='\n')


def search(list_of_elements, maze):
    # Finding the node with lowest F value to perform searching from that node
    list_of_elements.sort(key=sort)

    item = list_of_elements[0]
    list_of_elements.pop(0)
    node = item['node']
    path = item['path']

    path_copy = path.copy()

    h, v = node[:2]

    if node == goal:
        print("Maze has been solved")
        printMaze(path, maze)
        return

    if validate(h, v - 1, maze):  # left
        path_copy.append([h, v - 1])
        G = g_score(path)
        H = h_score(node)
        F = G + H
        node = [h, v - 1]
        maze[h, v - 1] = "o"  # indicate that node is visited
        list_of_elements.append({'F_score': F, 'node': node, 'path': path_copy})
        path_copy = path.copy()  # clear all changes we've done to path variable

    if validate(h, v + 1, maze):  # right
        path_copy.append([h, v + 1])
        G = g_score(path)
        H = h_score(node)
        F = G + H
        node = [h, v + 1]
        maze[h, v + 1] = "o"
        list_of_elements.append({'F_score': F, 'node': node, 'path': path_copy})
        path_copy = path.copy()

    if validate(h + 1, v, maze):  # up
        path_copy.append([h + 1, v])
        G = g_score(path)
        H = h_score(node)
        F = G + H
        node = [h + 1, v]
        maze[h + 1, v] = "o"
        list_of_elements.append({'F_score': F, 'node': node, 'path': path_copy})
        path_copy = path.copy()

    if validate(h - 1, v, maze):  # down
        path_copy.append([h - 1, v])
        G = g_score(path)
        H = h_score(node)
        F = G + H
        node = [h - 1, v]
        maze[h - 1, v] = "o"
        list_of_elements.append({'F_score': F, 'node': node, 'path': path_copy})
        path_copy = path.copy()

    # Then the function will be called again to expand next node
    search(list_of_elements, maze)


def main():
    maze = np.array([["#", "#", "#", " ", " ", " ", " ", " ", "#", "#", " ", "#", "#"],
                     ["#", " ", " ", " ", "#", "#", " ", "#", "#", "#", " ", "#", "#"],
                     ["#", " ", "#", "#", "#", "#", " ", " ", " ", " ", " ", "#", "#"],
                     [" ", " ", " ", " ", "#", "#", "#", "#", "#", "#", " ", " ", " "],
                     [" ", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#", " "],
                     [" ", "#", "#", " ", " ", " ", "#", " ", " ", " ", "#", " ", " "],
                     [" ", "#", "#", " ", "#", " ", "#", " ", "#", " ", "#", "G", "#"],
                     ["S", " ", " ", " ", "#", " ", " ", " ", "#", " ", " ", " ", "#"]])
    maze_copy = maze.copy()

    list_of_elements = [{'F_score': 0, 'node': start, 'path': [start]}]
    print("Original maze:", maze_copy, sep='\n')
    search(list_of_elements, maze)


if __name__ == "__main__":
    main()
