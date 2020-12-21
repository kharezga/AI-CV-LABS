import numpy as np


def validate(hori, vert, maze):
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
    if not 0 <= hori < 8 and 0 <= vert < 8:  # check if node is valid and it exits in the maze dimensions which was defined as 8
        return False
    elif maze[hori, vert] == "#" or maze[hori, vert] == "o":
        return False
    else:
        return True


def search(node, queue, goal, maze):
    """Solve given maze with use of Deep First Search Algorithm.
                        Parameters
                        ----------
                        node : __eq__
                           Node of the algorithm
                        queue : abstract data type
                           Queue needed for the algorithm perforation
                        goal : int array
                            Coordinates of the finish
                        maze : numpy array
                            Maze to be solved
                        """
    h, v = node[:2]

    if node == goal:  # Checking if the goal is reached
        print("Maze solved")
        return
    maze[h, v] = "o"  # Marking node as visited

    # Try to move in one of four directions
    if validate(h, v + 1, maze):  # right-side
        node = [h, v + 1]
        queue.append(node)
        search(node, queue, goal, maze)

    elif validate(h + 1, v, maze):  # up-side
        node = [h + 1, v]
        queue.append(node)
        search(node, queue, goal, maze)
    elif validate(h, v - 1, maze):  # left-side
        node = [h, v - 1]
        queue.append(node)
        search(node, queue, goal, maze)
    elif validate(h - 1, v, maze):  # down-side
        node = [h - 1, v]
        queue.append(node)
        search(node, queue, goal, maze)

    # If it is not possible, return to last visited node
    else:
        queue.pop()
        node = queue[len(queue) - 1]
        print("Returning to the previous node: ", node)
        search(node, queue, goal, maze)


def main():
    start = [0, 0]
    goal = [0, 0]
    height = 8
    width = 8

    maze = np.array([["#", "#", "#", " ", " ", " ", "#", "#"],
                     ["#", " ", " ", " ", "#", " ", "#", "#"],
                     ["#", " ", "#", "#", "#", " ", "#", "#"],
                     [" ", " ", " ", " ", "#", "G", "#", "#"],
                     [" ", "#", "#", "#", "#", " ", "#", "#"],
                     [" ", "#", "#", " ", " ", " ", "#", "#"],
                     [" ", "#", "#", " ", "#", "#", "#", "#"],
                     ["S", " ", " ", " ", "#", "#", "#", "#"]])

    # Finding star and goal points
    for h in range(0, height):
        for v in range(0, width):
            if maze[h, v] == "G":
                goal = [h, v]
            elif maze[h, v] == "S":
                start = [h, v]

    queue = [start]
    print("Original maze:", maze, sep='\n')
    search(start, queue, goal, maze)
    print("Solved maze:", maze, sep='\n')


if __name__ == "__main__":
    main()
