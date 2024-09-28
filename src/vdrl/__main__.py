import sys
import os


def main(args):
    pass


def run_minigrid_labyrinth():
    dirname = os.path.dirname(__file__)
    example_path = os.path.abspath(
        dirname + "/../../examples/run_minigrid_labyrinth.py"
    )
    os.system(f'python "{example_path}"')


if __name__ == "__main__":
    main(sys.argv)
