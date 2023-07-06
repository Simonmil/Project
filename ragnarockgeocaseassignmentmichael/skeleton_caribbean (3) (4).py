import numpy as np
from skimage import morphology


def generate_problem(H, W):
    # First God gathered the waters and called it "sea"
    world = np.zeros((H, W))

    # Then God gathered land
    mask = np.random.choice([0, 1], world.shape, p=(0.85, 0.15))
    world[mask] = np.random.randint(1, 10, mask.shape)[mask]

    n_islands = np.random.randint(0, 100)
    max_radius = 10
    Y0s = np.random.randint(0, H, n_islands)
    X0s = np.random.randint(0, W, n_islands)
    for x0, y0 in zip(X0s, Y0s):

        radius = np.random.randint(1, max_radius + 1)

        mask = morphology.disk(radius)
        # Rise!! said God, and made a Caribbean island
        mean, std = np.random.uniform(2, 6), np.random.uniform(1, 5)
        the_island = np.random.normal(mean, std, mask.shape) * mask
        the_island = np.clip(the_island, 0, 10).astype(int)

        y1 = min(H, y0 + 2 * radius + 1)
        x1 = min(W, x0 + 2 * radius + 1)
        YY, XX = np.split(np.mgrid[y0:y1, x0:x1], 2)

        # Too lazy to do masking and stuff like that, simply add.
        world[YY, XX] += the_island[:y1 - y0, :x1 - x0]
        world[YY, XX] = np.clip(world[YY, XX], 0, 10)

    # 6000 years later and Ragnar looks for his favourite Caribbean island.
    return world


def solve(problem, m, use_8_conn=True):
    '''
    problem: np.array, from `generate_problem`
    m: int, the minimum island size.
    use_8_conn: if True, then use 8-connectivity. Otherwise, use 4
    connectivity.

    output: float, np.ndarray
        the first is a float denoting the score of the island
        the second output is an np.array of shape (?, 2) where the first column
        denote the Y-coordinates and second column the X-coordinates of the
        island.
    '''
    # TODO: implement this stuff.
    max_score = -np.inf
    return max_score, np.empty((0, 2), np.int32)


if __name__ == '__main__':

    # just some example arguments...
    H, W = 256, 512
    problem = generate_problem(H, W)
    min_island_size = 100  # in square kilometers, i.e. pixels

    solve(problem, min_island_size)
