import numpy as np

# Uniform from nd ball
def random_uni_points_in_ball(num_points, radius=1, center=(0, 0, 0), dim=3):
    """
    Generate random points uniformly inside an n-dimensional ball (L2 norm).
    """
    point = np.random.normal(size=(num_points, dim))

    norm_point = np.linalg.norm(point, ord=2, axis=-1, keepdims=True)

    scaled_point = radius * point / norm_point * np.random.uniform(low=0, high=1, size=(num_points,1))**(1/dim) \
                   + np.array(center).reshape(1, -1)

    return scaled_point



def gaussian_disk(num_points=1):
    """
        Generate a random point in the unit disk using Gaussian sampling.
    """
    point = np.random.normal(loc=0.0, scale=1.0, size=2)
    norm_point = np.linalg.norm(point)
    scaled_point = point / norm_point * np.random.uniform(0, 1)**(1/2)
    return scaled_point

def gaussian_ball(num_points=1):
    """
        Generate a random point in the unit ball using Gaussian sampling.
    """
    point = np.random.normal(loc=0.0, scale=1.0, size=3)
    norm_point = np.linalg.norm(point)
    scaled_point = point / norm_point * np.random.uniform(0, 1)**(1/3)
    return scaled_point




def cutoff_and_rescale_noise(noise, cutoff_radius):
    """
    Cutoff noise vectors that exceed the given radius and rescale them.

    Arguments:
    - noise: (N, d) array of sampled noise vectors.
    - cutoff_radius: The threshold norm for noise.

    Returns:
    - Adjusted noise with norm constraint.
    """
    norms = np.linalg.norm(noise, axis=1)
    mask = norms > cutoff_radius  # Identify outliers

    # Rescale the noise vectors that exceed the cutoff
    noise[mask] = noise[mask] / norms[mask, np.newaxis] * cutoff_radius

    return noise

# Function to generate random points within a 3D ball of radius R
def random_points_in_ball(R, num_points, center = (0, 0, 0), dim=3):
    """
        Generate random points uniformly distributed in a ball of radius R in 'dim' dimensions.
    """
    points = []
    for _ in range(num_points):
        # generate a random point from a 3D Gaussian distribution
        point = gaussian_ball()
        # scale the point to be within the ball
        scaled_point = R*point
        points.append(scaled_point)
    return np.array(points)


def random_points_in_ball_with_noise(R, num_points, noise_std=0.1, dim=3):
    """
        Generate random points uniformly distributed in a ball of radius R in 'dim' dimensions
        and add Gaussian noise.
    """
    points = random_points_in_ball(R, num_points, dim)
    noise = np.random.normal(0, noise_std, size=points.shape)
    noisy_points = points + noise
    return noisy_points

def random_points_in_ball_with_cutoff_noise(R, num_points, noise_std=0.1, cutoff=0.2, dim=3):
    """
    Generate random points uniformly distributed in a ball of radius R in 'dim' dimensions
    and add Gaussian noise with a cutoff and rescaling.

    Arguments:
    - R: Radius of the ball.
    - num_points: Number of points to generate.
    - noise_std: Standard deviation of the noise.
    - cutoff: Maximum allowable norm for noise.
    - dim: Dimensionality of the space (default 3).

    Returns:
    - Noisy points inside the ball with constrained noise.
    """
    points = random_points_in_ball(R, num_points, dim=dim)
    noise = np.random.normal(0, noise_std, size=points.shape)
    
    # Apply cutoff and rescaling
    noise = cutoff_and_rescale_noise(noise, cutoff)

    noisy_points = points + noise
    return noisy_points


# Function to generate random points in the disk of radius R - 2d or 3d
def random_points_in_disk(R, num_points):
    """
        Generate random points within a 2D disk using Gaussian sampling.

        Arguments:
        - radius: The radius of the disk.
        - num_points: Number of random points to sample.

        Returns:
        - points: Randomly sampled points within the disk.
    """
    points = []

    for _ in range(num_points):
        # generate a random point from a 2D Gaussian distribution
        point = gaussian_disk()
        # scale the point to be within the disk
        scaled_point = R*point
        points.append(scaled_point)
    # Return points as an array of shape (num_points, 2)
    return np.array(points)

def random_points_in_disk_with_noise(R, num_points, noise_std=0.1):
    """
        Generate random points within a 2D disk and add Gaussian noise.
    """
    points = random_points_in_disk(R, num_points)
    noise = np.random.normal(0, noise_std, size=points.shape)
    noisy_points = points + noise
    return noisy_points

def random_points_in_disk_with_cutoff_noise(R, num_points, noise_std=0.1, cutoff=0.2):
    """
    Generate random points uniformly distributed in a disk of radius R in '2D' dimensions
    and add Gaussian noise with a cutoff and rescaling.

    Arguments:
    - R: Radius of the disk.
    - num_points: Number of points to generate.
    - noise_std: Standard deviation of the noise.
    - cutoff: Maximum allowable norm for noise.

    Returns:
    - Noisy points inside the disk with constrained noise.
    """
    points = random_points_in_disk(R, num_points)
    noise = np.random.normal(0, noise_std, size=points.shape)
    
    # Apply cutoff and rescaling
    noise = cutoff_and_rescale_noise(noise, cutoff)

    noisy_points = points + noise
    return noisy_points

def random_points_in_ellipse(a, b, num_points):
    """
        Generate random points within a 2D ellipse using Gaussian sampling.

        Arguments:
        - a: Semi-major axis length.
        - b: Semi-minor axis length.
        - num_points: Number of random points to sample.

        Returns:
        - points: Randomly sampled points within the ellipse.
    """
    points = []
    for _ in range(num_points):
        gaussian_point = gaussian_disk()
        ellipse_point = np.array([a*gaussian_point[0], b*gaussian_point[1]])
        points.append(ellipse_point)
    return np.array(points)

def random_points_in_ellipse_with_noise(a, b, num_points, noise_std=0.1):
    """
        Generate random points within a 2D ellipse and add Gaussian noise.
    """
    points = random_points_in_ellipse(a, b, num_points)
    noise = np.random.normal(0, noise_std, size=points.shape)
    noisy_points = points + noise
    return noisy_points

def random_points_in_ellipsoid(a, b, c, num_points):
    """
        Generate random points within a 3D ellipsoid using Gaussian sampling.

        Arguments:
        - a: Semi-principal axis along the x-axis.
        - b: Semi-principal axis along the y-axis.
        - c: Semi-principal axis along the z-axis.
        - num_points: Number of random points to sample.

        Returns:
        - points: Randomly sampled points within the ellipsoid.
    """
    points = []
    for _ in range(num_points):
        gaussian_point = gaussian_ball()
        ellipsoid_point = np.array([a*gaussian_point[0], b*gaussian_point[1], c*gaussian_point[2]])
        points.append(ellipsoid_point)
    return np.array(points)

def random_points_in_ellipsoid_with_noise(a, b, c, num_points, noise_std=0.1):
    """
        Generate random points within a 3D ellipsoid and add Gaussian noise.
    """
    points = random_points_in_ellipsoid(a, b, c, num_points)
    noise = np.random.normal(0, noise_std, size=points.shape)
    noisy_points = points + noise
    return noisy_points

def random_points_in_ellipsoid_with_cutoff_noise(a, b, c, num_points, noise_std=0.1, cutoff=0.2):
    """
    Generate random points in an ellipsoid and add Gaussian noise with cutoff and rescaling.

    Arguments:
    - a, b, c: Semi-axes of the ellipsoid.
    - num_points: Number of points to generate.
    - noise_std: Standard deviation of the noise.
    - cutoff: Maximum allowable norm for noise.

    Returns:
    - Noisy points inside the ellipsoid with constrained noise.
    """
    points = random_points_in_ellipsoid(a, b, c, num_points)
    noise = np.random.normal(0, noise_std, size=points.shape)
    
    # Apply cutoff and rescaling
    noise = cutoff_and_rescale_noise(noise, cutoff)

    noisy_points = points + noise
    return noisy_points