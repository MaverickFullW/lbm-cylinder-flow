import numpy as np

e = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [-1, 0],
    [0, -1],
    [1, 1],
    [-1, 1],
    [-1, -1],
    [1, -1]
])

w = np.array([
    4/9,
    1/9,
    1/9,
    1/9,
    1/9,
    1/36,
    1/36,
    1/36,
    1/36
])


def equilibrium(rho, u):
    eu = u @ e.T
    u2 = np.sum(u**2, axis=-1)
    feq = w * np.asarray(rho)[..., None] * (
        1 + 3*eu + 4.5*eu**2 - 1.5*u2[..., None]
    )

    return feq


def collision(f, feq, tau):
    f_post = f - (f - feq) / tau

    return f_post


def streaming(f_post):
    f_streamed = np.empty_like(f_post)

    for i in range(9):
        f_streamed[:, :, i] = np.roll(
            f_post[:, :, i], shift=(e[i, 1], e[i, 0]), axis=(0, 1)
        )

    return f_streamed


def streaming_with_bounce_back(f_post, solid):
    f_next = np.zeros_like(f_post)
    fluid = ~solid
    opposite = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

    f_next[:, :, 0][fluid] = f_post[:, :, 0][fluid]

    for i in range(1, 9):
        ex = e[i, 0]
        ey = e[i, 1]
        neighbor_solid = np.roll(
            solid, shift=(-ey, -ex), axis=(0, 1)
        )
        neighbor_fluid = np.roll(
            fluid, shift=(-ey, -ex), axis=(0, 1)
        )
        fluid_to_fluid = fluid & neighbor_fluid
        fluid_to_solid = fluid & neighbor_solid

        if ex == 1:
            fluid_to_fluid[:, -1] = False
            fluid_to_solid[:, -1] = False
        elif ex == -1:
            fluid_to_fluid[:, 0] = False
            fluid_to_solid[:, 0] = False

        moving_population = np.where(
            fluid_to_fluid, f_post[:, :, i], 0
        )
        f_next[:, :, i] += np.roll(
            moving_population, shift=(ey, ex), axis=(0, 1)
        )
        f_next[:, :, opposite[i]][fluid_to_solid] += (
            f_post[:, :, i][fluid_to_solid]
        )

    return f_next


def bounce_back(f_streamed, solid):
    opposite = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
    f_bounced = f_streamed.copy()
    f_bounced[solid] = f_streamed[solid][:, opposite]

    return f_bounced


def cylinder_force(f_post, solid):
    fluid = ~solid
    links_per_direction = []
    Fx = 0.0
    Fy = 0.0

    for i in range(1, 9):
        ex = e[i, 0]
        ey = e[i, 1]
        neighbor_solid = np.roll(
            solid, shift=(-ey, -ex), axis=(0, 1)
        )
        links = fluid & neighbor_solid
        links_per_direction.append(links)
        delta_px = 2 * f_post[:, :, i] * e[i, 0]
        delta_py = 2 * f_post[:, :, i] * e[i, 1]
        Fx += np.sum(delta_px[links])
        Fy += np.sum(delta_py[links])

    return Fx, Fy


def velocity_inlet_left(f, ux, uy=0.0):
    rho = (
        f[:, 0, 0]
        + f[:, 0, 2]
        + f[:, 0, 4]
        + 2 * (
            f[:, 0, 3]
            + f[:, 0, 6]
            + f[:, 0, 7]
        )
    ) / (1 - ux)
    f[:, 0, 1] = f[:, 0, 3] + (2.0 / 3.0) * rho * ux
    f[:, 0, 5] = (
        f[:, 0, 7]
        - 0.5 * (f[:, 0, 2] - f[:, 0, 4])
        + 0.5 * rho * uy
        + (1.0 / 6.0) * rho * ux
    )
    f[:, 0, 8] = (
        f[:, 0, 6]
        + 0.5 * (f[:, 0, 2] - f[:, 0, 4])
        - 0.5 * rho * uy
        + (1.0 / 6.0) * rho * ux
    )

    return rho


def open_outlet_right(f):
    f[:, -1, 3] = f[:, -2, 3]
    f[:, -1, 6] = f[:, -2, 6]
    f[:, -1, 7] = f[:, -2, 7]

    return f


def macroscopic(f):
    rho = np.sum(f, axis=-1)
    momentum = f @ e
    u = np.zeros_like(momentum)
    np.divide(
        momentum,
        rho[..., None],
        out=u,
        where=rho[..., None] > 0
    )

    return rho, u
