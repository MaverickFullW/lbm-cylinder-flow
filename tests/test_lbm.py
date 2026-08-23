import numpy as np

from src.lbm import w
from src.lbm import equilibrium
from src.lbm import e
from src.lbm import collision
from src.lbm import streaming
from src.lbm import bounce_back
from src.lbm import macroscopic
from src.lbm import cylinder_force
from src.lbm import streaming_with_bounce_back
from src.lbm import velocity_inlet_left
from src.lbm import open_outlet_right


def test_weights_sum_to_one():
    assert np.isclose(np.sum(w), 1)


def test_equilibrium_at_rest_equals_weights():
    rho = 1.0
    u = np.array([0.0, 0.0])
    feq = equilibrium(rho, u)
    assert np.allclose(feq, w)


def test_equilibrium_recovers_density_and_velocity():
    rho = 1.0
    u = np.array([0.05, 0.0])
    feq = equilibrium(rho, u)

    rho_recovered = np.sum(feq)
    momentum_recovered = feq @ e
    u_recovered = momentum_recovered / rho_recovered

    assert np.isclose(rho_recovered, rho)
    assert np.allclose(u_recovered, u)


def test_equilibrium_over_full_domain():
    ny = 4
    nx = 6
    rho = np.ones((ny, nx))
    u = np.zeros((ny, nx, 2))
    u[:, :, 0] = 0.05
    feq = equilibrium(rho, u)

    rho_recovered = np.sum(feq, axis=-1)
    momentum_recovered = feq @ e
    u_recovered = momentum_recovered / rho_recovered[..., None]

    assert np.allclose(rho_recovered, rho)
    assert np.allclose(u_recovered, u)


def test_collision_preserves_equilibrium():
    rho = 1.0
    u = np.array([0.05, 0.0])
    feq = equilibrium(rho, u)
    f = feq.copy()
    tau = 0.65
    f_post = collision(f, feq, tau)

    assert np.allclose(f_post, f)


def test_streaming_moves_population():
    f_post = np.zeros((5, 5, 9))
    f_post[2, 2, 1] = 1.0

    f_streamed = streaming(f_post)

    assert f_streamed[2, 3, 1] == 1.0
    assert np.sum(f_streamed) == 1.0


def test_streaming_moves_all_directions():
    f_post = np.zeros((7, 7, 9))

    for i in range(9):
        f_post[3, 3, i] = i + 1

    f_streamed = streaming(f_post)

    destinations = [
        (3, 3),
        (3, 4),
        (4, 3),
        (3, 2),
        (2, 3),
        (4, 4),
        (4, 2),
        (2, 2),
        (2, 4),
    ]

    for i, (y, x) in enumerate(destinations):
        assert f_streamed[y, x, i] == i + 1


def test_bounce_back_reverses_solid_population():
    f_streamed = np.zeros((5, 5, 9))
    solid = np.zeros((5, 5), dtype=bool)
    solid[2, 2] = True
    f_streamed[2, 2, 1] = 1.0

    f_bounced = bounce_back(f_streamed, solid)

    assert f_bounced[2, 2, 3] == 1.0
    assert f_bounced[2, 2, 1] == 0.0
    assert np.array_equal(f_bounced[~solid], f_streamed[~solid])


def test_macroscopic_recovers_density_and_velocity():
    rho = 1.0
    u = np.array([0.05, 0.0])
    f = equilibrium(rho, u)

    rho_recovered, u_recovered = macroscopic(f)

    assert np.isclose(rho_recovered, rho)
    assert np.allclose(u_recovered, u)


def test_cylinder_force_detects_positive_x_link():
    solid = np.zeros((5, 5), dtype=bool)
    solid[2, 3] = True
    f_post = np.zeros((5, 5, 9))
    f_post[2, 2, 1] = 1.0

    Fx, Fy = cylinder_force(f_post, solid)

    assert np.isclose(Fx, 2.0)
    assert np.isclose(Fy, 0.0)


def test_cylinder_force_positive_x_momentum_exchange():
    solid = np.zeros((5, 5), dtype=bool)
    solid[2, 3] = True
    f_post = np.zeros((5, 5, 9))
    f_post[2, 2, 1] = 1.0

    Fx, Fy = cylinder_force(f_post, solid)

    assert np.isclose(Fx, 2.0)
    assert np.isclose(Fy, 0.0)


def test_streaming_with_bounce_back_positive_x():
    solid = np.zeros((5, 5), dtype=bool)
    f_post = np.zeros((5, 5, 9))
    f_post[1, 1, 1] = 2.0
    solid[3, 3] = True
    f_post[3, 2, 1] = 1.0

    f_next = streaming_with_bounce_back(f_post, solid)

    assert f_next[1, 2, 1] == 2.0
    assert f_next[3, 2, 3] == 1.0
    assert f_next[3, 3, 1] == 0.0


def test_streaming_with_bounce_back_all_moving_directions():
    opposite = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

    for i in range(1, 9):
        ex = e[i, 0]
        ey = e[i, 1]
        neighbor_y = 3 + ey
        neighbor_x = 3 + ex

        solid = np.zeros((7, 7), dtype=bool)
        f_post = np.zeros((7, 7, 9))
        f_post[3, 3, i] = 1.0

        f_next = streaming_with_bounce_back(f_post, solid)

        assert f_next[neighbor_y, neighbor_x, i] == 1.0

        solid = np.zeros((7, 7), dtype=bool)
        solid[neighbor_y, neighbor_x] = True
        f_post = np.zeros((7, 7, 9))
        f_post[3, 3, i] = 1.0

        f_next = streaming_with_bounce_back(f_post, solid)

        assert f_next[3, 3, opposite[i]] == 1.0


def test_streaming_with_bounce_back_wraps_at_right_boundary():
    solid = np.zeros((5, 5), dtype=bool)
    f_post = np.zeros((5, 5, 9))
    f_post[2, 4, 1] = 1.0

    f_next = streaming_with_bounce_back(f_post, solid)

    assert f_next[2, 0, 1] == 0.0


def test_streaming_with_bounce_back_does_not_wrap_at_left_boundary():
    solid = np.zeros((5, 5), dtype=bool)
    f_post = np.zeros((5, 5, 9))
    f_post[2, 0, 3] = 1.0

    f_next = streaming_with_bounce_back(f_post, solid)

    assert f_next[2, -1, 3] == 0.0


def test_velocity_inlet_left_recovers_prescribed_velocity():
    ny = 5
    nx = 5
    rho = np.ones((ny, nx))
    u = np.zeros((ny, nx, 2))
    f = equilibrium(rho, u)

    rho_in = velocity_inlet_left(f, ux=0.05, uy=0.0)
    rho_new, u_new = macroscopic(f)

    assert np.allclose(rho_new[:, 0], rho_in)
    assert np.allclose(u_new[:, 0, 0], 0.05)
    assert np.allclose(u_new[:, 0, 1], 0.0)


def test_macroscopic_handles_zero_density_without_nan():
    f = np.zeros((2, 2, 9))

    rho, u = macroscopic(f)

    assert np.all(rho == 0.0)
    assert not np.any(np.isnan(u))


def test_open_outlet_right_reconstructs_incoming_populations():
    f = np.zeros((5, 5, 9))
    f[:, -2, 3] = 1.0
    f[:, -2, 6] = 2.0
    f[:, -2, 7] = 3.0
    f[:, -1, 1] = 4.0

    open_outlet_right(f)

    assert np.all(f[:, -1, 3] == 1.0)
    assert np.all(f[:, -1, 6] == 2.0)
    assert np.all(f[:, -1, 7] == 3.0)
    assert np.all(f[:, -1, 1] == 4.0)
