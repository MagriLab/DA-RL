import jax.numpy as jnp


def get_steps(t, dt):
    return int(jnp.round(t / dt))


def downsample(*y, t, new_dt):
    dt = t[1] - t[0]

    downsample = get_steps(new_dt, dt)
    if downsample == 0:
        raise ValueError(
            f"Desired new time step is smaller than the simulation: {new_dt}<{dt}"
        )

    y_new = [yy[::downsample] for yy in y]
    t_new = t[::downsample]
    return *y_new, t_new


def discard_transient(*y, t, transient_time):
    dt = t[1] - t[0]

    N_transient = get_steps(transient_time, dt)

    y_new = [yy[N_transient:, :] for yy in y]
    t_new = t[N_transient:] - t[N_transient]
    return *y_new, t_new


def create_input_output(u, y, full_state, p, t, N_washout, N_loop):
    # input
    u_washout = u[0:N_washout]
    u_loop = u[N_washout : N_washout + N_loop - 1]

    # parameter/action
    p_washout = p[0:N_washout]
    p_loop = p[N_washout : N_washout + N_loop]

    # output
    y_loop = y[N_washout + 1 : N_washout + N_loop]
    full_state_loop = full_state[N_washout + 1 : N_washout + N_loop]

    # output time
    t_loop = t[N_washout + 1 : N_washout + N_loop]

    return u_washout, u_loop, p_washout, p_loop, y_loop, full_state_loop, t_loop


def create_dataset(
    full_state,
    y,
    t,
    p,
    network_dt,
    transient_time,
    washout_time,
    loop_times,
    loop_names=None,
    start_idxs=None,
):
    full_state, y, t = downsample(full_state, y, t=t, new_dt=network_dt)
    full_state, y, t = discard_transient(
        full_state, y, t=t, transient_time=transient_time
    )

    # separate into washout and loops
    N_washout, *N_loops = [
        get_steps(t, network_dt) for t in [washout_time, *loop_times]
    ]

    # create a dictionary to store the data
    data = {}
    start_idx = 0
    for loop_idx, N_loop in enumerate(N_loops):
        if start_idxs is not None:
            start_idx = start_idxs[loop_idx]

        # get washout and loop data
        (
            u_washout,
            u_loop,
            p_washout,
            p_loop,
            y_loop,
            full_state_loop,
            t_loop,
        ) = create_input_output(
            y[start_idx:],
            y[start_idx:],
            full_state[start_idx:],
            p[start_idx:],
            t[start_idx:],
            N_washout,
            N_loop + 1,
        )
        if loop_names is not None:
            loop_name = loop_names[loop_idx]
        else:
            loop_name = f"loop_{loop_idx}"
        data[loop_name] = {
            "u_washout": u_washout,
            "p_washout": p_washout,
            "u": u_loop,
            "p": p_loop,
            "y": y_loop,
            "full_state": full_state_loop,
            "t": t_loop,
        }

        if start_idxs is None:
            start_idx += N_washout + N_loop

    return data
