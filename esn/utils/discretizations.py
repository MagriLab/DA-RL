def finite_differences(J, J_right, J_left, h, method):
    if method == "forward":
        return (J_right - J) / (h)
    elif method == "backward":
        return (J - J_left) / (h)
    elif method == "central":
        return (J_right - J_left) / (2 * h)
