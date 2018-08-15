def constant_zero(*args):
    return 0.0


def get_reciprocal(decay):
    def reciprocal(step_counter):
        return 1. / (decay * step_counter + 1)

    return reciprocal


def get_linear(steps, start_value, final_value):
    def linear(step_counter):
        diff = start_value - final_value

        if step_counter >= steps:
            return final_value

        return start_value - (step_counter / steps) * diff

    return linear
