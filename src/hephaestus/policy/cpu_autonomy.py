"""Predeclared finite adaptive policy, not a certified specialist-model policy."""


def learning_rate(plan, policy, cycle, memories):
    if not 1 <= cycle <= plan.cycles or policy not in plan.policies:
        raise ValueError("CPU policy/cycle outside the frozen plan")
    if policy == "fixed":
        return plan.fixed_learning_rate, "predeclared fixed recipe"
    if policy == "heuristic":
        return plan.heuristic_schedule[cycle-1], "predeclared time schedule; no outcome feedback"
    if not memories:
        return plan.initial_learning_rate, "first adaptive experiment; no prior outcome"
    last = memories[-1]["metadata"]
    previous = float(last["controlled_variables"]["selected_learning_rate"])
    grid = list(plan.learning_rate_grid)
    if previous not in grid:
        raise ValueError("experiment memory contains an unadmitted learning rate")
    index = grid.index(previous)
    if last["outcome"] == "improved":
        return grid[min(len(grid)-1,index+1)], "prior accepted development improvement; try next admitted rate"
    return grid[max(0,index-1)], "prior nonimprovement/regression; reduce admitted rate"


def accept_candidate(before, after, margin):
    return (after["mean_target_nll"] < before["mean_target_nll"]-margin
            and after["exact_correct"] >= before["exact_correct"])
