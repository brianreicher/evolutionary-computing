from collections import defaultdict

import numpy as np
import pandas as pd

import evo_v6 as evo
from profiler import profile, Profiler

SECTIONS: np.ndarray = np.genfromtxt("data/sections.csv", delimiter=",", skip_header=1)
SECTIONS_DF: pd.DataFrame = pd.read_csv('data/sections.csv', usecols=['section', 'daytime'])
TA_PREFS: np.ndarray = np.genfromtxt("data/tas.csv", delimiter=",", skip_header=1, usecols=range(3, 20), dtype='S20')
TA_MAX_ASSIGNED: np.ndarray = np.genfromtxt("data/tas.csv", delimiter=",", skip_header=1, usecols=[2])


@profile
def overallocation(sol) -> int:
    """ Objective function:

    Minimize overallocation of TAs (overallocation): Each TA specifies how many labs they can
    support (max_assigned column in tas.csv). If a TA requests at most 2 labs, and you assign to them 5
    labs, that’s an overallocation penalty of 3. Compute the objective by summing the overallocation
    penalty over all TAs. There is no minimum allocation.

    Args:
        sol (np.array): Numpy array where each row is a TA and each column is a lab. The value in each
                        cell is 1 if the TA is assigned to the lab and 0 otherwise.

    Returns:
        int: The overallocation penalty for the solution.

    """
    # number of labs assigned to each TA is just the sum of each row
    labs_assigned: np.ndarray = np.sum(sol, axis=1)

    # overallocation penalty for each TA (can't be negative, max with 0)
    overallocations: np.ndarray = np.maximum(labs_assigned - TA_MAX_ASSIGNED, 0)

    # sum the overallocation penalties
    return np.sum(overallocations).item()


@profile
def conflicts(sol) -> int:
    """ Objective function:

    Minimize time conflicts (conflicts): Minimize the number of TAs with one or more time conflicts. A
    time conflict occurs if you assign a TA to two labs meeting at the same time. If a TA has multiple
    time conflicts, still count that as one overall time conflict for that TA.

    Args:
        sol (np.array): Numpy array where each row is a TA and each column is a lab. The value in each
                        cell is 1 if the TA is assigned to the lab and 0 otherwise.

    Returns:
        int: The number of TAs with one or more time conflicts.

    """
    # get sections daytime
    sections: dict = dict(zip(SECTIONS_DF['section'],
                              SECTIONS_DF['daytime']))

    # labs assigned to each TA is the index of 1s
    labs_assigned: tuple = tuple(zip(np.where(sol)[0], np.where(sol)[1]))

    # extract assigned course daytimes (matching labs_assigned index to full sects dict)
    assigned_times: list = [(labs_assigned[k][0], sections[labs_assigned[k][1]]) for k in range(0, len(labs_assigned))]

    # dict to see every scheulded time for each TA
    times_dict: defaultdict = defaultdict(list)

    # add times to dict
    [times_dict[item[0]].append(item[1]) for item in assigned_times]

    # for each TA key, if there are any repeat time assignments, add a penalty
    conflicts: list[int] = [1 for key, value in times_dict.items() if len(times_dict[key]) != len(set(times_dict[key]))]

    return np.sum(conflicts).item()


@profile
def undersupport(sol) -> int:
    """ Objective function:

    Minimize Under-Support (undersupport): If a section needs at least 3 TAs and you only assign 1,
    count that as 2 penalty points. Minimize the total penalty score across all sections. There is no
    penalty for assigning too many TAs. You can never have enough TAs.

    Args:
        sol (np.array): Numpy array where each row is a TA and each column is a lab. The value in each
                        cell is 1 if the TA is assigned to the lab and 0 otherwise.

    Returns:
        int: The total undersupport penalty for the solution.

    """
    # number of TAs assigned to each lab is just the sum of each row
    labs_assigned = np.sum(sol, axis=0)

    # undersupport penalty for each TA (can't be negative, max with 0)
    undersupport = np.maximum(SECTIONS[:, 6] - labs_assigned, 0)

    # sum the undersupport penalties
    return np.sum(undersupport).item()


@profile
def unwilling(sol) -> int:
    """ Objective function:

    Minimize the number of times you allocate a TA to a section they are unwilling to support
    (unwilling): You could argue this is really a hard constraint, but we will treat it as an objective to be
    minimized instead.

    Args:
        sol (np.array): Numpy array where each row is a TA and each column is a lab. The value in each
                        cell is 1 if the TA is assigned to the lab and 0 otherwise.

    Returns:
        int: The number of times a TA is assigned to a section they are unwilling to support.

    """
    # number of labs assigned to each TA they're unwilling to support
    unwilling_labs: np.ndarray = np.sum(sol * (TA_PREFS == b'U'), axis=1)

    # sum it all up
    return np.sum(unwilling_labs).item()


@profile
def unpreferred(sol) -> int:
    """ Objective function:

    Minimize the number of times you allocate a TA to a section where they said “willing” but not
    “preferred” (unpreferred): In effect, we are trying to assign TAs to sections that they prefer. But we
    want to frame every objective a minimization objective. So, if your solution score has unwilling=0
    and unpreferred=0, then all TAs are assigned to sections they prefer.

    Args:
        sol (np.array): Numpy array where each row is a TA and each column is a lab. The value in each
                        cell is 1 if the TA is assigned to the lab and 0 otherwise.

    Returns:
        int: The number of times a TA is assigned to a section they do not prefer.

    """
    # number of labs assigned to each TA they're willing to support
    willing_labs = np.sum(sol * (TA_PREFS == b'W'), axis=1)

    # sum it all up
    return np.sum(willing_labs).item()


@profile
def swap_random_tas(sol) -> np.ndarray:
    """ Agent function:

    Swap two TAs at random by swapping two random rows in the solution.

    Args:
        sol (list): List containing a numpy array where each row is a TA and each column is a lab.
                    The value in each cell is 1 if the TA is assigned to the lab and 0 otherwise.

    Returns:
        np.array: The fixed solution.

    """
    # extract the solution from the list
    sol = sol[0]

    # get the index of two random TAs
    tas: np.ndarray = np.random.choice(len(sol), 2)
    ta1, ta2 = tas[0], tas[1]

    # swap the two TAs
    sol[[ta1, ta2]] = sol[[ta2, ta1]]

    return sol


@profile
def fix_overallocated_tas(sol) -> np.ndarray:
    """ Agent function:

    Fix any TAs that are overallocated by removing a random lab from the TA. If the TA is not overallocated
    then do nothing.

    Args:
        sol (list): List containing a numpy array where each row is a TA and each column is a lab.
                    The value in each cell is 1 if the TA is assigned to the lab and 0 otherwise.

    Returns:
        np.array: The fixed solution.

    """
    # extract the solution from the list
    sol = sol[0]

    # get the number of labs assigned to each TA
    labs_assigned = np.sum(sol, axis=1)

    def unassign_lab(ta) -> None:
        """ Helper function to unassign a random lab from a TA.

        Args:
            ta (int): The index of the TA to unassign a lab from.

        Returns:
            np.array: The solution with the lab unassigned from the TA.

        """
        # get the index of the labs assigned to the TA
        labs: np.ndarray = np.where(sol[ta] == 1)[0]

        # get the index of a random lab assigned to the TA
        lab = np.random.choice(labs)

        # remove the lab from the TA
        sol[ta][lab] = 0

    # get the index of the TAs that are overallocated
    overallocated_tas: np.ndarray = np.where(labs_assigned > TA_MAX_ASSIGNED)[0]

    # unassign a random lab from each overallocated TA
    [unassign_lab(ta) for ta in overallocated_tas]

    return sol


@profile
def fix_unwilling(sol) -> np.ndarray:
    """ Agent function:

    Unassign TAs from labs they are unwilling to support.

    Args:
        sol (list): List containing a numpy array where each row is a TA and each column is a lab.
                    The value in each cell is 1 if the TA is assigned to the lab and 0 otherwise.

    Returns:
        np.array: The fixed solution.

    """
    # extract the solution from the list
    sol = sol[0]

    # get the instances where a TA is unwilling to support a lab
    unwilling = TA_PREFS == b'U'

    # unassign the TAs from the labs they're unwilling to support
    sol[unwilling] = 0

    return sol


@profile
def mutate(sol) -> np.ndarray:
    """ Agent function:

    Mutate the solution by selecting a random TA and toggling a random lab.

    Args:
        sol (list): List containing a numpy array where each row is a TA and each column is a lab.
                    The value in each cell is 1 if the TA is assigned to the lab and 0 otherwise.

    Returns:
        np.array: The mutated solution.

    """
    # extract the solution from the list
    sol = sol[0]

    # get the index of a random TA
    ta: int = np.random.choice(len(sol))

    # get the index of a random lab
    lab: int = np.random.choice(len(sol[0]))

    # toggle the lab for the TA
    sol[ta][lab] = 1 - sol[ta][lab]

    return sol


@profile
def fix_undersupport(sol) -> np.ndarray:
    """ Agent function:

    Assign random TA to random undersupported labs, checking if they are willing to accept the section.

    Args:
        sol (list): List containing a numpy array where each row is a TA and each column is a lab.
                    The value in each cell is 1 if the TA is assigned to the lab and 0 otherwise.

    Returns:
        np.array: The fixed solution.

    """
    # extract the solution from the list
    sol = sol[0]

    # number of TAs assigned to each lab is just the sum of each row
    labs_assigned = np.sum(sol, axis=0)

    # check to see if lab is undersupported
    undersupport = SECTIONS[:, 6] - labs_assigned
    
    def assign_ta(lab) -> None:
        """ Helper function to assign a random TA to a specified lab.

        Args:
            lab (int): The index of the lab to assign a TA to.

        Returns:
            np.array: The solution with the TA assigned to the lab.

        """
        # get the index of the TAs that are willing to support the lab
        willing_tas: np.ndarray = np.where(TA_PREFS[:, lab] == b'W')[0]

        # if there are no willing TAs, then do nothing
        if len(willing_tas) != 0:
            # choose a random willing TA
            ta = np.random.choice(willing_tas)

            # assign the TA to the lab
            sol[ta][lab] = 1

    # toggle a random TA for every undersupported section
    [assign_ta(lab) for lab in np.where(undersupport > 0)[0]]

    return sol


def main() -> None:
    # create population
    env: evo.Environment = evo.Environment()

    # register the fitness criteria (objects)
    env.add_fitness_criteria("overallocation", overallocation)
    env.add_fitness_criteria("conflicts", conflicts)
    env.add_fitness_criteria("undersupport", undersupport)
    env.add_fitness_criteria("unwilling", unwilling)
    env.add_fitness_criteria("unpreferred", unpreferred)

    # register all agents
    env.add_agent("fix_unwilling", fix_unwilling)
    env.add_agent("swap_random_tas", swap_random_tas)
    env.add_agent("fix_overallocated_tas", fix_overallocated_tas)
    env.add_agent("mutate", mutate)
    env.add_agent("fix_undersupport", fix_undersupport)

    # seed the population with an initial random solution
    sol: np.ndarray = np.random.randint(2, size=(43, 17))  # TODO: replace with something smarter than random?
    env.add_solution(sol)

    # run the evolve function
    env.evolve(50000000, dom=1000, sync=10000, time_limit=600)

    # print result
    print(env)
    Profiler.report()

    # print the final assignments
    for i, sol in enumerate(env.pop):
        print(f"Solution {i + 1}:")
        print(sol)
        print(env.pop[sol])
        print("\n")

    # save the final scores to a csv file
    with open("scores.csv", "w") as f:
        f.write('groupname, overallocation, conflicts, undersupport, unwilling, unpreferred,')
        f.write("\n")
        for sol in env.pop:
            f.write("group13,")
            for i in range(5):
                f.write(f"{int(sol[i][1])},")
            f.write("\n")


if __name__ == '__main__':
    main()
