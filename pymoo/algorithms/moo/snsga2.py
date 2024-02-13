import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.docs import parse_doc_string
from pymoo.core.survival import Survival
from pymoo.core.sampling import Sampling
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.normalization import get_extreme_points_c
from pymoo.problems import get_problem


import matplotlib.pyplot as plt

# =========================================================================================================
# Implementation
# =========================================================================================================


class SNSGA2(NSGA2):

    def __init__(self, sparsity = (0.75, 1), **kwargs):
        """
        Parameters
        ----------
        sparsity : (sparsity_lower, sparsity_upper)

        """
        

        super().__init__(sampling=VSSPS(), **kwargs)

        


class RankAndModifiedCrowdingSurvival(Survival):

    def __init__(self, ref_points,
                 epsilon,
                 weights,
                 normalization,
                 extreme_points_as_reference_points
                 ) -> None:

        super().__init__(True)
        self.n_obj = ref_points.shape[1]
        self.ref_points = ref_points
        self.epsilon = epsilon
        self.extreme_points_as_reference_points = extreme_points_as_reference_points

        self.weights = weights
        if self.weights is None:
            self.weights = np.full(self.n_obj, 1 / self.n_obj)

        self.normalization = normalization
        self.ideal_point = np.full(self.n_obj, np.inf)
        self.nadir_point = np.full(self.n_obj, -np.inf)

    def _do(self, problem, pop, n_survive=None, **kwargs):

        # get the objective space values and objects
        F = pop.get("F")

        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = NonDominatedSorting().do(F)

        if self.normalization == "ever":
            # find or usually update the new ideal point - from feasible solutions
            self.ideal_point = np.min(np.vstack((self.ideal_point, F)), axis=0)
            self.nadir_point = np.max(np.vstack((self.nadir_point, F)), axis=0)

        elif self.normalization == "front":
            front = fronts[0]
            if len(front) > 1:
                self.ideal_point = np.min(F[front], axis=0)
                self.nadir_point = np.max(F[front], axis=0)

        elif self.normalization == "no":
            self.ideal_point = np.zeros(self.n_obj)
            self.nadir_point = np.ones(self.n_obj)

        if self.extreme_points_as_reference_points:
            self.ref_points = np.row_stack([self.ref_points, get_extreme_points_c(F, self.ideal_point)])

        # calculate the distance matrix from ever solution to all reference point
        dist_to_ref_points = calc_norm_pref_distance(F, self.ref_points, self.weights, self.ideal_point,
                                                     self.nadir_point)

        for k, front in enumerate(fronts):

            # save rank attributes to the individuals - rank = front here
            pop[front].set("rank", np.full(len(front), k))

            # number of individuals remaining
            n_remaining = n_survive - len(survivors)

            # the ranking of each point regarding each reference point (two times argsort is necessary)
            rank_by_distance = np.argsort(np.argsort(dist_to_ref_points[front], axis=0), axis=0)

            # the reference point where the best ranking is coming from
            ref_point_of_best_rank = np.argmin(rank_by_distance, axis=1)

            # the actual ranking which is used as crowding
            ranking = rank_by_distance[np.arange(len(front)), ref_point_of_best_rank]

            if len(front) <= n_remaining:

                # we can simply copy the crowding to ranking. not epsilon selection here
                crowding = ranking
                I = np.arange(len(front))

            else:

                # Distance from solution to every other solution and set distance to itself to infinity
                dist_to_others = calc_norm_pref_distance(F[front], F[front], self.weights, self.ideal_point,
                                                         self.nadir_point)
                np.fill_diagonal(dist_to_others, np.inf)

                # the crowding that will be used for selection
                crowding = np.full(len(front), np.nan)

                # solutions which are not already selected - for
                not_selected = np.argsort(ranking)

                # until we have saved a crowding for each solution
                while len(not_selected) > 0:

                    # select the closest solution
                    idx = not_selected[0]

                    # set crowding for that individual
                    crowding[idx] = ranking[idx]

                    # need to remove myself from not-selected array
                    to_remove = [idx]

                    # Group of close solutions
                    dist = dist_to_others[idx][not_selected]
                    group = not_selected[np.where(dist < self.epsilon)[0]]

                    # if there exists solution with a distance less than epsilon
                    if len(group):
                        # discourage them by giving them a high crowding
                        crowding[group] = ranking[group] + np.round(len(front) / 2)

                        # remove group from not_selected array
                        to_remove.extend(group)

                    not_selected = np.array([i for i in not_selected if i not in to_remove])

                # now sort by the crowding (actually modified rank) ascending and let the best survive
                I = np.argsort(crowding)[:n_remaining]

            # set the crowding to all individuals
            pop[front].set("crowding", crowding)

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])

        # inverse of crowding because nsga2 does maximize it (then tournament selection can stay the same)
        pop.set("crowding", -pop.get("crowding"))

        return pop[survivors]


class VSSPS(Sampling):

    def __init__(self, sparsity_range = (0.75, 1)): 

        self.s_lower = sparsity_range[0]
        self.s_upper = sparsity_range[1]


    def _do(self, problem, n_samples, base_sampler_class, **kwargs):

        ## Problem initialization

        base_sampler = base_sampler_class()

        # Initial population sampling 
        X = base_sampler._do(problem, n_samples, **kwargs)

        N = X.shape[0]
        D = X.shape[1]

        # non-zero mask
        mask = np.array(np.zeros(np.shape(X)), dtype = bool)

        ## Determine the positioning of each stripe per individual
        densityVector = 1 - np.linspace(self.s_lower, self.s_upper, N)

        widthVector = np.round(np.multiply(densityVector, D)).astype(int)


        # Put widths back into bound if rounding error occurred
        lb = np.floor((1- self.s_lower)*D)
        widthVector[widthVector > lb] = lb

        cumulativeWidths = np.cumsum(widthVector)
        
        # if all sparsities are 100%, then skip processing, since everything
        # will be zeros 
        if np.sum(widthVector == 0) == N:
            processedIndvs = N 
        else:
            processedIndvs = 0

        cycle_count = 0
        cycles = np.ones((N, D), dtype=int) * -1

        while processedIndvs < N:

            # Figure out how many stripes will fit in this cycle 
            spotsThatFitMask = np.logical_and(cumulativeWidths <= D, cumulativeWidths != 0)

            numThatFit = np.sum(spotsThatFitMask)
            
            largestFit = np.max(cumulativeWidths[spotsThatFitMask])

            cumulativeWidths = cumulativeWidths - largestFit 

            cumulativeWidths[cumulativeWidths < 0] = 0 

            processedIndvs = processedIndvs + numThatFit

            spotsThatFit = np.where(spotsThatFitMask)[0]

            cycles[cycle_count, 0:numThatFit] = spotsThatFit 

            cycle_count = cycle_count + 1

        ## Create density mask

        # Mask out non-zero values cycle-by-cycle
        currentIndv = 0

        for c in range(cycle_count):

            cycle = cycles[c, cycles[c, :] != -1]
            
            widths = widthVector[cycle]

            gapToFill = D - np.sum(widths)

            gapSize = np.ceil((D - np.sum(widths))/np.size(widths)).astype(int)

            position = 0

            for width in widths: 

                # Determine if a gap is needed
                gapWidth = 0
                if gapToFill > 0:
                    gapWidth = gapSize
                    gapToFill = gapToFill - gapWidth
                

                # Determine the position of the stripe
                startPoint = position

                # If we're in the final cycle of the striping, then switch from 
                # filling in the extra gap to spacing the stripes out
                if c == (cycle_count - 1):
                    endPoint = position+width
                else:
                    endPoint = position+width+gapWidth

                # Prevent overflow from a gap calculation
                if endPoint > D:
                    endPoint = D

                # Mask out stripe
                mask[currentIndv, startPoint:endPoint] = True

                # Go to the next individual
                position = position + width + gapWidth 

                currentIndv = currentIndv + 1

        # Zero out the necessary spots
        X[np.logical_not(mask)] = 0

        return X

        
    
    #  
    #  Nomenclature example
    #  N = 8
    #  D = 14
    #
    #          Cycle length of 14
    #          |
    #  |-------|-----------------|
    #  1 1 1 1                        -
    #          1 1 1 1                |-- One full cycle
    #                  1 1 1          |
    #                        1 1 1    -
    #     |-------|------|-----|---------------- Cycle count of 4
    #
    #   |---|--|-|----------------------Cycle count of 4
    #  1 1
    #      1 1
    #          1
    #            1
    #  |----|----|
    #       |
    #       Cyle length of 6

if __name__ == "__main__": 

    sampler = VSSPS()
    dummy_prob = get_problem("zdt1", n_var= 100)
   
    n_samples = 100

    X = sampler._do(dummy_prob, n_samples, FloatRandomSampling)

    masked_X = X
    masked_X[masked_X != 0] = 1

    fig, ax = plt.subplots()
    im = ax.imshow(X)

    plt.show()

    

