import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.docs import parse_doc_string
from pymoo.core.survival import Survival
from pymoo.core.sampling import Sampling
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.normalization import get_extreme_points_c
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

import matplotlib.pyplot as plt

# =========================================================================================================
# Implementation
# =========================================================================================================


class SNSGA2(NSGA2):

    def __init__(self, base_sampler_class, sparsity = (0.75, 1), **kwargs):
        """
        Parameters
        ----------
        sparsity : (sparsity_lower, sparsity_upper)

        """

        super().__init__(sampling=VSSPS(base_sampler_class), **kwargs)
        

class VSSPS(Sampling):

    def __init__(self, base_sampler_class, sparsity_range = (0.75, 1), nz_indices=[],  **kwargs): 

        self.s_lower            = sparsity_range[0]
        self.s_upper            = sparsity_range[1]
        self.base_sampler_class = base_sampler_class 
        self.nz_indices        = nz_indices  

        super().__init__(**kwargs)


    def _do(self, problem, n_samples, **kwargs):

        ## Problem initialization

        base_sampler = self.base_sampler_class()

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

        # Record the indices that should be non-zero in the mask 
        mask[:, self.nz_indices] = True;

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

    # Algorithm smoke test 
    sampler = VSSPS(FloatRandomSampling, nz_indices=[0, 5])

    prob = get_problem("zdt1", n_var= 100)

    #algorithm = NSGA2(sampling=sampler, pop_size=100)
    #
    #res = minimize(prob,
    #               algorithm,
    #               ('n_gen', 200),
    #               seed=1,
    #               verbose=False)


    #plot = Scatter()
    #plot.add(prob.pareto_front(), plot_type="line", color="black", alpha=0.7)
    #plot.add(res.F, facecolor="none", edgecolor="red")
    #plot.show()


    # Visual representation of the sampling
    n_samples = 100

    X = sampler._do(prob, n_samples)

    X[X != 0] = 1

    fig, ax = plt.subplots()
    im = ax.imshow(X)

    plt.show()


