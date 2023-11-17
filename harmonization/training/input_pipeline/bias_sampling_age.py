import numpy as np

def harm_probas(agesA, agesB, bins):
    """
    Computes a sampling probability given ages from two datasets and consecutive age ranges.
    ----
    PARAMETERS:
        agesA: list of ages in the first dataset
        agesB: list of ages in the second dataset
        bins: list of ages corresponding to the consecutive age ranges to harmonize
    RETURN:
        list of size len(bins)-1 corresponding to the sampling probabilities
    """
    classesA = np.digitize(ageA,bins[1:-1])
    classesB = np.digitize(ageB,bins[1:-1])
    countA = np.bincount(classesA, minlength=len(bins)-1)
    countB = np.bincount(classesB, minlength=len(bins)-1)
    
    probsA = countA/countA.sum()
    probsB = countB/countB.sum()
    fixed = np.zeros(len(countA), dtype='bool')
    
    def update_probas(probs,idx,p):
        delta = probs[idx]-p
        probs[idx] = p
        s = probs[~fixed].sum()
        for j in np.argwhere(~fixed).flatten():
            probs[j] += delta*probs[j]/s
    
    indices = np.argsort(np.minimum(countA,countB))
    for i in indices[:-1]:
        fixed[i] = True
        if countA[i] > countB[i]:
            update_probas(probsA, i, probsB[i])
        else:
            update_probas(probsB, i, probsA[i])
    return probsA