#!/usr/bin/env python

"""
Simulate Wright-Fisher population dynamics with selection
"""
# packages
import argparse
import numpy as np
import scipy.stats as stats
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import itertools

# global variables
pop_size = 1000
seq_length = 100
alphabet = ['A', 'T', 'G', 'C']
mutation_rate = float(input('Mutation rate [0.0001]: ') or 0.0001) # per gen per individual per site
recombination_rate = float(input('Recombination rate [0.02]: ') or 0.02)
replicate = input('Replicate number? ') or 0
generations = 500
print('Mutation rate: {}, Recombination rate: {}, Population size: {}, Sequence length: {}'.format(mutation_rate, recombination_rate, pop_size, seq_length))

model_type = str(input('Model type [c(lassic) or t(runcnorm)]? ') or 'c')
if model_type == 'c':
    fitness_effect = float(input('Fitness effect [1.1]: ') or 1.1) # fitness effect if a functional mutation occurs - not used anymore
    fitness_defect = float(input('Fitness defect [0.95]: ') or 0.95) # fitness effect if a deleterious mutation occurs
    fitness_chance = float(input('Fitness chance [0.1]: ') or 0.1) # chance that a mutation has a fitness effect
    print('Fitness effect: {}, defect: {}, chance: {}'.format(fitness_effect, fitness_defect, fitness_chance))
elif model_type == 't':
    lower, upper, mu, sigma = 0, 2, 0.71, 0.31 #mu=mean sigma=sd truncated normal distribution
    fitness_effect = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    print('lower: {}, upper: {}, mean: {}, stdev: {}'.format(lower, upper, mu, sigma))
else:
    print("You should enter <c> or <t> only.")
    sys.exit(0)

#opening file line
dvifile = open("diversity_{}_{}_{}.txt".format(mutation_rate, recombination_rate, replicate), mode = "w") # diversity graph
dvgfile = open("divergence_{}_{}_{}.txt".format(mutation_rate, recombination_rate, replicate), mode = "w") # divergence graph
hapfile = open("haplotype_{}_{}_{}.txt".format(mutation_rate, recombination_rate, replicate), mode = "w")
fitfile = open("meanfit_{}_{}_{}.txt".format(mutation_rate, recombination_rate, replicate), mode = "w")

# population
base_haplotype = ''.join(["A" for i in range(seq_length)])
pop = {}
pop[base_haplotype] = pop_size
fitness = {}
fitness[base_haplotype] = 1.0
history = []

# mutation
def get_mutation_count():
    mean = mutation_rate * pop_size * seq_length
    return np.random.poisson(mean)

def get_recomb_count():
    mean = recombination_rate * pop_size * seq_length
    return np.random.poisson(mean)

def get_random_haplotype():
    haplotypes = list(pop.keys())
    frequencies = [x/float(pop_size) for x in pop.values()]
    total = sum(frequencies)
    frequencies = [x / total for x in frequencies]
    return np.random.choice(haplotypes, p=frequencies)

def get_mutant(haplotype):
    site = np.random.randint(seq_length)
    possible_mutations = list(alphabet)
    possible_mutations.remove(haplotype[site])
    mutation = np.random.choice(possible_mutations)
    new_haplotype = haplotype[:site] + mutation + haplotype[site+1:]
    return new_haplotype

def get_recomb(haplotype_a, haplotype_b):
    site = np.random.randint(seq_length) #random integer
    new_haplotype = haplotype_a[:site] + haplotype_b[site:]   #base taken and changed into new one
    return new_haplotype

def get_fitness(haplotype):
    old_fitness = fitness[haplotype]
    if model_type == 't':
        return old_fitness * float(fitness_effect.rvs(size=1))
    elif model_type == 'c':
        if (np.random.random() < fitness_chance): # use this line (and the else lines below) is you are using fitness_chance
            return old_fitness * fitness_effect # use this is fitness_effect is a single number
        else:
            return old_fitness * fitness_defect

def get_recomb_fitness(haplotype_a, haplotype_b):
    old_fitness = ( fitness[haplotype_a] + fitness[haplotype_b] ) / 2
    return old_fitness

def mutation_event():
    haplotype = get_random_haplotype()
    if pop[haplotype] >= 1: # modification here
        pop[haplotype] -= 1
        new_haplotype = get_mutant(haplotype)
        if new_haplotype in pop:
            pop[new_haplotype] += 1
        else:
            pop[new_haplotype] = 1
        if new_haplotype not in fitness:
            fitness[new_haplotype] = get_fitness(haplotype)

def recomb_event():
    haplotype_a = get_random_haplotype()
    haplotype_b = get_random_haplotype()
    if haplotype_a != haplotype_b:
        new_haplotype = get_recomb(haplotype_a, haplotype_b)
        if new_haplotype in pop:
            pop[new_haplotype] += 1 # if the new haplotype is already there
        else:
            pop[new_haplotype] = 1 # if it is not already there
        if new_haplotype not in fitness:
            fitness[new_haplotype] = get_recomb_fitness(haplotype_a, haplotype_b)
        pop[haplotype_a] -= 1
        pop[haplotype_b] -= 1
        if pop[haplotype_a] == 0:
            del pop[haplotype_a]
        if pop[haplotype_b] == 0:
            del pop[haplotype_b]

def mutation_step():
    mutation_count = get_mutation_count()
    for i in range(mutation_count):
        mutation_event()

def recomb_step():
    recombination_count = get_recomb_count()
    for i in range(recombination_count):
        recomb_event()

# genetic drift and selection
def get_offspring_counts():
    haplotypes = list(pop.keys())
    frequencies = [pop[haplotype]/float(pop_size) for haplotype in haplotypes]
    fitnesses = [fitness[haplotype] for haplotype in haplotypes]
    weights = [x * y for x,y in zip(frequencies, fitnesses)]
    total = sum(weights)
    weights = [x / total for x in weights]
    return list(np.random.multinomial(pop_size, weights))

def offspring_step():
    counts = get_offspring_counts()
    for (haplotype, count) in zip(list(pop.keys()), counts):
        if (count > 0):
            pop[haplotype] = count
        else:
            del pop[haplotype]

def mean_fitness(popdict, fitdict):
    mfit = sum( [ tpl[1] * fitdict[tpl[0]] for tpl in popdict.items() ] ) / pop_size
    return mfit

# simulate
def time_step():
    mutation_step()
    recomb_step()
    offspring_step()

def simulate():
    clone_pop = dict(pop)
    history.append(clone_pop)
    for i in range(generations):
        time_step()
        print(i+1, "\t", mean_fitness(pop, fitness), file = fitfile)
        clone_pop = dict(pop)
        history.append(clone_pop)

# plot diversity
def get_distance(seq_a, seq_b):
    diffs = 0
    length = len(seq_a)
    assert len(seq_a) == len(seq_b)
    for chr_a, chr_b in zip(seq_a, seq_b):
        if chr_a != chr_b:
            diffs += 1
    return diffs / float(length)

def get_diversity(population):
    haplotypes = list(population.keys())
    haplotype_count = len(haplotypes)
    diversity = 0
    for i in range(haplotype_count):
        for j in range(haplotype_count):
            haplotype_a = haplotypes[i]
            haplotype_b = haplotypes[j]
            frequency_a = population[haplotype_a] / float(pop_size)
            frequency_b = population[haplotype_b] / float(pop_size)
            frequency_pair = frequency_a * frequency_b
            diversity += frequency_pair * get_distance(haplotype_a, haplotype_b)
    return diversity

def get_diversity_trajectory():
    trajectory = [get_diversity(generation) for generation in history]
    return trajectory

def diversity_plot(xlabel="generation"):
    mpl.rcParams['font.size']=14
    trajectory = get_diversity_trajectory()
    for f in trajectory: ## loop added for printing to file
        print(f, end='\n', file = dvifile)
    plt.plot(trajectory, "#447CCD")
    plt.ylabel("diversity")
    plt.xlabel(xlabel)

# plot divergence
def get_divergence(population):
    haplotypes = population.keys()
    divergence = 0
    for haplotype in haplotypes:
        frequency = population[haplotype] / float(pop_size)
        divergence += frequency * get_distance(base_haplotype, haplotype)
    return divergence

def get_divergence_trajectory():
    trajectory = [get_divergence(generation) for generation in history]
    return trajectory

def divergence_plot(xlabel="generation"):
    mpl.rcParams['font.size']=14
    trajectory = get_divergence_trajectory()
    for f in trajectory: ## loop added for printing to file
        print(f, end='\n', file = dvgfile)
    plt.plot(trajectory, "#447CCD")
    plt.ylabel("divergence")
    plt.xlabel(xlabel)

# plot trajectories
def get_frequency(haplotype, generation):
    pop_at_generation = history[generation]
    if haplotype in pop_at_generation:
        return pop_at_generation[haplotype]/float(pop_size)
    else:
        return 0

def get_trajectory(haplotype):
    trajectory = [get_frequency(haplotype, gen) for gen in range(generations)]
    return trajectory

def get_all_haplotypes():
    haplotypes = set()
    for generation in history:
        print(len(generation), end='\n', file=hapfile) ## printing to file
        for haplotype in generation:
            haplotypes.add(haplotype)
    return haplotypes

def get_nice_data():
    if model_type == 'c':
        par = (replicate, generations, pop_size, mutation_rate, recombination_rate, fitness_chance, fitness_defect, fitness_effect)
    elif model_type == 't':
        par = (replicate, generations, pop_size, mutation_rate, recombination_rate, lower, upper, mu, sigma)
    par_as_str = '_'.join(([str(x) for x in par]))
    with(open("results"+par_as_str+".txt", "w")) as nicedata:
        haplotypes = get_all_haplotypes()
        if model_type == 'c':
            print("replicate: {}, # gens: {}, pop_size: {}, mut_rate: {}, rec_rate: {}, mut_pr: {}, del_ef: {}, ben_ef: {}".format(*par), file=nicedata)
        elif model_type == 't':
            print("replicate: {}, # gens: {}, pop_size: {}, mut_rate: {}, rec_rate: {}, lower: {}, upper: {}, mu: {}, sigma: {}".format(*par), file=nicedata)
        print("haplotype\tfitness\tpersistence\tfirstgen\t", "\t".join("gen"+str(x+1) for x in range(generations)), file=nicedata)

        for haplotype in haplotypes:
            traj = get_trajectory(haplotype)
            not_zeroes = [x!=0 for x in traj]
            living_gen = sum(not_zeroes)
            if True in not_zeroes:
                first_gen = not_zeroes.index(True) + 1
            else:
                first_gen = 'NA'
            print(haplotype, "\t", fitness[haplotype], "\t", living_gen, "\t", first_gen, "\t", "\t".join([str(x * pop_size) for x in traj]), file=nicedata)

def stacked_trajectory_plot(xlabel="generation"):
    colors_lighter = ["#A567AF", "#8F69C1", "#8474D1", "#7F85DB", "#7F97DF", "#82A8DD", "#88B5D5", "#8FC0C9", "#97C8BC", "#A1CDAD", "#ACD1A0", "#B9D395", "#C6D38C", "#D3D285", "#DECE81", "#E8C77D", "#EDBB7A", "#EEAB77", "#ED9773", "#EA816F", "#E76B6B"]
    mpl.rcParams['font.size']=18
    haplotypes = get_all_haplotypes()
    trajectories = [get_trajectory(haplotype) for haplotype in haplotypes]
    plt.stackplot(range(generations), trajectories, colors=colors_lighter)
    plt.ylim(0, 1)
    plt.ylabel("frequency")
    plt.xlabel(xlabel)

# plot snp trajectories
def get_snp_frequency(site, generation):
    minor_allele_frequency = 0.0
    pop_at_generation = history[generation]
    for haplotype in pop_at_generation.keys():
        allele = haplotype[site]
        frequency = pop_at_generation[haplotype] / float(pop_size)
        if allele != "A":
            minor_allele_frequency += frequency
    return minor_allele_frequency

def get_snp_trajectory(site):
    trajectory = [get_snp_frequency(site, gen) for gen in range(generations)]
    return trajectory

def get_all_snps():
    snps = set()
    for generation in history:
        for haplotype in generation:
            for site in range(seq_length):
                if haplotype[site] != "A":
                    snps.add(site)
    return snps

def snp_trajectory_plot(xlabel="generation"):
    colors = ["#781C86", "#571EA2", "#462EB9", "#3F47C9", "#3F63CF", "#447CCD", "#4C90C0", "#56A0AE", "#63AC9A", "#72B485", "#83BA70", "#96BD60", "#AABD52", "#BDBB48", "#CEB541", "#DCAB3C", "#E49938", "#E68133", "#E4632E", "#DF4327", "#DB2122"]
    mpl.rcParams['font.size']=18
    snps = get_all_snps()
    trajectories = [get_snp_trajectory(snp) for snp in snps]
    data = []
    for trajectory, color in zip(trajectories, itertools.cycle(colors)):
        data.append(range(generations))
        data.append(trajectory)
        data.append(color)
    plt.plot(*data)
    plt.ylim(0, 1)
    plt.ylabel("frequency")
    plt.xlabel(xlabel)

if __name__=="__main__":

    simulate() # where code is ran
    get_nice_data()

    plt.figure(num=None, figsize=(14, 10), dpi=80, facecolor='w', edgecolor='k')
    plt.subplot2grid((3,2), (0,0), colspan=2)
    stacked_trajectory_plot(xlabel="")
    plt.subplot2grid((3,2), (1,0), colspan=2)
    snp_trajectory_plot(xlabel="")
    plt.subplot2grid((3,2), (2,0))
    diversity_plot()
    plt.subplot2grid((3,2), (2,1))
    divergence_plot()
    plt.savefig("evolplotsmutationandselection_{}_{}_{}.png".format(replicate, mutation_rate, recombination_rate))

dvifile.close()
dvgfile.close()
hapfile.close()
fitfile.close()
