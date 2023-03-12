# coding: utf-8

# # Exercise code for Monte Calro simulation of the 2d Ising model
# This module is for simulation of 2d Ising model on the square lattice, whose Hamiltonian is given by
# $$ \mathcal{H} = -J \sum_{\langle i,j\rangle} S_i S_j - h \sum_i S_i ,$$
# where $S_i = \pm 1$.
# 
# You can select three simulation algorithms:
# * metropolis
# * heatbath
# * cluster (Swendsen-Wang)
# 
# The outputs are:
# * Energy: $\langle E\rangle = \langle \mathcal{H}\rangle/N$.
# * Squared magnetization: $\langle M^2\rangle = \langle (\sum_i S_i)^2\rangle/N^2$.
# * Specific heat: $N(\langle E^2\rangle - \langle E\rangle^2)/T$
# * Magnetic susceptibility: $N(\langle M^2\rangle\rangle)/T$
# * Connected susceptibility: $N(\langle M^2\rangle - \langle |M|^2\rangle)/T$
# * Binder ratio: $(\langle M^4\rangle/\langle M^2\rangle)/T$
# 
# * This code works on python3 with numpy and numba modules 
#
# For usage of this code, please run it with -h option: python Ising_lib.py -h

# 2018 May Tsuyoshi Okubo
# 2019 May Tsuyoshi Okubo, updated
# 2020 July Tsuyoshi Okubo, updated

import numpy as np
from numba import jit, f8, i8, b1, void #for speed up, we use jit compile
import argparse
## Metropolis
@jit(nopython=True)
def metropolis(S,exps,L):
    N = L**2
    ran = np.random.rand(N).reshape(L,L)
    for ix in range(L):
        for iy in range(L):
            hm = (S[ix,iy]*(S[(ix + 1)%L,iy] + S[ix,(iy + 1)%L] + S[(ix - 1 + L)%L,iy] + S[ix,(iy - 1 + L )%L]) + 4)//2
            if ran[ix,iy] < exps[hm, (S[ix,iy]+1)//2]:
                S[ix,iy] *= -1
    return S

## Heat bath
@jit(nopython=True)
def heatbath(S,exps,L):
    N = L**2
    ran = np.random.rand(N).reshape(L,L)
    for ix in range(L):
        for iy in range(L):
            hm = ((S[(ix + 1)%L,iy] + S[ix,(iy + 1)%L] + S[(ix - 1 + L)%L,iy] + S[ix,(iy - 1 + L )%L]) + 4)//2
            if ran[ix,iy] < exps[hm, (S[ix,iy]+1)//2]:
                S[ix,iy] = 1
            else:
                S[ix,iy] = -1
    return S


## For Swendsen-Wang algorithm
@jit(nopython=True)
def get_cluster_num(num,cluster_num):
    if num == cluster_num[num]:
        return num
    else:
        return get_cluster_num(cluster_num[num],cluster_num)


@jit(nopython=True)
def update_cluster_num(ni,nj,cluster_num):
    ci = get_cluster_num(ni,cluster_num)
    cj = get_cluster_num(nj,cluster_num)
    if ci < cj:
        cluster_num[cj] = ci
    else:
        cluster_num[ci] = cj

@jit(nopython=True)
def make_bond(S,prob,L):
    N = L**2
    bond = np.zeros((L,L,2),dtype=np.int64)
    ran = np.random.rand(2*N).reshape(L,L,2)
    for ix in range(L):
        for iy in range(L):
            if S[ix,iy] *S[(ix+1)%L, iy] > 0: ## spins are parallel
                if ran[ix,iy,0] < prob:
                    bond[ix,iy,0] = 1
            if S[ix,iy] *S[ix, (iy+1)%L] > 0: ## spins are parallel
                if ran[ix,iy,1] < prob:
                    bond[ix,iy,1] = 1
    return bond

@jit(nopython=True)
def make_cluster(bond,L):
    N = L**2
    cluster_num = np.arange(N)
    for ix in range(L):
        for iy in range(L):
            if bond[ix,iy,0] >0: ## connected
                ni = ix + iy *L
                nj = (ix + 1)%L + iy * L
                update_cluster_num(ni,nj,cluster_num)
            if bond[ix,iy,1] >0: ## connected
                ni = ix + iy *L
                nj =  ix+ (iy + 1)%L * L
                update_cluster_num(ni,nj,cluster_num)
        ## count total cluster number
        
    cluster_num_count = np.zeros(N,dtype=np.int64)
    for i in range(N):
        nc = get_cluster_num(i,cluster_num)
        cluster_num[i] = nc
        cluster_num_count[nc] += 1

    total_cluster_num = 0

    true_cluster_num = np.zeros(N,dtype=np.int64)
    true_cluster_num_count = np.zeros(N,dtype=np.int64)
    for nc in range(N):
        if cluster_num_count[nc] > 0:
            true_cluster_num[nc] = total_cluster_num
            true_cluster_num_count[total_cluster_num] = cluster_num_count[nc]
            total_cluster_num += 1

    for i in range(N):
        cluster_num[i] = true_cluster_num[cluster_num[i]]
    return cluster_num.reshape(L,L), true_cluster_num_count[:total_cluster_num]

@jit(nopython=True)
def flip_spin(S, cluster_num,cluster_num_count,flip,L):    
    total_cluster_num = cluster_num_count.shape[0]
    ran = np.random.rand(total_cluster_num)
    spin_direction = np.zeros(total_cluster_num,dtype=np.int64)


    for i in range(total_cluster_num):
        if ran[i] < 1.0 / (1.0 + np.exp(-2.0 * flip * cluster_num_count[i])):
            spin_direction[i] = 1
        else:
            spin_direction[i] = -1

    for ix in range(L):
        for iy in range(L):
            S[ix,iy] = spin_direction[cluster_num[ix,iy]]

def Swendsen_Wang(S,prob,flip,L):
    N = L**2
    
    ## make bond configulations    
    bond = make_bond(S,prob,L)

    ## make clusters
    cluster_num, cluster_num_count = make_cluster(bond,L)
    ## update spin

    flip_spin(S,cluster_num,cluster_num_count,flip,L)
    ## for imporoved estimator
    Nc2 = np.sum(cluster_num_count.astype(float)**2)

    return S,Nc2/N**2


## for Main MCMC simulation
@jit(nopython=True)
def Calc_local_energy(S,L):
    local_ene = 0.0
    for ix in range(L):
        for iy in range(L):
            local_ene += S[ix,iy] * (S[(ix + 1)%L ,iy] + S[ix,(iy + 1)%L])
    return local_ene
@jit(nopython=True)
def Initialize(seed,L):
    N = L**2
    if seed is not None:
        np.random.seed(seed)
    mp = np.array((-1,1),dtype=np.int64)
    return np.random.choice(mp,N).reshape(L,L)


## Main MCMC simulation
def MC(L,T,h,thermalization,observation,seed=11,algorithm="metropolis",output_snapshots=False):
    N = L**2
    S = Initialize(seed,L)
    exps=np.zeros((5,2)) ## for heatbath and metropolis
    
    if algorithm=="heatbath":
        for i in range(5):
            hm = -4.0 + 2.0*i
            for j in range(2):
                sh = h * (2 * j - 1)
                exps[i,j]= 1.0/(1.0 + np.exp(-2.0*(hm + sh)/T))
    elif algorithm == "cluster":
        prob = 1.0 - np.exp(-2.0/T)
        flip = h/T
    else: ## metropolis
        for i in range(5):
            hm = -4.0 + 2.0*i
            for j in range(2):
                sh = h * (2 * j - 1)
                exps[i,j]= np.exp(-2.0*(hm+sh)/T)


        
    for i in range(thermalization):
        if algorithm=="heatbath":
            S = heatbath(S,exps,L)
        elif algorithm == "cluster":
            S, Nc2 = Swendsen_Wang(S,prob,flip,L)
        else:
            S = metropolis(S,exps,L)
    mag = []
    mag2 = []
    mag_abs = []
    ene = []
    ene2 = []
    mag2_imp = []
    mag4 = []
    for i in range(observation):
        if algorithm=="heatbath":
            S = heatbath(S,exps,L)
        elif algorithm == "cluster":
            S, Nc2 = Swendsen_Wang(S,prob,flip,L)
            mag2_imp.append(Nc2)
        else:
            S = metropolis(S,exps,L)

        local_mag = np.sum(S) / N
        mag.append(local_mag)
        mag2.append(local_mag**2)
        mag4.append(local_mag**4)
        mag_abs.append(np.abs(local_mag))
        
        local_ene = -Calc_local_energy(S,L) / N - h * local_mag
        ene.append(local_ene)
        ene2.append(local_ene**2)

        
    mag = np.array(mag)
    mag2 = np.array(mag2)
    ene = np.array(ene)
    ene2 = np.array(ene2)
    mag2_imp = np.array(mag2_imp)
    mag4 = np.array(mag4)
    mag_abs = np.array(mag_abs)
    if output_snapshots:
        return mag,mag2,mag2_imp,mag4,mag_abs,ene,ene2,S
    else:
        return mag,mag2,mag2_imp,mag4,mag_abs,ene,ene2

## For analysis
def Jackknife(data,bin_size=100, func=None, data2=None):
    def make_bin(data,bin_size_in):
        bin_size = bin_size_in
        data_size = len(data) 
        bin_num = data_size // bin_size
        if bin_num < 10:
            bin_size = data_size // 10
            bin_num = 10
        bin_data_temp = np.zeros(bin_num)
        for i in range(bin_num):
            bin_data_temp[i] = np.sum(data[i * bin_size: (i+1)*bin_size]) / bin_size
        bin_data = (np.sum(bin_data_temp) - bin_data_temp[:]) / (bin_num-1)
        return bin_data,bin_size

    if func is not None:
        if data2 is not None:
            ## we assume that func is a function of data and data2
            bin_data, bin_size_out = make_bin(data,bin_size)
            bin_data2, bin_size_out = make_bin(data2,bin_size)

            average = np.sum(func(bin_data,bin_data2))/(len(bin_data))
            error = np.sqrt( (np.sum(func(bin_data,bin_data2)**2)/len(bin_data) - average **2) * (len(bin_data)-1))

        else:
            ## we assume that func is a function of data
            bin_data, bin_size_out = make_bin(data,bin_size)
            average = np.sum(func(bin_data))/(len(bin_data))
            error = np.sqrt( (np.sum(func(bin_data)**2)/len(bin_data) - average **2) * (len(bin_data)-1))

    else:
        ## binning
        bin_data, bin_size_out = make_bin(data,bin_size)
        average = np.sum(bin_data)/(len(bin_data))
        error = np.sqrt( (np.sum(bin_data**2)/len(bin_data) - average **2) * (len(bin_data)-1))

    return average, error

def main():

    def parse_args():
        Tc = 2.0/np.log(1.0+np.sqrt(2.0)) 
        parser = argparse.ArgumentParser(description='Monte Carlo simulation of the square lattice Ising model')
        parser.add_argument('-L',metavar='L',dest='L', type=int, default=16,
                            help='the size of square lattice. (default: L = 16)')
        parser.add_argument('-t', '--thermalization',metavar='thermalization',dest='thermalization', type=int,default=10000,
                            help='MC steps for thermalization. (default: 10000)')
        parser.add_argument('-o', '--observation',metavar='observation',dest='observation', type= int,default=10000,
                            help='MC steps for observation. (default: 10000)')
        parser.add_argument('-T', '--Temperature',metavar='T',dest='T', type=float,default=Tc,
                            help='Temperature. (default: T= Tc)')
        parser.add_argument('-hz', metavar='hz',dest='hz', type=float,default=0.0,
                            help='External magnetic field. (default: h= 0)')
        parser.add_argument( '-a','--algorithm', metavar='algorithm',dest='algorithm',default="metropolis",
                             help='Algorithms for MC simulation. You can use "metropolis", "heatbath" or "cluster"(Swendsen-Wang) (default: metropolis)')
        parser.add_argument('-s', '--seed',metavar='seed',dest='seed', type=int,default=None,
                            help='seed for random number generator. (default: seed= None)')

        return parser.parse_args()


    args = parse_args()
    L = args.L
    N = L**2
    T = args.T
    h = args.hz

    algorithm = args.algorithm
    random_seed = args.seed
    thermalization = args.thermalization
    observation = args.observation

    print("## Simulation conditions:")
    if algorithm == "heatbath":
        print("## Algorithm = Heat bath")
    elif algorithm == "cluster":
        print("## Algorithm = Swendsen-Wang")
    else:
        print("## Algorithm = Metropolis")
    print("## L = "+repr(L))
    print("## T = "+repr(T))
    print("## h = "+repr(h))
    print("## random seed = "+repr(random_seed))
    print("## thermalization steps = "+repr(thermalization))
    print("## observation steps = "+repr(observation))

        
    ## run MC simulation
    mag, mag2, mag2_imp, mag4, mag_abs,ene,ene2 = MC(L,T,h,thermalization,observation,random_seed,algorithm)
    
    ## output averages and erros

    def variance(e,e2):
        return e2 -e**2
    def binder(m2,m4):
        return m4 / m2 **2    
    
    E, E_err = Jackknife(ene,bin_size=max(100,observation//100))
    E2,E2_err = Jackknife(ene2,bin_size=max(100,observation//100))
    M,M_err = Jackknife(mag,bin_size=max(100,observation//100))
    M2,M2_err = Jackknife(mag2,bin_size=max(100,observation//100))
    M4,M4_err = Jackknife(mag4,bin_size=max(100,observation//100))
    C, C_err = Jackknife(ene,bin_size=max(100,observation//100),func=variance, data2=ene2)
    C *= N/T**2
    C_err *= N/T**2
    b, b_err = Jackknife(mag2,bin_size=max(100,observation//100),func=binder, data2=mag4)
    chi, chi_err = Jackknife(mag_abs,bin_size=max(100,observation//100),func=variance, data2=mag2)
    

    print ("### Outputs with errors estimated by Jackknife method ###")
    print ("T = " + repr(T))
    print ("Energy = " + repr(E) + " +- " +repr(E_err))
    print ("Energy^2 = " + repr(E2) + " +- " +repr(E2_err))
    print ("Magnetization = " + repr(M) + " +- " +repr(M_err))
    print ("Magnetization^2 = " + repr(M2) + " +- " +repr(M2_err))
    print ("Magnetization^4 = " + repr(M4) + " +- " +repr(M4_err))
    print ("Specific heat = " + repr(C) + " +- " +repr(C_err))
    print ("Susceptibility = " + repr(M2/T * N) + " +- " +repr(M2_err/T * N))
    print ("Connected Susceptibility = " + repr(chi/T * N) + " +- " +repr(chi_err/T * N))
    print ("Binder ratio = " + repr(b) + " +- " +repr(b_err))

    if algorithm == "cluster":
        M2_imp, M2_imp_err = Jackknife(mag2_imp,bin_size=max(100,observation/100))
        print ("Magnetization^2: improved estimator = " + repr(M2_imp) + " +- " +repr(M2_imp_err))
        print ("Susceptibility: improved estimator  = " + repr(M2_imp/T * N) + " +- " +repr(M2_imp_err/T * N))
    


if __name__ == '__main__':
    main()

