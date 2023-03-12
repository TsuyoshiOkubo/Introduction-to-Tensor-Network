# Tensor Renormalization Groupe
# based on M. Levin and C. P. Nave, PRL 99 120601 (2007)
# and X.-C. Gu, M. Levin, and X.-G. Wen, PRB 78, 205116(2008).
# 2017 April Tsuyoshi Okubo
# Updated on July, 2019

import numpy as np
import scipy as scipy
import scipy.linalg as linalg

def Trace_tensor(A):
    # contraction of a single Tensor with the perioic boundary condition
    Trace_A = np.trace(A,axis1=0,axis2=2)
    Trace_A = np.trace(Trace_A)
    return Trace_A


def initialize_A(T,Energy_flag=False):
    # Make initial tensor of square lattice Ising model at a temperature T
    A =np.empty((2,2,2,2))
    if Energy_flag:
        Energy_A = np.empty((2,2,2,2))
    
    for i in range(0,2):
        si = (i - 0.5) * 2
        for j in range(0,2):
            sj = (j - 0.5) * 2
            for k in range(0,2):
                sk = (k - 0.5) * 2
                for l in range(0,2):
                    sl = (l - 0.5) * 2

                    A[i,j,k,l] = np.exp((si*sj + sj*sk + sk*sl + sl*si)/T)
                    if Energy_flag:
                        ## impurity tensor for Energy
                        Energy_A[i,j,k,l] = -0.5 * (si*sj + sj*sk + sk*sl + sl*si) * A[i,j,k,l]
                    
    ## normalization of tensor
    factor = Trace_tensor(A)
    A /= factor
    if Energy_flag:
        Energy_A /= factor
        return A,Energy_A,factor
    else:
        return A,factor

def SVD_type1(A,D):
    ## sigular value decomposition and truncation of A
    A_mat1=np.reshape(A,(A.shape[0]*A.shape[1],A.shape[2]*A.shape[3]))
        
    U,s,VT = linalg.svd(A_mat1)

    ## truncate singular values at D_cut
    D_cut = np.min([s.size,D])
    s_t = np.sqrt(s[0:D_cut])
    
    S3 = np.dot(U[:,0:D_cut],np.diag(s_t))
    S1 = np.dot(np.diag(s_t),VT[0:D_cut,:])
    S3 = np.reshape(S3,(A.shape[0],A.shape[1],D_cut))
    S1 = np.reshape(S1,(D_cut,A.shape[2],A.shape[3]))

    return S1,S3
def SVD_type2(A,D):
    ## sigular value decomposition and truncation of A
    A_mat2 = np.transpose(A,(0,3,1,2))
    A_mat2 = np.reshape(A_mat2,(A.shape[0]*A.shape[3],A.shape[1]*A.shape[2]))
        
    U,s,VT = linalg.svd(A_mat2)
    
    ## truncate singular values at D_cut
    D_cut = np.min([s.size,D])
    s_t = np.sqrt(s[0:D_cut])

    S2 = np.dot(U[:,0:D_cut],np.diag(s_t))
    S4 = np.dot(np.diag(s_t),VT[0:D_cut,:])
    S2 = np.reshape(S2,(A.shape[0],A.shape[3],D_cut))
    S4 = np.reshape(S4,(D_cut,A.shape[1],A.shape[2]))
    return S2,S4
    
def Update_Atensor(A,D):    

    S1,S3 = SVD_type1(A,D)
    S2,S4 = SVD_type2(A,D)

    A = Combine_four_S(S1,S2,S3,S4)
    factor = Trace_tensor(A)

    A /= factor
    return A, factor

def Update_Atensor_with_E(A,E1,E2,E3,E4,D):
    
    S1,S3 = SVD_type1(A,D)
    S2,S4 = SVD_type2(A,D)

    A = Combine_four_S(S1,S2,S3,S4)
    factor = Trace_tensor(A)

    A /= factor

    ## calclate impurity part
    E1_S1,E1_S3 = SVD_type1(E1,D)
    E2_S2,E2_S4 = SVD_type2(E2,D)
    E3_S1,E3_S3 = SVD_type1(E3,D)
    E4_S2,E4_S4 = SVD_type2(E4,D)
    
    E1 = Combine_four_S(E1_S1,S2,S3,E2_S4)
    E2 = Combine_four_S(E3_S1,E2_S2,S3,S4)
    E3 = Combine_four_S(S1,E4_S2,E3_S3,S4)
    E4 = Combine_four_S(S1,S2,E1_S3,E4_S4)

    E1 /= factor
    E2 /= factor
    E3 /= factor
    E4 /= factor
    return A,E1,E2,E3,E4, factor

def Update_Etensor_4site(E1,E2,E3,E4,D,factor):
    ## calclate impurity part
    E1_S1,E1_S3 = SVD_type1(E1,D)
    E2_S2,E2_S4 = SVD_type2(E2,D)
    E3_S1,E3_S3 = SVD_type1(E3,D)
    E4_S2,E4_S4 = SVD_type2(E4,D)
    
    E1 = Combine_four_S(E1_S1,E4_S2,E3_S3,E2_S4)
    E2 = Combine_four_S(E3_S1,E2_S2,E1_S3,E4_S4)

    E1 /= factor
    E2 /= factor
    return E1,E2

def Update_Etensor_2site(E1,E2,D,factor):
    ## calclate impurity part
    E1_S1,E1_S3 = SVD_type1(E1,D)
    E2_S2,E2_S4 = SVD_type2(E2,D)
    
    E1 = Combine_four_S(E1_S1,E2_S2,E1_S3,E2_S4)

    E1 /= factor
    return E1

def Combine_four_S(S1,S2,S3,S4):
    S12 = np.tensordot(S1,S2,axes=(1,0))
    S43 = np.tensordot(S4,S3,axes=(2,0))

    A = np.tensordot(S12,S43,axes=([1,2],[1,2]))
    A = np.transpose(A,(0,1,3,2))
    return A
    
##### Main part of TRG ####
def TRG_Square_Ising(T,D,TRG_steps,Energy_flag):

    ##  Initialization ##
    if Energy_flag:
        A, Energy_A,factor = initialize_A(T,Energy_flag)
        E1=Energy_A
        E2=A
        E3=A
        E4=A
    else:
        A,factor = initialize_A(T)

    TRG_factors = [factor]
    
    ## TRG iteration ##
    for i_TRG in range(0,TRG_steps):

        if Energy_flag:

            if i_TRG == TRG_steps - 2:
                ## special treatment for 4 tensors sytem
                A,factor = Update_Atensor(A,D)
                E1,E2 = Update_Etensor_4site(E1,E2,E3,E4,D,factor)
            elif i_TRG == TRG_steps -1:
                ## special treatment for 2 tensors sytem
                A,factor = Update_Atensor(A,D)
                E1 = Update_Etensor_2site(E1,E2,D,factor)
            else:
                A,E1,E2,E3,E4,factor = Update_Atensor_with_E(A,E1,E2,E3,E4,D)

            
        else:
            A, factor = Update_Atensor(A,D)
        TRG_factors.append(factor)

    ## End of TRG iteration

    #Calclation of free energy
    free_energy_density = 0.0
    for i_TRG in range(TRG_steps+1):
        free_energy_density +=  np.log(TRG_factors[i_TRG]) * 0.5**i_TRG
    ## note: we normalize A so that Trace(A) = 1.0
    free_energy_density = -T * 0.5 * (free_energy_density + 0.5**TRG_steps*np.log(Trace_tensor(A))) 
    print("T, free_energy_density = "+repr(T)+" "+repr(free_energy_density))
    if Energy_flag:
        energy_density = Trace_tensor(E1)/Trace_tensor(A)
        print("T, energy_density = "+repr(T)+" "+repr(energy_density))
        return free_energy_density,energy_density
    else:
        return free_energy_density

