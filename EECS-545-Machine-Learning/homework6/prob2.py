import numpy as np
np.random.seed(17)

# for K = 2 , you should use the following parameters to initialize
# transition matrix
Initial_A_2 = np.array([
    [0.4,0.6],
    [0.6,0.4]
])

# emission matrix
Initial_phi_2 = np.array([
    [0.5, 0.1, 0.2, 0.2],
    [0.1, 0.5, 0.1, 0.3]
])

#对于一列输入，为了记录从头到尾的alpha，需要一个N*K的矩阵
X1 = [1,1,3,0,1,0,1,2,1,0]
X2 = [1,3,0,1,2,1,0,0,3]

def forward(X,K,trans,emiss,pi):
    alpha = [[emiss[i,X[0]]*pi[i] for i in range(K)]]
    for i in range(1,len(X)):
        alpha.append([emiss[k,X[i]]*sum([alpha[i-1][l]*trans[l,k] 
                             for l in range(K)]) for k in range(K)])
    return np.array(alpha)

def backward(X,K,trans,emiss,pi):
    beta = [[1]*K]
    for i in range(len(X)-1,0,-1):
        beta.append([sum([beta[-1][l]*emiss[l,X[i]]*trans[k,l] 
                          for l in range(K)]) for k in range(K)])
    beta.reverse()
    return np.array(beta)

def cal_kesi(X,alpha,beta,emiss,trans):
    new_trans = np.array(trans)
    K = len(trans)
    kesi = []
    for l in range(1,len(X)):
        for i in range(K):
            for j in range(K):
                new_trans[i,j] = alpha[l-1,i]*emiss[j,X[l]]*trans[i,j]*beta[l,j]
        kesi.append(new_trans/new_trans.sum())
    return kesi

def cal_gamma(alpha,beta):
    gamma = alpha*beta
    return gamma/gamma.sum(axis = 1).reshape(len(gamma),1)

def EM(trans,emiss,pi):
    global X1,X2
    K = len(trans)    
    for n in range(10):
        alpha1 = forward(X1,K,trans,emiss,pi)
        beta1 = backward(X1,K,trans,emiss,pi)
        alpha2 = forward(X2,K,trans,emiss,pi)
        beta2 = backward(X2,K,trans,emiss,pi)    
        gamma1 = cal_gamma(alpha1,beta1)
        gamma2 = cal_gamma(alpha2,beta2)
        kesi1 = cal_kesi(X1,alpha1,beta1,emiss,trans)
        kesi2 = cal_kesi(X2,alpha2,beta2,emiss,trans)
        #??????????????????????
        #pi = (gamma1[0, :] + gamma2[0, :]) / 2
        #update trans
        for i in range(K):
            for j in range(K):
                trans[i,j] = sum([kesi1[l-1][i,j] for l in range(1,len(X1))]+
                              [kesi2[l-1][i,j] for l in range(1,len(X2))])/\
                            sum([kesi1[l-1][i,k] for k in range(K) for l in range(1,len(X1))]
                                +[kesi2[l-1][i,k] for k in range(K) for l in range(1,len(X2))])
        #update emiss
        for i in range(K):
            for j in range(4):
                emiss[i,j] = sum([gamma1[l][i] for l in range(len(X1)) if X1[l] == j]
                             +[gamma2[l][i] for l in range(len(X2)) if X2[l] == j])/\
                            sum([gamma1[l][i] for l in range(len(X1))]
                                +[gamma2[l][i] for l in range(len(X2))])
    return trans,emiss                        
# for K = 4 , you should use the following parameters to initialize
# transition matrix
Initial_A_4 = np.array([
    [0.3, 0.1, 0.2, 0.4],
    [0.1, 0.2, 0.4, 0.3],
    [0.2, 0.4, 0.3, 0.1],
    [0.4, 0.3, 0.1, 0.2]]
)

# emission matrix
Initial_phi_4 = np.array([
    [0.5, 0.1, 0.2, 0.2],
    [0.1, 0.5, 0.1, 0.3],
    [0.1, 0.2, 0.5, 0.2],
    [0.3, 0.1, 0.1, 0.5]
])

#copy from github
def viterbi(N,M,A,B,P,hidden,observed):
    sta = []
    LEN = len(observed)
    Q = [([0]*N) for i in range(LEN)]
    path = [([0]*N) for i in range(LEN)]
    for j in range(N):
        Q[0][j]=P[j]*B[j][observation.index(observed[0])]
        path[0][j] = -1
    for i in range(1,LEN):
        for j in range(N):
            max = 0.0
            index = 0
            for k in range(N):
                if(Q[i-1][k]*A[k][j] > max):
                    max = Q[i-1][k]*A[k][j]
                    index = k
            Q[i][j] = max * B[j][observation.index(observed[i])]
            path[i][j] = index
    max = 0.0
    idx = 0
    for i in range(N):
        if(Q[LEN-1][i]>max):
            max = Q[LEN-1][i]
            idx = i
    sta.append(hidden[idx])
    for i in range(LEN-1,0,-1):
        idx = path[i][idx]
        sta.append(hidden[idx])
    sta.reverse()
    return sta
    
trans,emiss = EM(Initial_A_2,Initial_phi_2,[0.5,0.5])

trans,emiss = EM(Initial_A_4,Initial_phi_4,[0.25,0.25,0.25,0.25])
    