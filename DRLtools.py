"""
@Peimou Sun
Discrete RL model
"""
import numpy as np

class FMDP(object):
    """
    Finite Markov Decision Process.
    In this model, the transition matrix and reward are
    all deterministic functions or values. In this program, I did
    not consider exploration-policy. (Of course , we dont need to consider
    exploration if the transition and reward are deterministic)

    Parameters
    ----------
    nstates: |S|, must be a finite int.
    nacts: |A|, must be a finite int.
    gamma: discount rate, must be a constant numeric variable.
    acts_dict: The dictionary of index:act. For example {0:"up", 1:"down"}
    states_dict: The dictionary of index:state. For example {0:"State1", 1:"State2"}

    Reference
    ---------
    'Reinforcement Learning: An Introduction', Richard S. Sutton and Andrew G. Barto

    """
    def __init__(self, nstates, nacts, gamma, **kwargs):
        self.nstates = nstates
        self.nacts = nacts
        self.__VFuncVec = np.empty(nstates)
        self.__Policy = np.empty((nstates, nacts))
        self.__Transition = np.empty((nstates, nacts, nstates))
        self.__Reward = np.zeros((nstates, nstates))
        self.Q = np.zeros((nstates, nacts))
        self.gamma = gamma if gamma >=0 else 0
        self.acts_dict = {}
        self.states_dict = {}
        self.deterministic = True
        self.epsilon = 0.3

        for item in kwargs.items():
            if hasattr(self, item[0]):
                setattr(self, item[0], item[1])


    @property
    def Policy(self):
        return self.__Policy


    @Policy.setter
    def Policy(self, value):
        if value.shape != self.__Policy.shape:
            raise ValueError("Invalid shape")
        if not np.allclose(np.sum(value, axis=1), np.ones(self.nstates)):
            raise ValueError("Invalid Policy func")
        if not self.deterministic:
            raise ValueError("Only available in deterministic model")
        self.__Policy = value


    @property
    def Transition(self):
        return self.__Transition


    @Transition.setter
    def Transition(self, value):
        if value.shape != self.__Transition.shape:
            raise ValueError("Invalid shape")
        if not np.alltrue(np.einsum("ijk->ij",value)<=1):
            raise ValueError("Invalid Transition Matrix")
        if not self.deterministic:
            raise ValueError("Only available in deterministic model")
        self.__Transition = value


    @property
    def Reward(self):
        return self.__Reward


    @Reward.setter
    def Reward(self, value):
        if value.shape != self.__Reward.shape:
            raise ValueError("Invalid shape")
        if not self.deterministic:
            raise ValueError("Only available in deterministic model")
        self.__Reward = value


    def GreedyPolicy(self, state):
        u = np.random.uniform()
        if u > self.epsilon:
            return self.__Policy[state]
        else:
            return np.random.randint(0, self.nstates, dtype = int)


    def FitPolicy(self):
        self.__Policy = np.zeros((self.nstates, self.nacts))
        self.__Policy[:, np.argmax(self.Q, axis = 1)] = 1


    def FitQ(self, states, acts, nstates, rewards, alpha):
        """
        On-line learning
        """
        if self.deterministic: self.deterministic = False
        na = self.GreedyPolicy(nstates)
        self.Q[states, acts] += alpha * (rewards[i] + self.gamma
                                           * self.Q[nstates, na]
                                        - self.Q[states, acts])

        self.FitPolicy()


    def ValueEval(self):
        A = np.eye(self.nstates) - self.gamma * \
            np.einsum('ij, ijk->ik', self.Policy, self.Transition)
        b = np.einsum('ij, ikj, ik->i', self.Reward,
                      self.Transition, self.Policy)[:, np.newaxis]
        return np.linalg.pinv(A) @ b


    def ValueIter(self, tol = 1e-6, max_iter = 1e6):
        """
        Deterministic model method
        """
        if not self.deterministic:
            raise ValueError("ValueIter method can only be used in deterministic model")
        OptValue = np.zeros(self.nstates)
        Delta = 1
        niter = 0
        while(Delta >= tol):
            self.Q = self.gamma * np.einsum("ijk, k->ij",self.Transition, OptValue)
            self.Q += np.einsum("ijk, ik->ij", self.Transition, self.Reward)
            Ovf = OptValue
            OptValue = np.maximum.reduce(self.Q, axis=1)
            Delta = np.sum(np.power(OptValue - Ovf,2)) #I used l2 norm here.
            niter += 1
            if niter >= max_iter:
                print(f"Iterate more than max_iter: {max_iter}")
                break
        self.OptValueFunc = OptValue
        return OptValue


    @staticmethod
    def DSample(distr, urv):
        distr = np.cumsum(distr)
        state = np.arange(0, len(distr))
        return np.min(state[distr>=urv])


    def OptPolicy(self, state, nsteps = 50, random_seed = None):
        if not hasattr(self, "Q"):
            self.ValueIter()
        ss = np.empty(nsteps+1).astype(int)
        ss[0] = state
        acts = np.empty(nsteps).astype(int)
        res = np.empty(nsteps)

        if not random_seed: np.random.seed(random_seed)
        rs = np.random.uniform(size=nsteps)

        for i in range(1, nsteps+1):
            acts[i-1] = np.argmax(self.Q[ss[i-1]])
            ss[i] = self.DSample(self.Transition[ss[i-1], acts[i-1]], rs[i-1])
            res[i-1] = self.Reward[ss[i-1], ss[i]]

        if len(self.acts_dict) != 0:
            acts = [self.acts_dict[con] for con in acts]
        if len(self.states_dict) != 0:
            ss = [self.states_dict[con] for con in acts]

        return ss, acts, res



if  __name__ == "__main__":
    # test for einsum, you can run this sample code to check the answer
    p = np.array([[0.2,0.8],[0.3,0.7]])
    t = np.array([[0.3,0.7],[0.5,0.5]]).reshape(2,1,2)
    ans1 = np.einsum('ij, ijk->ik', p, t)
    ans2 = np.empty((2,2))
    for i in range(2):
        for j in range(2):
            ans2[i][j] = np.sum(t[i,:,j] * p[i,:])
    if np.allclose(ans1, ans2):
        print("Pass 1")
    r = np.linspace(1,4,4).reshape(2,2)
    ans3 = np.einsum('ij, ikj, ik->i', r, t, p)
    ans4 = np.einsum("ik, ijk->ij", r, t)
    ans4 = np.einsum("ik,ik->i",p, ans4)
    if np.allclose(ans3, ans4):
        print("Pass 2")



