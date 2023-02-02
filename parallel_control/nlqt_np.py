"""
Numpy-based Nonlinear (iterated) Linear Quadratic Tracker.

@author: Simo Särkkä
"""

class NLQT:

    def __init__(self,lqt,model):
        self.lqt = lqt
        self.model = model

    def linearize(self, u_list, x_list):
        for k in range(len(u_list)):
            x = x_list[k]
            u = u_list[k]
            f = self.model.f(x, u)
            Fx = self.model.Fx(x, u)
            Fu = self.model.Fu(x, u)

            # f(xp,up) = f(x,u) + Fx(x,u) (xp - x) + Fu(x,u) (up - u)
            F = Fx
            L = Fu
            c = f - Fx @ x - Fu @ u

            self.lqt.F[k] = F
            self.lqt.c[k] = c
            self.lqt.L[k] = L

    def iterate(self, u_list, x_list, lqt_method=0):
        self.linearize(u_list, x_list)
        if lqt_method == 0:
            Kx_list, d_list, S_list, v_list = self.lqt.parBackwardPass()
            u_list, x_list = self.lqt.parForwardPass(x_list[0], Kx_list, d_list)
        elif lqt_method == 1:
            Kx_list, d_list, S_list, v_list = self.lqt.parBackwardPass()
            u_list, x_list = self.lqt.parFwdBwdPass(x_list[0], Kx_list, d_list, S_list, v_list)
        else:
            Kx_list, d_list, S_list, v_list = self.lqt.seqBackwardPass()
            u_list, x_list = self.lqt.seqForwardPass(x_list[0], Kx_list, d_list)
        return u_list, x_list

    def simulate(self, x0, u_data):
        x_list = [x0]
        x = x0
        for k in range(len(u_data)):
            x = self.model.f(x, u_data[k])
            x_list.append(x)

        return x_list

    def cost(self, x0, u_list):
        x_list = self.simulate(x0, u_list)
        return self.lqt.cost(x_list, u_list)


