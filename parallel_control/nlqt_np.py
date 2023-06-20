"""
Numpy-based Nonlinear (iterated) Linear Quadratic Tracker.

@author: Simo Särkkä
"""

###########################################################################
#
# Nonlinear iterated Linear Quadratic Tracker
#
###########################################################################

class NLQT:

    def __init__(self,lqt,model):
        """ Form nonlinear iterated linear quadratic tracker.

        Parameters:
            lqt: Linear quadratic tracker.
            model: NonlinearModel object.
        """
        self.lqt = lqt
        self.model = model

    def linearize(self, u_list, x_list):
        """ Linearize model around trajectory.

        Parameters:
            u_list: List of control inputs.
            x_list: List of states.
        """
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
        """ Iterate the nonlinear linear quadratic tracker.

        Parameters:
            u_list: List of control inputs.
            x_list: List of states.
            lqt_method: 0 = parallel, 1 = parallel fw/bw, 2 = sequential.

        Returns:
            u_list: List of updated control inputs.
            x_list: List of updated states.
        """
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
        """ Simulate the controlled nonlinear model.

        Parameters:
            x0: Initial state.
            u_data: List of control inputs.

        Returns:
            x_list: List of states.
        """
        x_list = [x0]
        x = x0
        for k in range(len(u_data)):
            x = self.model.f(x, u_data[k])
            x_list.append(x)

        return x_list

    def cost(self, x0, u_list):
        """ Evaluate the cost given a control input sequence.

        Parameters:
            x0: Initial state.
            u_list: List of control inputs.

        Returns:
            J: Cost.
        """
        x_list = self.simulate(x0, u_list)
        return self.lqt.cost(u_list, x_list)


