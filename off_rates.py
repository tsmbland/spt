# Adapted from Lars Hubatsch

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fmin
from scipy.integrate import odeint

"""
Off rate analysis

"""


class OffRateFitter:
    """

    """

    def __init__(self, features, timestep):

        # Data
        self.features = features

        # Parameters
        self.timestep = timestep

        # Results
        self.partCount, _ = np.histogram(self.features.frame, bins=self.features.frame.max() + 1)
        self.fitTimes = np.arange(0, len(self.partCount) * self.timestep, self.timestep)

        # self.koff = None
        # self.kph = None
        # self.nss = None

    def fit_off_rate(self, variant=1):
        """
        Fit differential equation to data by solving with odeint and
        using fmin to parameters that best fit time/intensity data or
        by using optimize.curve_fit. Different variants have use
        different free parameters.

        Adapted from Lars's code

        """

        '''
        Variant 1: fits differential equation to data, free parameters
        kOff, Nss, kPh, assumes infinite cytoplasmic pool
        '''
        if variant == 1:
            def dy_dt(y, t, kOff, Nss, kPh):
                # Calculates derivative for known y and params
                return kOff * Nss - (kOff + kPh) * y

            def objFunc(params, fitTimes, fitData):
                # Returns distance between solution of diff. equ. with
                # parameters params and the data fitData at times fitTimes
                # Do integration of dy_dt using parameters params
                y = odeint(dy_dt, params[1], fitTimes, args=(params[0], params[1], params[2]))
                # Get y-values at the times needed to compare with data
                return sum((y.flatten() - fitData) ** 2)

            # Set reasonable starting values for optimization
            kOffStart, NssStart, kPhStart = 0.01, self.partCount[0], 0.01
            # Optimize objFunc to find optimal kOffStart, NssStart, kPhStart
            x = fmin(objFunc, [kOffStart, NssStart, kPhStart],
                     args=(self.fitTimes, self.partCount),
                     disp=False)

            self.kOffVar1, self.NssVar1, self.kPhVar1 = (x[0], x[1], x[2])
            # Get solution using final parameter set determined by fmin
            self.fitSolVar1 = odeint(dy_dt, self.NssVar1, self.fitTimes,
                                     args=(self.kOffVar1, self.NssVar1, self.kPhVar1))

        '''
        Variant 2: fits solution of DE to data, fitting N(0), N(Inf) and koff,
        therefore being equivalent to variant=1
        '''
        if variant == 2:
            def exact_solution(times, koff, count0, countInf):
                return (count0 - countInf) * np.exp(-koff * count0 / countInf * times) + countInf

            popt, pcov = curve_fit(exact_solution, self.fitTimes, self.partCount)
            self.kOffVar2 = popt[0]
            self.fitSolVar2 = exact_solution(self.fitTimes, popt[0], popt[1], popt[2])

        '''
        Variant 3: fits solution of DE to data, assuming Nss=N(0) and
        Nss_bleach=N(end), only one free parameter: koff
        '''
        if variant == 3:
            def exact_solution(count0, countInf):
                def curried_exact_solution(times, koff):
                    return ((count0 - countInf) *
                            np.exp(-koff * count0 / countInf * times) + countInf)

                return curried_exact_solution

            popt, pcov = curve_fit(exact_solution(self.partCount[0], self.partCount[-1]), self.fitTimes,
                                   self.partCount)
            self.kOffVar3 = popt[0]
            func = exact_solution(self.partCount[0], self.partCount[-1])
            self.fitSolVar3 = [func(t, popt[0]) for t in self.fitTimes]

        '''
        Variant 4: fits solution of DE to data, fitting off rate and N(Inf),
        leaving N(0) fixed at experimental value
        '''
        if variant == 4:
            def exact_solution(count0):
                def curried_exact_solution(times, koff, countInf):
                    return ((count0 - countInf) *
                            np.exp(-koff * count0 / countInf * times) + countInf)

                return curried_exact_solution

            kOffStart, countInf = 0.005, 50
            popt, pcov = curve_fit(exact_solution(self.partCount[0]), self.fitTimes, self.partCount,
                                   [kOffStart, countInf])
            self.kOffVar4 = popt[0]
            func = exact_solution(self.partCount[0])
            self.fitSolVar4 = [func(t, popt[0], popt[1]) for t in self.fitTimes]

        '''
        Variant 5 (according to supplement Robin et al. 2014):
        Includes cytoplasmic depletion, fixes N(0). N corresponds to R,
        Y corresponds to cytoplasmic volume
        '''
        if variant == 5:
            def exact_solution(count0):
                def curried_exact_solution(times, r1, r2, kPh):
                    return (count0 * ((kPh + r2) / (r2 - r1) * np.exp(r1 * times) +
                                      (kPh + r1) / (r1 - r2) * np.exp(r2 * times)))

                return curried_exact_solution

            popt, pcov = curve_fit(exact_solution(self.partCount[0]), self.fitTimes, self.partCount,
                                   [-0.006, -0.1, 0.1], maxfev=10000)

            self.kPhVar5 = popt[2]
            self.kOnVar5 = (popt[0] * popt[1]) / self.kPhVar5
            self.kOffVar5 = -(popt[0] + popt[1]) - (self.kOnVar5 + self.kPhVar5)
            func = exact_solution(self.partCount[0])
            self.fitSolVar5 = [func(t, popt[0], popt[1], popt[2]) for t in self.fitTimes]

        '''
        Variant 6: This tries to circumvent the error made by fixing the
        starting condition to the first measurement. This point already has a
        statistical error affiliated with it. Fixing it propagates this
        error through the other parameters/the whole fit. Otherwise
        equivalent to variant 5.
        '''
        if variant == 6:
            def curried_exact_solution(times, r1, r2, kPh, count0):
                return (count0 * ((kPh + r2) / (r2 - r1) * np.exp(r1 * times) +
                                  (kPh + r1) / (r1 - r2) * np.exp(r2 * times)))

            popt, pcov = curve_fit(curried_exact_solution, self.fitTimes, self.partCount, [-0.1, -0.2, -0.01, 200],
                                   maxfev=10000)
            self.count0Var6 = popt[3]
            self.kPhVar6 = popt[2]
            self.kOnVar6 = (popt[0] * popt[1]) / self.kPhVar6
            self.kOffVar6 = -(popt[0] + popt[1]) - (self.kOnVar6 + self.kPhVar6)
            self.fitSolVar6 = [curried_exact_solution(t, popt[0], popt[1], self.kPhVar6, self.count0Var6) for t in
                               self.fitTimes]

    def plot_off_rate_fit(self, variant):
        font = {'weight': 'bold', 'size': 'larger'}
        fig, ax = plt.subplots()
        # fig.suptitle('Variant ' + str(variant), fontsize=20, fontdict=font, bbox=dict(facecolor='green', alpha=0.3))
        if variant == 1:
            ax.plot(self.fitTimes, self.partCount, self.fitTimes,
                    self.fitSolVar1)
            ax.set_title(self.kOffVar1)
            print(self.kOffVar1)
        elif variant == 2:
            ax.plot(self.fitTimes, self.partCount, self.fitTimes,
                    self.fitSolVar2)
            ax.set_title(self.kOffVar2)
            print(self.kOffVar2)
        elif variant == 3:
            ax.plot(self.fitTimes, self.partCount, self.fitTimes,
                    self.fitSolVar3)
            ax.set_title(self.kOffVar3)
            print(self.kOffVar3)
        elif variant == 4:
            ax.plot(self.fitTimes, self.partCount, self.fitTimes,
                    self.fitSolVar4)
            ax.set_title(self.kOffVar4)
            print(self.kOffVar4)
        elif variant == 5:
            ax.plot(self.fitTimes, self.partCount, self.fitTimes,
                    self.fitSolVar5)
            ax.set_title(self.kOffVar5)
            print(self.kOffVar5)
        elif variant == 6:
            ax.plot(self.fitTimes, self.partCount, self.fitTimes,
                    self.fitSolVar6)
            ax.set_title(self.kOffVar6)
            print(self.kOffVar6)
        else:
            print('Variant ' + str(variant) + ' does not exist.')
        ax.set(ylabel='# particles', xlabel='t [s]')
