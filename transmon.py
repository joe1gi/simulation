#
#
# Transmon class for Qutip
# 
#

import functools
import numpy as np
import qutip

def cached_property(deps, maxsize=16, typed=False):
    '''
    Convert a method that depends on the attributes listed in `deps` to a
    property with an lru_cache.
    '''
    def cached_decorator(meth):
        @functools.lru_cache(maxsize=maxsize, typed=typed)
        def prop_args(self, *args):
            return meth(self)
        @property
        def prop(self):
            args = tuple(getattr(self, dep) for dep in deps)
            return prop_args(self, *args)
        return prop
    return cached_decorator


class Transmon(object):
    '''
    Simulate a Transmon qubit.

    Arguments
    ---------
    Ej, Ec: float
        Josephson and charging energy of the Transmon
    N: int
        Size of the associated Hilbert space
    '''
    def __init__(self, Ej, Ec, N):
        self.Ej = Ej
        self.Ec = Ec
        self.N = N

    @property
    def energies(self):
        return self.H.eigenenergies()

    @property
    def states(self):
        return self.H.eigenstates()[1]

    def quorum(self, t=None):
        '''
        Return ground/+x/+y/excited states (at time t)
        '''
        ground, excited = self.states[:2]
        if t is not None:
            omega = np.diff(self.energies[:2])[0]
            excited *= np.exp(-1j*omega*t)
        return [
            ground, 
            (ground + excited) / np.sqrt(2), 
            (ground + 1j*excited) / np.sqrt(2), 
            excited
        ] 



class PhaseSpaceTransmon(Transmon):
    '''
    Simulate a flux-tunable Transmon in phase space.

    Arguments
    ---------
    Same as for `Transmon`, plus
    ng: float
        Static gate charge
    flux: float
        Flux applied to the SQUID

    '''
    def __init__(self, Ej, Ec, ng, flux, N):
        super().__init__(Ej, Ec, N)
        self.ng = ng
        self.flux = flux

    @cached_property(['Ej', 'Ec', 'ng', 'flux', 'N'])
    def states(self):
        return self.H.eigenstates()[1]

    @cached_property(['Ej', 'Ec', 'ng', 'flux', 'N'])
    def energies(self):
        return self.H.eigenenergies()

    @property
    def phase(self):
        '''Phase-across-the-junction operator'''
        phases = np.linspace(-np.pi, np.pi, self.N, endpoint=False)
        return qutip.qdiags([phases], [0])

    @property
    def dphase(self):
        '''d/dphase operator'''
        N = self.N
        c = N/(2*np.pi)
        # 1st order, standard and periodic
        #return c*qutip.qdiags([[-1]*(N), [1]*(N-1)], [0,1])
        #return c*qutip.qdiags([[-1]*(N), [1]*(N-1), [1]], [0,1,-N+1])
        # 2nd order, standard and periodic
        #return c*qutip.qdiags([[-1]*(N-1), [1]*(N-1)], [-1, 1])
        return c*qutip.qdiags([[-1]*(N-1), [1]*(N-1), [-1], [1]], 
                              [-1, 1, N-1, -N+1])
        # 4th order
        #return c*qutip.qdiags([[1]*(N-2), [-8]*(N-1), [8]*(N-1), [-1]*(N-1)], 
        #                      [-2,-1,1,2])

    @property
    def dphase2(self):
        '''d^2/dphase^2 operator'''
        N = self.N
        c = (N/(2*np.pi))**2
        # 1st order, standard and periodic
        #return qutip.qdiags([[1]*(N-1), [-2]*N, [1]*(N-1)], [-1,0,1])
        return c*qutip.qdiags([[1]*(N-1), [-2]*N, [1]*(N-1), [1], [1]], 
                              [-1,0,1,-N+1,N-1])

    @property
    def n(self):
        '''island charge operator'''
        return -1j*self.dphase

    @property
    def n2(self):
        '''squared island charge operator'''
        return -self.dphase2

    @property
    def H(self):
        '''Hamiltonian of the bare qubit'''
        return (4*self.Ec*(self.n2-2*self.ng*self.n+self.ng**2) - 
                self.Ej*np.abs(np.cos(self.flux))*self.phase.cosm())

    def H_charge(self, dcharge=1., transition=None, frequency=0., phase=0.):
        '''
        Charge drive Hamiltonian, normalized to g01=2*Pi

        If dcharge, transition or frequency are provided, the charge is 
        modulated as dcharge(t, args)*Cos[(E_j-E_i+2*Pi*frequency)*t + phase].

        H_charge is a first order expansion at the current bias point, 
        it is valid only for large Ej/Ec and smallish amplitudes.

        Arguments
        ---------
        dcharge: `float`, `complex` or `callable`, optional
            Constant amplitude or envelope function of the drive with signature 
            dcharge(t, args).
        transition: (`int`, `int`), optional
            If transition is set to (i, j), the charge is modulated at a 
            frequency corresponding to the energy difference between the
            i'th and j'th states.
        frequency: `float`, optional
            Modulation frequency or offset to the frequency given by transition
        phase: `float`, optional         
            Modulation phase

        Returns
        `qutip.Qobj` or [`qutip.Qobj`, `callable`] list
        '''
        if (transition is not None) or (frequency != 0.) or (phase != 0.):
            frequency = 2*np.pi*frequency
            if transition is not None:
                energies = self.energies
                frequency += energies[transition[1]] - energies[transition[0]]
            if callable(dcharge):
                dcharge_ = lambda t, args: (dcharge(t, args)*
                                           np.cos(frequency*t+phase))
            else:
                dcharge_ = lambda t, args: dcharge*np.cos(frequency*t+phase)
            return [self._H_charge, dcharge_]
        if callable(dcharge):
            return [self._H_charge, dcharge]
        else:
            return dcharge*self._H_charge

    @property
    def _H_charge(self):
        '''The charge operator in the selected basis.'''
        # normalize coupling strength
        states = self.states
        beta = 2*np.pi/np.abs((states[1].dag()*self.n*states[0])[0,0])
        return beta*self.n

    @property
    def _H_flux(self):
        '''The Josephson energy operator in the selected basis.'''
        return self.Ej*self.phase.cosm()

    def H_flux(self, dflux):
        '''
        Flux drive Hamiltonian

        Arguments
        ---------
        dflux: `float`, `complex` or `callable`
            dflux(t, args) is called by qutip's solvers to determine the flux
            offset at time t in units of Phi0/(2*Pi).
        
        Returns
        -------
        [`qutip.Qobj`, `callable`] list that can be passed to qutip solvers
        '''
        if callable(dflux):
            def amplitude(t, *args):
                return (np.abs(np.cos(self.flux + dflux(t, *args))) - 
                        np.abs(np.cos(self.flux)))
            return [self._H_flux, amplitude]
        else:
            return (np.abs(np.cos(self.flux+dflux)) - 
                    np.abs(np.cos(self.flux))) * self._H_flux



class TruncatedTransmon(object):
    '''
    Simulate a Transmon qubit in the energy eigenbasis

    Transforms the operators of a `PhaseSpaceTransmon` into the energy 
    eigenbasis at its bias point and truncates the Hilbert space to `Ntr` 
    levels.
    '''
    def __init__(self, transmon, Ntr):
        self.full = transmon
        self.N = Ntr

    @property
    def energies(self):
        return self.full.energies[:self.N]

    @property
    def states(self):
        return [qutip.basis(self.N, n) for n in range(self.N)]

    quorum = Transmon.quorum

    @property
    def flux(self):
        return self.full.flux

    @property
    def H(self):
        return qutip.qdiags([self.energies], [0])

    @property
    def _H_charge(self):
        full_op = self.full._H_charge
        return qutip.Qobj(
            [[#self.full.states[i].dag()*self.full._H_charge*self.full.states[j] 
              full_op.matrix_element(self.full.states[i].dag(), 
                                     self.full.states[j])
              for i in range(self.N)] 
             for j in range(self.N)]
        )

    @property
    def _H_flux(self):
        full_op = self.full._H_flux
        return qutip.Qobj(
            [[full_op.matrix_element(self.full.states[i].dag(), 
                                     self.full.states[j])
              for i in range(self.N)] 
             for j in range(self.N)]
        )

    H_charge = PhaseSpaceTransmon.H_charge
    H_flux = PhaseSpaceTransmon.H_flux
