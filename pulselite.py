from copy import copy

import numpy as np

def gauss(t, sigma, mu):
    '''Gaussian with width `sigma` centered around `mu`.'''
    return np.exp(-(np.power((t - mu), 2))/(2*sigma*sigma))

def gausstruncated(t, sigma, truncate, mu):
    '''
    Gaussian with width `sigma` centered around `mu`, and limited
    to [`mu`-`truncate`*`sigma`, `mu`+`truncate`*`sigma`].
    '''
    gaussian = lambda t: (
        (gauss(t, sigma, mu) - gauss(sigma*truncate, sigma, 0.)) / 
        (1. - gauss(sigma*truncate, sigma, 0.))
    )
    if (t >= mu-truncate*sigma) and (t <= mu+truncate*sigma):
        return gaussian(t)
    return 0.

def gausstruncated_derivative(t, sigma, truncate, mu):
    '''
    Derivative of a truncated Gaussian used for DRAG pulses.
    '''
    gaussian_der = lambda t: (
        -(t - mu)/sigma**2 * gauss(t, sigma, mu) / 
        (1. - gauss(sigma*truncate, sigma, 0.))
    )
    return np.piecewise(t, [(t >= mu-truncate*sigma) & (t <= mu+truncate*sigma)], [gaussian, 0.])
    

class Pulse(object):
    '''
    Pulse base class
    
    Must have `length` and `operator` properties.
    '''
    operator = None
        
    def function(self, t, tstop):
        pass

    def __neg__(self):
        r = copy(self)
        r.amplitude = -self.amplitude
        return r

class ZeroPulse(Pulse):
    '''Zero amplitude pulse.'''
    def __init__(self, length):
        self.length = length
        
    def function(self, t, tstop):
        return 0.    

class SquarePulse(Pulse):
    def __init__(self, operator, amplitude, length, **op_kwargs):
        self.operator = operator
        self.amplitude = amplitude
        self.length = length
        self.op_kwargs = op_kwargs
        
    def function(self, t, tstop):
        #return self.amplitude*np.piecewise(t, [(t >= tstop-self.length) & (t < tstop)], [1., 0.])
        if (t >= tstop-self.length) and (t < tstop):
            return self.amplitude
        return 0.
    
class GaussianPulse(Pulse):
    def __init__(self, operator, amplitude, sigma, truncate, **op_kwargs):
        self.operator = operator
        self.amplitude = amplitude
        self.sigma = sigma
        self.truncate = truncate
        self.op_kwargs = op_kwargs
        
    def function(self, t, tstop):
        return self.amplitude*gausstruncated(t, self.sigma, self.truncate, tstop - self.sigma*self.truncate)
    
    @property
    def length(self):
        return 2*self.truncate*self.sigma

class DRAGCorrection(GaussianPulse):
    '''amplitude <= amplitude * qscale / anharmonicity from pulsegen'''
    def function(self, t, tstop):
        return self.amplitude*gausstruncated_derivative(t, self.sigma, self.truncate, tstop - self.sigma*self.truncate)



class Sequence(object):
    '''
    A sequence of pulses on a qubit in a multi-qubit system.
    
    Converts a sequence of abstract Pulse objects to operators
    by looking up Pulse.operator strings 'flux', 'charge', 
    'charge-x', 'charge-y' to the corresponding `Qobj` held by
    the `qubit` object. `channel` and `dimensions` are used to
    promote the single-qubit operator to the correct subspace
    in a larger Hilbert space of all simulated systems.

    Parameters
    ----------
    qubit - `Transmon` object or compatible
        used by render to resolve operator names to Qobjs
    channel - `int`
        Subsystem number
    dimensions - [`int`]
        Dimensions of all subsystems in the Hamiltonian
    separation - `float`
        Separation between pulses
        
    Methods
    -------
    render -> [(`qutip.Qobj`, function), ...], float
        Generate Hamiltonian components for all pulses.
        Returns a tuple containing a list of time-dependent operators 
        in (op, func) form and the total length of the sequence.
    '''

    def __init__(self, qubit, channel, dimensions, separation=0.):
        self.qubit = qubit
        self.channel = channel
        self.dimensions = dimensions
        self.separation = separation
        self.pulses = []
        
    def append(self, pulse):
        '''Append a pulse to the queue.'''
        self.pulses.append(pulse)
        
    def extend(self, pulses):
        '''Append a list of pulses ot the queue'''
        self.pulses.extend(pulses)
    
    @staticmethod
    def make_lambda(function, tstop):
        return lambda t, args: function(t, tstop, **args)
    
    def render(self):
        '''
        Calculate contributions to the Hamiltonian as (Qobj, function) pairs.
        '''
        H = []
        t = 0.
        for pulse in self.pulses:
            t += pulse.length
            operator, function = pulse.operator, pulse.function
            # Don't add ZeroPulse spacers to the Hamiltonian
            if operator is not None:
                if callable(operator):
                    operator, function = operator(function, **pulse.op_kwargs)
                if len(self.dimensions) > 1:
                    operator = qutip.tensor(*[operator if ch==self.channel else qutip.qeye(dim)
                                              for ch, dim in enumerate(self.dimensions)])
                H.append([operator, self.make_lambda(function, t)])
            t += self.separation
        t -= self.separation
        return H, t