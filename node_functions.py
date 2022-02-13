import math
import numpy as np

from scipy import signal
from scipy.special import betainc, expit

# We used a modified sigmoidal transfer function, φ(x) = 1/(1+e^−4.9x) , at all nodes.

def clamp(num, min_value, max_value):
   return np.max(np.min(num, max_value), min_value) # arrays
#    return max(min(num, max_value), min_value)

def sqr(x):
    return np.square(x) # arrays
    # return x**2

def lin(x):
    return x

def cubic(x):
    return np.power(x, 3) # arrays
    # return  math.pow(x,3)

def sawtooth(x):
    return signal.sawtooth(x*10)

def noise(x):
    # x = max(-255.0, min(255.0, 2.5 * x))
    return np.random.normal(loc=x, scale=1.0, size=None)

def sigmoid(x):
    # y = 1/(1 + np.exp(-5*x))
    return expit(x) # faster

def tanh(x):
    y = np.tanh(2.5*x) # arrays
    # y = math.tanh(2.5*x)
    return y

def tanh_sig(x):
    return sigmoid(tanh(x))

def sin(x):
    # y =  np.sin(x*math.pi)
    # y =  np.sin(x*2* math.pi)
    # return .5 * (y + 1)
    
    y =  np.sin(x*math.pi) # faster for arrays
    # y = math.sin(x*math.pi)
    return y
    
def cos(x):
    # y =  np.cos(x*2* math.pi)
    # return .5 * (y + 1)
    y =  np.cos(x*math.pi) # faster for arrays
    # return math.cos(x)
    return y

def gauss(x):
    # y = np.exp(-50.0 * (x-.5) ** 2)
    y = 2*np.exp(-20.0 * (x) ** 2)-1 # faster for arrays
    # y = 2*math.exp(-20.0 * (x) ** 2)-1
    return y

def inv_gauss(x):
    return 0- gauss(x)

def beta(x):
    k = .45
    return betainc(k, k, x)

def log(x):
    x = max(1e-7, x)
    # return clamp(y, 0, 1)
    y = 1+ np.log(x) / 16 # faster for arrays
    # y = 1+ math.log(x) / 16
    return y

def identity(x):
    return x

def relu(x):
    return x * (x > 0)

def abs_activation(x):
    return np.abs(x) # arrays
    # return abs(x)

def hat(x):
    try:
        x = 1.0 - np.abs(x)
        x= np.clip(x, 0, 1)
        return x
    except Exception as e:
        print(e)
        return max(0.0, 1 - abs(x)) # not arrays

def pulse(x):
    return 2*(x % 1 < .5) -1

def inv_log(x):
    if(x == 0): return x
    if(x == 1): return x
    return 1/(1+np.exp(x/(1-x), -1)) # arrays
    # return 1/(1+math.exp(x/(1-x), -1))

def sig_transfer(x):
    return 1/(1+np.exp(x*-4.9)) # arrays
    # return 1/(1+math.exp(x*-4.9))

def clamped_activation(x):
    return np.max(-1.0, np.min(1.0, x)) # arrays
    # return max(-1.0, min(1.0, x))

def elu(x):
    return [xi if xi > 0.0 else math.exp(xi) - 1 for xi in x] # array
    # return x if x > 0.0 else math.exp(x) - 1

def inv(x):
    try: x = 1.0 / x
    except ArithmeticError: return 0.0
    else: return x

def round_activation(x):
    # return round(x)
    return np.round(x) # arrays


# In both systems, function outputs range between
# [−1,1]. However, ink level is darker the closer the output
# is to zero. Therefore, an output of either -1 or 1 produces
# white. 
def output_to_value_picbreeder(x):
    # The network output w was in the range [−1 ...1] 
    # and the corresponding grayscale value was calculated as 256(1−|w|). (Woolley and Stanley, 2011)
    # return 1-np.subtract(1, np.abs(x))
    # return np.subtract(1, np.abs(x)) # 1 - |x| # arrays
    return 1 - abs(x)
 
def output_to_value_gauss(x):
    y = np.exp(-5.0 * np.power(x,2))-1 # arrays
    # y = math.exp(-5.0 * math.pow(x,2))-1
    return -y 

def output_to_value_jackson(x):
    return np.clip(1-np.subtract(1, np.abs(x)), -1, 1) # arrays
    # return clamp(1-np.subtract(1, np.abs(x)), -1, 1) 

all_node_functions = [
    # picbreeder:
        cos,
        sin,
        gauss,
        identity,
        sigmoid,

    # others:
        relu,
        hat,
        pulse,
        tanh,
        abs_activation,
        round_activation,
        # elu,
        # inv_gauss,
        # tanh_sig,
        # inv,
        # sawtooth # slow

    # can return outside [-1, 1]
        sqr,
        # cubic,

        # noise,
        
        # beta,
        # log,
        # inv_log # sometimes throws "result too large"

    ]





