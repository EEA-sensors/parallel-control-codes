"""
Simulated associative scan.

@author: Simo Särkkä
"""

import math


def my_assoc_scan(op,a,reverse=False):
    a = a.copy()
    
    if reverse:
        list.reverse(a)
        a = my_assoc_scan(lambda x,y: op(y,x),a,reverse=False)
        list.reverse(a)
        return a
    
    orig_n = len(a)
    log_n  = math.ceil(math.log2(orig_n))
    n = 2**log_n
    
    if n != orig_n:
        #print('Expanding array to length %d -> %d.' % (orig_n, n))
        while len(a) < n:
            a.append(None)
            
    a0 = a.copy()
    
    # Up pass
    for d in range(log_n):
        for i in range(0,n,2**(d+1)): # This would be a parallel loop
            i1 = i + 2**d - 1
            i2 = i + 2**(d+1) - 1
            
            if a[i2] is None:
                a[i2] = a[i1]
            elif a[i1] is None:
                pass
            else:
                a[i2] = op(a[i1],a[i2])
            
    a[n-1] = None
    
    # Down pass
    for d in reversed(range(log_n)):
        for i in range(0,n,2**(d+1)): # This would be a parallel loop
            i1 = i + 2**d - 1
            i2 = i + 2**(d+1) - 1
        
            t = a[i1]
            a[i1] = a[i2]
            
            if a[i2] is None:
                a[i2] = t
            elif t is None:
                pass
            else:
                a[i2] = op(a[i2],t)
                
    # Extra pass
    for i in range(n): # This is a parallel loop
        if a[i] is None:
            a[i] = a0[i]
        elif a0[i] is None:
            pass
        else:
            a[i] = op(a[i],a0[i])
    
    a = a[0:orig_n]
    
    return a

