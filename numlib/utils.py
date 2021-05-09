
import math

__author__ = 'Scott Simmons'
__version__ = '0.1'
__status__ = 'Development'
__date__ = '5/1/21'
__copyright__ = """
  Copyright 2014-2021 Scott Simmons

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
"""
__license__= 'Apache 2.0'

def gcd(a, b):
    """Return the gcd of a and b.

    """
    if b == 0:
        return abs(a)
    return abs(b) if a % b == 0 else gcd(b, a % b)

def xgcd(a, b):
    """Return [gcd(a,b), s, t] satisfying gcd(a,b) = r*a + s*b.

    """
    s = [1, 0]; t = [0, 1]
    while True:
        quot = -(a // b)
        a = a % b
        s[0] += quot * s[1]; t[0] += quot * t[1]
        if a == 0:
            return [b, s[1], t[1]]
        quot = -(b // a)
        b = b % a
        s[1] += quot * s[0]; t[1] += quot * t[0]
        if b == 0:
            return [a, s[0], t[0]]

def sieve(n = 1000000):
    """Return list of primes <= n.

    Uses Sieve of Eratosthenes.
    """
    assert n >= 1
    primes = []
    sieve = [True] * (n + 1)
    for p in range (2, n+1):
        if sieve[p]:
            primes.append(p)
            for i in range(p * p, n + 1, p):
                sieve[i] = False
    return primes

def isprime(n):
    """Return True/False according to whether n is very likely prime.

    Using a variety of pseudoprime tests.
    """
    assert n >= 1
    if n in [2,3,5,7,11,13,17,19,23,29]:
        return True
    return isprimeE(n, 2) and isprimeE(n, 3) and isprimeE(n, 5)

def isprimeF(n, b):
    """Return True if n is a prime or a Fermat pseudoprime to base b; False, otherwise.

    """
    return pow(b, n-1, n) == 1

def isprimeE(n, b):
    """Return True if n is a prime or an Euler pseudoprime to base b; False, otherwise.

    """
    if not isprimeF(n, b): return False
    r = n-1
    while r % 2 == 0: r //= 2
    c = pow(b, r, n)
    if c == 1: return True
    while True:
        if c == 1: return False
        if c == n-1: return True
        c = pow(c, 2, n)

def factor_(n):
    """Return, with fair likelihood of success, a prime factor of n.

    """
    assert n > 1
    if isprime(n): return n
    for fact in [2,3,5,7,11,13,17,19,23,29]:
        if n % fact == 0: return fact
    return factorPR(n)  # Needs work - no guarantee that a prime factor will be returned

def factor(n):
    """Return, with fair likelihood of success, a sorted list of the prime factors of n.

    """
    assert n > 1
    if isprime(n):
        return [n]
    fact = factor_(n)
    assert fact != 1, "Unable to factor "+str(n)
    facts = factor(n // fact) + factor(fact)
    facts.sort()
    return facts

def factorPR(n):
    """Return a non-trivial factor of n using the Pollard Rho method.

    Note: This method will occasionally fail.
    """
    for slow in [2,3,4,6]:
        numsteps = 2 * math.floor(math.sqrt(math.sqrt(n))); fast=slow; i=1
        while i < numsteps:
            slow = (slow*slow + 1) % n
            i = i + 1
            fast = (fast*fast + 1) % n
            fast = (fast*fast + 1) % n
            g = gcd(fast-slow,n)
            if (g != 1):
                if (g == n):
                    break
                else:
                    return g
    return 1

def phi(n):
    """Return the number of positive integers less than and coprime to n.

    This computes Euler's totient function. Group theoretically, phi(n) is the order of
    (Z/nZ)^*, the multiplicative group of units in ring, Z/nZ, of integers modulo n.
    """
    assert n>1
    factors = factor(n)
    factors.sort()
    phi_ = 1
    prevfact = 1
    for fact in factors:
        if fact == prevfact:
            phi_ *= fact
        else:
            phi_ *= fact - 1
            prevfact = fact
    return phi_

