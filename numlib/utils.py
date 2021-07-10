from __future__ import annotations  # for TYPE_CHECKING to work below; remove Pthon 3.9?
import os
import pickle
import copy
import math
import decimal
import functools
import operator
from itertools import combinations
from typing import List, Tuple, TYPE_CHECKING, cast, TypeVar, Generator
if TYPE_CHECKING:
    from polylib import FPolynomial

__author__ = "Scott Simmons"
__version__ = "0.1"
__status__ = "Development"
__date__ = "5/1/21"
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
__license__ = "Apache 2.0"

#def gcd(a: int, b: int) -> int:
#    if b == 0:
#        return abs(a)
#
#    return abs(b) if a % b == 0 else gcd_(b, a % b)

Euclidean = TypeVar('Euclidean', int, 'FPolynomial') # Both Euclidean rings

def gcd_(a: Euclidean, b: Euclidean) -> Euclidean:
    """Return a greatest common divisor of a and b.

    If the arguments are ints, this returns either the usual, positive
    greatest common divisor, or its negative. So, recover the usual gcd
    with abs(gcd_(a,b).

    For polynomials over a field, greatest common divisors are defined
    only up to multiplication by a unit in the field; this function
    returns one of them, which you can then make monic, if you wish.

    Examples:

        For integers:

        >>> gcd_(15, -35)
        -5

        Polynomials are assumed to be defined using the polylib library:

        >>> from polylib import FPolynomial
        >>> from numlib import Zmod
        >>> GF = Zmod(43)  # a finite field
        >>> x = FPolynomial([0, GF(1)])  # indeterimant for Z/34Z[x]
        >>> p1 = 33 * (x - 3) * (7 * x + 6) * (5 * x**4 - 9)
        >>> p2 = 100 * (x - 8) * (7 * x + 6) * (x**2 - 11)
        >>> print('p1 = '+str(p1)); print('p2 = '+str(p2))
        p1 = 14 + 26x + 28x^2 + 40x^4 + 19x^5 + 37x^6
        p2 = 39 + 3x + 13x^2 + 31x^3 + 12x^4
        >>> g = gcd_(p1, p2)
        >>> print(g)  # a gcd
        30 + 35x
        >>> print(g * g[-1]**-1)  # the unique monic gcd
        7 + x

        >>> from numlib import GaloisField
        >>> GF = GaloisField(2,3)
        >>> t = GF.t()
        >>> p1 = 33 * (t - 3) * (7 * t + 6) * (5 * t**4 - 9)
        >>> p2 = 100 * (t - 8) * (7 * t + 6) * (t**2 - 11)
        >>> print(gcd_(p1, p2))
        1 + t

        >>> #GF = GaloisField(43)
        >>> #x = FPolynomial([GF([-3]), GF([1])])
        >>> #p1 = 33 * (x - 3) * (7 * x + 6) * (5*x**4 - 9)
        >>> #p2 = 100 * (x - 8) * (7 * x + 6) * (x**2 - 11)
        >>> #print(gcd_(p1, p2))
        -4 + <-3 + t>

    """
    if b == 0:
        return a

    return b if a % b == 0 else gcd_(b, a % b)

def lcm_(a: Euclidean, b: Euclidean) -> Euclidean:
    """Return a least common multiple of a and b.

    Examples:

        >>> from polylib import FPolynomial
        >>> from numlib import Zmod
        >>> GF = Zmod(43)  # a finite field
        >>> x = FPolynomial([0, GF(1)])  # indeterimant for Z/34Z[x]
        >>> p1 = 33 * (x - 3) * (7 * x + 6) * (5*x**3 - 9)
        >>> p2 = 100 * (x - 8) * (7 * x + 6) * (x**2 - 11)
        >>> print(lcm_(p1,p2))
        14 + 35x + x^2 + 4x^3 + x^4 + 23x^5 + 20x^6 + 12x^7 + 40x^8
    """
    return a * b // gcd_(a,b)

def xgcd(a: Euclidean, b: Euclidean) -> Tuple[Euclidean, ...]:
    """Return tuple (gcd(a,b), s, t) satisfying gcd(a,b) = r*a + s*b.

    This works as expected for ints, but also with polynomials defined
    over fields.

    Examples:

        Polynomials are assumed to be defined using the polylib library.

        >>> from polylib import FPolynomial
        >>> from numlib import Zmod

        >>> GF = Zmod(43)  # a finite field
        >>> x = FPolynomial([0, GF(1)])  # indeterimant for Z/34Z[x]
        >>> p1 = 8 + 100*x**2 + x**4; p2 = 30 + 26*x**3
        >>> print('p1 = '+str(p1)+';  p2 = '+str(p2))
        p1 = 8 + 14x^2 + x^4;  p2 = 30 + 26x^3
        >>> print(",  ".join([str(poly) for poly in xgcd(p1, p2)]))
        40,  42 + 2x + 7x^2,  36 + 31x + 33x^2 + 8x^3

        >>> from fractions import Fraction    # generate polynomials
        >>> x = FPolynomial([0, Fraction(1)]) # over the rationals
        >>> p1 = 5*x+Fraction(7,3)*x**2
        >>> p2 = Fraction(1,11)*x+x**2+x**5
        >>> print('p1 = '+str(p1)+';  p2 = '+str(p2))
        p1 = 5x + 7/3x^2;  p2 = 1/11x + x^2 + x^5
        >>> print(",  ".join([str(poly) for poly in xgcd(p1, p2)]))
        502681/26411x,  9096/2401 - 675/343x + 45/49x^2 - 3/7x^3,  1
    """

    s0 = cast(Euclidean, 1); s1 = cast(Euclidean, 0)
    t0 = cast(Euclidean, 0); t1 = cast(Euclidean, 1)
    while True:
        quot = -(a // b)
        a = a % b
        s0 += quot * s1
        t0 += quot * t1
        if a == 0:
            return (b, s1, t1)
        quot = -(b // a)
        b = b % a
        s1 += quot * s0
        t1 += quot * t0
        if b == 0:
            return (a, s0, t0)

    # Below, the original, works fine, save typing
    #s = [1, 0]
    #t = [0, 1]
    #while True:
    #    quot = -(a // b)
    #    a = a % b
    #    s[0] += quot * s[1] #type: ignore
    #    t[0] += quot * t[1] #type: ignore
    #    if a == 0:
    #        return (b, s[1], t[1])
    #    quot = -(b // a)
    #    b = b % a
    #    s[1] += quot * s[0] #type: ignore
    #    t[1] += quot * t[0] #type: ignore
    #    if b == 0:
    #        return (a, s[0], t[0])


def sieve(n: int = 1000000) -> Tuple[int, ...]:
    """Return list of primes <= n.

    Uses Sieve of Eratosthenes.
    """
    assert n >= 1
    primes = []
    sieve = [True] * (n + 1)
    for p in range(2, n + 1):
        if sieve[p]:
            primes.append(p)
            for i in range(p * p, n + 1, p):
                sieve[i] = False
    return tuple(primes)

def truefactor(n:int) -> int:
    """Return smallest prime factor > 1 of n > 1.

    >>> truefactor(143)
    11
    >>> truefactor(2027)
    2027
    """
    assert n > 1
    for p in sieve(int(decimal.Decimal(n).sqrt()+1)):
        if n % p == 0:
            return p
    return n

def istrueprime(n:int) -> int:
    """Return True/False according to whether a positive int n  is prime.

    This is slow for very large n.

    >>> istrueprime(2027)
    True
    >>> istrueprime(2027*2017)
    False
    """
    return n > 1 and truefactor(n) == n

def isprime(n: int) -> int:
    """Return True/False according to whether n is likely prime.

    Uses a variety of pseudoprime tests. This is fast.

    >>> isprime(2027)
    True
    >>> isprime(2027*2017)
    False
    >>> isprime(258001471497710271176990892852404413747)
    True
    """
    assert n >= 1
    if n in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]:
        return True
    return isprimeE(n, 2) and isprimeE(n, 3) and isprimeE(n, 5)


def isprimeF(n: int, b: int) -> bool:
    """True if n is prime or a Fermat pseudoprime to base b."""

    return pow(b, n - 1, n) == 1


def isprimeE(n: int, b: int) -> bool:
    """True if n is prime or an Euler pseudoprime to base b."""

    if not isprimeF(n, b):
        return False
    r = n - 1
    while r % 2 == 0:
        r //= 2
    c = pow(b, r, n)
    if c == 1:
        return True
    while True:
        if c == 1:
            return False
        if c == n - 1:
            return True
        c = pow(c, 2, n)


def factor_(n: int) -> int:
    """Return, with fair likelihood of success, a prime factor of n.

    >>> factor_(8) == factor_(4) == 2
    True
    """

    assert n > 1
    if isprime(n):
        return n
    for fact in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]:
        if n % fact == 0:
            return fact
    return factorPR(n)


def factor(n: int) -> List[int]:
    """Return, with fair likelihood of success, the prime factors of n.

    >>> factor(2017*2027*12353948231)  # product of primes
    [2017, 2027, 12353948231]
    >>> factor(2017**4*(2027*12353948231)**2)
    [2017, 2017, 2017, 2017, 2027, 2027, 12353948231, 12353948231]
    """

    assert n > 1
    if isprime(n):
        return [n]
    fact = factor_(n)
    assert fact != 1, "Unable to factor " + str(n)
    facts = factor(n // fact) + factor(fact)
    facts.sort()
    return facts


def factorPR(n: int) -> int:
    """Return a factor of n using the Pollard Rho method.

    The return value is 1, if n is prime, and a non-trivial factor,
    otherwise.  Note: This method will occasionally fail to find a
    non-trivial factor when one exists.

    >>> factorPR(2017*2027*12353948231)  # product of primes
    2017
    >>> factorPR(8) == factorPR(4) == 2  # fails
    False
    """
    numsteps = 2 * int(decimal.Decimal(n).sqrt().sqrt())
    for slow in [2, 3, 4, 6]:
        fast = slow
        for _ in range(numsteps):
            slow = (slow * slow + 1) % n
            fast = (fast * fast + 1) % n
            fast = (fast * fast + 1) % n
            g = math.gcd(fast - slow, n)
            if g != 1:
                if g == n:
                    break
                else:
                    return g
    return 1

def truephi(n: int) -> int:
    """Return the number of positive integers less than and coprime to n.

    This computes Euler's totient function. Group theoretically, phi(n) is
    the order of (Z/nZ)^*, the multiplicative group of units in the ring,
    Z/nZ, of integers modulo n.

    Slow for very large n.

    Examples:

        >>> truephi(2**10-1)
        600
    """
    assert n > 1
    phi_ = 1
    prevfact = 1
    while n > 1:
        fact = truefactor(n)
        n //= fact
        if fact == prevfact:
            phi_ *= fact
        else:
            phi_ *= fact - 1
            prevfact = fact
    return phi_


def phi(n: int) -> int:
    """Return the number of positive integers less than and coprime to n.

    This computes Euler's totient function. Group theoretically, phi(n) is
    the order of (Z/nZ)^*, the multiplicative group of units in the ring,
    Z/nZ, of integers modulo n.

    Fast but, technically, can fail.

    Examples:

        >>> phi(2**10-1)
        600
    """
    assert n > 1
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

def mu(n: int) -> int:
    """Return the value of the Moebius function on n.

    >>> mu(3*5*2)
    -1
    >>> mu(3*5*2*17)
    1
    >>> mu(3*3*5*2)
    0
    >>> mu(1)
    1
    >>> mu(5)
    -1
    >>> mu(2**10-1)
    -1
    """
    if n == 1:
        return 1
    else:
        #return -reduce(add, [mu(d) for d in range(2, n) if n % d == 0], 1)
        # below is faster (for all n?)
        count = 1
        prevfact = truefactor(n)
        n //= prevfact
        while n > 1:
            factor = truefactor(n)
            if prevfact == factor:
                return 0
            count += 1
            n //= factor
            prevfact = factor
        return (-1)**count

def divisors(n: int) -> Generator[int]:
    """Return all divisors greater than 1.

    Examples:

       >>> list(divisors(7))
       [7]
       >>> list(divisors(12))
       [2, 3, 6, 4, 12]
    """
    facts = factor(n)
    for r in range(1, len(facts)+1):
        for fact in set(combinations(facts, r)):
            yield functools.reduce(operator.mul, fact, 1)

def addorder(element: object, exponent = None):
    """Return the additive order of element.

    Args:

        element: An element in an additive group.

        exponent: An integer such that n*element is the additive
            identity.

    Returns:

        (int). The additive order of element.

    If exponent is not None, this is O(order) in the order of element.

    If a (minimal) exponent is provided, this is O(log2(order)) where
    the constant is larger for smoother and (lees efficient) exponents.
    This assumes that mulitiplication by n in the ambient group is
    O(log2(n)).

    Examples:

        >>> from numlib import Zmod
        >>> R = Zmod(24)
        >>> addorder(R(5))
        24
        >>> addorder(R(2))
        12

        Find the order of an element of an Elliptic Curve:

        >>> from numlib import GaloisField, EllCurve
        >>> from polylib import FPolynomial
        >>> F = GaloisField(7, 3)
        >>> F  # GaloisField of order 7^3
        Z/7[t]/<-3 + 3t^2 + t^3>
        >>> t = F([0,1])  # generator of the unit group of GF(7^3)
        >>> E = EllCurve(t+1, t**2)
        >>> E  # An elliptic curve over GF(7^3)
        y^2 = (t^2) + (1 + t)x + x^3 over Z/7[t]/<-3 + 3t^2 + t^3>

        Let us check that the j-invariant is non-zero:

        >>> E.j
        1 + 2t^2 + <-3 + 3t^2 + t^3>

        Given that (2+2t+2t^2, 1+3t-t^2) is a point on the curve, let
        us compute its order:

        >>> pt = E(2 + 2*t - 2*t**2, 1 + 3*t - t**2)
        >>> addorder(pt)
        339

        This curve, in fact, consists of 339 points (and so is cyclic).
        If we know the order of the curve then we should use that fact
        (since, then, addorder is O(log(order)) instead of O(order)):

        >>> for pt in E:
        ...     addorder(pt, 339)
        ...     break
        ...
        113
    """
    identity = 0*element
    accum = copy.copy(element)
    if exponent:
        if element == identity:
            return 1
        prev_divisor = 1
        for divisor in sorted(list(divisors(exponent))[:-1]):
            accum += (divisor - prev_divisor) * element
            if accum == identity:
                return divisor
            prev_divisor = divisor
        return exponent
    else:
        order = 1
        while accum != identity:
            order += 1
            accum += element
        return order

def mulorder(element, exponent = None):
    """Return the order of element.

    Args:

        element: An element in an multiplicative group.

        exponent: An integer such that n*element is the multiplicative
            identity.

    Returns:

        (int). The multiplicative order of element.

    If exponent is not None, this is O(order) in the order of element.

    If a (minimal) exponent is provided, this is O(log2(order)) where
    the constant is larger for smoother (and less efficient) exponents.
    This assumes that exponentiation by n in the ambient group is
    O(log2(n)).

    Examples:

        >>> from numlib import Zmod
        >>> R = Zmod(24)
        >>> mulorder(R(5), phi(24))
        2
        >>> R = Zmod(60)
        >>> mulorder(R(7), phi(60))
        4
        >>> R = Zmod(60)
        >>> mulorder(R(7))
        4

        Find the order of an element in a Galois field:

        >>> from polylib import FPolynomial
        >>> from numlib import Zmod, FPmod
        >>> PF = Zmod(13)   # The primefield is Z/13Z.
        >>> irred = FPolynomial([PF(1)]*5, 't')  # 1+t+t^2+t^3+t^4
        >>> GF = FPmod(irred)  #  Galois field of order 13^4

        Let us check if irred is primitive for GF:

        >>> t = GF((0, 1))  # t + <1 + t + t^2 + t^3 + t^4>
        >>> mulorder(t, 13**4-1)
        5

        No, t is not a generator of the unit group of GF(13^4)
        and so 1+x+x^2+x^3+x^4 is not a primitive polynomial for
        GF(13^4). This is not surprising: any time that we quot-
        ient by an irreducible of the form 1+x+x^2+...+x^n, we
        have

         x^(n+1) = x*x^n = -x (x^(n-1)+ x^(n-2)+ ... + x^2 + x)
                         = - x^n - x^(n-1) - ... - x^2 - x
                         = 1.

        Let us check that the irreducible 4+3x^2+x^3 in Z/7Z[x]
        is primitive for GF(7^3):

        >>> PF = Zmod(7)   # The primefield is Z/7Z.
        >>> t = FPolynomial((0, PF(1)), 't')
        >>> irred = 4 + 3*t**2 + t**3
        >>> GF = FPmod(irred)  #  Galois field of order 7^3
        >>> t = GF(t)
        >>> mulorder(t, GF.order() - 1) == 7**3-1
        True
    """
    identity = element**0
    accum = copy.copy(element)
    if exponent:
        if element == identity:
            return 1
        prev_divisor = 1
        for divisor in sorted(list(divisors(exponent))[:-1]):
            accum *= element ** (divisor - prev_divisor)
            if accum == identity:
                return divisor
            prev_divisor = divisor
        return exponent
    else:
        order = 1
        while accum != identity:
            order += 1
            accum *= element
        return order

def serialize(obj, filename, directory = '.'):
    filename = ''.join(c for c in filename if c.isalnum())
    if os.path.exists(directory+'/'+filename):
        raise ValueError(f"file {directory}/{filename} exists")
    pickle.dump(obj , open(directory+'/'+filename, "wb"))

def unserialize(filename, directory = '.'):
    filename = ''.join(c for c in filename if c.isalnum())
    if not os.path.exists(directory+'/'+filename):
        raise ValueError(f"file {directory}/{filename} does not exist")
    return pickle.load(open(directory+'/'+filename, "rb"))

if __name__ == "__main__":

    import doctest

    doctest.testmod()
