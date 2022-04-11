from __future__ import (
    annotations,
)  # for TYPE_CHECKING to work below; remove Python 3.9?
import os
import pickle
import random
import copy
import math
import decimal
import functools
import operator
from itertools import combinations, cycle, product, tee
from typing import List, Tuple, cast, TypeVar, Generator, Callable, Optional
from polylib import FPolynomial, Field, Ring
import numlib as nl  # fixes circular import

__author__ = "Scott Simmons"
__version__ = "0.2"
__status__ = "Development"
__date__ = "03/24/22"
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

# def gcd(a: int, b: int) -> int:
#    if b == 0:
#        return abs(a)
#
#    return abs(b) if a % b == 0 else gcd(b, a % b)

R = TypeVar("R", bound=Ring)
F = TypeVar("F", bound=Field)

Euclidean = TypeVar("Euclidean", int, FPolynomial[F])  # Both are Euclidean rings
# Euclidean = TypeVar('Euclidean', int, 'FPolynomial[Field]') # Both Euclidean rings


def gcd(a: Euclidean, b: Euclidean) -> Euclidean:
    """Return a greatest common divisor of a and b.

    If the arguments are ints, this returns either the usual, positive
    greatest common divisor, or its negative. So, recover the usual gcd
    with abs(gcd(a,b)).

    For polynomials over a field, greatest common divisors are defined
    only up to multiplication by a unit in the field; this function
    returns one of them, which you can then make monic, if you wish.

    Examples:

        For integers:

        >>> gcd(15, -35)
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
        >>> g = gcd(p1, p2)
        >>> print(g)  # a gcd
        30 + 35x
        >>> print(g * g[-1]**-1)  # the unique monic gcd
        7 + x

        >>> from numlib import GaloisField
        >>> GF = GaloisField(2, 3)
        >>> t = GF()
        >>> p1 = 33 * (t - 3) * (7 * t + 6) * (5 * t**4 - 9)
        >>> p2 = 100 * (t - 8) * (7 * t + 6) * (t**2 - 11)
        >>> print(gcd(p1, p2))
        t+1

        >>> GF = GaloisField(43)
        >>> t = GF()
        >>> p1 = 33 * (t - 3) * (7 * t + 6) * (5*t**4 - 9)
        >>> p2 = 100 * (t - 8) * (7 * t + 6) * (t**2 - 11)
        >>> print(gcd(p1, p2))
        -4
    """
    if b == 0:
        return a

    return b if a % b == 0 else gcd(b, a % b)


def lcm(a: Euclidean, b: Euclidean) -> Euclidean:
    """Return a least common multiple of a and b.

    Examples:

        >>> from polylib import FPolynomial
        >>> from numlib import Zmod
        >>> GF = Zmod(43)  # a finite field
        >>> x = FPolynomial([0, GF(1)])  # indeterimant for Z/34Z[x]
        >>> p1 = 33 * (x - 3) * (7 * x + 6) * (5*x**3 - 9)
        >>> p2 = 100 * (x - 8) * (7 * x + 6) * (x**2 - 11)
        >>> print(lcm(p1,p2))
        14 + 35x + x^2 + 4x^3 + x^4 + 23x^5 + 20x^6 + 12x^7 + 40x^8
    """
    return (a * b) // gcd(a, b)


def xgcd(a: Euclidean, b: Euclidean) -> Tuple[Euclidean, ...]:
    """Return tuple (gcd(a,b), s, t) satisfying gcd(a,b) = s*a + t*b.

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

    s0 = cast(Euclidean, 1)
    s1 = cast(Euclidean, 0)
    t0 = cast(Euclidean, 0)
    t1 = cast(Euclidean, 1)
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
    # s = [1, 0]
    # t = [0, 1]
    # while True:
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


def leastdivisor(n: int) -> int:
    """Return smallest prime factor > 1 of n > 1.

    Examples:

        >>> leastdivisor(143)
        11
        >>> leastdivisor(2027)
        2027
    """
    assert n > 1
    for p in sieve(int(decimal.Decimal(n).sqrt() + 1)):
        if n % p == 0:
            return p
    return n


def istrueprime(n: int) -> bool:
    """Return True/False according to whether a positive int n  is prime.

    This is slow for very large n.

    Examples:

        >>> istrueprime(2027)
        True
        >>> istrueprime(2027*2017)
        False
    """
    return n > 1 and leastdivisor(n) == n


def isprime(n: int) -> bool:
    """Return True/False according to whether n is likely prime.

    Uses a variety of pseudoprime tests. This is fast.

    Examples:

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

    return cast(int, pow(b, n - 1, n)) == 1


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

    Example:

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

    Examples:

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
    if isprime(fact):
        facts = [fact] + factor(n // fact)
    else:
        facts = factor(fact) + factor(n // fact)
    facts.sort()
    return facts


def factorPR(n: int) -> int:
    """Return a factor of n using the Pollard Rho method.

    The return value is 1, if n is prime, and a non-trivial factor,
    otherwise.  Note: This method will occasionally fail to find a
    non-trivial factor when one exists.

    Examples:

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
        fact = leastdivisor(n)
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

    Examples:

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
        facts = factor(n)
        len_ = len(facts)
        if len(set(facts)) < len_:
            return 0
        return cast(int, (-1) ** len_)


def truemu(n: int) -> int:
    """Return the value of the Moebius function on n.

    Examples:

        >>> truemu(3*5*2)
        -1
        >>> truemu(3*5*2*17)
        1
        >>> truemu(3*3*5*2)
        0
        >>> truemu(1)
        1
        >>> truemu(5)
        -1
        >>> truemu(2**10-1)
        -1
    """
    if n == 1:
        return 1
    else:
        # return -reduce(add, [mu(d) for d in range(2, n) if n % d == 0], 1)
        # below is faster (for all n?)
        count = 1
        prevfact = leastdivisor(n)
        n //= prevfact
        while n > 1:
            factor = leastdivisor(n)
            if prevfact == factor:
                return 0
            count += 1
            n //= factor
            prevfact = factor
        return cast(int, (-1) ** count)


def divisors_(n: int) -> Generator[int, None, None]:
    """Return all divisors greater than 1.

    Examples:

        >>> list(divisors(7))
        [7]
        >>> list(divisors(12))
        [2, 3, 4, 6, 12]
    """
    facts = factor(n)
    for r in range(1, len(facts) + 1):
        for fact in set(combinations(facts, r)):
            yield functools.reduce(operator.mul, fact, 1)


def divisors(n: int) -> List[int]:
    """Return sorted, increasing list of divisors > 1."""

    return sorted(list(divisors_(n)))


def addorder_(element: R, possible_orders: List[int]) -> int:
    """Helper for addorder that accepts a list of possible orders.

    Args:

        element: An element of an additive group.

        possible_orders: List of possible orders sorted and in
            increasing order.

    Returns:

        (int). The additive order.
    """
    identity = 0 * element
    accum = copy.copy(element)
    if element == identity:
        return 1
    prev_divisor = 1
    for divisor in possible_orders[:-1]:
        accum += (divisor - prev_divisor) * element
        if accum == identity:
            return divisor
        prev_divisor = divisor
    return possible_orders[-1]


def addorder(element: R, exponent: Optional[int] = None) -> int:
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
        >>> F = GaloisField(7, 3)  # GaloisField of order 7^3
        >>> print(F)
        Z/7[t]/<t^3+3t^2-3>
        >>> t = F([0,1])  # generator of the unit group of GF(7^3)
        >>> E = EllCurve(t+1, t**2)
        >>> E  # An elliptic curve over GF(7^3)
        y^2 = x^3 + (t+1)x + (t^2) over Z/7[t]/<t^3+3t^2-3>

        Let us check that E is non-singular:

        >>> E.disc
        -3t^2+3t-1 + <t^3+3t^2-3>

        Given that (2+2t+2t^2, 1+3t-t^2) is a point on the curve, let
        us compute its order:

        >>> pt = E(2 + 2*t - 2*t**2, 1 + 3*t - t**2)
        >>> addorder(pt)
        339

        This curve, in fact, consists of 339 points (and so is cyclic).
        If we know the order of the curve then we should use that fact
        (since, then, addorder is O(log(order)) instead of O(order)):

        >>> for pt in affine(E):
        ...     addorder(pt, 339)
        ...     break
        ...
        113
    """
    if exponent:
        return addorder_(element, divisors(exponent))
    else:
        identity = 0 * element
        accum = copy.copy(element)
        order = 1
        while accum != identity:
            order += 1
            accum += element
        return order


def mulorder_(element: R, possible_orders: List[int]) -> int:
    """Helper for mulorder that accepts a list of possible orders.

    Args:

        element: An element of a multiplicative group.

        possible_orders: List of possible orders sorted and in
            increasing order.

    Returns:

        (int). The multiplicative order.
    """
    identity = element**0
    accum = copy.copy(element)
    if element == identity:
        return 1
    prev_divisor = 1
    for divisor in possible_orders[:-1]:
        accum *= element ** (divisor - prev_divisor)
        if accum == identity:
            return divisor
        prev_divisor = divisor
    return possible_orders[-1]


def mulorder(element: R, exponent: Optional[int] = None):
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
        >>> mulorder(t, 7**3-1) == 7**3-1
        True
    """
    if exponent:
        return mulorder_(element, divisors(exponent))
    else:
        identity = element**0
        accum = copy.copy(element)
        order = 1
        while accum != identity:
            order += 1
            accum *= element
        return order


def iproduct(*iterables, repeat: int = 1):
    """Cartesian product of large or infinite iterables (per MarkCBell)

    Examples:

        >>> print(list(iproduct(range(2), range(2))))
        [(0, 0), (1, 0), (0, 1), (1, 1)]
        >>> print(list(iproduct(range(2), repeat=2)))
        [(0, 0), (1, 0), (0, 1), (1, 1)]
        >>> list1 = list(iproduct(range(2), range(2), repeat=2))
        >>> list2 = list(iproduct(range(2), repeat=4))
        >>> list1 == list2
        True
    """

    iterables = [
        item
        for row in zip(*(tee(iterable, repeat) for iterable in iterables))
        for item in row
    ]
    N = len(iterables)
    saved = [[] for _ in range(N)]  # All the items that we have seen of each iterable.
    exhausted = set()  # The set of indices of iterables that have been exhausted.
    for i in cycle(range(N)):
        if i in exhausted:  # Just to avoid repeatedly hitting that exception.
            continue
        try:
            item = next(iterables[i])
            yield from product(*saved[:i], [item], *saved[i + 1 :])  # Finite product.
            saved[i].append(item)
        except StopIteration:
            exhausted.add(i)
            if (
                not saved[i] or len(exhausted) == N
            ):  # Product is empty or all iterables exhausted.
                return
    yield ()  # There are no iterables.


def affine(E: nl.EllipticCurve[F]) -> Generator[Tuple[int, int], None, None]:
    """Return a generator that yields the affine points of E.

    This works fine for small curves. But it pre computes two
    dictionaries that each of size of order of the curve. So
    this of course takes forever on large curves.

    Consider using affine2, instead.

    Examples:

        >>> from numlib import Zmodp, EllCurve
        >>> F = Zmodp(71)
        >>> E = EllCurve(F(2), F(3)); E
        y^2 = x^3 + 2x + 3 over Z/71
        >>> len(list(affine(E)))
        87

        >>> F = Zmodp(73)
        >>> E = EllCurve(F(2), F(3))
        >>> len(list(affine(E)))
        69

        >>> from numlib import GaloisField
        >>> F = GaloisField(5, 2)  # a field of order 25
        >>> t = F()
        >>> E = EllCurve(2+t, 4*t**0); E
        y^2 = x^3 + (t+2)x - 1 over Z/5[t]/<t^2+t+2>
        >>> len(list(affine(E)))
        34
    """
    coefcls = E.disc.__class__
    # b = E.f(0)
    # a = E.f.derivative()(0)
    b = E.f[0]
    a = E.f[1]

    if hasattr(coefcls, "char") and coefcls.char and hasattr(coefcls, "__iter__"):

        y2s = {}  # map all squares y2 in the field k to {y | y^2 = y2}
        fs = {}  # map all outputs fs = f(x), x in k, to {x | f(x) = fs}

        # build y2s and fs
        for x in coefcls:
            x2 = x**2
            y2s.setdefault(x2, []).append(x)
            fs.setdefault(x2 * x + a * x + b, []).append(x)

        # yield all points of the curve
        for y2 in y2s.keys():
            for f in fs.keys():
                if y2 == f:
                    for x in fs[f]:
                        for y in y2s[y2]:
                            yield E(x, y)
    else:
        return NotImplemented


def sqrt(a: F, q: int, p: int) -> F:
    """Return square root of the given square a.

    The prime p must be odd, currently.

    Args:
        a (Field): a square in the form of instance of an
            implementation of a finite field F.
        q (int): the order of F.
        p (int): the (odd) characteristic of F.

    Returns:

        (Field). An element of F whose square is the given a.
    """
    if q % 4 == 3:
        return a ** ((q + 1) // 4)
    elif q > p:
        return NotImplemented
    else:
        one = a**0
        t = random.randint(0, p - 1) * one
        while (t**2 - 4 * a) ** ((p - 1) // 2) != -1:
            t = random.randint(0, p - 1) * one
        Fp2 = nl.FPmod(FPolynomial([a, t, one]))
        x = Fp2([0, 1])
        return (x ** ((p + 1) // 2))[0]


def affine2(E: nl.EllipticCurve[F]) -> Generator[Tuple[int, int], None, None]:
    """Yield roughly half of the affine points of E.

        This yields one of each pair {(x, y), (x, -y)} of points
        not on the line y=0 and works by checking if f(x) is a
        quadratic residue where y^2 = f(x) defines E. If f(x) is
        a quadratic residue then one of the corresponding points
        on the curve is yielded.

        Examples:

            >>> from numlib import Zmodp, EllCurve

            >>> F = Zmodp(71)
            >>> E = EllCurve(F(2), F(3)); E
            y^2 = x^3 + 2x + 3 over Z/71
            >>> len(list(affine2(E)))
            42
            >>> E.disc
            2 + <71>

            The curve E above is non-singular and in fact has 3
            points with y-coordinate equal to 0 so that, in tot-
            al, that curve has 2 * 42 + 3 = 87 finite points.

            The curve below has only one point with y=0; hence
            it has 2 * 34 + 1 = 79 finite points.

            >>> F = Zmodp(73)
            >>> E = EllCurve(F(2), F(3), debug = True)
            >>> E.disc != 0
            True
            >>> aff = affine2(E)
            >>> len(list(affine2(E)))
            34

    #        >>> from numlib import GaloisField
    #        >>> F = GaloisField(5, 2)
    #        >>> t = F()
    #        >>> E = EllCurve(2+t, 4*t**0); E
    #        y^2 = x^3 + (t+2)x - 1 over Z/5[t]/<t^2+t+2>
    #        >>> len({pt for pt in affine2(E)})  # TODO: implement this
    #        0
    """
    coefcls = E.disc.__class__
    p = coefcls.char
    q = p if not hasattr(coefcls, "order") else coefcls.order
    if q % 2 == 0:
        return NotImplemented

    for x in coefcls:
        fx = E.f(x)
        if fx ** ((q - 1) // 2) == 1:
            yield E(x, y=sqrt(fx, q=q, p=p))


def frobenious2(
    E: nl.EllipticCurve[F], m: int
) -> Callable[[nl.EllipticCurve[F]], nl.EllipticCurve[F]]:
    """Return the mth iterate of the  q^r-power Frobenius isogeny.

    Args:

        E (AlgebraicCurve): an elliptic curve over a finite field
            K of order q^r where q = p^n and E is defined in terms
            of short Weierstrauss coefficients a and b in the ord-
            er q subfield of K.

        m (int): a positive integer.

    Returns:

        (Callable). A function that maps a given pt = (x, y) in E
            to (x^(q^m), y^(q^m)).

    Examples:

        The Frobenious endomophism of a field of order q^r as an ex-
        tension of a subfield of order q = p^n maps like x -> x^q;
        it has order q^r (and in fact generates the associated Galois
        group).

        Given E as above, the Frobenious isogeny E -> E maps like

                            (x, y) -> (x^q, y^q);

        clearly, its rth iterate is the identity.

        Example 1: q = p = 7; r = 3

        >>> from numlib import GaloisField, EllCurve, affine2
        >>> GF = GaloisField(7, 3); t = GF()
        >>> E = EllCurve(2*t**0, 3*t**0); E  # A curve E over GF(7,3)
        y^2 = x^3 + 2x + 3 over Z/7[t]/<t^3+3t^2-3>
        >>> pt = next(affine2(E)); print(pt)  # A point on E
        (-3t^2-3t-2, -t^2-t+1)

        The 3rd iterate is the identity isogeny:

        >>> from numlib import affine
        >>> E_affine = affine(E)  # affine points  of E
        >>> frob = frobenious2(E, 3)
        >>> print(f"{pt} maps to {frob(pt)}")
        (-3t^2-3t-2, -t^2-t+1) maps to (-3t^2-3t-2, -t^2-t+1)
        >>> all(frob(pt) == pt for pt in E_affine)
        True

        But the 2nd one is non-trivial:

        >>> frob = frobenious2(E, 2)
        >>> print(f"{pt} maps to {frob(pt)}")
        (-3t^2-3t-2, -t^2-t+1) maps to (3t^2-t+3, t^2+2t-2)

        Example 2: q = 7^3; r = 2

        >>> from polylib import FPolynomial
        >>> from numlib import FPmod
        >>> GF = GaloisField(7, 3, indet='(t)'); t = GF()
        >>> # the polynomial below has coefficients in GF(7, 3)
        >>> # and is irreducible over GF(7,3)
        >>> irred = FPolynomial([-t**2-3, -2*t**2-t, t**0])
        >>> print(irred)
        (-t^2-3) + (-2t^2-t)x + x^2
        >>> F = FPmod(irred); F
        Z/7[(t)]/<(t^3+3t^2-3)>[x]/<(-t^2-3) + (-2t^2-t)x + x^2>
        >>> x = F([0*t, 1*t**0])
        >>> x
        x + <(-t^2-3) + (-2t^2-t)x + x^2>
        >>> #E = EllCurve(2*t*x**0, 3*t**2*x**0, debug = True)

    """
    PF = E.j.__class__
    t = PF()
    if PF.char == t.char:  # then the primefield is Z/p
        PF_frob = lambda x: x ** (PF.char**m)
    else:  # the primefield is an extension of Z/p
        PF_frob = lambda x: x ** (PF.order // PF.char**m)
    return lambda pt: E(*tuple(map(PF_frob, tuple(pt.co))))


def frobenious(
    E: nl.EllipticCurve[F], r: int
) -> Callable[[nl.EllipticCurve[F]], nl.EllipticCurve[F]]:
    """Return the q^r-power Frobenious isogeny.

    E must be Weierstrass curve. TODO: Generalize.

    Args:

        E (AlgebraicCurve): an elliptic curve over a finite field
            K of order q = p^n.

        r (int): a positive integer.

    Returns:

        (Callable). A function that maps a given pt = (x, y) in E
            to (x^(q^r), y^(q^r)).

    Examples:

    Given E defined the field K = GF(p,n) of order q = p^n, this re-
    turns the function E(K') -> E(K') on K'-rational points of E over
    where K' is a field of order q^r, i.e., GF(p,nr), that maps like

                          (x, y) -> (x^r, y^r).

    In other words, on the level of coefficients x and y, the map is
    just the rth iterate of the endormorphism F: K -> K defined by
    F(x) = x^p^n which, in turn, is just the nth iteratate of the
    Frobenious endomorphism Z/p -> Z/p.

    If K has order q=p^n, then F^n(x) = x^(p^n) = x for all x so that
    the map x -> x^(p^n) is the identity on k; hence frobenious(E,1)
    is the identity on K-rational points of E. For instance,

        >>> import numlib as nl
        >>> GF = nl.GaloisField(7, 3); t = GF()
        >>> E = nl.EllCurve(1+t, 2*t); E  # A curve E over GF(7,3)
        y^2 = x^3 + (t+1)x + (2t) over Z/7[t]/<t^3+3t^2-3>
        >>> E_affine = nl.affine(E)  # affine points  of E

        >>> pt = next(nl.affine2(E)); print(pt)  # A point on E
        (-3t^2-2t-2, -3t^2-2t-2)
        >>> frob = frobenious(E, 3)
        >>> print(f"{pt} maps to {frob(pt)}")
        (-3t^2-2t-2, -3t^2-2t-2) maps to (-3t^2-2t-2, -3t^2-2t-2)
        >>> all(frob(pt) == pt for pt in E_affine)
        True

    As an endomorphism of K=GF(p,n), F has order n (and, in fact, gen-
    erates the Galois group of GF(p,n) over GF(p,1)).



    If we think of F as a map on the algebraic closure of k, and frob(r) is just frob(1) composed with itself r times, and

        - frob(r) is the identity when restricted to GF(p,r);
          i.e, frob = frob(1) has order r when on GF(p,r).

        >>> frob = frobenious(E, 1)
        >>> print(f"{pt} maps to {frob(pt)}")
        (-3t^2-2t-2, -3t^2-2t-2) maps to (-t^2+2t+3, -t^2+2t+3)
        >>> frob = frobenious(E, 2)
        >>> print(f"{pt} maps to {frob(pt)}")
        (-3t^2-2t-2, -3t^2-2t-2) maps to (-3t^2, -3t^2)
        >>> frob = frobenious(E, 3)
        >>> print(f"{pt} maps to {frob(pt)}")
        (-3t^2-2t-2, -3t^2-2t-2) maps to (-3t^2-2t-2, -3t^2-2t-2)
        >>> frob = frobenious(E, 4)
        >>> print(f"{pt} maps to {frob(pt)}")
        (-3t^2-2t-2, -3t^2-2t-2) maps to (-t^2+2t+3, -t^2+2t+3)
    """
    # b = E.f(0)
    # a = E.f.derivative()(0)
    b = E.f[0]
    a = E.f[1]
    pr = E.j.char**r
    E_codomain = nl.EllCurve(a**pr, b**pr)
    F = lambda x: x**pr
    return lambda pt: E_codomain(*tuple(map(F, tuple(pt.co))))


def serialize(obj: object, filename: str, directory: str = ".") -> None:
    filename = "".join(c for c in filename if c.isalnum())
    if os.path.exists(directory + "/" + filename):
        raise ValueError(f"file {directory}/{filename} exists")
    print("serializing", filename, "to", directory)
    pickle.dump(obj, open(directory + "/" + filename, "wb"))


def unserialize(filename: str, directory: str = ".") -> object:
    filename = "".join(c for c in filename if c.isalnum())
    if not os.path.exists(directory + "/" + filename):
        raise ValueError(f"file {directory}/{filename} does not exist")
    print("unserializing", filename, "from", directory)
    return pickle.load(open(directory + "/" + filename, "rb"))


if __name__ == "__main__":

    import doctest

    doctest.testmod()
