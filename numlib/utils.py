from __future__ import (
    annotations,
)  # for TYPE_CHECKING to work below; remove Python 3.9?
import os
import pickle
#import copy
import math
import decimal
import functools
import operator
from itertools import combinations, cycle, product, tee
from typing import cast, TypeVar, Optional, Any, Generator, overload, Iterable
from polylib.polynomial import FPolynomial, Ring, Field

__author__ = "Scott Simmons"
__version__ = "0.3"
__status__ = "Development"
__date__ = "01/27/24"
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

#R = TypeVar("R", bound=Ring[Any])
#F = TypeVar("F", bound=Field[None])
F = TypeVar("F", bound=Field)
#FPoly = FPolynomial[F]

#Don't do this; can't use generic types in TypeVar's
#Euclidean = TypeVar('Euclidean', int, 'FPolynomial[Field]')
#Euclidean = TypeVar('Euclidean', int, 'FPolynomial[F]') # type: ignore[valid-type]

@overload
def gcd(a: int, b: int) -> int: ...
@overload
def gcd(a: FPolynomial[F], b: FPolynomial[F]) -> FPolynomial[F]: ...
def gcd(a: Any, b: Any) -> Any:
#def gcd(a: Euclidean, b: Euclidean) -> Euclidean:
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
        p1 = 14 - 17x - 15x^2 - 3x^4 + 19x^5 - 6x^6
        p2 = -4 + 3x + 13x^2 - 12x^3 + 12x^4
        >>> g = gcd(p1, p2)
        >>> print(g)  # a gcd
        -13 - 8x
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

@overload
def lcm(a: int, b: int) -> int: ...
@overload
def lcm(a: FPolynomial[F], b: FPolynomial[F]) -> FPolynomial[F]: ...
def lcm(a: Any, b: Any) -> Any:
    """Return a least common multiple of a and b.

    Examples:

        >>> from polylib import FPolynomial
        >>> from numlib import Zmod
        >>> GF = Zmod(43)  # a finite field
        >>> x = FPolynomial([0, GF(1)])  # indeterimant for Z/34Z[x]
        >>> p1 = 33 * (x - 3) * (7 * x + 6) * (5*x**3 - 9)
        >>> p2 = 100 * (x - 8) * (7 * x + 6) * (x**2 - 11)
        >>> print(lcm(p1,p2))
        14 - 8x + x^2 + 4x^3 + x^4 - 20x^5 + 20x^6 + 12x^7 - 3x^8
    """
    return (a * b) // gcd(a, b)

@overload
def xgcd(a: int, b: int) -> tuple[int, ...]: ...
@overload
def xgcd(a: FPolynomial[F], b: FPolynomial[F]) -> tuple[FPolynomial[F], ...]: ...
def xgcd(a: Any, b: Any) -> Any:
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
        p1 = 8 + 14x^2 + x^4;  p2 = -13 - 17x^3
        >>> print(",  ".join([str(poly) for poly in xgcd(p1, p2)]))
        -3,  -1 + 2x + 7x^2,  -7 - 12x - 10x^2 + 8x^3

        >>> from fractions import Fraction    # generate polynomials
        >>> x = FPolynomial([0, Fraction(1)]) # over the rationals
        >>> p1 = 5*x+Fraction(7,3)*x**2
        >>> p2 = Fraction(1,11)*x+x**2+x**5
        >>> print('p1 = '+str(p1)+';  p2 = '+str(p2))
        p1 = 5x + 7/3x^2;  p2 = 1/11x + x^2 + x^5
        >>> print(",  ".join([str(poly) for poly in xgcd(p1, p2)]))
        502681/26411x,  9096/2401 - 675/343x + 45/49x^2 - 3/7x^3,  1
    """
    s0 = 1; s1 = 0
    t0 = 0; t1 = 1
    while True:
        quot = -(a // b)
        a = a % b
        s0 += quot * s1
        t0 += quot * t1
        if a == 0:
            return (b,s1,t1)
        quot = -(b // a)
        b = b % a
        s1 += quot * s0
        t1 += quot * t0
        if b == 0:
            return (a,s0,t0)

def sieve(n: int = 1000000) -> tuple[int, ...]:
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


def factor(n: int) -> list[int]:
    """Return, with fair likelihood of success, the prime factors of n.

    The factors are returned in increasing order.

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


def factor2(n: int) -> list[tuple(int, int)]:
    """Return, with fair likelihood of success, the prime factors of n.

    The factors are returned in the form [(p_1, e_1), (p_2, e_2), ...]
    where n = p_1 ^ e_1 p_2 ^ e_2 ... and the p_i are distinct and in
    increasing order.

    Examples:

        >>> factor2(2017*2027*12353948231)
        [(2017, 1), (2027, 1), (12353948231, 1)]
        >>> factor2(2017**4*(2027*12353948231)**2)
        [(2017, 4), (2027, 2), (12353948231, 2)]
    """

    facts = factor(n)
    return list((item, facts.count(item)) for item in set(facts))


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


def divisors(n: int) -> list[int]:
    """Return sorted, increasing list of divisors > 1."""

    return sorted(list(divisors_(n)))


#def addorder_(element: Ring[Any], possible_orders: list[int]) -> int:
def addorder_(element: Ring, possible_orders: list[int]) -> int:
    """Helper for addorder that accepts a list of possible orders.

    Args:

        element: An immutable element of an additive group.

        possible_orders: List of possible orders sorted and in
            increasing order.

    Returns:

        (int). The additive order.
    """
    identity = 0 * element
    if element == identity:
        return 1
    #accum = copy.copy(element)
    accum = element
    prev_divisor = 1
    for divisor in possible_orders[:-1]:
        accum += (divisor - prev_divisor) * element
        if accum == identity:
            return divisor
        prev_divisor = divisor
    return possible_orders[-1]


#def addorder(element: Ring[Any], exponent: Optional[int] = None) -> int:
def addorder(element: Ring, exponent: Optional[int] = None) -> int:
    """Return the additive order of element.

    Args:

        element: An immutable element in an additive group.

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

        >>> from numlib import affine
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
        #accum = copy.copy(element)
        accum = element
        order = 1
        while accum != identity:
            order += 1
            accum += element
        return order


#def mulorder_(element: Ring[Any], possible_orders: list[int]) -> int:
def mulorder_(element: Ring, possible_orders: list[int]) -> int:
    """Helper for mulorder that accepts a list of possible orders.

    Args:

        element: An immutable element of a multiplicative group.

        possible_orders: List of possible orders sorted and in
            increasing order.

    Returns:

        (int). The multiplicative order.
    """
    identity = element**0
    if element == identity:
        return 1
    #accum = copy.copy(element)
    accum = element
    prev_divisor = 1
    for divisor in possible_orders[:-1]:
        accum *= cast(Ring, element ** (divisor - prev_divisor))
        if accum == identity:
            return divisor
        prev_divisor = divisor
    return possible_orders[-1]


#def mulorder(element: Ring[Any], exponent: Optional[int] = None) -> int:
def mulorder(element: Ring, exponent: Optional[int] = None) -> int:
    """Return the order of element.

    Args:

        element: An immutable element in an multiplicative group.

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
        #accum = copy.copy(element)
        accum = element
        order = 1
        while accum != identity:
            order += 1
            accum *= element
        return order


#def iproduct(*iterables: Iterable, repeat: int = 1) -> Generator[tuple[Any, ...], None, None]:
def iproduct(*iterables: Any, repeat: int = 1) -> Generator[tuple[Any, ...], None, None]:
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
    iterables_ = [
        item
        for row in zip(*(tee(iterable, repeat) for iterable in iterables))
        for item in row
    ]
    N = len(iterables_)
    saved: list[list[Any]] = [[] for _ in range(N)]  # All the items we've seen.
    exhausted: set[int] = set() # Set of indices of iterables that're exhausted.
    for i in cycle(range(N)):
        if i in exhausted:  # Just to avoid repeatedly hitting that exception.
            continue
        try:
            item = next(iterables_[i])
            yield from product(*saved[:i], [item], *saved[i + 1 :]) # Finite prod.
            saved[i].append(item)
        except StopIteration:
            exhausted.add(i)
            if (
                not saved[i] or len(exhausted) == N
            ):  # Product is empty or all iterables exhausted.
                return
    yield ()  # There are no iterables.

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
