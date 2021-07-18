from copy import copy
from typing import NewType, Type
from numlib import isprime, gcd, xgcd, iproduct
from polylib import Polynomial, FPolynomial

__author__ = "Scott Simmons"
__version__ = "0.1"
__status__ = "Development"
__date__ = "06/23/21"
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

ModularInt = NewType("ModularInt", int)

def Zmod(n: int, mp=False, prime=False, negatives=False) -> Type[int]:
    """Quotient the integers by the principal ideal generated by n.

    This returns a class that can be used to work in the ing Z/p, the
    integers modulo the prime p.  In other words, instances of this class
    are elements of the ring Z/p.

    Args:

        n (int): generator of the principal ideal by which we quotient.

        mp (bool): whether to use multiprecision integers (requires the
            gmpy2 module).

        prime (bool): for large n, put True here if n is known to be
            prime; doing so offers speed up and can avoid errors.

        negatives (bool): whether to balance representations by using
            negative numbers; i.e., -1 instead of n-1, etc.

    Returns:

        (type). A class, instances of which represent the equivalence
            classes of integers modulo the ideal <n>.

    Examples:

        Example 1: Z/n, the ring of integers modulo a composite n

        >>> Zn = Zmod(143)  # the ring integers modulo 143
        >>> print(Zn)
        Z/143
        >>> Zn(2)**8
        113 + <143>
        >>> Zn(3)/Zn(4)
        108 + <143>
        >>> Zn(13)**-1  # Z/13 has zero divisors
        Traceback (most recent call last):
        AssertionError: 13 is not invertible modulo 143
        >>> Zn(3)/Zn(11)
        Traceback (most recent call last):
        AssertionError: 11 is not invertible modulo 143
        >>> len(list(Zn))  # Zn is a class but also an iterator
        143
        >>> len(list(Zn.units()))  # Zn.units() is also an iterator
        120
        >>> Zn.isField()
        False
        >>> Zn(142) == -1
        True

        Example 2: the Galois field of integers modulo a prime

        >>> GF = Zmod(43)  # the field integers modulo 43
        >>> print(GF)
        Z/43
        >>> GF.isField()
        True

        Let us find a generator of the multiplicative group of units:

        >>> def order(x):
        ...     for j in range(1, 43):
        ...         if x**j == 1:
        ...             return j
        >>> for x in GF.units():
        ...     if order(x) == 42:
        ...         print(x, "is a generator")
        ...         break
        3 is a generator
    """
    global ZModPrime, Z_Mod

    if not isinstance(n, int):
        raise TypeError(f"n must be of type integer, not {type(n)}")
    if n <= 0:
        raise ValueError(f"n must be a positive integer, not {n}")

    #if not mp:

    class Z_Mod_(int):

        def __new__(cls, value: int):
            #return super(cls, cls).__new__(cls, value % n)

            #value = value % n
            #value = value - n if negatives and n//2 + 1 <= value < n else value
            # below is equivalent to above but doesn't take mod unless necessary;
            # does not # appear to be faster
            if negatives:
                half = - ((n - 1) // 2)
                if value < half or value >  - half + (n + 1) % 2:
                    value = value % n
                    if value >= n//2 + 1:
                        value = value - n
            elif value < 0 or value >= n:
                value = value % n

            return super(Z_Mod_, cls).__new__(cls, value)

        #def __new__(metacls, cls, bases, classdict, value):
        #    return super(metacls, metacls).__new__(metacls, value % n)

        def __add__(self, other):
            if isinstance(other, int):
                return self.__class__(super(Z_Mod_, self).__add__(other))
            else:
                return NotImplemented

        def __radd__(self, other):
            if isinstance(other, int):
                return self.__class__(super(Z_Mod_, self).__radd__(other))
            else:
                return NotImplemented

        def __neg__(self):
            return self.__class__(super(Z_Mod_, self).__neg__())

        def __sub__(self, other):
            if isinstance(other, int):
                return self.__class__(super(Z_Mod_, self).__sub__(other))
            else:
                return NotImplemented

        def __rsub__(self, other):
            if isinstance(other, int):
                return self.__class__(super(Z_Mod_, self).__rsub__(other))
            else:
                return NotImplemented

        def __mul__(self, other):
            if isinstance(other, int):
                return self.__class__(super(Z_Mod_, self).__mul__(other))
            else:
                return NotImplemented

        def __rmul__(self, other):
            if isinstance(other, int):
                return self.__class__(super(Z_Mod_, self).__rmul__(other))
            else:
                return NotImplemented

        def __pow__(self, m):
            if isinstance(m, int):
                if m >= 0:
                    return self.__class__(pow(int(self), m, n))
                else:
                    return self.__class__(1)/self.__class__(pow(int(self), -m, n))
            else:
                return NotImplemented

        def __truediv__(self, other):
            g, inv, _ = xgcd(int(other), n)
            assert g == 1 or g == -1, str(int(other))+" is not invertible modulo "+str(n)
            return self * inv

        def __rtruediv__(self, other):
            return self.__class__(other).__truediv__(self)

        def __eq__(self, other):
            return (int(self) - int(other)) % n == 0
            #return NotImplemented

        def __repr__(self):
            return super(Z_Mod_, self).__repr__() + " + <" + repr(n) +">"

        def __str__(self):
            return super(Z_Mod_, self).__repr__()  # for Python 3.9

    class Z_ModIterable(type):

        def __iter__(self, units_: bool = False):
            #for i in range(n//2 + 1, n//2 + n + 1) if negatives else range(n):
            for i in range(- ((n - 1) // 2) , n - ((n - 1) // 2) ) if negatives else range(n):
                if i == 0 and units_:
                    continue
                #if not units and not prime or gcd(i, n) == 1:
                if not units_ or self.__class__ == 'ZModPrime' or gcd(i, n) == 1:
                    #yield Z_Mod_(i)
                    yield self(i)

        def units(self):
            if n == 1:
                return []
            else:
                return self.__iter__(units_=True)

        @classmethod
        def isField(self) -> bool:
            return False

        @classmethod
        def __len__(self):
            return n

        @classmethod
        def __str__(self):
            return "Z/"+str(n)

        @classmethod
        def __repr__(self):
            return "Z/"+str(n)

    class Z_Mod(Z_Mod_, metaclass = Z_ModIterable):

        def isinvertible(self):
            return self != 0 and gcd(self, n) == 1

    class ZModPrime(Z_Mod_, metaclass = Z_ModIterable):

        def __truediv__(self, other):
            return self * xgcd(int(other), n)[1]

        def __rtruediv__(self, other):
            return self.__class__(other).__truediv__(self)

        def isinvertible(self):
            return self != 0

        def __hash__(self):
            return hash((int(self), n))

        @classmethod
        def isField(self) -> bool:
            return True

        @classmethod
        def char(self):
            return n

    return ZModPrime if prime or isprime(n) else Z_Mod

def Pmod(monic: Polynomial) \
        -> Type[Polynomial]:
    """Quotient a univariate polynomial ring by a principal ideal.

    If the polynomial ring consists of polynomials with coefficients in
    a field, consider calling FPmod instead of this.

    Args:

        monic: A monic polynomial with coefficients in some ring, R.

    Returns:

        (type). A class whose constructor yields elements of the quotient
            ring R[x]/<monic>, the polynomial ring over R modulo the ideal
            generated by monic.

            Note: the string for the indeterminant used to represent ele-
                  ments of the returned quotient ring is that of monic.

    Examples:

        (See, also, the examples in the docstring for FPmod)

        >>> from polylib import Polynomial # For polynomials over a ring.
        >>> i = Polynomial([0, 1], 'i') # The coeff. ring is Z, the integers.
        >>> monic = i**2 + 1  # With this monic polynomial, the quotient
        >>> GZ = Pmod(monic)  # ring is just the Gaussian integers, Z[i];
        >>> i = GZ(i)    # This is the same as i = GZ([0, 1], 'i').
        >>> print((1 + 2*i)**3)
        -11 - 2i
        >>> (5-i)/(3-i+i**3)
        Traceback (most recent call last):
        TypeError: unsupported operand type(s) for /: 'Pmod_' and 'Pmod_'
    """
    global Pmod_

    class Pmod__(type):

        def __iter__(self):
            for coeffs in iproduct(fpoly[0].__class__, repeat=fpoly.degree()):
                yield(self(coeffs))

        @classmethod
        def __str__(self):
            if monic[0].__class__.__name__ in {'ZModPrime', 'Zmod'}:
                class_string = monic[0].__class__.__class__.__str__()
            else:
                class_string = monic[0].__class__.__name__
            return f"{class_string}[{monic.x}]/<{monic}>"

        @classmethod
        def order(self):   # is this ok over a non-field??
            if hasattr(monic[0].__class__,'__len__'):
                return len(monic[0].__class__) ** monic.degree()
            else:
                return None

    class Pmod_(Polynomial, metaclass = Pmod__):

        def __init__(self, coeffs, x = None, spaces = True, increasing = False):

            if not (isinstance(coeffs, Polynomial) or hasattr(type(coeffs), '__iter__')):
                raise ValueError(
                    "The argument to coeffs must be an iterable (e.g., a list or a "
                    f"tuple) not a {type(coeffs).__name__}. Try wrapping {coeffs} in "
                    f"square brackets."
                )

            if x and not (x == monic.x or x == f"({monic.x})" or x == f"[{monic.x}]"):
                raise ValueError(
                    f"The indeterminant string must be {monic.x}, ({monic.x}), "
                    f"or [{monic.x}], not {x}."
                )

            if len(coeffs) == 0:
                raise ValueError("coeffs cannot be empty")

            if monic._degree is None:
                raise ValueError("no need to quotient by <0>")
            one = (monic**0)[0]
            poly = Polynomial([one*elt for elt in coeffs], x = monic.x, spaces = monic.spaces) % monic
            super().__init__(poly._coeffs, x = x if x else monic.x, spaces =  monic.spaces)

        def __eq__(self, other):
            return ((self - other) % monic)._degree is None


        def __hash__(self):
            return hash((self._coeffs, monic._coeffs))

        def __repr__(self):
            monic_ = copy(monic)
            monic_.x = self.x
            nomic_.x_unwrapped = self.x_unwrapped
            monic_.spaces = self.spaces
            monic_.increasing = self.increasing
            s = str(monic_)
            if s[0] == "(" and s[-1] == ")":
                s = s[1:-1]
            return f"<{irred}>" if self._degree is None  else f"{self} + <{s}>"

    return Pmod_

def FPmod(fpoly: FPolynomial) -> Type[FPolynomial]:
    """Quotient a univariate polynomial ring over a field by a principal ideal.

    Args:

        monic: A monic polynomial with coefficients in some field, K.

    Returns:

        (type). A class whose constructor yields elements of the quotient
            ring K[x]/<monic>, the polynomial ring over K modulo the ideal
            generated by monic.

            Note: the string for the indeterminant used to represent ele-
                  ments of the returned quotient ring is that of monic.

    Examples:

        >>> from polylib import FPolynomial  # defines polys over a field

        Example 1: constructing the Gaussian rationals

        >>> from fractions import Fraction as F
        >>> x = FPolynomial([0, F(1)]) # builds Q[x], polys over rationals
        >>> monic = x**2 + 1   # an (irreducible) monic polynomial
        >>> GQ = FPmod(monic)  # Quotient Q[x] by ideal generated by monic
        >>> print(GQ)   # the quotient is just the Gaussian rationals
        Fraction[x]/<1 + x^2>

        One way to now define an element of the Gaussian rationals:

        >>> elt = FPolynomial([F(2,3), -F(5,4)])
        >>> print(elt)
        2/3 - 5/4x

        A better way is to use an indeterminant called 'i':

        >>> i = FPolynomial([0, F(1)], 'i')
        >>> GQ = FPmod(i**2 + 1)
        >>> i = GQ(i)
        >>> elt = F(2,3) - F(5,4)*i
        >>> print(elt)
        2/3 - 5/4i
        >>> elt
        2/3 - 5/4i + <1 + i^2>

        Now we can compute in the Gaussian rationals:
        >>> print(elt**-1)
        96/289 + 180/289i
        >>> print((1+2*i)**3)
        -11 - 2i
        >>> print((5-i)/(3-i+i**3))
        17/13 + 7/13i

        Example 2: constructing a Galois field

        >>> PF = Zmod(5)  # the prime field, Z/5
        >>> t = FPolynomial([0, PF(1)],'t') # coefficients in Z/5
        >>> monic = 2-t+t**3 # irreducibe - primitive, in fact - over Z/5
        >>> GF = FPmod(monic)# GF(5^3) represented as: Z/5[t]/<2-t+t^3>
        >>> GF.order()
        125
        >>> len({elt for elt in GF})  # GF is also an iterator
        125
        >>> GF.char()
        5

        GF is a class, the constructor of which returns elements of our im-
        plementation of GF(5^3) which, as a set - and, in fact, as a vector
        space over our implemenation of the prime field, Z/5 - can be rep-
        resented as all polynomials over Z/5 of degree 2 or less.

        To convieniently work in GF, let us define an indeterminant from
        which to build elements.

        >>> t = GF(t, 't')  # equivalently, t = GF([0, 1], 't')

        Now t is an element of the quotient ring Z/5[t]/<2-t+t^3>: it is
        the element t + <2-x+x^3>, or t + <monic>, in general.

        So t is an element of GF(5^3) that, of course, is a root of the
        cubic polynomial monic:

        >>> monic.of(t)  #  0 + <2 + 4t + t^3> = <2 + 4t + t^3>
        <2 + 4t + t^3>

        Morevover, monic is a polynomial with coeffiecients in GF(5^3)'s
        prime field; hence, {1, t, t^2} is a basis for GF(5^3) over GF(5)
        = Z/5.

        But the polynomial monic above is primitive, which means that t
        generates the multiplicative group of units in GF(5^3); hence, we
        can also write GF(5^3) = {0, 1, t, t^2, ..., t^(5^3-2}. Let us
        verify that t has order 5^3-1.

        >>> for j in range(1, 5**3):
        ...     if t**j == 1:
        ...         print(f"t has order {j}")
        t has order 124

        Example 3: quotient by a non-maximal ideal

        >>> PF = Zmod(5)  # the prime field
        >>> t = FPolynomial([0, PF(1)], 't') # indeterminant for Z/5[t]
        >>> p1 = 2-t+t**3  # this is primitive polynomial for GF(5^3)
        >>> p2 = 3+3*t+t**2  # this is primitive polynomial for GF(5^2)
        >>> QuotientRing = FPmod(p1*p2)  # this has zero divisors

        Here, we quotient Z/5[x] by the ideal generated by the reducible
        polynomial p1*p2 = (2-x+x^3)(2+x^2). However, each factor is irr-
        educible in Z/5[x], and the factors are relatively prime:

        >>> from numlib import xgcd, lcm
        >>> g, m1 , m2 = xgcd(p1, p2)
        >>> g
        FPolynomial((1 + <5>,))

        Hence, by the Chinese Remainder Theorem, QuotientRing is the dir-
        ect product GF(5^3) x GF(5^2) of Galois fields,

           Z/5[x]/<p1*p2> = Z/5[x]/<p1> x Z/5[x]/<p2>
                           = Z/5[x]/<2-x+x^3> x Z/5[x]/<3+3x+x^2>.

        The decomposition into a product is equivalent to a decomposi-
        tion as a direct sum of ideals in Z/5[x]/<p1*p2>.

        Such a direct sum decomposition corresponds exactly to an idem-
        potent element e of Z/5[x]<p1*p2>. Now, we can take e = m2*p2
        which, from Bezout's identy, 1=m1*p1+m2*p2, is idempotent since
        m2*p2 - (m2*p2)^2 = m1*m2*p1*p2 is in <p1*p2>; then 1-e = m1*p1,
        which is also idempotent. Then the direct sum decomposition is

          Z/5[x]/<p1*p2> = e * Z/5[x]/<p1*p2> + (1-e) * Z/5[x]/<p1*p2>
                         = <m2*p2>/<p1*p2> + <m1*p1>/<p1*p2>

        The multiplicative identity in the left summand is e = m2*p2;
        in the right, it is 1-e = m1*p1.

        Recall that p1, respectively p2, is a primitive polynomial for
        GF(5^3), resp. GF(5^2). Let us define t as an element of the
        quotient ring GF(5^5).

        >>> t = QuotientRing(t, 't')  # indeterminant for QuotientRing

        Explicitly, we have, for e and 1-e,

        >>> e = (m2*p2).of(t)
        >>> e; 1-e
        4 + t + 4t^3 + <1 + 3t + 4t^2 + 2t^3 + 3t^4 + t^5>
        2 + 4t + t^3 + <1 + 3t + 4t^2 + 2t^3 + 3t^4 + t^5>

        Since (e*t)^n = e*t^n, the order of e*t in the left summand,
        <m2*p2>/<p1*p2> is 5^3-1:

        >>> def order(elt, multiplicative_id = 1):
        ...     for j in range(1, 5**5):
        ...         if elt**j == multiplicative_id:
        ...             return j
        >>> order(e*t, multiplicative_id = e) == 5**3-1
        True

        Similarly, the order of (1-e)*t in <m1*p1>/<p1*p2> is 5^2-1:

        >>> order((1-e)*t, multiplicative_id = 1-e) == 5**2-1
        True

        There are 5^2 multiples of p1 = 2-x+x**3 with degree <= 5; like-
        wise, there are 5^3 multiples of p2 = 3+3*x+x**2 with degree <-5.
        There are, then, 5^2 + 5^3 - 1 zero divisors in GF(5^3) x GF(5^2)
        (since we counted zero twice); so that the unit group has order
        5^5 - 5^2 - 5^3 + 1 = (5^3-1) * (5^2-1) = 2976. The unit group in
        GF(5^3) x GF(5^2) is not cyclic: gcd(5**2-1, 5**3-1) = 5-1 = 4.

        The order of t is:

        >>> order(t) == lcm(5**3-1, 5**2-1) == 744
        True

        Example 4: extension of an extension field

        >>> from polylib import FPolynomial

        >>> from numlib import Zmod, FPmod
        >>> PF = Zmod(5)
        >>> t = FPolynomial([0,PF(1)], 't')
        >>> GF = FPmod(1+t**2+t**3)
        >>> GF
        Z/5[t]/<1 + t^2 + t^3>
        >>> t = GF([0, 1], 't')
        >>> t**3
        4 + 4t^2 + <1 + t^2 + t^3>

        >>> from numlib import GaloisField
        >>> GF = GaloisField(5,3)
        >>> GF
        Z/5[t]/<t^3+t^2+2>
        >>> t = GF.t()
        >>> t**3
        -t^2-2 + <t^3+t^2+2>

        >>> #F = FPmod(3-t)
        >>> #F

        >>> #F = FPmod(GF([3])-GF([1])*x)  error here, maybe correctly?
        >>> #F

        >>> #s = F([0, 1], 's')
        >>> #s

        >>> #s**2
    """
    global FPmod_

    class FPmod__(type):

        def __iter__(self):
            for coeffs in iproduct(fpoly[0].__class__, repeat=fpoly.degree()):
                yield(self(coeffs))

        @classmethod
        def __str__(self):
            if fpoly[0].__class__.__name__ in {'ZModPrime', 'Zmod'}:
                class_string = fpoly[0].__class__.__class__.__str__()
            else:
                class_string = fpoly[0].__class__.__name__
            return f"{class_string}[{fpoly.x}]/<{fpoly}>"

        @classmethod
        def __repr__(self):
            return self.__str__()

        @classmethod
        def order(self):   # is this a good idea?
            if hasattr(fpoly[0].__class__,'__len__'):
                return len(fpoly[0].__class__) ** fpoly.degree()
            else:
                return None

        @classmethod
        def char(self):
            return fpoly[-1].char()

    class FPmod_(FPolynomial, metaclass = FPmod__):

        def __init__(self, coeffs, x = None, spaces = True, increasing = False):

            if not (isinstance(coeffs, Polynomial) or hasattr(type(coeffs), '__iter__')):
                raise ValueError(
                    "The argument to coeffs must be an iterable (e.g., a list or a "
                    f"tuple) not a {type(coeffs).__name__}. Try wrapping {coeffs} in "
                    f"square brackets."
                )

            if x and not (x == fpoly.x or x == f"({fpoly.x})" or x == f"[{fpoly.x}]"):
                raise ValueError(
                    f"The indeterminant string must be {fpoly.x}, ({fpoly.x}), "
                    f"or [{fpoly.x}], not {x}."
                )

            if len(coeffs) == 0:
                raise ValueError("coeffs cannot be empty")

            if fpoly._degree is None:
                raise ValueError("no need to quotient by <0>")
            one = (fpoly**0)[0]
            poly = FPolynomial([one*elt for elt in coeffs], x = fpoly.x, spaces = fpoly.spaces, increasing = fpoly.increasing) % fpoly
            super().__init__(poly._coeffs, x = x if x else fpoly.x, spaces = fpoly.spaces, increasing = fpoly.increasing)

        def __eq__(self, other):
            return ((self - other) % fpoly)._degree is None

        def __hash__(self):
            return hash((self._coeffs, fpoly._coeffs))

        def __repr__(self):
            irred = copy(fpoly)
            irred.x = self.x
            irred.x_unwrapped = self.x_unwrapped
            irred.spaces = self.spaces
            irred.increasing = self.increasing
            s = str(irred)
            if s[0] == "(" and s[-1] == ")":
                s = s[1:-1]
            return f"<{irred}>" if self._degree is None  else f"{self} + <{s}>"

        def __truediv__(self, other):
            if isinstance(other, int):
                return (other * (self**0)[0])**-1 * self
            elif isinstance(other, FPolynomial):
                gcd, inv, _ = xgcd(other, fpoly)
                if not gcd._degree == 0:
                    raise ValueError(f"{other} is not invertible modulo {fpoly}")
                return self * inv * (1/gcd[0])
            else:
                return NotImplemented

        def __rtruediv__(self, other):
            return self.__class__([other], self.x, self.spaces, self.increasing) / self

        def __pow__(self, m):
            if m < 0:
                assert self != 0, "cannot invert 0"
                return super(FPmod_, 1/self).__pow__(-m)
            else:
                #print("here",self.__class__.__name__)
                return super(FPmod_, self).__pow__(m)

    return FPmod_

def GaloisField(p: int, r: int = 1, negatives = True) -> Type[FPolynomial]:
    """Return an implemention a Galois field of order p^n.

    Rather than calling this with n = 1 to implement GF(p), you may pre-
    fer to call Zmod like GF = Zmod(p); or, if p is a large prime, like
    GF = Zmod(p, prime=True). The constructor of a class GF defined by
    means of Zmod will return essentially ints.

    Alternatively, if you are, say, writing a program that wants
    GF(p) or any Galois field GF(p^r), all of the same type, then you
    may wish to call use this even with r=1.

    Note: t is always a generator of the unit group, even when r = 1.

    Args:

        p (int): a prime.

        r (int): a positive integer.

        negatives (bool): whether to use negative numbers when represent-
            ing large number in Z/p.

    Returns:

        (type). A class, instance of which are elements in the Galois
            field of order p^n represented as Z/p[t] modulo the ideal
            generated by a primitive polynomial (so the t generates the
            multiplicative group of units).

    Examples:

        If GF = GaloisField(p, r), then GF is a class (but think type or
        set). For any p and r, one can easily construct elements of the
        corresponding Galois field. But remember that such elements are
        (equivalences classes of) polynomials of degree r.

        >>> GF = GaloisField(5, 3)
        >>> GF   # elements are cubic polynomials over Z/5
        Z/5[t]/<t^3+t^2+2>

        To instantiate an element of Z/5[t]/<2 + t^2 + t^3> , we need to
        specify its coefficients as a polynomial. One way to to this is:

        >>> GF([3, 4, 17])
        2t^2-t-2 + <t^3+t^2+2>

        Alternatively, one can use a indeterminant:

        >>> t =  GF([0, 1])
        >>> -2 - t + 2*t**2
        2t^2-t-2 + <t^3+t^2+2>

        For convenience, the indeterminant is already available:

        >>> t = GF.t()
        >>> -2 - t + 2*t**2
        2t^2-t-2 + <t^3+t^2+2>

        More notes:

        GF = GaloisField(p, r) is a generator for the entire field:

        >>> print(', '.join(str(elt) for elt in GF)) # doctest: +ELLIPSIS
        -2t^2-2t-2, -2t^2-2t-1, -2t^2-t-2, ...

        For all p and r (including r=1), t is actually a generator for
        the multiplicative group of units of GF = GaloisField(p, r):

        >>> len({t**i for i in range(GF.order() - 1)})
        124

        When working with, say, elliptic curves, one may want to define
        a polynomial f(x) whose coefficients are in a Galois field. This
        is different that the examples above -- in those we defined poly-
        nomials that are elements of the Galois field.

        For this we will need to wrap our elements of GF in parenthesis
        if we want an unambiguous string version of f.

        >>> t = GF.t('(t)')

        Then, we can do something like this:

        >>> from polylib import FPolynomial
        >>> f = FPolynomial([1 + t**2, 2*t**0])
        >>> f
        FPolynomial(((t^2+1) + <t^3+t^2+2>, (2) + <t^3+t^2+2>))
        >>> print(f)
        (t^2+1) + (2)x

        But notice that we must use t**0; i.e., this is not the same:

        >>> FPolynomial([1 + t**2, 2])
        FPolynomial(((t^2+1) + <t^3+t^2+2>, 2))

        Perhaps more safely, we can do this:

        >>> FPolynomial([GF([1, 0, 1]), GF([2])])
        FPolynomial((t^2+1 + <t^3+t^2+2>, 2 + <t^3+t^2+2>))

        Alternatively, we can create an indeterminant for x:

        >>> from polylib import Polynomial
        >>> x = Polynomial([0, 1])
        >>> p = GF([1, 0, 1], '(t)') * x**0 + GF([2], '(t)') * x
        >>> p
        Polynomial(((t^2+1) + <t^3+t^2+2>, (2) + <t^3+t^2+2>))
        >>> print(p)
        (t^2+1) + (2)x

        or, this:

        >>> print((1 + t**2)*x**0 + (2*t**0)*x)
        (t^2+1) + (2)x

        More examples:

        Case: r = 1

        To demonstrate this case, let us not us negative representations.

        >>> GF = GaloisField(17, negatives = False)
        >>> GF
        Z/17[t]/<t+14>
        >>> GF([10])  # note the you need brackets around the 10
        10 + <t+14>
        >>> print(GF([10]))
        10
        >>> GF([10])**-1
        12 + <t+14>
        >>> print(GF([11]) + GF([12]))
        6
        >>> print(GF([11]) * GF([12]))
        13
        >>> t = GF((0, 1))  # t is a generator of the unit group
        >>> print(', '.join(str(t**i) for i in range(1,17)))
        3, 9, 10, 13, 5, 15, 11, 16, 14, 8, 7, 4, 12, 2, 6, 1

        So t is a generator when r = 1, as in all other cases. But if
        you with to work with single elements of GF (as opposed to say
        all units), you may end up doing:

        >>> 9*t**0
        9 + <t+14>

        Alternatively, you may want simply:

        >>> GF = Zmod(17)
        >>> print(GF)
        Z/17
        >>> GF(9)
        9 + <17>

        Case: r > 1

        >>> GF = GaloisField(2, 4)
        >>> print(GF)
        Z/2[t]/<t^4+t^3+1>

        One can define elements of GF like this:

        >>> GF([1, 2, 345, 0, 0, 0, 1])
        t^3+t + <t^4+t^3+1>

        Or, perhaps more conveniently, by using an indeter-
        minant

        >>> t = GF.t() # same as t=GF([0, 1])
        >>> 1 + 2*t + 345*t**2 + t**6
        t^3+t + <t^4+t^3+1>

        One can find all generators for the unit group:

        >>> from numlib import iproduct
        >>> def order(x):
        ...     for j in range(1, 2 ** 4):
        ...         if x**j == 1:
        ...             return j
        >>> generators = []
        >>> for coeffs in iproduct(Zmod(2), repeat=4):
        ...     elt = GF(coeffs)
        ...     if order(elt) == 2 ** 4 - 1:
        ...         generators.append(str(elt))
        >>> print(', '.join(generators))
        t, t^2, t^2+t, t^2+t+1, t^3+t^2, t^3+t^2+t, t^3+1, t^3+t^2+1

        Since t is a generator, 1 + t^3 + t^4 is a primitive polynomial
        for GF(2^4).
    """
    if not isinstance(p, int):
        raise TypeError(f"p must be of type integer, not {type(p)}")
    if not isprime(p):
        raise ValueError("p must be prime")
    if not isinstance(r, int):
        raise TypeError(f"r must be of type integer, not {type(r)}")
    if r <= 0:
        raise ValueError(f"r must be a positive integer, not {r}")

    t = FPolynomial([0, Zmod(p, negatives = negatives)(1)], 't', spaces = False, increasing = False)
    if r == 1:
        # Find the first generator of (Z/p)*:
        a = 1
        for a in range(2, p):
            for i in range(1, p - 1):
                if a**i % p == 1:
                    break
            if i == p - 2:
                break
        irred = t - a*t**0
    elif p == 2:
        if r == 2:
            irred = 1 + t + t**2
        elif r == 3:
            irred = 1 + t**2 + t**3
        elif r == 4:
            irred = 1 + t**3 + t**4
        elif r == 5:
            irred = 1 + t**3 + t**5
        else:
            return NotImplemented
    elif p == 5:
        if r == 2:
             irred = 2 + t + t**2
        elif r == 3:
             irred = 2 + t**2 + t**3
        elif r == 4:
             irred = 2 - t + t**2 - t**4
        else:
            return NotImplemented
    elif p == 7:
        if r == 2:
             irred = 3 - t + t**2
        elif r == 3:
             irred = 4 + 3*t**2 + t**3
        elif r == 4:
             irred = 2 - t - 3*t**2 + 3*t**4
        else:
            return NotImplemented
    elif p == 23:
        if r == 2:
             irred = -4 - t + t**2
        elif r == 3:
             irred = -11 - t + t**3
        else:
            return NotImplemented
    elif p == 31:
        if r == 2:
             irred = 3 - 7*t + t**2
        elif r == 3:
             irred = -12 - 11*t + t**3
        else:
            return NotImplemented
    elif p == 43:
        if r == 2:
            irred = -20 - t + t**3
        if r == 3:
            irred = -19 + 16*t + t**3
        else:
            return NotImplemented
    elif p == 71:
        if r == 2:
            irred = 7 + 7*t + t**2
        elif r == 3:
            irred = 2 + 9*t + t**3
        else:
            return NotImplemented
    elif p == 113:
        if r == 2:
            irred = 10 + t + t**2
        elif r ==3:
            irred = -55 - 50*t + t**3
        else:
            return NotImplemented
    elif p == 503:
        if r == 2:
            irred = -201 - t + t**2
        else:
            return NotImplemented
    else:
        return NotImplemented

    GF = FPmod(irred)

    def indeterminant(x = irred.x):
        #return GF([0, 1], x)
        return GF(t, x)

    GF.t = indeterminant

    return GF

def squareroot(n: int) -> str:
    """Return string vertion of int prepended with a unicode radical symbol.

    Useful for pretty printing elements of, for example, a quadratic field.

    Example:

        >>> print(squareroot(5))
        √5

    """
    return u"\u221A"+str(n)

if __name__ == "__main__":

    import doctest

    doctest.testmod()

