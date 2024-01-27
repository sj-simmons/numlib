"""
The functions included here can be used to create various rings.

    Zmod - constructs the ring of integers modulo n
    Zmodp - constructs the field integers modulo a prime p
    GaloisField - constructs a finite field of order p^r

For instance, one can define the (Galois) field of order 7^3:

    >>> from numlib import Zmodp
    >>> GF = Zmodp(17)
    >>> a = GF(3); a
    3 + <17>
    >>> a ** -1
    6 + <17>

See the docstrings for each function for more examples; e.g., at your
commandline type:

    pydoc3 numlib.GaloisField

Each of the three functions listed above returns a class. In turn, each
class has a method called indet that returns an indeterminant that can
be used to build polynomials over that ring or field. E.g.,

    >>> x = GF.indet()
    >>> f = 3*x**2-7*x+1; print(f); print(type(f))
    3x^2-7x+1
    <class 'polylib.polynomial.FPolynomial'>

One can use the following two functions, both included here, to real-
ize various rings and fields as quotients of polynomial rings.

    PMod - quotient the ring of polynomials R[x] with coefficients
           in ring R by any monic polynomial with coeffiecients in
           that ring, leading in general to another ring.
    FPMod - quotient a polynomial ring F[x] with coefficients in a
            field F by a polynomial with coefficients in F; if the
            polynomial is irreducible, the quotient is a field.

For instance, since f defined above is irreducible over Z/17:

    >>> all(f(a) != 0 for a in GF)
    True

we can define a field of order 17^2 by:

    >>> FPmod(f)
    Z/17[x]/<3x^2-7x+1>
"""
from __future__ import annotations
import copy
import random
import re
from typing import Type, TypeVar, Optional, Sequence, cast, Callable, Iterator, Generic, overload, Iterable, Tuple
from polylib.polynomial import Polynomial, FPolynomial, Ring, Field
from numlib.utils import gcd, xgcd, divisors, mulorder, mulorder_, iproduct

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

#  NOTE: Always multiply polynomials on the right by instance of ZMod or ZModP

class ZModMeta(type, Iterable['ZMod']):
    #char: int
    order: Tuple[int, int]
    @classmethod
    def __iter__(cls, units_: bool = False) -> Iterator['ZMod']: ...
    @classmethod
    def units(cls) -> Iterator['ZMod']: ...
    #@classmethod
    #def __len__(cls) -> int: ...
    @classmethod
    def __str__(cls) -> str: ...
    @classmethod
    def __repr__(cls) -> str: ...
    @classmethod
    def indet(cls, indet: str) -> Polynomial['ZMod']: ...

class ZMod(int, metaclass=ZModMeta):
    def __add__(self, other: int|'ZMod') -> 'ZMod': ...
    def __radd__(self, other: int) -> 'ZMod': ...
    def __sub__(self, other: int|'ZMod') -> 'ZMod': ...
    def __rsub__(self, other: int) -> 'ZMod': ...
    def __neg__(self) -> 'ZMod': ...
    def __mul__(self, other: int|'ZMod') -> 'ZMod': ...
    def __rmul__(self, other: int) -> 'ZMod': ...
    def __pow__(self, m: int) -> Optional['ZMod']: ... # type: ignore[override]
    def __truediv__(self, other: int|'ZMod') -> Optional['ZMod']: ... # type: ignore[override]
    def isunit(self) -> bool: ...

class ZModPMeta(type, Iterable['ZModP']):
    #char: int
    order: Tuple[int, int]
    @classmethod
    def __iter__(cls, units_: bool = False) -> Iterator['ZModP']: ...
    @classmethod
    def units(cls) -> Iterator['ZModP']: ...
    @classmethod
    #def __len__(cls) -> int: ...
    #@classmethod
    def __str__(cls) -> str: ...
    @classmethod
    def __repr__(cls) -> str: ...
    @classmethod
    def indet(cls, indet: str) -> FPolynomial['ZModP']: ...

class ZModP(int, metaclass=ZModPMeta):
    def __add__(self, other: int|'ZModP') -> 'ZModP': ...
    def __radd__(self, other: int) -> 'ZModP': ...
    def __sub__(self, other: int|'ZModP') -> 'ZModP': ...
    def __rsub__(self, other: int) -> 'ZModP': ...
    def __neg__(self) -> 'ZModP': ...
    def __mul__(self, other: int|'ZModP') -> 'ZModP': ...
    def __rmul__(self, other: int) -> 'ZModP': ...
    def __pow__(self, m: int) -> 'ZModP': ... # type: ignore[override]
    def __truediv__(self, other: int|'ZModP') -> 'ZModP': ...
    def __rtruediv__(self, other: int) -> 'ZModP': ... # type: ignore[misc]
    def isunit(self) -> bool: ...

#class ZModP_(ZModP, metaclass=ZModPMeta):

R = TypeVar('R', bound = Ring)
F = TypeVar('F', bound = Field)

class PModMeta(type):
    @classmethod
    def __str__(cls) -> str: ...
    @classmethod
    def __repr__(cls) -> str: ...

class PMod(Generic[R], metaclass=PModMeta): ...
    #def __init__(self, *args, **kwargs): ...
    #def __add__(self, other: Union[R, int, 'PMod[R]']) -> 'PMod[R]': ...
    #def __radd__(self, other: Union[R, int]) -> 'PMod[R]': ...
    #def __sub__(self, other: Union[R, int, 'PMod[R]']) -> 'PMod[R]': ...
    #def __rsub__(self, other: Union[R, int]) -> 'PMod[R]': ...
    #def __neg__(self) -> 'PMod[R]': ...
    #def __mul__(self, other: R|int|'PMod[R]']) -> 'PMod[R]': ...
    #def __rmul__(self, other: Union[R, int]) -> 'PMod[R]': ...
    #def __pow__(self, m: int) -> Optional['PMod[R]']: ... # type: ignore[override]

class FPModMeta(type):
    @classmethod
    def __str__(cls) -> str: ...
    @classmethod
    def __repr__(cls) -> str: ...

class FPMod(Generic[F], metaclass=FPModMeta):
    def __init__(self, coeffs: Sequence[F], x: Optional[str] = ...) -> None: ...
    #indet: # NOTE: implement this here and in Pmod and GF_PFMod??
    # NOTE:  This (and Pmod) should probably induce F (and R)
    #def __init__(self, *args, **kwargs): ...
    #def __add__(self, other: Union[F, int, 'FPMod[F]']) -> 'FPMod[F]': ...
    #def __radd__(self, other: Union[F, int]) -> 'FPMod[F]': ...
    #def __sub__(self, other: Union[F, int, 'FPMod[F]']) -> 'FPMod[F]': ...
    #def __rsub__(self, other: Union[F, int]) -> 'FPMod[F]': ...
    #def __neg__(self) -> 'FPMod[F]': ...
    def __mul__(self, other: F|int|'FPMod[F]') -> 'FPMod[F]': ...
    def __rmul__(self, other: F|int) -> 'FPMod[F]': ...
    def __pow__(self, m: int) -> 'FPMod[F]': ...
    #def __truediv__(self, other: Union[F, int, 'FPMod[F]']) -> 'FPMod[F]': ...
    #def __rtruediv__(self, other: Union[F, int]) -> 'FPMod[F]': ... # type: ignore[misc]

class GF_ZModPMeta(type, Iterable['GF_ZModP']):
    #char: int
    order: Tuple[int, int]
    def __iter__(self) -> Iterator['GF_ZModP']: ...
    @classmethod
    def __str__(cls) -> str: ...
    @classmethod
    def __repr__(cls) -> str: ...
    @classmethod
    def indet(cls, letter: str) -> FPolynomial['GF_ZModP']: ...

class GF_ZModP(metaclass=GF_ZModPMeta):
    #def __init__(self, *args, **kwargs): ...
    def __add__(self, other: int|'GF_ZModP') -> 'GF_ZModP': ...
    def __radd__(self, other: int) -> 'GF_ZModP': ...
    def __sub__(self, other: int|'GF_ZModP') -> 'GF_ZModP': ...
    def __rsub__(self, other: int) -> 'GF_ZModP': ...
    def __neg__(self) -> 'GF_ZModP': ...
    def __mul__(self, other: int|'GF_ZModP') -> 'GF_ZModP': ...
    def __rmul__(self, other: int) -> 'GF_ZModP': ...
    def __pow__(self, m: int) -> 'GF_ZModP': ...
    def __truediv__(self, other: int|'GF_ZModP') -> 'GF_ZModP': ...
    def __rtruediv__(self, other: int) -> 'GF_ZModP': ...
    def __eq__(self, other: int|'GF_ZModP') -> bool: ... # type: ignore[override]

def Zmod(n: int, mp: bool = False, negatives: bool = True,) -> Type[ZMod]:
    """Quotient the integers by the principal ideal generated by n.

    This returns a class whose instances are elements of Z/n, the ring
    of integers modulo the prime n.

    Args:

        n (int): the generator of the principal ideal by which we quot-
            ient.

        mp (bool): whether to use multiprecision integers (requires the
            gmpy2 module).

        negatives (bool): whether to balance representations by using
            negative numbers; i.e., -1 instead of n-1, etc.

    Returns:

        (type). A class, instances of which represent the equivalence
            classes of integers modulo the ideal <n>.

    Examples:

        Working in Z/143:

        >>> R = Zmod(143)  # the ring integers modulo 143
        >>> print(R)
        Z/143
        >>> R(2)**8
        -30 + <143>
        >>> R(3)*R(4)**-1
        -35 + <143>
        >>> print(R(13)**-1)  # Z/143 has zero divisors
        Traceback (most recent call last):
        AssertionError: 13 is not invertible modulo 143
        >>> sum(1 for _ in R) # R is a class but also a generator
        143
        >>> len(list(R.units()))  # Zn.units() is is a generator
        120
        >>> R(142) == -1
        True

        Working in polynomials over Z/143:

        >>> x = Zmod(143).indet() # same as x=polylib.Polynomial([R(0),R(1)])
        >>> print(3*x**2 + 5*x + 100)
        3x^2+5x-43
        >>> 3*x**2 + 5*x + 100
        Polynomial((-43 + <143>, 5 + <143>, 3 + <143>))

        >>> R = Zmod(143)
        >>> y = R.indet('y -')
        >>> print(3*y**2 + 5*y + 100)
        3y^2 + 5y - 43

        >>> y = Zmod(143, negatives=False).indet('y')
        >>> print(3*y**2+5*y+100)
        100+5y+3y^2
    """
    if not isinstance(n, int) or n < 1:
        raise TypeError(f"n must be a positive int, not {n} of {type(n)}")

    #class ZModMeta_(type):
    class ZModMeta_(ZModMeta):
        #char = n
        order = (n, 1)

        @classmethod
        def __iter__(cls, units_: bool = False) -> Iterator['ZMod']:
            # for i in range(n//2 + 1, n//2 + n + 1) if negatives else range(n):
            for i in (
                range(-((n - 1) // 2), n - ((n - 1) // 2)) if negatives else range(n)
            ):
                if i == 0 and units_:
                    continue
                if not units_ or abs(gcd(i, n)) == 1:
                    yield ZMod_(i)

        @classmethod
        def units(cls) -> Iterator['ZMod']:
            if n == 1:
                return iter([])
            else:
                return cls.__iter__(units_=True) # type: ignore[call-arg]

        #@classmethod
        #def __len__(cls) -> int:
        #    return n

        @classmethod
        def __str__(cls) -> str:
            return f"Z/{n}"

        @classmethod
        def __repr__(cls) -> str:
            return f"Z/{n}"

        @classmethod
        def indet(cls, indet: str = 'x-') -> Polynomial['ZMod']:
            return Polynomial([ZMod_(0), ZMod_(1)], x=indet)

    class ZMod_(ZMod, metaclass=ZModMeta_):
        if negatives:
            def __new__(cls: Type[ZMod], value: int) -> 'ZMod_':
                # # return super(cls, cls).__new__(cls, value % n)
                # if value is None:
                #     return Polynomial(
                #         [
                #             #super(Z_Mod_, cls).__new__(cls, 0),
                #             #super(Z_Mod_, cls).__new__(cls, 1),
                #             super().__new__(cls, 0),
                #             super().__new__(cls, 1),
                #         ],
                #         x=indet,
                #     )
                # # value = value % n
                # # value = value - n if negatives and n//2 + 1 <= value < n else value
                # # below is equivalent to above but doesn't take mod unless necessary;
                # # does not appear to be faster
                half = -((n - 1) // 2)
                if value < half or value > -half + (n + 1) % 2:
                    value = value % n
                    if value >= n // 2 + 1:
                        value = value - n

                #return super(Z_Mod_, cls).__new__(cls, value)
                return cast(ZMod_, ZMod.__new__(cls, value))
                #return ZMod.__new__(cls, value)
        else:
            def __new__(cls: Type[ZMod], value: int) -> 'ZMod_':
                if value < 0 or value >= n:
                    value = value % n
                return cast(ZMod_, ZMod.__new__(cls, value))

        # def __new__(metacls, cls, bases, classdict, value):
        #    return super(metacls, metacls).__new__(metacls, value % n)

        def __add__(self, other: int|'ZMod_') -> 'ZMod_':
            return ZMod_(super(ZMod, self).__add__(other))

        def __radd__(self, other: int) -> 'ZMod_':
            return ZMod_(super(ZMod, self).__radd__(other))

        def __neg__(self) -> 'ZMod_':
            return ZMod_(super(ZMod, self).__neg__())

        def __sub__(self, other: int|'ZMod_') -> 'ZMod_':
            return ZMod_(super(ZMod, self).__sub__(other))

        def __rsub__(self, other: int) -> 'ZMod_':
            return ZMod_(super(ZMod, self).__rsub__(other))

        def __mul__(self, other: int|'ZMod_') -> 'ZMod_':
            return ZMod_(super(ZMod, self).__mul__(other))

        def __rmul__(self, other: int) -> 'ZMod_':
            return ZMod_(super(ZMod, self).__rmul__(other))

        def __pow__(self, m: int) -> Optional['ZMod_']: # type: ignore[override]
            if m > 0:
                return ZMod_(pow(int(self), m, n))
            elif m < 0:
                g, inv, _ = xgcd(int(self), n)
                assert abs(g) == 1, f"{int(self)} is not invertible modulo {n}"
                #return ZMod_(pow(inv * g, -m, n)) if abs(g) == 1 else None
                return ZMod_(pow(inv * g, -m, n))
            else:
                return ZMod_(1)

        def __truediv__(self, other: int|'ZMod_') -> Optional['ZMod_']: # type: ignore[override]
            inv = ZMod_(other) ** -1
            return ZMod_(super(ZMod, self).__mul__(inv)) if inv else None

        def __rtruediv__(self, other: int) -> Optional['ZMod_']: # type: ignore[misc, override]
            inv = self ** -1
            return  ZMod_(inv * other) if inv else None

        def __eq__(self, other: int|'ZMod_') -> bool: # type: ignore[override]
            return (int(self) - int(other)) % n == 0

        # NOTE: do you need __eq__?  YES, AND ALSO __ne__
        def __ne__(self, other: int|'ZMod_') -> bool: # type: ignore[override]
            return (int(self) - int(other)) % n != 0

        def __hash__(self) -> int:
            return hash((int(self), n))

        def __repr__(self) -> str:
            return f"{super().__repr__()} + <{n}>"

        def __str__(self) -> str:
            return super().__repr__()  # for Python 3.9

        def isunit(self) -> bool:
            return self != 0 and abs(gcd(self, n)) == 1

    ZMod_.__name__ = f"Z/{n}"

    return ZMod_

def Zmodp(p: int, mp: bool =False, negatives: bool =True) -> Type[ZModP]:
    """Quotient the integers by the principal ideal generated by n.

    This returns a class that can be used to work in the ring of int-
    egers modulo the prime p: instances of the returned class are
    elements of the ring, Z/p.

    Args:

        p (int): a prime defining the ideal ideal by which we quotient.

        mp (bool): whether to use multiprecision integers (requires the
            gmpy2 module).

        negatives (bool): whether to balance representations by using
            negative numbers; i.e., -1 instead of n-1, etc.

    Returns:

        (type). A class, instances of which represent the equivalence
            classes of integers modulo the ideal <n>.

    Examples (see also the examples for Zmod):

        >>> F = Zmodp(43, negatives=False)  # the integers modulo 43
        >>> print(F)
        Z/43

        Let us find a generator of the multiplicative group of units:

        >>> def order(x):
        ...     for j in range(1, 43):
        ...         if x**j == 1:
        ...             return j
        >>> for x in F:
        ...     if order(x) == 42:
        ...         print(x, "is a generator")
        ...         break
        3 is a generator
    """
    if not isinstance(p, int) or p < 2:
        raise TypeError(f"p must be a positive prime int, not {p} of {type(p)}")

    class ZModPMeta_(ZModPMeta):

        order = (p, 1)

        @classmethod
        def __iter__(cls, units_: bool = False) -> Iterator['ZModP']:
            # for i in range(n//2 + 1, n//2 + n + 1) if negatives else range(n):
            for i in (
                range(-((p - 1) // 2), p - ((p - 1) // 2)) if negatives else range(p)
            ):
                if i == 0 and units_:
                    continue
                yield ZModP_(i)

        @classmethod
        def units(cls) -> Iterator['ZModP']:  # NOTE: REMOVE THIS?
            if p == 1:
                return iter([])
            else:
                return cls.__iter__(units_=True) # type: ignore[call-arg]

        #@classmethod
        #def __len__(cls) -> int:
        #    return p

        @classmethod
        def __str__(cls) -> str:
            return f"Z/{p}"

        @classmethod
        def __repr__(cls) -> str:
            return f"Z/{p}"

        @classmethod
        def indet(cls, indet: str = 'x-') -> FPolynomial['ZModP']:
            return FPolynomial([ZModP_(0), ZModP_(1)], x=indet)

    class ZModP_(ZModP, metaclass=ZModPMeta_):
        if negatives:
           def __new__(cls: type, value: int) -> 'ZModP_':
           #def __new__(cls: Type[ZModP], value: int) -> 'ZModP_':
               # return super(cls, cls).__new__(cls, value % n)
               # if value == ():
               #     return FPolynomial(
               #         [
               #             super(Z_Mod_, cls).__new__(cls, 0),
               #             super(Z_Mod_, cls).__new__(cls, 1),
               #         ],
               #         x=indet,
               #     )
               # # value = value % p
               # # value = value - p if negatives and p//2 + 1 <= value < p else value
               # # below is equivalent to above but doesn't take mod unless necessary;
               # # does not # appear to be faster
               half = -((p - 1) // 2)
               if value < half or value > -half + (p + 1) % 2:
                   value = value % p
                   if value >= p // 2 + 1:
                       value = value - p
               #return super(Z_Mod_, cls).__new__(cls, value)
               return cast(ZModP_, ZModP.__new__(cls, value))
        else:
           def __new__(cls: type, value: int) -> 'ZModP_':
               if value < 0 or value >= p:
                   value = value % p

               #return super(Z_Mod_, cls).__new__(cls, value)
               return cast(ZModP_, ZModP.__new__(cls, value))

        def __add__(self, other: int|'ZModP_') -> 'ZModP_':
            return ZModP_(super(ZModP, self).__add__(other))

        def __radd__(self, other: int) -> 'ZModP_':
            return ZModP_(super(ZModP, self).__radd__(other))

        def __neg__(self) -> 'ZModP_':
            return ZModP_(super(ZModP, self).__neg__())

        def __sub__(self, other: int|'ZModP_') -> 'ZModP_':
            return ZModP_(super(ZModP, self).__sub__(other))

        def __rsub__(self, other: int) -> 'ZModP_':
            return ZModP_(super(ZModP, self).__rsub__(other))

        def __mul__(self, other: int|'ZModP_') -> 'ZModP_':
            return ZModP_(super(ZModP, self).__mul__(other))

        def __rmul__(self, other: int) -> 'ZModP_':
            return ZModP_(super(ZModP, self).__rmul__(other))

        def __truediv__(self, other: int|'ZModP_') -> 'ZModP_':
            assert other != 0, "Cannot divide by zero."
            g, inv, _ = xgcd(int(other), p)
            return ZModP_(super(ZModP, self).__mul__(inv * g))

        def __rtruediv__(self, other: int) -> 'ZModP_': # type: ignore[misc]
            assert self != 0, "Cannot divide by zero."
            return ZModP_(other).__truediv__(self)

        def __pow__(self, m: int) -> None|'ZModP_': # type: ignore[override]
            if m > 0:
                return ZModP_(pow(int(self), m, p))
            elif m < 0:
                return ZModP_(1)/ZModP_(pow(int(self), -m, p))
            else:
                return ZModP_(1)

        def __eq__(self, other: int|'ZModP_') -> bool: # type: ignore[override]
            return (int(self) - int(other)) % p == 0

        def __ne__(self, other: int|'ZModP_') -> bool: # type: ignore[override]
            return (int(self) - int(other)) % p != 0

        def __hash__(self) -> int:
            return hash((int(self), p))

        def __repr__(self) -> str:
            return f"{super().__repr__()} + <{p}>"

        def __str__(self) -> str:
            return super().__repr__()  # for Python 3.9

        def isunit(self) -> bool:
            return self != 0

    ZModP_.__name__ = f"Z/{p}"

    return ZModP_

R1 = TypeVar('R1', bound=Ring, covariant=True)

def Pmod(monic: Polynomial[R]) -> Type[PMod[R]]:
    """Quotient a univariate polynomial ring by a principal ideal.

    If the polynomial ring consists of polynomials with coefficients in
    a field, consider calling FPmod instead of this.

    Args:

        monic (Polynomial): q monic polynomial with coefficients in some
            ring, R.

    Returns:

        (type). A class whose constructor yields elements of the quotient
            ring R[x]/<monic>, the polynomial ring over R modulo the ideal
            generated by monic.

            Note: the string for the indeterminant used to represent ele-
                  ments of the returned quotient ring is that of monic.

    Examples (see, also, the examples in the docstring for FPmod):

        The Gaussian integers:

        >>> from polylib import Polynomial # For polynomials over a ring.
        >>> i = Polynomial([0, 1], 'i') # The coeff. ring is Z, the integers.
        >>> monic = i**2 + 1  # With this monic polynomial, the quotient
        >>> GZ = Pmod(monic)  # ring is just the Gaussian integers, Z[i];
        >>> GZ
        int[i]/<1+i^2>
        >>> i = GZ(i)    # This is the same as i = GZ([0, 1], 'i').
        >>> print((2*i+1 )**3)
        -11-2i
        >>> (5-i)/(3-i+i**3) # GZ is not a division ring
        Traceback (most recent call last):
        TypeError: unsupported operand type(s) for /: 'Pmod_' and 'Pmod_'
    """
    mdeg = monic._degree
    mx = monic.x
    one = cast(R, monic[-1]**0)

    class PmodMeta_(PModMeta, type):
        @classmethod
        def __str__(cls) -> str:
            return "{0}[{1}]/<{2}>".format(type(one).__name__, re.sub(' +|-+|\\++','',mx), monic)

        @classmethod
        def __repr__(cls) -> str:
            return "{0}[{1}]/<{2}>".format(type(one).__name__, re.sub(' +|-+|\\++','',mx), monic)

    #class Pmod_(Polynomial, metaclass=PmodMeta_):
    class Pmod_(Polynomial[R1], PMod[R1], metaclass=PmodMeta_):

        __slots__ = ()

        def __init__(self, coeffs: Sequence[R1], x: str = mx):

            if not (
                isinstance(coeffs, Polynomial) or hasattr(type(coeffs), "__iter__")
            ):
                raise ValueError(
                    f"The argument to coeffs must be an iterable (e.g., a list or a tuple) "
                    f"not a {type(coeffs).__name__}. Try wrapping {coeffs} in square brackets."
                )

            #if x and not (x == mx or x == f"({mx})" or x == f"[{mx}]"):
            #    raise ValueError(
            #        f"The indeterminant string must be {mx}, ({mx}), or [{mx}], not {x}."
            #    )

            if len(coeffs) == 0:
                raise ValueError("coeffs cannot be empty")

            if mdeg < 0:
                raise ValueError("no need to quotient by <0>")

            #poly = Polynomial(
            poly = Polynomial[R]([one * cast(R, elt) for elt in coeffs], x=mx)
            polydeg = poly._degree
            if polydeg and polydeg >= mdeg:
                poly %= monic
            super().__init__(cast(Sequence[R1], poly._coeffs), x=x if x else mx)

        def __eq__(self, other: int|'Pmod_[R1]') -> bool:  # type: ignore[override]
            return ((self - other).__mod__(cast(Polynomial[R1], monic)))._degree < 0

        def __ne__(self, other: int|'Pmod_[R1]') -> bool:  # type: ignore[override]
            return ((self - other).__mod__(cast(Polynomial[R1], monic)))._degree > -1

        def __hash__(self) -> int:
            return hash((self._coeffs, monic._coeffs))

        def __repr__(self) -> str:
            monic_ = copy.copy(monic)
            monic_.x = self.x
            return f"<{monic}>" if self._degree < 0 else f"{self} + <{str(monic_)}>"

    return Pmod_

#F1 = TypeVar('F1', bound=Field[Any])
F1 = TypeVar('F1', bound=Field)

def FPmod(fpoly: FPolynomial[F]) -> Type[FPMod[F]]:
    """Quotient a univariate polynomial ring over a field by a principal ideal.

    Args:

        fpoly: a monic polynomial with coefficients in some field, K.

    Returns:

        (type). A class whose constructor yields elements of the quotient
            ring K[x]/<monic>, the polynomial ring over K modulo the ideal
            generated by monic.

            Note: the string for the indeterminant used to represent ele-
                  ments of the returned quotient ring is that of monic.

    Examples:

        First we import this so we can build polys over fields:

        >>> from polylib import FPolynomial

        Example 1: constructing the Gaussian rationals

        >>> from fractions import Fraction as F
        >>> x = FPolynomial([0, F(1)]) # builds Q[x], polys over rationals
        >>> monic = x**2 + 1   # an irreducible monic polynomial in Q[x]
        >>> GQ = FPmod(monic)  # Quotient Q[x] by ideal generated by monic
        >>> GQ   # the quotient is just the Gaussian rationals
        Fraction[x]/<1 + x^2>

        One way to now define an element of the Gaussian rationals:

        >>> elt = GQ([F(2,3), -F(5,4)])
        >>> elt
        2/3 - 5/4x + <1 + x^2>

        A more natural way to define GQ, using an indeterminant called 'i':

        >>> i = FPolynomial([0,F(1)],'i ') # FPolynomial((0, Fraction(1,1)))
        >>> GQ = FPmod(i**2 + 1)
        >>> i = GQ(i)  # i + <1 + i^2>
        >>> elt = F(2,3) - F(5,4)*i
        >>> elt
        2/3 - 5/4i + <1 + i^2>
        >>> print(elt)
        2/3 - 5/4i

        Now we can compute in the Gaussian rationals:

        >>> print(elt**-1)
        96/289 + 180/289i
        >>> print((1+2*i)**3)
        -11 - 2i
        >>> print((5-i)/(3-i+i**3))
        17/13 + 7/13i

        Example 2: constructing a Galois field

        >>> PF = Zmodp(5)  # the prime field, Z/5
        >>> t = FPolynomial([0, PF(1)],'t') # coefficients in Z/5
        >>> monic = 2-t+t**3 # irreducibe - primitive, in fact - over Z/5
        >>> GF = FPmod(monic)# GF(5^3) represented as: Z/5[t]/<2-t+t^3>

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

        >>> monic(t)  #  0 + <2 + 4t + t^3> = <2 + 4t + t^3>
        <2-t+t^3>

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

        >>> PF = Zmodp(5)  # the prime field
        >>> t = FPolynomial([0, PF(1)], 't') # indeterminant for Z/5[t]
        >>> p1 = 2-t+t**3  # this is primitive polynomial for GF(5^3)
        >>> p2 = 3+3*t+t**2  # this is primitive polynomial for GF(5^2)
        >>> QuotientRing = FPmod(p1*p2)  # this has zero divisors

        Here, we quotient Z/5[x] by the ideal generated by the reducible
        polynomial p1*p2 = (2-x+x^3)(2+x^2). However, each factor is irr-
        educible in Z/5[x], and the factors are relatively prime:

        >>> from numlib import xgcd
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
        which is also idempotent. The direct sum decomposition is

          Z/5[x]/<p1*p2> = e * Z/5[x]/<p1*p2> + (1-e) * Z/5[x]/<p1*p2>
                         = <m2*p2>/<p1*p2> + <m1*p1>/<p1*p2>

        The multiplicative identity in the left summand is e = m2*p2;
        in the right, it is 1-e = m1*p1.

        Recall that p1, respectively p2, is a primitive polynomial for
        GF(5^3), resp. GF(5^2). Let us define t as an element of the
        quotient ring GF(5^5).

        >>> t = QuotientRing(t, 't')  # indeterminant for QuotientRing

        Explicitly, we have, for e and 1-e,

        >>> e = (m2*p2)(t)
        >>> e; 1-e
        -1+t-t^3 + <1-2t-t^2+2t^3-2t^4+t^5>
        2-t+t^3 + <1-2t-t^2+2t^3-2t^4+t^5>

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
        wise, there are 5^3 multiples of p2 = 3+3*x+x**2 with degree <=5.
        There are, then, 5^2 + 5^3 - 1 zero divisors in GF(5^3) x GF(5^2)
        (since we counted zero twice); so that the unit group has order
        5^5 - 5^2 - 5^3 + 1 = (5^3-1) * (5^2-1) = 2976. The unit group in
        GF(5^3) x GF(5^2) is not cyclic: gcd(5**2-1, 5**3-1) = 5-1 = 4.

        The order of t is:

        >>> from numlib import lcm
        >>> order(t) == lcm(5**3-1, 5**2-1) == 744
        True

        Example 4: extension of an extension field

        >>> from fractions import Fraction as Q
        >>> from polylib import FPolynomial
        >>> from numlib import FPmod, rootsymb

        Adjoin the square root of 2 to Q, obtaining F1:

        >>> x = FPolynomial([Q(0), Q(1)], rootsymb(2))
        >>> F1 = FPmod(2-x**2)
        >>> root2 = F1(x); root2
        √2 + <2-√2^2>

        Now, adjoin the square root of 3 to the field F1:

        >>> y = FPolynomial((F1([0]), F1([1])), rootsymb(3))
        >>> F2 = FPmod(3-y**2)
        >>> root3 = F2(y); root3
        √3 + <(3) + -√3^2>

        Note that type is inferred from left summand:

        >>> root2+root3  # not correct
        √3 + √2 + <2 - √2^2>

        >>> root2*root3**0 + root3

        >>> root3+root2  # correct
        √2 + √3 + <3 + -√3^2>

        >>> (1+root3+root2)**5
        296 + 224√2 + 184 + 120√2√3 + <3 + -√3^2>

        >>> (root3+root2)**4-10*(root3+root2)**2+1
        <3 + -√3^2>
    """
    fdeg = fpoly._degree
    fx = fpoly.x
    one = fpoly[-1]**0

    class FPModMeta_(FPModMeta):

        @classmethod
        def __str__(cls) -> str:
            return "{0}[{1}]/<{2}>".format(type(one).__name__, re.sub(' +|-+|\\++','',fx), fpoly)

        @classmethod
        def __repr__(cls) -> str:
            return "{0}[{1}]/<{2}>".format(type(one).__name__, re.sub(' +|-+|\\++','',fx), fpoly)

    #class FPmod_(FPMod[F1], metaclass=FPModMeta_):
    class FPmod_(FPolynomial[F1], FPMod[F1], metaclass=FPModMeta_): #type: ignore[misc]

        __slots__ = ()

        def __init__(self, coeffs: Sequence[F1], x: str = fx) -> None:

            if not (
                isinstance(coeffs, Polynomial) or hasattr(type(coeffs), "__iter__")
            ):
                raise ValueError(
                    f"The argument to coeffs must be an iterable (e.g., a list or a tuple) "
                    f"not a {type(coeffs).__name__}. Try wrapping {coeffs} in square brackets."
                )

            #if x and not (x == fx or x == f"({fx})" or x == f"[{fx}]"):
            #    raise ValueError(
            #        f"The indeterminant string must be {fx}, ({fx}) or [{fx}], not {x}."
            #    )

            if len(coeffs) == 0:
                raise ValueError("coeffs cannot be empty")

            if fdeg < 0:
                raise ValueError("no need to quotient by <0>")

            # poly = FPolynomial([type(one)(elt) for elt in coeffs], x = fx)
            poly = FPolynomial[F1](coeffs, x=fx)
            polydeg = poly._degree
            if polydeg and polydeg >= fdeg:
                poly %= cast(FPolynomial[F1], fpoly)
            super().__init__(poly._coeffs, x=x if x else fx)

        def __truediv__(self, other: F1 |int | FPMod[F1]) -> 'FPmod_[F1]':
            #if isinstance(other, int):
            if not isinstance(other, FPolynomial):
                return self * (cast(FPmod_[F1], other) * (self**0)[0]) ** -1
            else:
                g, inv, _ = xgcd(other, fpoly)
                if not g._degree == 0:
                    raise ValueError(f"{other} is not invertible modulo {fpoly}")
                return self * FPmod_[F1]((inv * g[0] ** -1)._coeffs)

        def __rtruediv__(self, other: int | F1) -> 'FPmod_[F1]':
            return type(self)((cast(F1, other),)) / self

        def __pow__(self, m: int) -> 'FPmod_[F1]':
            if m < 0:
                assert self != 0, "cannot invert 0"
                # return super(FPmod_, 1/self).__pow__(-m)
                return (1 / self).__pow__(-m)
            else:
                return super().__pow__(m)
                #return cast(FPmod_, super(FPmod_, self).__pow__(m))

        #def __eq__(self, other: 'FPmod_[F1]') -> bool: # type: ignore[override]
            #NOTE: Induce from int here an Pmod GaloisField?? Trying this
        def __eq__(self, other: int|'FPmod_[F1]') -> bool: # type: ignore[override]
            return ((self - other).__mod__(cast(FPolynomial[F1], fpoly)))._degree < 0

        # NOTE: Do you want __ne__ here and Pmod GaloisField??  Trying this
        def __ne__(self, other: int|'FPmod_[F1]') -> bool: # type: ignore[override]
            return ((self - other).__mod__(cast(FPolynomial[F1], fpoly)))._degree > -1

        def __hash__(self) -> int:
            return hash((self._coeffs, fpoly._coeffs))

        def __repr__(self) -> str:
            fpoly_ = copy.copy(fpoly)
            fpoly_.x = self.x
            return f"<{fpoly_}>" if self._degree < 0 else f"{self} + <{str(fpoly_)}>"

    return FPmod_

def GaloisField(p: int, r: int = 1, negatives: bool = True, x: str = "t-") -> Type[GF_ZModP]:
    """Return an implemention of a Galois field of order p^r.

    Rather than calling this with r = 1 to implement GF(p), you may want
    instead to call Zmodp like GF=Zmodp(p) in which case the constructor
    for GF will then return essentially ints.

    Alternatively, if you are, say, writing a program that wants either
    GF(p) or any Galois field GF(p^r), all of the same type, then you may
    wish to call use this even with r=1.

    Note: the polynomial t=GF() is always a generator of the unit group,
    even when r=1 (see examples).

    Args:

        p (int): a prime.

        r (int): a positive integer.

        negatives (bool): whether to use negative numbers when represent-
            ing large numbers in Z/p.

        x (str): a string specifying, e.g., the indeterminant.

    Returns:

        (type). A class, instances of which are elements in the Galois
            field of order p^r represented as Z/p[t] modulo the ideal
            generated by a primitive polynomial (so that t generates
            the multiplicative group of units).

    Examples:

        If GF = GaloisField(p, r), then GF is a class with which one can
        easily construct elements of the corresponding Galois field; rem-
        ember that such elements are (equivalence classes of) polynomials.

        >>> GF = GaloisField(5, 3)
        >>> print(GF)
        Z/5[t]/<t^3+t^2+2>

        To instantiate an element of Z/5[t]/<2+t^2+t^3>, we need to spec-
        ify its coefficients as a polynomial. One way to do this is:

        >>> PF = Zmodp(5, negatives=True)  # instantiate the primefield
        >>> GF([PF(3), PF(4), PF(17)])  # increasing degree order
        2t^2-t-2 + <t^3+t^2+2>

        Alternatively, one can use an indeterminant:

        >>> t =  GF([PF(0), PF(1)])
        >>> 3 + 4*t + 17*t**2
        2t^2-t-2 + <t^3+t^2+2>

        For convenience, the indeterminant is provided like this:

        >>> t = GF()
        >>> 3 + 4*t + 17*t**2
        2t^2-t-2 + <t^3+t^2+2>

        More notes:

        GF = GaloisField(p, r) is a Python  generator object for the ent-
        ire Galois field:

        >>> print(', '.join(str(elt) for elt in GF)) # doctest: +ELLIPSIS
        -2t^2-2t-2, -2t^2-2t-1, -2t^2-t-2, ...
        >>> len(list(GF))
        125

        For all p and r (including r=1), t is a generator for the multi-
        plicative group of units in GF = GaloisField(p, r):

        >>> p, r = GF.order
        >>> len({t**i for i in range(p**r - 1)})
        124

        When working with, say, elliptic curves, one may want to define
        a polynomial f(x) whose coefficients are in a Galois field. This
        is different than the examples above -- in those we defined poly-
        nomials that are elements of the Galois field.

        For this we will need to wrap our elements of GF in parenthesis
        if we want an unambiguous string version of f.

        >>> GF = GaloisField(5, 3, 't')
        >>> t = GF()

        Then, we can do something like this:

        >>> from polylib import FPolynomial
        >>> f = FPolynomial([1 + t**2, 2*t**0])
        >>> f
        FPolynomial((t^2+1 + <t^3+t^2+2>, 2 + <t^3+t^2+2>))
        >>> print(f)
        (t^2+1) + (2)x

        But notice that we must use t**0; i.e., this is not the same:

        >>> FPolynomial([1 + t**2, 2])
        FPolynomial((t^2+1 + <t^3+t^2+2>, 2))

        Alternatively, we can create an indeterminant for x:

        >>> from polylib import Polynomial
        >>> x = Polynomial([0, 1])
        >>> print(x**0*(1 + t**2) + x*(2*t**0))
        (t^2+1) + (2)x

        More examples:

        Case: r = 1

        To demonstrate this case, let us not use negative representations.

        >>> GF = GaloisField(17, negatives = False)
        >>> print(GF)
        Z/17[t]/<t+14>
        >>> t = GF()
        >>> x = GF(10*t**0)
        >>> x
        10 + <t+14>
        >>> print(x)
        10
        >>> x**-1
        12 + <t+14>
        >>> y = GF(12*t**0)
        >>> print(x + y)
        5
        >>> print(x * y)
        1
        >>> # t is a generator of the unit group
        >>> print(', '.join(str(t**i) for i in range(1,17)))
        3, 9, 10, 13, 5, 15, 11, 16, 14, 8, 7, 4, 12, 2, 6, 1

        So t is a generator when r = 1, as in all other cases. But if
        you wish to work with single elements of GF, you end up doing
        say:

        >>> 9*t**0
        9 + <t+14>

        Alternatively, instead of GF = GaloisField(17, 1) you may
        want simply:

        >>> GF = Zmodp(17)
        >>> GF
        Z/17
        >>> GF(9)
        -8 + <17>

        Case: r > 1

        >>> GF = GaloisField(2, 4)
        >>> print(GF)
        Z/2[t]/<t^4+t^3+1>

        One can define elements of GF like this:

        >>> t = GF()
        >>> 1 + 2*t + 345*t**2 + t**6
        t^3+t + <t^4+t^3+1>

        GF is a generator, which is fastest way to iterate though
        all elements of GF:

        >>> print(', '.join(str(x) for x in GF)) # doctest: +ELLIPSIS
        0, 1, t, t+1, t^2, t^2+t, t^2+1, t^2+t+1, t^3, ...
    """
    if not isinstance(p, int) or p < 0:
        raise TypeError(f"p must be a positive prime, not {type(p)}")
    if not isinstance(r, int) or r < 0:
        raise TypeError(f"r must be a positive integer, not {type(r)}")

    PF = Zmodp(p, negatives=negatives)
    t = PF.indet(x)

    if r == 1:
        # Find the first generator of (Z/p)*:
        divs = divisors(p - 1)
        a = 0  # solely for typing
        for a in range(2, p):
            if mulorder_(PF(a), divs) == p - 1:
                break
        # a = 1
        # for a in range(2, p):
        #    for i in range(1, p - 1):
        #        if a**i % p == 1:
        #            break
        #    if i == p - 2:
        #        break
        irred = t - a

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
            return NotImplemented #type:ignore
    elif p == 5:
        if r == 2:
            irred = 2 + t + t**2
        elif r == 3:
            irred = 2 + t**2 + t**3
        elif r == 4:
            irred = 2 - t + t**2 - t**4
        else:
            return NotImplemented #type:ignore
    elif p == 7:
        if r == 2:
            irred = 3 - t + t**2
        elif r == 3:
            irred = 4 + 3 * t**2 + t**3
        elif r == 4:
            irred = 2 - t - 3 * t**2 + 3 * t**4
        else:
            return NotImplemented #type:ignore
    elif p == 23:
        if r == 2:
            irred = -4 - t + t**2
        elif r == 3:
            irred = -11 - t + t**3
        else:
            return NotImplemented #type:ignore
    elif p == 31:
        if r == 2:
            irred = 3 - 7 * t + t**2
        elif r == 3:
            irred = -12 - 11 * t + t**3
        else:
            return NotImplemented #type:ignore
    elif p == 43:
        if r == 2:
            irred = 3 + 21 * t + t**2
        elif r == 3:
            irred = -19 + 16 * t + t**3
        else:
            return NotImplemented #type:ignore
    elif p == 71:
        if r == 2:
            irred = 7 + 7 * t + t**2
        elif r == 3:
            irred = 2 + 9 * t + t**3
        else:
            return NotImplemented #type:ignore
    elif p == 113:
        if r == 2:
            irred = 10 + t + t**2
        elif r == 3:
            irred = -55 - 50 * t + t**3
        else:
            return NotImplemented #type:ignore
    elif p == 503:
        if r == 2:
            irred = -201 - t + t**2
        else:
            return NotImplemented #type:ignore
    else:
        return NotImplemented #type:ignore

    class GF_ZModPMeta_(GF_ZModPMeta):
        order = (p, r)

        def __iter__(self) -> Iterator[GF_ZModP] :
            for coeffs in iproduct(PF, repeat=r):
                yield (self(coeffs))

        @classmethod
        def __str__(cls) -> str:
            return "{0}[{1}]/<{2}>".format(PF, re.sub(' +|-+|\\++','',x), irred)

        @classmethod
        def __repr__(cls) -> str:
            return "{0}[{1}] mod {2}".format(PF, re.sub(' +|-+|\\++','',x), repr(irred))

        @classmethod
        def indet(cls, letter: str = 'x ') -> FPolynomial[GF_ZModP]:
            t = cast(GF_ZModP, GF())
            return FPolynomial[GF_ZModP]([t*0, t**0], x=letter)

    class GF_ZModP_(FPolynomial[ZModP], GF_ZModP, metaclass=GF_ZModPMeta_): # type: ignore[misc]

        __slots__ = ()

        def __init__(self, coeffs: Sequence[ZModP] = (), x: str = x) -> None:
            # if not (isinstance(coeffs, Polynomial) or hasattr(type(coeffs), '__iter__')):
            #    raise ValueError(
            #        f"The argument to coeffs must be an iterable (e.g., a list or a tuple) "
            #        f"not a {type(coeffs).__name__}. Try wrapping {coeffs} in square brackets."
            #    )

            # if len(coeffs) == 0:
            #    raise ValueError("coeffs cannot be empty")

            len_ = len(coeffs)
            if len_ > r:
                cast('GF_ZModP_', super().__init__(
                    (
                        FPolynomial(coeffs, x=x)
                        % irred
                    )._coeffs,
                    x=x,
                ))
            elif len_ > 0:
                cast('GF_ZModP_', super().__init__(coeffs, x=x))
            else:
                cast('GF_ZModP_', super().__init__(t._coeffs, x=x))

        def __hash__(self) -> int:
            return hash((self._coeffs, irred._coeffs))

        # def __str__(self):
        #    s = str(irred)
        #    if s[0] == "(" and s[-1] == ")":
        #        s = s[1:-1]
        #    return f"<{s}>" if self._degree < 0  else f"{super().__str__()} + <{s}>"

        def __repr__(self) -> str:
            s = str(irred)
            #if s[0] == "(" and s[-1] == ")":
            #    s = s[1:-1]
            return f"<{s}>" if self._degree < 0 else f"{super().__str__()} + <{s}>"

        def __truediv__(self, other: int|GF_ZModP) -> 'GF_ZModP_':
            if not isinstance(other, GF_ZModP):
                #return self * (other * (self**0)[0]) ** -1
                return self * PF(other) ** -1
            #elif isinstance(other, FPolynomial):
            else:
                g, inv, _ = xgcd(cast(FPolynomial[ZModP], other), irred)
                return self * GF_ZModP_((inv * g[0] ** -1)._coeffs)

        def __rtruediv__(self, other: int|ZModP) -> 'GF_ZModP_':
            return GF_ZModP_([PF(other)], self.x) / self

        def __pow__(self, m: int) -> 'GF_ZModP_':
            if m < 0:
                assert self != 0, "cannot invert 0"
                return super(GF_ZModP_, 1 / self).__pow__(-m)
            else:
                return super().__pow__(m)

        def __eq__(self, other: int|'GF_ZModP_') -> bool: # type: ignore[override]
            return ((self - other).__mod__(irred))._degree < 0

        def __ne__(self, other: int|'GF_ZModP_') -> bool: # type: ignore[override]
            return ((self - other).__mod__(irred))._degree > -1

    GF = GF_ZModP_
    GF.__name__ = str(GF)

    return GF


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

    Example:

        >>> from numlib import Zmodp
        >>> p = 31; F = Zmodp(p)
        >>> a = F(5)
        >>> assert a ** ((p-1)//2) == 1
        >>> sqrt(a, p, p)
        -6 + <31>
    """
    if q % 4 == 3:
        return a ** ((q + 1) // 4)
    assert q == p, "not yet implemented"
    one = a**0
    t = random.randint(0, p - 1) * one
    while (t**2 - 4 * a) ** ((p - 1) // 2) != -1:
        t = random.randint(0, p - 1) * one
    Fp2 = FPmod(FPolynomial[F]([a, t, one]))
    x = Fp2([cast(F, 0), cast(F, 1)])
    return cast(FPolynomial[F], (x ** ((p + 1) // 2)))[0]


def rootsymb(n: int) -> str:
    """Return string version of int prepended with a unicode radical symbol.

    Useful for pretty printing elements of, for example, a quadratic field.

    Example:

        >>> print(rootsymb(5))
        √5
    """
    return "\u221A" + str(n)

if __name__ == "__main__":
    import doctest
    doctest.testmod()

