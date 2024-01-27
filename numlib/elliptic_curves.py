from __future__ import annotations
import copy
from typing import Type, cast, TypeVar, Generic, Callable, Optional, Any, overload, Literal, Generator
from polylib.polynomial import Field, Polynomial, FPolynomial
from numlib.quotient_rings import ZModP, GF_ZModP, sqrt

__author__ = "Scott Simmons"
__version__ = "0.3.1"
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


class AlgebraicCurve:
    pass

#    def __init(self, f: FPolynomial):  # f(x,y) over algebraically closed field
#        pass

#F = TypeVar("F", bound=Field[Any])
F = TypeVar("F", bound=Field, contravariant=True)

class EllipticCurve(Generic[F]):
    def __init__(self, x: int|F, y: int|F|None, z: int|F|None = None) -> None: ...
    disc: F
    j: Optional[F]
    f: Polynomial[F]
    point: tuple[F, ...]
    co: tuple[F|Optional[F], ...]

#class EllipticCurve: ...
#class EllipticCurve(type, Generic[F]):
class WeierstrassCurveMeta(type, Generic[F]):
    #def __init__(self, *args, **kwargs): ...
    #def __init__(self, x: Union[int, F], y: Union[int, F, None], z: Union[int, F, None] = None) -> None: ...
    disc: F
    j: Optional[F]
    f: Polynomial[F]
    order: int
    point: tuple[F, ...]
    pointorder: int
    @classmethod
    def __repr__(cls) -> str: ...
    #co: tuple[Optional[F]]

#class WeierstrassCurve(EllipticCurve[F]):
#class WeierstrassCurve(Generic[F], metaclass = EllipticCurve):
class WeierstrassCurve(EllipticCurve[F], metaclass = WeierstrassCurveMeta):
    #@overload
    #def __init__(self, x: Union[int, F1], y: Union[int, F1]) -> None: ...
    #@overload
    #def __init__(self, x: int, y: int) -> None: ...
    #@overload
    #def __init__(self, x: F, y: F) -> None: ...
    #@overload
    ##def __init__(self, x: int, y: int, z: int) -> None: ...
    #@overload
    #def __init__(self, x: F, y: F, z: F = None) -> None: ...
    def __init__(self, x: int|F, y: int|F, z: int|F|None = None) -> None:
        self.co: tuple[F, F, F]
    def __add__(self, other: 'WeierstrassCurve[F]') -> 'WeierstrassCurve[F]': ...
    def __neg__(self) -> 'WeierstrassCurve[F]': ...
    def __mul__(self, other: int) -> 'WeierstrassCurve[F]': ...
    def __rmul__(self, other: int) -> 'WeierstrassCurve[F]': ...
    def double(self) -> 'WeierstrassCurve[F]': ...

class MontgomeryCurveMeta(type, Generic[F]):
    #def __init__(self, *args, **kwargs): ...
    #def __init__(self, x: Union[int, F], y: Union[int, F, None], z: Union[int, F, None] = None) -> None: ...
    disc: F
    j: Optional[F]
    f: Polynomial[F]
    @classmethod
    def __repr__(cls) -> str: ...
    order: int
    point: tuple[F, ...]
    pointorder: int
    #co: tuple[Optional[F]]

class MontgomeryCurve(EllipticCurve[F], metaclass = MontgomeryCurveMeta):
    def __init__(self, x: int|F, y: int|F|None = None, z: int|F|None = None) -> None:
        self.co: tuple[F, Optional[F], F]

F1 = TypeVar("F1", bound=Field)

#def Weierstrass(a: F, b: F, debug: bool = False) -> EllipticCurve[F]:
#def Weierstrass(a: F, b: F, debug: bool = False) -> Type[EllipticCurve[F]]:
def Weierstrass(a: F, b: F, debug: bool = False) -> Type[WeierstrassCurve[F]]:
    """Return a class whose instances are elements of y^2=x^3+ax+b.

    This implements an elliptic curve in short Weierstrass form. The re-
    turned class allows one to work in the curves k-rational points where
    the field k is that of the arguments to the paramters a and b.

    Examples:

        >>> from numlib import Zmodp
        >>> GF = Zmodp(7)
        >>> E = EllCurve(GF(1), GF(1), debug = True)
        >>> E
        y^2 = x^3 + x + 1 over Z/7
        >>> E.j  # the j-invariant of the curve
        1 + <7>

        When defining points on E, the type of the coefficient is inferred:

        >>> pt = E(2, 5)  # No need for E(GF(2), GF(5))
        >>> pt
        (2, -2) on y^2 = x^3 + x + 1 over Z/7
        >>> print(pt)
        (2, -2)
        >>> print(-pt)
        (2, 2)

        The point at infinity, (0: 1: 0), is the identity; for instance:

        >>> print(E(0,1,0) + E(2,5))
        (2, -2)

        Points are in displayed non-homogeneous coordinates, except for
        the point at infinity:
        >>> print(0 * pt)
        [0: 1: 0]

        Not a point on E:

        >>> E(0,0) # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ValueError: (0, 0) = [0: 0: 1] is not on y^2 = x^3 + x + 1...

        Find all points on E:

        >>> from numlib import iproduct
        >>> for pair in iproduct(GF, GF): # doctest: +ELLIPSIS
        ...     try:
        ...         print(E(*pair), end=', ')
        ...     except:
        ...         pass
        (0, -1), (0, 1), (2, -2), (2, 2), ...

        Alternatively,

        >>> from numlib import affine
        >>> len({pt for pt in affine(E)}) # affine(E) is a Python generator
        4

        This curve has order 5 and hence is cyclic. Every non-identity
        element is a generator:

        >>> for i in range(1, 6):
        ...     print(i * (E(2, 5)))
        (2, -2)
        (0, -1)
        (0, 1)
        (2, 2)
        [0: 1: 0]
    """
    one = a ** 0
    zero = one * 0
    f_ = Polynomial((b, a, zero, one), "x -")

    class Meta(WeierstrassCurveMeta[F1]):

        #f = Polynomial((b, a, zero, one), "x -")
        f = cast(Polynomial[F1], f_)
        disc = cast(F1, -16 * (4 * a**3 + 27 * b**2))
        j = cast(F1, -110592 * a**3) / disc if disc != zero else None

        @classmethod
        def __repr__(cls) -> str:
            return f"y^2 = {f_} over {type(one).__name__}"

    class WeierstrassCurve_(WeierstrassCurve[F1], metaclass=Meta):
        if debug:
            def __init__(self, x: int|F1, y: int|F1, z: int|F1 = cast(F1, one)) -> None:
                """projective coordinates [x: y: z]"""

                x = cast(F1, one) * x; y = cast(F1, one) * y; z = cast(F1, one) * z

                if z != zero:
                    if (y / z) ** 2 != f_(cast(F, x / z)):
                        raise ValueError(
                            (
                                f"({x/z}, {y/z}) = [{x}: {y}: {z}] is not on y^2 = {f_}: "
                                f"y^2 = {(y/z)**2} != {f_(cast(F, x/z))}"
                            )
                        )
                else:
                    if not (x == 0 and y != 0):
                        raise ValueError(f"[{x}: {y}: {z}] is not on {y**2} = {f_}")

                self.co = (x, y, z)
        else:
            def __init__(self, x: int|F1, y: int|F1, z: int|F1 = cast(F1, one)) -> None:

                self.co = (cast(F1, one) * x, cast(F1, one) * y, cast(F1, one) * z)

        def __add__(self, other: WeierstrassCurve[F1]) -> WeierstrassCurve[F1]:

            # compare with eq below and fix this
            if other.co[2] == 0:
                return self
            if self.co[2] == 0:
                return other

            u0 = self.co[0] * other.co[2]
            u1 = other.co[0] * self.co[2]
            t0 = self.co[1] * other.co[2]
            t1 = other.co[1] * self.co[2]

            if u0 == u1:
                return self.double() if t0 == t1 else self.__class__(cast(F1, zero), cast(F1, one), cast(F1, zero))

            u = u0 - u1
            u2 = u * u
            u3 = u2 * u
            t = t0 - t1
            v = self.co[2] * other.co[2]
            w = t * t * v - u2 * (u0 + u1)

            return self.__class__(u * w, t * (u0 * u2 - w) - t0 * u3, u3 * v)

        def __eq__(self, other: object) -> bool:
            if not isinstance(other, WeierstrassCurve_):
                if other == 0:
                    return self.co[2] == zero
                else:
                    return NotImplemented
            if self.co[2] == 0 == other.co[2]:
                return True
            if self.co[2] == other.co[2]:
                return all([self.co[i] == other.co[i] for i in range(2)])
            if self.co[2] * other.co[2] != 0:
                return all(
                    [
                        self.co[i] / self.co[2] == other.co[i] / other.co[2]
                        for i in range(2)
                    ]
                )
            else:
                return False

        def __mul__(self, n: int) -> WeierstrassCurve[F1]:
            if n == 0:
                return self.__class__(cast(F1, zero), cast(F1, one), cast(F1, zero))
            elif n == 1:
                return self
            factor = self.__mul__(n // 2)
            if n % 2 == 0:
                return factor.double()
            else:
                return self + factor.double()

        def double(self) -> WeierstrassCurve[F1]:
            if self.co[2] == 0 or self.co[1] == 0:
                return self.__class__(cast(F1, zero), cast(F1, one), cast(F1, zero))
            else:
                t = 3 * self.co[0] ** 2 + cast(F1, a) * self.co[2] ** 2
                u = 2 * self.co[1] * self.co[2]
                v = 2 * u * self.co[0] * self.co[1]
                w = t * t - 2 * v
                return self.__class__(
                    u * w, t * (v - w) - 2 * (u * self.co[1]) ** 2, u**3
                )

        def __rmul__(self, n: int) -> WeierstrassCurve[F1]:
            return self.__mul__(n)

        def __neg__(self) -> WeierstrassCurve[F1]:
            if self.co[2] == 0 or self.co[1] == 0:
                return self
            else:
                return self.__class__(self.co[0], -self.co[1], self.co[2])

        def __sub__(self, other: WeierstrassCurve[F1]) -> WeierstrassCurve[F1]:
            return self.__add__(-other)

        def __str__(self: WeierstrassCurve[F1]) -> str:
            if self.co[2] == 0:
                return f"[{self.co[0]}: {self.co[1]}: {self.co[2]}]"
            else:
                return f"({self.co[0]/self.co[2]}, {self.co[1]/self.co[2]})"

        def __repr__(self: WeierstrassCurve[F1]) -> str:
            if self.co[2] == 0:
                return f"{self} on {self.__class__}"
            else:
                return f"{self} on {self.__class__}"

        def __hash__(self: WeierstrassCurve[F1]) -> int:
            # need to hash unique coords so do something like this:
            if self.co[2] == 0:
                return hash((zero, one, zero))
            else:
                return hash((self.co[0] / self.co[2], self.co[1] / self.co[2]))

    return WeierstrassCurve_

def Montgomery(a: F, b: F, debug: bool = False) -> Type[MontgomeryCurve[F]]:
    """Return a class whose instances are elements of by^2=x^3+ax^2+x.

    This implements an elliptic curve in Montgomery form. The returned
    class allows one to work in the curves k-rational points where the
    field k is that of the arguments to the paramters a and b.

    Multiplication of a point on the curve by a positive integer is
    the implemented using the ladder.

    If the coefficients are in a finite field, then one can work with
    the compressed curve by giving the constructor only one coordinate,
    x; or, equivalently, by specifying y = None. See examples.

    Examples:

        Curve25519 (over Z/(2**255-19)):

        >>> from numlib import Zmodp
        >>> p = 2**255-19
        >>> GF = Zmodp(p, negatives = True)
        >>> E = Montgomery(GF(486662), GF(1), debug = True)
        >>> E                           # doctest: +ELLIPSIS
        y^2 = x^3 + 486662x^2 + x over Z/578960446186580977...
        >>> E.j != 0
        True

        Standard basepoint:

        >>> x = GF(9)
        >>> from numlib import sqrt
        >>> y = sqrt(E.f(x), p, p)
        >>> y = -y if int(y) < 0 else y
        >>> g = E(x, y)
        >>> print(g)           # doctest: +ELLIPSIS
        (9, 147816194475895447910205935684099868872646061346...

        The basepoint g generates a subgroup whose (prime) order is:

        >>> n = 2**252 + 27742317777372353535851937790883648493

        Let us check that g has the correct order:

        >>> #print(n*g)
        [0: 1: 0]
        >>> from numlib import addorder
        >>> #order(g, n) == n
        True

        Working on the compressed curve:

        >>> g = E(GF(9)) # same as g = E(GF(9), None)
        >>> print(g)
        (9, _)
        >>> #print(n*g)
        [0: 1: 0]

        warning: when working with the compressed curve, the
                 coordinate x is not verified to the x-coord
                 of a valid point on the curve, even if deb-
                 ug is True.
    """
    one = (a * b) ** 0
    zero = one * cast(F, 0)
    f_ = Polynomial((zero, one, a, one), "x -")
    ypoly = Polynomial((zero, zero, b), "y -")

    class Meta(MontgomeryCurveMeta[F1]):

        f = cast(Polynomial[F1], f_)
        disc = cast(F1, a)**2 - 4
        j = cast(F1, 256 * (a**2 - 3)**3) / disc if disc != zero else None

        @classmethod
        def __repr__(cls) -> str:
            return f"{ypoly} = {f_} over {type(one).__name__}"

    class MontgomeryCurve_(MontgomeryCurve[F1], metaclass=Meta):
        if debug:
            def __init__(self, x: F1, y: Optional[F1] = None, z: F1 = cast(F1, one)) -> None:
                """projective coordinates [x: y: z]"""

                x = cast(F1, one) * x; y = y if y is None else cast(F1, one) * y; z = cast(F1, one) * z

                if z != zero:
                    # not a great idea:
                    #if y is None:
                    #    if hasattr(type(one), 'char'):
                    #        order = type(one).order if hasattr(type(one), 'order') else type(one).char
                    #        if (f_(x / z)*bb**-1) ** ((order-1)//2) == -1:
                    #            raise ValueError(f"if x = {x}, then f(x)/b = {f_(x)*bb**-1} is not a square")
                    #    else:
                    #        raise ValueError("please provide a y value")
                    #elif bb * (y / z) ** 2 != f_(x / z):
                    #    raise ValueError(
                    #        (
                    #            f"({x/z}, {y/z}) = [{x}: {y}: {z}] is not on {ypoly} = {f_}: "
                    #            f"{ypoly} = {(y/z)**2} != {f_(x/z)}"
                    #        )
                    #    )

                    # NOTE:  nothing is checked if y is None.
                    if y is not None:
                        if cast(F1, b) * (y / z) ** 2 != f_(cast(F, x / z)):
                            raise ValueError(
                                (
                                    f"({x/z}, {y/z}) = [{x}: {y}: {z}] is not on {ypoly} = {f_}: "
                                    f"{ypoly} = {(y/z)**2} != {f_(cast(F, x/z))}"
                                )
                            )
                else:
                    if not (x == 0 and y != 0):
                        raise ValueError(f"[{x}: {y}: {z}] is not on {ypoly} = {f_}")

                self.co = (x, y, z)

        else:
            def __init__(self, x: F1, y: Optional[F1] = None, z: F1 = cast(F1, one)) -> None:
                self.co = (cast(F1, one) * x, y if y is None else cast(F1, one) * y, cast(F1, one) * z)

        def __mul__(self, n: int) -> MontgomeryCurve_[F1]:
            def ladder(tup: tuple[F1, F1], k: int) -> tuple[F1, F1]:
                if k == 1:
                    return tup
                elif k % 2 == 0:
                    X, Z = ladder(tup, k // 2)
                    return (
                        (X - Z)**2 * (X + Z)**2,
                        ((X + Z)**2 - (X - Z)**2) * ((X + Z)**2 + (cast(F1, a)-2)/4*((X + Z)**2 - (X - Z)**2))
                    )
                else:
                    return tup

            x, z = ladder((self.co[0], self.co[2]), n)
            return self.__class__(x, self.co[1], z)

        def __rmul__(self, n: int) -> MontgomeryCurve_[F1]:
            return self.__mul__(n)

        def __str__(self: MontgomeryCurve_[F1]) -> str:
            if self.co[2] == 0:
                if self.co[1]:
                    return f"[{self.co[0]}: {self.co[1]}: {self.co[2]}]"
                else:
                    return f"[{self.co[0]}: {one}: {self.co[2]}]"
            else:
                if self.co[1] is None:
                    return f"({self.co[0]/self.co[2]}, _)"
                else:
                    return f"({self.co[0]/self.co[2]}, {self.co[1]/self.co[2]})"

        def __repr__(self: MontgomeryCurve_[F1]) -> str:
            if self.co[2] == 0:
                return f"{self} on {self.__class__}"
            else:
                return f"{self} on {self.__class__}"

        def __hash__(self: MontgomeryCurve_[F1]) -> int:
            # need to hash unique coords so do something like this:
            if self.co[2] == 0:
                return hash((zero, one, zero))
            elif self.co[1] is None:
                return hash((self.co[0] / self.co[2], None))
            else:
                return hash((self.co[0] / self.co[2], self.co[1] / self.co[2]))

    return MontgomeryCurve_

EllCurve = Weierstrass

#FF = TypeVar('FF', ZModP, GF_ZModP)

EC = TypeVar('EC', bound = EllipticCurve[Any])

#@overload
#def affine(E: EllipticCurve[ZModP]) -> Generator[tuple[ZModP, ZModP], None, None]: ...
#@overload
#def affine(E: EllipticCurve[GF_ZModP]) -> Generator[tuple[GF_ZModP, GF_ZModP], None, None]: ...
#
#def affine(E: EllipticCurve[FF]) -> Generator[tuple[FF, FF], None, None]:

#@overload
#def affine(E: Type[EllipticCurve[ZModP]]) -> Generator[EllipticCurve[ZModP], None, None]: ...
#@overload
#def affine(E: Type[EllipticCurve[GF_ZModP]]) -> Generator[EllipticCurve[GF_ZModP], None, None]: ...
#
#def affine(E: Type[EllipticCurve[FF]]) -> Generator[EllipticCurve[FF], None, None]:
def affine(E: Type[EC]) -> Generator[EC, None, None]:
    """Return a generator that yields the affine points of E.

    This works fine for small curves. But it pre computes two diction-
    aries each of size the order of the curve; thus, this has horrify-
    ing order and takes essentially forever on large curves.  Consider
    using affine2 instead, though this could come in handy in checking
    correctness of more sophisticated algorithms on small curves.

    Args:

       E (EllipticCurve). An elliptic curve over a finite field.

    Returns:

       (Generator).  A generator yielding the finite points on E.

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
        y^2 = x^3 + (t+2)x + (-1) over Z/5[t]/<t^2+t+2>
        >>> len(list(affine(E)))
        34
    """
    coefcls = E.f[-1].__class__
    # b = E.f(0)
    # a = E.f.derivative()(0)
    b = E.f[0]
    a = E.f[1]

    y2s: dict[EC, list[EC]] = {}  # map all squares y2 in the field k to {y | y^2 = y2}
    fs: dict[EC, list[EC]] = {}  # map all outputs fs = f(x), x in k, to {x | f(x) = fs}

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


def affine2(E: Type[EC]) -> Generator[EC, None, None]:
    """Yield roughly half of the affine points of E.

    This yields one of each pair {(x, y), (x, -y)} of points not on the
    line y=0 and works by checking if f(x) is a quadratic residue where
    y^2=f(x) defines E.  If f(x) is a quadratic residue then one of the
    corresponding points on the curve is yielded.

    Args:

       E (EllipticCurve). An elliptic curve over a finite field.

    Returns:

       (Generator).  A generator yielding the finite points on E.


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
    #coefcls = E.disc.__class__
    coefcls = E.f[-1].__class__
    order = coefcls.order
    p = order[0]
    q = p ** order[1]
    assert q % 2 != 0

    for x in coefcls:
        fx = E.f(x)
        if fx ** ((q - 1) // 2) == 1:
            yield E(x, y=sqrt(fx, q=q, p=p))

def frobenious(E: Type[EC], n: int = 1, m: int = 1) -> Callable[[EC], EC]:
    """Return the mth iterate of the q^r-power Frobenious isogeny.

    Args:

        E (EllipticCurve): an elliptic curve over a finite field K of
            order q^r where q = p^n is the order of a subfield of K
            containing the coefficients of the equation defining E.

        n (int): the dimension of said subfield over Z/p. Default: 1.

        m (int): a positive integer. Default: 1.

    Returns:

        (Callable). A function that maps a given pt = (x, y) in E
            to (x^(q^m), y^(q^m)) in E.

    Examples:

        The Frobenious endomophism of a field of order q^r as an ex-
        tension of a subfield of order q = p^n maps like x -> x^q;
        it has order r (and in fact generates the associated Galois
        group).

        Given E as above, the Frobenious isogeny E -> E maps like

                            (x, y) -> (x^q, y^q);

        so that its rth iterate is the identity.

        Example 1: q = p = 7 (i.e. n = 1); r = 3

        >>> from numlib import GaloisField, EllCurve
        >>> GF = GaloisField(7, 3); t = GF()

        Let us define a curve over a field GF of order 7^3 but with
        coeffients in Z/7:

        >>> E = EllCurve(2*t**0, 3*t**0); E
        y^2 = x^3 + (2)x + (3) over Z/7[t]/<t^3+3t^2-3>
        >>> pt = next(affine2(E)); print(pt)  # A point on E
        (-3t^2-3t-2, -t^2-t+1)

        The 3rd iterate is the identity isogeny:

        >>> from numlib import affine2
        >>> E_affine = affine2(E)  # affine points of E
        >>> frob = frobenious(E, m=3)
        >>> print(f"{pt} maps to {frob(pt)}")
        (-3t^2-3t-2, -t^2-t+1) maps to (-3t^2-3t-2, -t^2-t+1)
        >>> all(frob(pt) == pt for pt in E_affine)
        True

        But the 2nd interate is non-trivial:

        >>> frob = frobenious(E, m=2)
        >>> print(f"{pt} maps to {frob(pt)}")
        (-3t^2-3t-2, -t^2-t+1) maps to (3t^2-t+3, t^2+2t-2)

        Example 2: q = 7^3 (so n = 3); r = 2

        >>> from polylib import FPolynomial
        >>> from numlib import FPmod
        >>> GF = GaloisField(7, 3, x='t-'); t = GF()
        >>> # an irreducible quadratic in GF(7,3)[x]:
        >>> irred = FPolynomial([-t**2-3, -2*t**2-t, t**0])
        >>> print(irred)
        (-t^2-3) + (-2t^2-t)x + x^2
        >>> F = FPmod(irred); F
        Z/7[t]/<t^3+3t^2-3>[x]/<(-t^2-3) + (-2t^2-t)x + x^2>
        >>> x = F([0*t, 1*t**0])
        >>> x
        x + <(-t^2-3) + (-2t^2-t)x + x^2>
        >>> a = x**0*(t*2); a
        (2t) + <(-t^2-3) + (-2t^2-t)x + x^2>
        >>> b = x**0 *(3*t**2); b
        (3t^2) + <(-t^2-3) + (-2t^2-t)x + x^2>
        >>> #a/b # <- NOTE: needs work on multivariate

        >>> #(12 + 3 * a * b ** 2 + 23)/ (b + 3)

        >>> #E = EllCurve(a, b, debug = True)

    """
    p, _ = type(E.disc).order
    PF_frob = lambda x: x ** ((p ** n) ** m )
    #if r == 1:  # then the basefield is Z/p
    #    PF_frob = lambda x: x ** (p**m)
    #else:  # the basefield is an extension of Z/p
    #    PF_frob = lambda x: x ** (PF.order // PF.char ** m)
    return lambda pt: E(*tuple(map(PF_frob, tuple(pt.co))))


#def frobenious(E1: EC, r: int) -> Callable[[EC], EC]:
#    """Return the q^r-power Frobenious isogeny.
#
#    E must be Weierstrass curve. TODO: Generalize.
#
#    Args:
#
#        E (EllipticCurve): an elliptic curve over a finite field
#            K of order q = p^n.
#
#        r (int): a positive integer.
#
#    Returns:
#
#        (Callable). A function that maps a given pt = (x, y) in E
#            to (x^(q^r), y^(q^r)).
#
#    Examples:
#
#    Given E defined over the field K = GF(p, n) of order q = p^n, this
#    returns the function E(K') -> E(K') on K'-rational points of E
#    where K' is a field of order q^r, i.e., GF(p, nr) that maps like
#
#                          (x, y) -> (x^r, y^r).
#
#    In other words, on the level of coefficients x and y, the map is
#    just the rth iterate of the endormorphism F: K -> K defined by
#    F(x) = x^(p^n) which, in turn, is just the nth iteratate of the
#    Frobenious endomorphism Z/p -> Z/p.
#
#    If K has order q=p^n, then F^n(x) = x^(p^n) = x for all x so that
#    the map x -> x^(p^n) is the identity on k; hence frobenious(E,1)
#    is the identity on K-rational points of E. For instance,
#
#        >>> import numlib as nl
#        >>> GF = nl.GaloisField(7, 3); t = GF()
#        >>> E = nl.EllCurve(1+t, 2*t); E  # A curve over GF(7,3)
#        y^2 = x^3 + (t+1)x + (2t) over Z/7[t]/<t^3+3t^2-3>
#        >>> E_affine = nl.affine(E)  # affine points of E
#
#        >>> pt = next(nl.affine2(E)); print(pt)  # A point on E
#        (-3t^2-2t-2, -3t^2-2t-2)
#        >>> frob = frobenious(E, 3)
#        >>> print(f"{pt} maps to {frob(pt)}")
#        (-3t^2-2t-2, -3t^2-2t-2) maps to (-3t^2-2t-2, -3t^2-2t-2)
#        >>> all(str(frob(pt)) == str(pt) for pt in E_affine)
#        True
#
#    As an endomorphism of K=GF(p,n), F has order n (and, in fact,
#    generates the Galois group of GF(p,n) over GF(p,1)).
#
#    If we think of F as a map on the algebraic closure of k, and
#    frob(r) is just frob(1) composed with itself r times, and
#
#        - frob(r) is the identity when restricted to GF(p,r);
#          i.e, frob = frob(1) has order r when on GF(p,r).
#
#        >>> frob = frobenious(E, 1)
#        >>> print(f"{pt} maps to {frob(pt)}")
#        (-3t^2-2t-2, -3t^2-2t-2) maps to (-t^2+2t+3, -t^2+2t+3)
#        >>> frob = frobenious(E, 2)
#        >>> print(f"{pt} maps to {frob(pt)}")
#        (-3t^2-2t-2, -3t^2-2t-2) maps to (-3t^2, -3t^2)
#        >>> frob = frobenious(E, 3)
#        >>> print(f"{pt} maps to {frob(pt)}")
#        (-3t^2-2t-2, -3t^2-2t-2) maps to (-3t^2-2t-2, -3t^2-2t-2)
#        >>> frob = frobenious(E, 4)
#        >>> print(f"{pt} maps to {frob(pt)}")
#        (-3t^2-2t-2, -3t^2-2t-2) maps to (-t^2+2t+3, -t^2+2t+3)
#    """
#    # b = E.f(0)
#    # a = E.f.derivative()(0)
#    b = E.f[0]
#    a = E.f[1]
#    p, r = type(E.disc).order
#    pr = p ** r
#    E_codomain = Weierstrass(a**pr, b**pr)
#    Frob = lambda x: x**pr
#    return lambda pt: E_codomain(*tuple(map(Frob, tuple(pt.co))))

if __name__ == "__main__":

    import doctest
    doctest.testmod()
