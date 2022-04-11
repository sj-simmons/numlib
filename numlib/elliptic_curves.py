from __future__ import annotations
from polylib import Field, Polynomial, FPolynomial
from typing import Type, cast, Tuple, TypeVar, Union, Generic, Callable
import copy

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


class AlgebraicCurve:
    pass


#    def __init(self, f: FPolynomial):  # f(x,y) over algebraically closed field
#        pass

F = TypeVar("F", bound=Field)


class EllipticCurve(Generic[F]):
    j: F
    disc: F
    f: FPolynomial[F]

    def __init__(self) -> None:
        self.co: Tuple[F, F, F]

    def __add__(self, other: EllipticCurve[F]) -> EllipticCurve[F]:
        pass

    def __neg__(self) -> EllipticCurve[F]:
        pass

    def __mul__(self, other: int) -> EllipticCurve[F]:
        pass

    def double(self) -> EllipticCurve[F]:
        pass


class WeierstrassCurve(EllipticCurve[F]):
    pass


class MontgomeryCurve(EllipticCurve[F]):
    pass


def Weierstrass(a: F, b: F, debug: bool = False) -> Type[EllipticCurve[F]]:
    """Return a class whose instances are elements of y^2=x^3+ax+b.

    This implements an elliptic curve in short Weierstrass form. The re-
    turned class allows one to work in the curves k-rational points where
    the field k is that of the arguments to the paramters a and b.

    Examples:

        >>> from numlib import Zmod
        >>> GF = Zmod(7)
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
        >>> len({pt for pt in affine(E)}) # finite(E) is a generator
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
    # global WeierstrassCurve  # not needed now?

    one = (a * b) ** 0
    zero = one * cast(F, 0)

    aa = copy.copy(a)
    if (
        isinstance(a, Polynomial)
        and a._degree
        and a._degree > 0
        and a.x.find("(") < 0
        and a.x.find(")") < 0
    ):
        aa.x = "(" + a.x + ")"

    bb = copy.copy(b)
    if (
        isinstance(b, Polynomial)
        and b._degree
        and b._degree > 0
        and b.x.find("(") < 0
        and b.x.find(")") < 0
    ):
        bb.x = "(" + b.x + ")"

    f_ = Polynomial((bb, aa, zero, one), "x", spaces=True, increasing=False)

    class WeierstrassCurve_(type):

        f = Polynomial((bb, aa, zero, one), "x", increasing=False)
        disc = cast(F, -16) * (cast(F, 4) * a**3 + cast(F, 27) * b**2)
        j = cast(F, -110592) * a**3 / disc if disc != zero else None

        @classmethod
        def __repr__(self) -> str:
            return f"y^2 = {f_} over {one.__class__}"

        # @classmethod
        # def discriminant(self):
        #    s = str(self.disc)
        #    if s[0] == '(' and s[-1] == ')' and s[1:-1].find('(') < 0:
        #        return s[1:-1]
        #    else:
        #        return s

        # @classmethod
        # def j_invariant(self):
        #    s = str(self.j)
        #    if s[0] == '(' and s[-1] == ')' and s[1:-1].find('(') < 0:
        #        return s[1:-1]
        #    else:
        #        return s

    class WeierstrassCurve(EllipticCurve[F], metaclass=WeierstrassCurve_):
        def __init__(self, x: F = zero, y: F = zero, z: F = one) -> None:
            """projective coordinates [x: y: z]"""

            # x = one * x; y = one * y; z = one * z

            if debug:
                if z != zero:
                    if (y / z) ** 2 != f_(x / z):
                        raise ValueError(
                            (
                                f"({x/z}, {y/z}) = [{x}: {y}: {z}] is not on y^2 = {f_}: "
                                f"y^2 = {(y/z)**2} != {f_(x/z)}"
                            )
                        )
                else:
                    if not (x == 0 and y != 0):
                        raise ValueError(f"[{x}: {y}: {z}] is not on {y**2} = {f_}")

            self.co = (x, y, z)

        def __add__(self, other: EllipticCurve[F]) -> EllipticCurve[F]:

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
                return self.double() if t0 == t1 else self.__class__(zero, one, zero)

            u = u0 - u1
            u2 = u * u
            u3 = u2 * u
            t = t0 - t1
            v = self.co[2] * other.co[2]
            w = t * t * v - u2 * (u0 + u1)

            return self.__class__(u * w, t * (u0 * u2 - w) - t0 * u3, u3 * v)

        def __eq__(self, other: object) -> bool:
            if isinstance(other, EllipticCurve):
                if self.co[2] == 0 == other.co[2]:
                    return True
                if self.co[2] == other.co[2]:
                    return all([self.co[i] == other.co[i] for i in range(2)])
                elif self.co[2] * other.co[2] != 0:
                    return all(
                        [
                            self.co[i] / self.co[2] == other.co[i] / other.co[2]
                            for i in range(2)
                        ]
                    )
            if isinstance(other, int) and other == 0:
                return self.co[2] == zero
            return NotImplemented

        def __mul__(self, n: int) -> EllipticCurve[F]:
            if n == 0:
                return self.__class__(zero, one, zero)
            elif n == 1:
                return self
            factor = self.__mul__(n // 2)
            if n % 2 == 0:
                return factor.double()
            else:
                return self + factor.double()

        def double(self) -> EllipticCurve[F]:
            if self.co[2] == 0 or self.co[1] == 0:
                return self.__class__(zero, one, zero)
            else:
                t = 3 * self.co[0] ** 2 + a * self.co[2] ** 2
                u = 2 * self.co[1] * self.co[2]
                v = 2 * u * self.co[0] * self.co[1]
                w = t * t - 2 * v
                return self.__class__(
                    u * w, t * (v - w) - 2 * (u * self.co[1]) ** 2, u**3
                )

        def __rmul__(self, n: int) -> EllipticCurve[F]:
            return self.__mul__(n)

        def __neg__(self) -> EllipticCurve[F]:
            if self.co[2] == 0 or self.co[1] == 0:
                return self
            else:
                return self.__class__(self.co[0], -self.co[1], self.co[2])

        def __sub__(self, other: EllipticCurve[F]) -> EllipticCurve[F]:
            return self.__add__(-other)

        def __str__(self: EllipticCurve[F]) -> str:
            if self.co[2] == 0:
                return f"[{self.co[0]}: {self.co[1]}: {self.co[2]}]"
            else:
                return f"({self.co[0]/self.co[2]}, {self.co[1]/self.co[2]})"

        def __repr__(self: EllipticCurve[F]) -> str:
            if self.co[2] == 0:
                return f"[{self.co[0]}: {self.co[1]}: {self.co[2]}] on {self.__class__}"
            else:
                return f"({self.co[0]/self.co[2]}, {self.co[1]/self.co[2]}) on {self.__class__}"

        def __hash__(self: EllipticCurve[F]) -> int:
            # need to hash unique coords so do something like this:
            if self.co[2] == 0:
                return hash((zero, one, zero))
            else:
                return hash((self.co[0] / self.co[2], self.co[1] / self.co[2]))

    return WeierstrassCurve


EllCurve = Weierstrass

if __name__ == "__main__":

    import doctest

    doctest.testmod()
