from polylib.polynomial import Field
from polylib import Polynomial
import copy

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

class AlgebraicCurve:
    pass

#    def __init(self, f: FPolynomial):  # f(x,y) over algebraically closed field
#        pass


def EllCurve(a: Field, b: Field) -> AlgebraicCurve:
    """Return a class whose instances are elements of y^2=x^3+ax+b.

    This implements an elliptic curve in Weierstrauss form. The returned
    class allows one to work in the curves k-rational points where the
    field k is that of the arguments to the paramters a and b.

    Examples:

        >>> from numlib import Zmod
        >>> GF = Zmod(7)
        >>> E = EllCurve(GF(1), GF(1))
        >>> E
        y^2 = x^3 + x + 1 over Z/7
        >>> E.j  # the j-invariant for the curve
        1 + <7>

        When defining points on E, the type of the coefficient is inferred:

        >>> pt = E(2, 5)  # No need for E(GF(2), GF(5))
        >>> pt
        (2, 5) on y^2 = x^3 + x + 1 over Z/7
        >>> print(pt)
        (2, 5)
        >>> print(-pt)
        (2, 2)

        The point at infinity, (0: 1: 0), is the identity; for instance:

        >>> print(E(0,1,0) + E(2,5))
        (2, 5)

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
        >>> for pair in iproduct(GF, GF):
        ...     try:
        ...         print(E(*pair))
        ...     except:
        ...         pass
        (0, 1)
        (2, 2)
        (2, 5)
        (0, 6)

        Alternatively, use a faster builtin method:

        >>> len({pt for pt in E})  # E generates the finite points
        4

        This curve has order 5 and hence is cyclic. Every non-identity
        element is a generator:

        >>> for i in range(1, 6):
        ...     print(i * (E(2, 5)))
        (2, 5)
        (0, 6)
        (0, 1)
        (2, 2)
        [0: 1: 0]
    """
    global Weierstrass

    one = (a*b)**0

    aa = copy.copy(a)
    if isinstance(a,Polynomial) and a.x.find('(') < 0 and a.x.find(')') < 0:
        aa.x  = '('+a.x+')'

    bb = copy.copy(b)
    if isinstance(b,Polynomial) and b.x.find('(') < 0 and b.x.find(')') < 0:
        bb.x  = '(' + b.x + ')'

    f_ =  Polynomial((bb, aa, 0*one, one), 'x', spaces = True, increasing = False)

    class EllipticCurve(type):

        f =  Polynomial((b, a, 0*one, one), 'x', increasing = False)
        disc = -16 * (4 * a ** 3 + 27 * b ** 2)
        j = -110592 * a ** 3 / disc if disc != one * 0 else None

        def __iter__(self):
            """Yield the finite points of the curve."""

            y2s = {} # map all squares y2 in the field k to {y | y^2 = y2}
            fs = {}  # map all outputs fs = f(x), x in k, to {x | f(x) = fs}

            # build y2s and fs
            for x in (a * b).__class__:
                x2 = x ** 2
                y2s.setdefault(x2, []).append(x)
                fs.setdefault(x2 * x + a * x + b, []).append(x)

            # yield all points of the curve
            for y2 in y2s.keys():
                for f in fs.keys():
                    if y2 == f:
                        for x in fs[f]:
                            for y in  y2s[y2]:
                                yield self(x, y)

        #@classmethod
        #def random_point(self):
        #    F = (a*b).__class__
        #    if F.char() % 4 == 3
        #        for x in F:
        #            fx = self.f.of(x)
        #            if fx ** (p - 1) // 2 == 1
        #            y = self.f.of(x) ** (p + 1) // 4
        #        rand
        #    else:
        #        return NotImplemented

        @classmethod
        def __repr__(self):
            if (a*b).__class__.__name__ in {'ZModPrime', 'Zmod', 'FPmod_'}:
                class_string = (a*b).__class__.__class__.__str__()
            else:
                class_string = (a*b).__class__.__name__
            return f"y^2 = {f_} over {class_string}"

        @classmethod
        def discriminant(self):
            s = str(self.disc)
            if s[0] == '(' and s[-1] == ')' and s[1:-1].find('(') < 0:
                return s[1:-1]
            else:
                return s

        @classmethod
        def j_invariant(self):
            s = str(self.j)
            if s[0] == '(' and s[-1] == ')' and s[1:-1].find('(') < 0:
                return s[1:-1]
            else:
                return s

    class Weierstrass(AlgebraicCurve, metaclass = EllipticCurve):

        def __init__(self, x = 0, y = 0, z = None):
            """projective coordinates [x: y: z]"""

            z = one if z is None else z

            x = one * x; y = one * y; z = one * z

            if z != 0*one:
                if (y/z)**2 != f_.of(x/z):
                    raise ValueError(
                        (f"({x/z}, {y/z}) = [{x}: {y}: {z}] is not on y^2 = {f_}: "
                        f"y^2 = {(y/z)**2} != {f_.of(x/z)}")
                    )
            else:
                if not (x == 0 and y != 0):
                    raise ValueError(f"[{x}: {y}: {z}] is not on {y2} = {f}")

            self.co = (x, y, z)

        def __add__(self: EllCurve, other: EllCurve) -> EllCurve:

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
                return self.double() if t0 == t1 else self.__class__(0, 1, 0)

            u = u0 - u1
            u2 = u * u
            u3 = u2 * u
            t = t0 - t1
            v = self.co[2] * other.co[2]
            w = t * t * v - u2 * (u0 + u1)

            return self.__class__(u * w, t * (u0 * u2 - w) - t0 * u3, u3 * v)

        def __eq__(self, other):
            if self.co[2] ==  0 == other.co[2]:
                return True
            if self.co[2] == other.co[2]:
                return all([self.co[i] == other.co[i] for i in range(2)])
            elif self.co[2] * other.co[2] != 0:
                return all([self.co[i]/self.co[2] == other.co[i]/other.co[2] for i in range(2)])
            return False

        def __mul__(self: EllCurve, n: int) -> EllCurve:

            if n == 0:
                return self.__class__(0, 1 ,0)
            elif n == 1:
                return self
            factor = self.__mul__(n//2)
            if n % 2 == 0:
                return factor.double()
            else:
                return self + factor.double()

        def double(self):
            if self.co[2] == 0 or self.co[1] == 0:
                return self.__class__(0,1,0)
            else:
                t = 3 * self.co[0]**2 + a * self.co[2]**2
                u = 2 * self.co[1] * self.co[2]
                v = 2 * u * self.co[0] * self.co[1]
                w = t * t - 2 * v
                return self.__class__(
                    u * w,
                    t * (v - w) - 2 * (u * self.co[1]) ** 2,
                    u**3)

        def __rmul__(self: EllCurve, n: int) -> EllCurve:
            return  self.__mul__(n)

        def __neg__(self):
            if self.co[2] == 0 or self.co[1] == 0:
                return self
            else:
                return self.__class__(self.co[0], -self.co[1], self.co[2])

        def __sub__(self: EllCurve, other: EllCurve):
            return self.__add__ (-other)

        def __str__(self):
            if self.co[2] == 0:
                return f"[{self.co[0]}: {self.co[1]}: {self.co[2]}]"
            else:
                return f"({self.co[0]/self.co[2]}, {self.co[1]/self.co[2]})"

        def __repr__(self):
            if self.co[2] == 0:
                return f"[{self.co[0]}: {self.co[1]}: {self.co[2]}] on {self.__class__}"
            else:
                return f"({self.co[0]/self.co[2]}, {self.co[1]/self.co[2]}) on {self.__class__}"

        def __hash__(self) -> int:
            return hash(self.co)

    return Weierstrass

def Edwards(a: Field, b: Field) -> AlgebraicCurve:
    pass

def Montgomery(a: Field, d: Field) -> AlgebraicCurve:
    pass

if __name__ == "__main__":

    import doctest

    doctest.testmod()
