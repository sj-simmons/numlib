# numlib

### Contents
* [Installation](#installation)
* [Quick start](#quick-start)
* [Basic examples](#basic-examples)
* [Finite fields](#finite-fields)
* [Algebraic number fields](#algebraic-number-fields)
* [Cyclotomic fields](#cyclotomic-fields)
* [Elliptic Curves](#elliptic-curves)

## Installation

From your command line:

```shell
pip install numlib --user
```
or, for the latest development version:
```shell
pip install git+https://github.com/sj-simmons/numlib.git --user
```

This library depends heavily on another Python library,
[polylib](https://github.com/sj-simmons/polylib),
which can be installed with **pip install polylib --user**.

Note: for your Python installation, you may need to type **pip** instead of **pip3**
in the commands above. You can check whether **pip** points to its Python3 version
by issuing the command  **pip -V** at your commandline.

## Quick start

#### Define and work in a Galois field

```pycon
>>> import numlib as nl
>>> GF = nl.GaloisField(7, 3)
>>> print(GF)
Z/7[t]/<-3+3t^2+t^3>
>>> GF.order  # 7^3
(7, 3)
```
Element of GF are (equivalence classes of) polynomials.
The easiest way to define a particular such element is
to use an indeterminant.
```
>>> t = GF()
>>> 1 + 2*t**2
1+2t^2 + <-3+3t^2+t^3>
>>> t**100
2+t-3t^2 + <-3+3t^2+t^3>
```
The element **t**, created as above, is always a
generator for GaloisField's multiplicative group of
units:
```pycon
>>> nl.mulorder(t)  # compute the multiplicative order
342
```
If you want something in GF's prime field when using
an indeterminant:
```pycon
>>> 2*t**0  
2 + <-3+3t^2+t^3>
>>> # the line above is equivalent to
>>> Zmod7 = Zmodp(7)
>>> GF([Zmod7(2)])
2 + <-3+3t^2+t^3>
```
You can run through all non-zero elements of GF by
taking powers of **t**; but this is faster:
```pycon
>>> for element in GF:
...     print(element)
...
-3t^2-3t-3
-3t^2-3t-2
-3t^2-2t-3
-3t^2-2t-2
-2t^2-3t-3
   ...
```

#### Define an elliptic curve using its short Weierstrass form over the Galois field GF

```pycon
>>> E = nl.EllCurve(2*t**2+t+5, 3*t**0)
>>> E
y^2 = x^3 + (2t^2+t-2)x + 3 over Z/7[t]/<t^3+3t^2-3>
```
For small order curves, one can generate all
finite <img alt="$\operatorname{GF}(7^3)$" src="svgs/183c0fcbf5555289982cc209850085a7.svg" valign=-4.109589000000009px width="52.00929029999999pt" height="17.4904653pt"/>-rational points:
```pycon
>>> for point in nl.finite(E):
...    print(point)
...
(t^2+2t+3, -3t^2-3t-2)
(t^2+2t+3, 3t^2+3t+2)
(2t^2-2, -3t^2-2t-2)
(2t^2-2, 3t^2+2t+2)
(3t^2+t-1, -2t^2-3t-3)
        ...
```
Counting the point at infinity, the total number of <img alt="$\operatorname{GF}(7^3)$" src="svgs/183c0fcbf5555289982cc209850085a7.svg" valign=-4.109589000000009px width="52.00929029999999pt" height="17.4904653pt"/>-rational
points on this curve is 378:
```pycon
>>> len(list(nl.affine(E)))
377
```
Warning: the last command can take essentially forever for large curves even if the
coefficients lie in field of somewhat manageable size &mdash; which is to say that,
when implementing crypto-systems, one gets more encryption bang for one's buck by
replacing a finite field with an elliptic curve over a finite field.

## Basic examples

First, numlib provides various number theoretic utilities that may prove useful.
```pycon
>>> from numlib import gcd, xgcd
>>> gcd(143, 2662)  # 11 since 143 = 11*13 and 2662 = 2*11^3
>>> xgcd(143, 2662)  # (11, -93, 5) since 11 = -93 * 143 + 5 * 2662
>>> -93 * 143 + 5 * 2662  #  = 11
```
Peruse all available utilities and see more examples by issuing the command
**pydoc3 numlib.utils** at your commandline or by typing, in the interpreter,
```pycon
>>> import numlib
>>> help(numlib.utils)
```

To work, in the interpreter, with the integers, <img alt="$\mathbb{Z}$" src="svgs/b9477ea14234215f4d516bad55d011b8.svg" valign=0.0px width="10.95894029999999pt" height="11.324195849999999pt"/>, modulo a positive integer <img alt="$n:$" src="svgs/2373b2f6e7b0d17338529bb5ba49121a.svg" valign=0.0px width="18.99919889999999pt" height="7.0776222pt"/>
```pycon
>>> from numlib import Zmod
>>>
>>> R = Zmod(15)  # R is a class that returns instances of integers modulo 15
>>> R(37)  # 7 mod 15
>>> print(R(37))  # 7
>>> x = R(9); y = R(30)
>>> z = 301*x + y**2 + x*y + 1000  # We can do arithmetic and integers such as 301
>>> z  # 4 mod 15                    and 1000 are coerced to integers mod 15
>>> z.isunit()  # True
```
R in the previous code block is now a class but think of it as a
type: the ring, <img alt="$\mathbb{Z}/15$" src="svgs/a3477f7ac2c6287b1119be1b65751b9f.svg" valign=-4.109589000000009px width="35.61656834999999pt" height="16.438356pt"/>, of integers modulo 15.
```pycon
>>> print(R)  # Z/15
```
Some simple class-level methods are available:
```pycon
>>> print(', '.join(str(x) for x in R.units())) # R.units() is a Python generator
1, 2, 4, 7, 8, 11, 13, 14  # the multiplicative group of units in Z/15
>>>
>>> phi = len(list(R.units()))  # the order of the group of units
>>> phi  # phi(15) = 8 where phi is Euler's totient function
>>>
>>> R(7)**phi == 1  # True
>>>
>>> # Number theoretically, the last equality holds by Euler's Theorem;
>>> # group theoretically, it follows from Lagrange's Theorem that the
>>> # order of an element of a group must divide the order of the group.
>>>
>>> # Let us find the actual order of R(7):
>>> for i in range(phi):
...     if R(7)**(i+1) == 1:
...         print(i+1)
...         break
4
```
If we prefer, we can balance the representatives of **Z/n** about 0:
```pycon
>>> list(Zmod(10, negatives=True))  # [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
>>> list(Zmod(10, negatives=True).units())  # [-3, -1, 1, 3]
```

#### Exercises
1. Find the multiplicative inverse of 2022 modulo 2027 or show that no such inverse exists.
2. Write a program that lists all non-zero elements of <img alt="$\mathbb{Z}/60$" src="svgs/0a9a6cf9aecafe4d6b38b15898ed04e7.svg" valign=-4.109589000000009px width="35.61656834999999pt" height="16.438356pt"/> along
   with their multiplicative orders.  
  (a) Is the multiplicative group of units in <img alt="$\mathbb{Z}/60$" src="svgs/0a9a6cf9aecafe4d6b38b15898ed04e7.svg" valign=-4.109589000000009px width="35.61656834999999pt" height="16.438356pt"/> cyclic?  
  (b) If so, is the group of units in <img alt="$\mathbb{Z}/n$" src="svgs/5a25068b686730b0d5c6d3c047688395.svg" valign=-4.109589000000009px width="29.04502589999999pt" height="16.438356pt"/> always cyclic?  
3. In the ring <img alt="$\mathbb{Z}/27720$" src="svgs/436dfd6ce001d9cbca71419c5cb63186.svg" valign=-4.109589000000009px width="60.27419804999999pt" height="16.438356pt"/>, solve each of the following equations for <img alt="$x,$" src="svgs/380aab7befb490c9e8b8027e557ed545.svg" valign=-3.1963502999999895px width="13.96121264999999pt" height="10.2739725pt"/>
   or argue that no solution exists.  
  (a) <img alt="$26x = 1$" src="svgs/7deb653917a22092554633006944ac27.svg" valign=0.0px width="55.97024729999999pt" height="10.5936072pt"/>  
  (b) <img alt="$833x = 1$" src="svgs/9fec9726d67beb2280de654054069be9.svg" valign=0.0px width="64.18945664999998pt" height="10.5936072pt"/>  
  (c) <img alt="$143x -7  = 2655$" src="svgs/9b15c3cd44a56b827518f28f8558800c.svg" valign=-1.3698745499999938px width="117.15748439999997pt" height="11.96348175pt"/>  
4. What are the conditions under which <img alt="$ax = b$" src="svgs/2d669bf55f3460fc469e923f439de136.svg" valign=0.0px width="47.05656944999999pt" height="11.4155283pt"/> always has a solution in <img alt="$\mathbb{Z}/n$" src="svgs/5a25068b686730b0d5c6d3c047688395.svg" valign=-4.109589000000009px width="29.04502589999999pt" height="16.438356pt"/>?
   Are those solutions unique?

### [Finite fields](https://en.wikipedia.org/wiki/Finite_field)

The quotient ring <img alt="$\mathbb{Z}/p$" src="svgs/73c9c321abfaf561fdbde1304ada12c1.svg" valign=-4.109589000000009px width="27.448716899999987pt" height="16.438356pt"/> is a *field* if
<img alt="$p$" src="svgs/2ec6e630f199f589a2402fdf3e0289d5.svg" valign=-3.1963502999999895px width="8.270567249999992pt" height="10.2739725pt"/> is prime since then every non-zero element has a multiplicative inverse. For example:
```pycon
>>> GF = Zmod(23)  # the field Z/23
>>> GF(8)**-1  #  8 has inverse 3 mod 23; equivalently, we could have typed 1/GF(8)
>>> len(list(PF.units())) == 22  # every non-zero element is a unit (i.e., invertible)
>>> GF.isField()   # True
```
Finite fields are often called Galois Fields, hence our notation **GF** in the last code block.

Both **GF.units()** above and, in fact, **GF** itself are Python generators.  This can be useful,
for instance, if we want to brute-force verify that the cubic <img alt="$1 + x^2 + 3x^3$" src="svgs/ea3a9f0df58be04c4230e07c439287f4.svg" valign=-1.3698729000000083px width="89.33778314999999pt" height="14.750749199999998pt"/>
is irreducible over <img alt="$\mathbb{Z}/17$" src="svgs/7b7a91d08bdb0f2c3ad080ff4ef2ae37.svg" valign=-4.109589000000009px width="35.61656834999999pt" height="16.438356pt"/>. For this, we use
[polylib](https://github.com/sj-simmons/polylib) (install with **pip install polylib --user**).
```python
from numlib import Zmod
from polylib import FPolynomial

f = FPolynomial([1, 0, 1, 3])  # 1 + x^2 +3x^3
GF = Zmod(17)
result = "irreducible"
for x in GF:
    if f.of(x) == 0:
        result = "reducible"
        break

print(f"{f} is {result} in Z/17[x]")
```
The only way that a cubic <img alt="$f(x)$" src="svgs/7997339883ac20f551e7f35efff0a2b9.svg" valign=-4.109589000000009px width="31.99783454999999pt" height="16.438356pt"/> with coefficients in a field <img alt="$\mathbb{K}$" src="svgs/9ebeacdd09c18ad447a4e29b9039c3b0.svg" valign=0.0px width="12.785434199999989pt" height="11.324195849999999pt"/> can be
reducible if it has a factor of the form <img alt="$x-\alpha$" src="svgs/ad1baf7900c88bcd43bffd2907b2a45e.svg" valign=-1.3698745499999996px width="40.06268309999999pt" height="10.958925449999999pt"/> for some <img alt="$\alpha\in\mathbb{K}$" src="svgs/0160bef1ca5087d29cf21b4cb7388b47.svg" valign=-0.6427030499999994px width="43.45307009999999pt" height="11.966898899999999pt"/>;
in the program above, we have used that observation along with the fact that
<img alt="$x-\alpha$" src="svgs/ad1baf7900c88bcd43bffd2907b2a45e.svg" valign=-1.3698745499999996px width="40.06268309999999pt" height="10.958925449999999pt"/> divides <img alt="$f(x)$" src="svgs/7997339883ac20f551e7f35efff0a2b9.svg" valign=-4.109589000000009px width="31.99783454999999pt" height="16.438356pt"/> if and only if <img alt="$f(\alpha)=0$" src="svgs/a9f5159b525b671711a095ce0f11b1e5.svg" valign=-4.109589000000009px width="63.31618424999999pt" height="16.438356pt"/> (which follows from unique
factorization in the polynomial ring <img alt="$\mathbb{K}[x]$" src="svgs/a9a65fe3bdf866d78c9305d487aa20d4.svg" valign=-4.109589000000009px width="31.31287004999999pt" height="16.438356pt"/> &mdash; concretely, since the
division algorithm holds in <img alt="$\mathbb{K}[x]$" src="svgs/a9a65fe3bdf866d78c9305d487aa20d4.svg" valign=-4.109589000000009px width="31.31287004999999pt" height="16.438356pt"/>, we have <img alt="$f(x) = q(x)(x-\alpha) + f(\alpha)$" src="svgs/9c440d556f6597991e8b3319c558ab2e.svg" valign=-4.109589000000009px width="190.14261914999997pt" height="16.438356pt"/>).

#### Practical matters involving large integers

When we issue the command **Zmod(n)** where **n** is some integer, **numlib** tries
to figure out (using the function **numlib.isprime**) whether **n** is prime (so whether
**Zmod(n)** is in fact a field).  The function **numlib.isprime(n)** is fast but
it returns **True** when **n** is only *likely* prime.  Unexpected behavior may result
in the rare case that **numlib.isprime** is wrong.

If we already know that <img alt="$p$" src="svgs/2ec6e630f199f589a2402fdf3e0289d5.svg" valign=-3.1963502999999895px width="8.270567249999992pt" height="10.2739725pt"/> is a prime, then we can avoid problems by indicating that:
```pycon
>>> import numlib
>>> n = 258001471497710271176990892852404413747
>>> GF = numlib.Zmod(n, prime = True)  # no need for Zmod to check if n is prime
>>>
>>> numlib.isprime(n)   # True, so isprime() gets it right this time
>>> GF = numlib.Zmod(n) # so this is equivalent to above for this n
```

Moreover, if **R = Zmod(n)**, then the generator **R.units()** simply yields in
turn those elements of the generator **R** which are relatively prime
with **n**. If **n** is large, this leads to many applications of the
Euclidean algorithm.

If the large **n** is known to be prime, then we already know that
everything but zero in **R** is a unit; consider indicating that
(which also saves time when computing inverses) as above:
```pycon
>>> PF = Zmod(2**3021377-1, prime=True)  # a Mersenne prime with about 900K digits
>>> PF(2)**-1  # this still takes several seconds
```
#### Constructing finite fields

Above, we used **numlib** to quickly construct a finite field of
prime order:
```pycon
>>> from numlib import Zmod
>>> GF = Zmod(17)  # a primefield of order 17
>>> GF.isField()  # True
>>> len(list(GF.units()))  # 16
>>> GF(5)/GF(10)  # since GF is a field, you can divide, this is 9 mod 17
>>> 5/GF(10)  # integers are coerced to the primefield so this is equivalent
>>> GF(5)/10  # so is this
>>> GF(3)+50  # 2, coercion is implemented w/r to all field operations
```
Of course, we can replace 17 with any prime, <img alt="$p,$" src="svgs/88dda0f87c1bb00d3c39ce2f369504cf.svg" valign=-3.1963502999999895px width="12.836790449999992pt" height="10.2739725pt"/> thereby obtaining the field
<img alt="$\mathbb{Z}/p$" src="svgs/73c9c321abfaf561fdbde1304ada12c1.svg" valign=-4.109589000000009px width="27.448716899999987pt" height="16.438356pt"/> of order <img alt="$p.$" src="svgs/bb44dfbad95f04997776cfe375c4eac3.svg" valign=-3.1963502999999895px width="12.836790449999992pt" height="10.2739725pt"/>

Not all finite fields have prime order.  Any field, <img alt="$\mathbb{K}$" src="svgs/9ebeacdd09c18ad447a4e29b9039c3b0.svg" valign=0.0px width="12.785434199999989pt" height="11.324195849999999pt"/>, finite or
not, admits a ring homomorphism <img alt="$\mathbb{Z} \rightarrow \mathbb{K}$" src="svgs/79b8355e58e5550f35e57bba647b281e.svg" valign=0.0px width="49.31497394999999pt" height="11.324195849999999pt"/> defined by
<img alt="$n \mapsto n\cdot 1_\mathbb{K},$" src="svgs/3e9b41ef93f28c8a47aebc4c9a8659cf.svg" valign=-3.196350299999994px width="79.73338064999999pt" height="13.789957499999998pt"/> where <img alt="$1_\mathbb{K}$" src="svgs/9e2b12b4e9c581f122d9e198e257e958.svg" valign=-2.4657286499999937px width="17.16890339999999pt" height="13.059335849999998pt"/> denotes the multiplicative
identity of <img alt="$\mathbb{K}.$" src="svgs/27fe5f6477b011a64df56ef0dfde6c26.svg" valign=0.0px width="17.35165739999999pt" height="11.324195849999999pt"/> The kernel of this homomorphism is either <img alt="$\{0\}$" src="svgs/7d91d603191db068c91b88364ae8b148.svg" valign=-4.109589000000009px width="24.657628049999992pt" height="16.438356pt"/> or
the ideal <img alt="$\langle p\rangle = p\cdot\mathbb{Z}=\{np~|~n\in\mathbb{Z}\}$" src="svgs/9eea1126e077f3ea7ab6d11e894147eb.svg" valign=-4.109589000000009px width="187.01067824999998pt" height="16.438356pt"/> for some
prime <img alt="$p$" src="svgs/2ec6e630f199f589a2402fdf3e0289d5.svg" valign=-3.1963502999999895px width="8.270567249999992pt" height="10.2739725pt"/> called the *characteristic* of <img alt="$\mathbb{K};$" src="svgs/a2ae22f0c875f5402d1086a59df6854d.svg" valign=-3.1963502999999998px width="17.35165739999999pt" height="14.52054615pt"/> hence, the image of the
natural ring homomorphism <img alt="$\mathbb{Z} \rightarrow \mathbb{K}$" src="svgs/79b8355e58e5550f35e57bba647b281e.svg" valign=0.0px width="49.31497394999999pt" height="11.324195849999999pt"/> is a copy
of either <img alt="$\mathbb{Z}$" src="svgs/b9477ea14234215f4d516bad55d011b8.svg" valign=0.0px width="10.95894029999999pt" height="11.324195849999999pt"/> or <img alt="$\mathbb{Z}/p$" src="svgs/73c9c321abfaf561fdbde1304ada12c1.svg" valign=-4.109589000000009px width="27.448716899999987pt" height="16.438356pt"/> living inside <img alt="$\mathbb{K}$" src="svgs/9ebeacdd09c18ad447a4e29b9039c3b0.svg" valign=0.0px width="12.785434199999989pt" height="11.324195849999999pt"/>
(in the latter case, <img alt="$p$" src="svgs/2ec6e630f199f589a2402fdf3e0289d5.svg" valign=-3.1963502999999895px width="8.270567249999992pt" height="10.2739725pt"/> must be prime since otherwise <img alt="$\mathbb{K}$" src="svgs/9ebeacdd09c18ad447a4e29b9039c3b0.svg" valign=0.0px width="12.785434199999989pt" height="11.324195849999999pt"/> would have
non-trivial, non-invertible zero-divisors).

In the case that the natural map <img alt="$\mathbb{Z} \rightarrow \mathbb{K}$" src="svgs/79b8355e58e5550f35e57bba647b281e.svg" valign=0.0px width="49.31497394999999pt" height="11.324195849999999pt"/> is injective,
so that its image is <img alt="$\mathbb{Z},$" src="svgs/faeb8f2042ba0fdd215f3dc0540f7293.svg" valign=-3.1963502999999998px width="15.52516514999999pt" height="14.52054615pt"/> then the characteristic of <img alt="$\mathbb{K}$" src="svgs/9ebeacdd09c18ad447a4e29b9039c3b0.svg" valign=0.0px width="12.785434199999989pt" height="11.324195849999999pt"/> is
defined to be zero and the field of fractions of the image, which is isomorphic
to <img alt="$\mathbb{Q}$" src="svgs/0f452ec0bcf578fa387e4857f80f03f4.svg" valign=-2.739730950000001px width="12.785434199999989pt" height="14.0639268pt"/>, is called the *prime field* of the infinite field <img alt="$\mathbb{K}.$" src="svgs/27fe5f6477b011a64df56ef0dfde6c26.svg" valign=0.0px width="17.35165739999999pt" height="11.324195849999999pt"/>
If <img alt="$\mathbb{Z} \rightarrow \mathbb{K}$" src="svgs/79b8355e58e5550f35e57bba647b281e.svg" valign=0.0px width="49.31497394999999pt" height="11.324195849999999pt"/> is not injective then the characteristic,
<img alt="$\operatorname{char}(K),$" src="svgs/5754eaf5f3833d216505138560f9876b.svg" valign=-4.109589000000009px width="63.12797204999998pt" height="16.438356pt"/> is *positive* &mdash; some prime <img alt="$p$" src="svgs/2ec6e630f199f589a2402fdf3e0289d5.svg" valign=-3.1963502999999895px width="8.270567249999992pt" height="10.2739725pt"/> &mdash; and the
*prime field* of <img alt="$\mathbb{K}$" src="svgs/9ebeacdd09c18ad447a4e29b9039c3b0.svg" valign=0.0px width="12.785434199999989pt" height="11.324195849999999pt"/> is the image of
<img alt="$\mathbb{Z}$" src="svgs/b9477ea14234215f4d516bad55d011b8.svg" valign=0.0px width="10.95894029999999pt" height="11.324195849999999pt"/> in <img alt="$\mathbb{K}$" src="svgs/9ebeacdd09c18ad447a4e29b9039c3b0.svg" valign=0.0px width="12.785434199999989pt" height="11.324195849999999pt"/> (which is isomorphic to <img alt="$\mathbb{Z}/p$" src="svgs/73c9c321abfaf561fdbde1304ada12c1.svg" valign=-4.109589000000009px width="27.448716899999987pt" height="16.438356pt"/>).

Whether or not <img alt="$\mathbb{K}$" src="svgs/9ebeacdd09c18ad447a4e29b9039c3b0.svg" valign=0.0px width="12.785434199999989pt" height="11.324195849999999pt"/> is finite, it is a vector space over its prime field.
Suppose that a finite field, <img alt="$\mathbb{K}$" src="svgs/9ebeacdd09c18ad447a4e29b9039c3b0.svg" valign=0.0px width="12.785434199999989pt" height="11.324195849999999pt"/>, has dimension <img alt="$n$" src="svgs/55a049b8f161ae7cfeb0197d75aff967.svg" valign=0.0px width="9.86687624999999pt" height="7.0776222pt"/> as a finite-dimensional
vector space over its prime field of order <img alt="$p$" src="svgs/2ec6e630f199f589a2402fdf3e0289d5.svg" valign=-3.1963502999999895px width="8.270567249999992pt" height="10.2739725pt"/>, then <img alt="$K$" src="svgs/d6328eaebbcd5c358f426dbea4bdbf70.svg" valign=0.0px width="15.13700594999999pt" height="11.232861749999998pt"/> necessarily has order
<img alt="$q = p^n$" src="svgs/2959e1db6cc745aa33a34574db47eec4.svg" valign=-3.1963519500000044px width="46.24230764999999pt" height="14.116037099999998pt"/>. How can we construct a finite field of prime power order?

To construct such a field, we need a  polynomial <img alt="$f(x)$" src="svgs/7997339883ac20f551e7f35efff0a2b9.svg" valign=-4.109589000000009px width="31.99783454999999pt" height="16.438356pt"/> of degree
<img alt="$n$" src="svgs/55a049b8f161ae7cfeb0197d75aff967.svg" valign=0.0px width="9.86687624999999pt" height="7.0776222pt"/> that is irreducible over <img alt="$\mathbb{Z}/p.$" src="svgs/62cd3ae2cb973630fe60c1b010f2057e.svg" valign=-4.109589000000009px width="32.014941749999984pt" height="16.438356pt"/>  Then we simply quotient
the polynomial ring <img alt="$\mathbb{Z}/p[x]$" src="svgs/2012540586a66e3dc43394a218c44338.svg" valign=-4.109589000000009px width="45.97615274999999pt" height="16.438356pt"/> (consisting of all polynomials with
coefficients in <img alt="$\mathbb{Z}/p$" src="svgs/73c9c321abfaf561fdbde1304ada12c1.svg" valign=-4.109589000000009px width="27.448716899999987pt" height="16.438356pt"/>) by the ideal <img alt="$\langle f(x)\rangle$" src="svgs/c6084175338a8a0f04a0ff398ca276b0.svg" valign=-4.109589000000009px width="44.78326709999998pt" height="16.438356pt"/>
generated by the irreducible <img alt="$f(x)\in \mathbb{Z}/p[x].$" src="svgs/74a3db77517bf8b7d263df20d598b009.svg" valign=-4.109589000000009px width="102.63134969999999pt" height="16.438356pt"/>

Above, we checked that <img alt="$1 + x^2 + 3x^3$" src="svgs/ea3a9f0df58be04c4230e07c439287f4.svg" valign=-1.3698729000000083px width="89.33778314999999pt" height="14.750749199999998pt"/> is irreducible over
<img alt="$\mathbb{Z}/17.$" src="svgs/e2d61fa49b6fc21b56dc34ecbfeb0580.svg" valign=-4.109589000000009px width="40.18279319999999pt" height="16.438356pt"/>  We can construct elements of the finite field of
order <img alt="$17^3$" src="svgs/d07c190856b363b8ab7555fd37127d7d.svg" valign=0.0px width="22.990966349999994pt" height="13.380876299999999pt"/> as follows.

```pycon
>>> from numlib import Zmod, FPmod
>>> from polylib import FPolynomial
>>>
>>> PF = Zmod(17)  # the prime field
>>> x = FPolynomial([0, PF(1)])  # indeterminant for polys over PF
>>> f = 1+x**2+3*x**3  # FPolynomial((1 mod 17, 0 mod 17, 1 mod 17, 3 mod 17))
>>> GF = FPmod(f)
>>> print(GF)  # Z/17[x]/<1 + x^2 + 3x^3>
>>>
>>> # now we can use the class GF to define elements of the Galois field:
>>> p1 = GF([1,2,3,4])  # 11 + 2x + 13x^2 mod 1 + x^2 + 3x^3
>>> p1**-1  # 4 + 12x + 14x^2 mod 1 + x^2 + 3x^3
```
Suppose, though, that we want to work conveniently in
<img alt="$\mathbb{Z}/p[x] / \langle 1 + x^2 + 3x^3\rangle$" src="svgs/9746574a81479c795f4b26e528256ce7.svg" valign=-4.109589000000009px width="157.14049229999998pt" height="17.4904653pt"/>
with an indeterminant.  Continuing with the interactive session above
```pycon
>>> # currently x is an element of Z/17[x]
>>> x  # FPolynomial((0 mod 17, 1 mod 17))
>>> x**100 # this is x^100 in Z/17[x]
>>> GF(x**100) # 16 + 7x + x^2 mod 1 + x^2 + 3x^3  <- now it's in the Galois field
>>>
>>> # the notation is clearer if we change the indeterminant to 't'
>>> t = GF([0, 1], 't')
>>> t**100  # 16 + 7t + t^2 mod 1 + t^2 + 3t^3
>>> print(t**100)  # 16 + 7t + t^2
>>>
>>> # alternatively, we can just do this:
>>> t = GF(x, 't')  # t mod 1 + t^2 + 3t^3
>>> t**100  # 16 + 7t + t^2 mod 1 + t^2 + 3t^3
```

Quotienting <img alt="$\mathbb{Z}/p[x]$" src="svgs/2012540586a66e3dc43394a218c44338.svg" valign=-4.109589000000009px width="45.97615274999999pt" height="16.438356pt"/> by the ideal
<img alt="$\langle f(x)\rangle$" src="svgs/c6084175338a8a0f04a0ff398ca276b0.svg" valign=-4.109589000000009px width="44.78326709999998pt" height="16.438356pt"/> generated by an irreducible is wholly analogous to
quotienting <img alt="$\mathbb{Z}$" src="svgs/b9477ea14234215f4d516bad55d011b8.svg" valign=0.0px width="10.95894029999999pt" height="11.324195849999999pt"/> by the ideal <img alt="$\langle p\rangle=p\mathbb{Z}$" src="svgs/54b1866a4895aaa9ac5506d0f2ebdef8.svg" valign=-4.109589000000009px width="62.20313879999999pt" height="16.438356pt"/> generated by
a prime number.  We get a field in both cases, roughly due to the fact that
dividing by an irreducible, respectively, a prime, leaves no chance for a
zero divisor.  In both cases, we are quotienting in a
[principal ideal domain](https://en.wikipedia.org/wiki/Principal_ideal_domain).
In fact, both <img alt="$\mathbb{K}[x],$" src="svgs/97c55d9840f914b3324602cf9722eba1.svg" valign=-4.109589000000009px width="35.87909324999999pt" height="16.438356pt"/> where <img alt="$\mathbb{K}$" src="svgs/9ebeacdd09c18ad447a4e29b9039c3b0.svg" valign=0.0px width="12.785434199999989pt" height="11.324195849999999pt"/> is a field, and <img alt="$\mathbb{Z}$" src="svgs/b9477ea14234215f4d516bad55d011b8.svg" valign=0.0px width="10.95894029999999pt" height="11.324195849999999pt"/> are
[Euclidean domains](https://en.wikipedia.org/wiki/Euclidean_domain) with Euclidean
function <img alt="$f(x) \mapsto \operatorname{degree}(f(x)),$" src="svgs/6807c902d5db5f79d67e0d27346102d6.svg" valign=-4.109589000000009px width="152.6257425pt" height="16.438356pt"/> respectively, <img alt="$n \mapsto |n|.$" src="svgs/5599d77a12164c2fdbb576da7b9399b3.svg" valign=-4.109589000000009px width="59.003024849999996pt" height="16.438356pt"/>
In a Euclidean domain, we can carry out the division algorithm.

Why is every ideal in <img alt="$\mathbb{K}[x]$" src="svgs/a9a65fe3bdf866d78c9305d487aa20d4.svg" valign=-4.109589000000009px width="31.31287004999999pt" height="16.438356pt"/> a principal ideal?  Let <img alt="$f(x)$" src="svgs/7997339883ac20f551e7f35efff0a2b9.svg" valign=-4.109589000000009px width="31.99783454999999pt" height="16.438356pt"/> be a
fixed polynomial (not necessarily irreducible) of minimal degree in a given proper
ideal of <img alt="$\mathbb{K}[x].$" src="svgs/832c5c8e7f5fd14b4d7ef25fba499c72.svg" valign=-4.109589000000009px width="35.87909324999999pt" height="16.438356pt"/>  and let <img alt="$g(x)$" src="svgs/ffcbbb391bc04da2d07f7aef493d3e2a.svg" valign=-4.109589000000009px width="30.61077854999999pt" height="16.438356pt"/> be any other element of that ideal.  Then
the division algorithm (that is, long division of polynomials) yields <img alt="$q(x)$" src="svgs/03fdf3c6a83ab1f3f304bbc20f6cdadf.svg" valign=-4.109589000000009px width="30.108508649999987pt" height="16.438356pt"/> and
<img alt="$r(x)$" src="svgs/c73b6615f0c7bd519371e439b4efff6d.svg" valign=-4.109589000000009px width="30.05337719999999pt" height="16.438356pt"/> satisfying <img alt="$g(x) = q(x)\cdot f(x)+ r(x)$" src="svgs/1cbbdfe2c0c14d03cc2dd3865710a522.svg" valign=-4.109589000000009px width="176.65130175pt" height="16.438356pt"/> where either <img alt="$r(x)=0$" src="svgs/6827753f2932ceb8262b4e98e47efe28.svg" valign=-4.109589000000009px width="60.19021634999999pt" height="16.438356pt"/> or <img alt="$r(x)$" src="svgs/c73b6615f0c7bd519371e439b4efff6d.svg" valign=-4.109589000000009px width="30.05337719999999pt" height="16.438356pt"/>
has degree strictly less than that of <img alt="$f(x).$" src="svgs/b9a2b0716bafb328e9ab3e48ce228ef1.svg" valign=-4.109589000000009px width="36.56405774999999pt" height="16.438356pt"/>  But <img alt="$r(x) = g(x) - q(x)\cdot f(x)$" src="svgs/8d45dac10d3d2f82004f5ee67eda748d.svg" valign=-4.109589000000009px width="176.65130175pt" height="16.438356pt"/>
is an element of the given proper ideal; so that <img alt="$r(x)$" src="svgs/c73b6615f0c7bd519371e439b4efff6d.svg" valign=-4.109589000000009px width="30.05337719999999pt" height="16.438356pt"/> must be the zero polynomial
since, otherwise, the minimality of the degree of <img alt="$f(x)$" src="svgs/7997339883ac20f551e7f35efff0a2b9.svg" valign=-4.109589000000009px width="31.99783454999999pt" height="16.438356pt"/> would be violated.

We can always represent an equivalence class in the quotient ring
<img alt="$\mathbb{K}[x]/\langle f(x) \rangle$" src="svgs/d47bce196f301cd32daa2326d19d6309.svg" valign=-4.109589000000009px width="84.31534649999999pt" height="16.438356pt"/> with a polynomial of degree less than that of
<img alt="$f(x)$" src="svgs/7997339883ac20f551e7f35efff0a2b9.svg" valign=-4.109589000000009px width="31.99783454999999pt" height="16.438356pt"/> &mdash; just pick any polynomial <img alt="$g(x)$" src="svgs/ffcbbb391bc04da2d07f7aef493d3e2a.svg" valign=-4.109589000000009px width="30.61077854999999pt" height="16.438356pt"/> in the equivalence class and use the
division algorithm to write <img alt="$g(x) = q(x) \cdot f(x) + r(x);$" src="svgs/39d731602a444c958313357e58dfaed4.svg" valign=-4.109589000000009px width="181.21752659999999pt" height="16.438356pt"/> then <img alt="$r(x)$" src="svgs/c73b6615f0c7bd519371e439b4efff6d.svg" valign=-4.109589000000009px width="30.05337719999999pt" height="16.438356pt"/> is the
specified representative the equivalence class defined by <img alt="$g(x).$" src="svgs/6749db005d4148cf024c52c9e461f4d9.svg" valign=-4.109589000000009px width="35.17700339999999pt" height="16.438356pt"/> The difference
of such representatives of any two distinct classes, <img alt="$r_1(x)$" src="svgs/ceccee930abc3ac847c55d6c15da44b3.svg" valign=-4.109589000000009px width="36.971201849999986pt" height="16.438356pt"/> and <img alt="$r_2(x)$" src="svgs/a12792fdc6ba41c1ff8202fc6516507c.svg" valign=-4.109589000000009px width="36.971201849999986pt" height="16.438356pt"/>, cannot be
an element of the ideal <img alt="$\langle f(x)\rangle$" src="svgs/c6084175338a8a0f04a0ff398ca276b0.svg" valign=-4.109589000000009px width="44.78326709999998pt" height="16.438356pt"/>; hence, as a vector space,
<img alt="$\mathbb{K}[x]/\langle f(x) \rangle$" src="svgs/d47bce196f301cd32daa2326d19d6309.svg" valign=-4.109589000000009px width="84.31534649999999pt" height="16.438356pt"/> is precisely the set of polynomials in
<img alt="$\mathbb{K}[x]$" src="svgs/a9a65fe3bdf866d78c9305d487aa20d4.svg" valign=-4.109589000000009px width="31.31287004999999pt" height="16.438356pt"/> of degree less than the degree of <img alt="$f(x)$" src="svgs/7997339883ac20f551e7f35efff0a2b9.svg" valign=-4.109589000000009px width="31.99783454999999pt" height="16.438356pt"/> with addition and scalar
multiplication that of polynomials.

Returning to the case in which <img alt="$f(x)$" src="svgs/7997339883ac20f551e7f35efff0a2b9.svg" valign=-4.109589000000009px width="31.99783454999999pt" height="16.438356pt"/> is irreducible in <img alt="$\mathbb{K}[x],$" src="svgs/97c55d9840f914b3324602cf9722eba1.svg" valign=-4.109589000000009px width="35.87909324999999pt" height="16.438356pt"/> let
<img alt="$r(x)$" src="svgs/c73b6615f0c7bd519371e439b4efff6d.svg" valign=-4.109589000000009px width="30.05337719999999pt" height="16.438356pt"/> represent a non-zero element of <img alt="$\mathbb{K}[x]/\langle f(x) \rangle$" src="svgs/d47bce196f301cd32daa2326d19d6309.svg" valign=-4.109589000000009px width="84.31534649999999pt" height="16.438356pt"/> and
consider the ideal generated by <img alt="$r(x)$" src="svgs/c73b6615f0c7bd519371e439b4efff6d.svg" valign=-4.109589000000009px width="30.05337719999999pt" height="16.438356pt"/> and <img alt="$f(x).$" src="svgs/b9a2b0716bafb328e9ab3e48ce228ef1.svg" valign=-4.109589000000009px width="36.56405774999999pt" height="16.438356pt"/> This is a principal ideal,
of course, generated by some polynomial that divides both <img alt="$r(x)$" src="svgs/c73b6615f0c7bd519371e439b4efff6d.svg" valign=-4.109589000000009px width="30.05337719999999pt" height="16.438356pt"/> and <img alt="$f(x).$" src="svgs/b9a2b0716bafb328e9ab3e48ce228ef1.svg" valign=-4.109589000000009px width="36.56405774999999pt" height="16.438356pt"/>
An irreducible <img alt="$f(x)$" src="svgs/7997339883ac20f551e7f35efff0a2b9.svg" valign=-4.109589000000009px width="31.99783454999999pt" height="16.438356pt"/> is only divisible by units in <img alt="$\mathbb{K}$" src="svgs/9ebeacdd09c18ad447a4e29b9039c3b0.svg" valign=0.0px width="12.785434199999989pt" height="11.324195849999999pt"/>  so that
the ideal in question is the whole ring <img alt="$\mathbb{K}[x]$" src="svgs/a9a65fe3bdf866d78c9305d487aa20d4.svg" valign=-4.109589000000009px width="31.31287004999999pt" height="16.438356pt"/>. In other words,
there exist polynomials <img alt="$a(x)$" src="svgs/0786a7b732d7242200d22552fe3630f1.svg" valign=-4.109589000000009px width="30.869576099999993pt" height="16.438356pt"/> and <img alt="$b(x)$" src="svgs/dfe1b1c707e7d2ffd71fdb37b9b32166.svg" valign=-4.109589000000009px width="29.23521809999999pt" height="16.438356pt"/> in <img alt="$\mathbb{K}[x]$" src="svgs/a9a65fe3bdf866d78c9305d487aa20d4.svg" valign=-4.109589000000009px width="31.31287004999999pt" height="16.438356pt"/> satisfying
<img alt="$a(x)\cdot r(x) + b(x)\cdot f(x) = 1_\mathbb{K},$" src="svgs/2e98f24ca08f9751e34a81008647d1f5.svg" valign=-4.109589000000009px width="210.46583415pt" height="16.438356pt"/> so that <img alt="$a(x)$" src="svgs/0786a7b732d7242200d22552fe3630f1.svg" valign=-4.109589000000009px width="30.869576099999993pt" height="16.438356pt"/> inverts
<img alt="$r(x)$" src="svgs/c73b6615f0c7bd519371e439b4efff6d.svg" valign=-4.109589000000009px width="30.05337719999999pt" height="16.438356pt"/> modulo <img alt="$f(x).$" src="svgs/b9a2b0716bafb328e9ab3e48ce228ef1.svg" valign=-4.109589000000009px width="36.56405774999999pt" height="16.438356pt"/> In practice, one finds <img alt="$a(x)$" src="svgs/0786a7b732d7242200d22552fe3630f1.svg" valign=-4.109589000000009px width="30.869576099999993pt" height="16.438356pt"/> using
the Euclidean algorithm in <img alt="$\mathbb{K}[x].$" src="svgs/832c5c8e7f5fd14b4d7ef25fba499c72.svg" valign=-4.109589000000009px width="35.87909324999999pt" height="16.438356pt"/> For example:
```pycon
>>> from numlib import Zmod, xgcd
>>> from polylib import FPolynomial
>>>
>>> x = FPolynomial([0, Zmod(17)(1)])
>>> # let us invert, in Z/17[x], 1 + 2x + 3x^2 + 4x^3 modulo 1 +x^2 + 3x^4
>>> tup = xgcd(1+2*x+3*x**2+4*x**3, 1+x**2+3*x**3)
>>> tup[0]  # FPolynomial((16 mod 17,)), the gcd is a constant polynomial
>>> print(tup[1]*tup[0][0]**-1)  # 4 + 12x + 14x^2, the inverse
```
Note that this last interactive session agrees with our previous session in which
we worked in the Galois field
<img alt="$\mathbb{Z}/p[x] / \langle 1 + x^2 + 3x^3\rangle.$" src="svgs/2f2a39abe93d5a2780057f61f8cc195b.svg" valign=-4.109589000000009px width="161.70671714999997pt" height="17.4904653pt"/> In fact, the
**FPmod** class simply calls **xgcd** when it needs to invert something.

#### Field extensions

From the previous discussion, we know that every finite field has prime power order
and that we can construct such a finite field with order <img alt="$q=p^n$" src="svgs/a45d222501990a73ef2943bc648355e0.svg" valign=-3.1963519500000044px width="46.24230764999999pt" height="14.116037099999998pt"/> by quotienting
the polynomial ring <img alt="$\mathbb{Z}/p[x]$" src="svgs/2012540586a66e3dc43394a218c44338.svg" valign=-4.109589000000009px width="45.97615274999999pt" height="16.438356pt"/> with the ideal <img alt="$\langle f(x)\rangle$" src="svgs/c6084175338a8a0f04a0ff398ca276b0.svg" valign=-4.109589000000009px width="44.78326709999998pt" height="16.438356pt"/>
generated by an irreducible polynomial in <img alt="$\mathbb{Z}/p[x].$" src="svgs/4e0688d80c924b32de25ae472f5bd95c.svg" valign=-4.109589000000009px width="50.542377599999995pt" height="16.438356pt"/>
Of course, in general, there are different candidates for the irreducible <img alt="$f.$" src="svgs/327b2cbbade2d2154eacafe4501096e8.svg" valign=-3.1963503000000055px width="13.47039209999999pt" height="14.611878599999999pt"/> Do
different choices of <img alt="$f$" src="svgs/190083ef7a1625fbc75f243cffb9c96d.svg" valign=-3.1963503000000055px width="9.81741584999999pt" height="14.611878599999999pt"/> lead to different finite fields of order <img alt="$q?$" src="svgs/b682918429c04844ba596497e9c5ca61.svg" valign=-3.1963503000000055px width="15.69067829999999pt" height="14.611878599999999pt"/>

Notice that any homomorphism <img alt="$\lambda: \mathbb{K}_1 \rightarrow \mathbb{K}_2$" src="svgs/f5154f9b679f2e1078996e625cdbd1f4.svg" valign=-2.4657286500000066px width="88.35597869999998pt" height="13.881256950000001pt"/>
between two fields is a homomorphism of rings whose kernel is an ideal; but a field
has no proper ideals, so either the kernel is all of <img alt="$\mathbb{K}_1$" src="svgs/aa8aa44d38149964bb22c870ceade4e3.svg" valign=-2.465728650000001px width="19.33798019999999pt" height="13.7899245pt"/> in which case
<img alt="$\lambda$" src="svgs/fd8be73b54f5436a5cd2e73ba9b6bfa9.svg" valign=0.0px width="9.58908224999999pt" height="11.4155283pt"/> is the zero map, or the kernel is <img alt="$\{0_{\mathbb{K}_1}\}$" src="svgs/4a77b1413156365a88684adbf977283b.svg" valign=-4.109589000000009px width="40.84485404999999pt" height="16.438356pt"/> so that <img alt="$\lambda$" src="svgs/fd8be73b54f5436a5cd2e73ba9b6bfa9.svg" valign=0.0px width="9.58908224999999pt" height="11.4155283pt"/>
is an injection.  Hence, there are no quotient fields; only subfields &mdash; or, said
differently, there are only extension fields.

Furthermore, if <img alt="$\lambda: \mathbb{K}_1 \rightarrow \mathbb{K}_2$" src="svgs/f5154f9b679f2e1078996e625cdbd1f4.svg" valign=-2.4657286500000066px width="88.35597869999998pt" height="13.881256950000001pt"/> maps injectively between
finite fields, then lambda must be an isomorphism &mdash; in fact essentially the identity
since lambda must map <img alt="$1_{\mathbb{K}_1} to $" src="svgs/01cd673fdadd974ed49cb0028a4ddf6d.svg" valign=-4.109564249999995px width="38.31058274999999pt" height="14.70317145pt"/>1_{\mathbb{K}_2} &mdash; on prime fields. If
<img alt="$\mathbb{K}_1$" src="svgs/aa8aa44d38149964bb22c870ceade4e3.svg" valign=-2.465728650000001px width="19.33798019999999pt" height="13.7899245pt"/> and <img alt="$\mathbb{K}_2$" src="svgs/d3e7b197c5f616bedc5826b75725f445.svg" valign=-2.465728650000001px width="19.33798019999999pt" height="13.7899245pt"/> have the same size, then such a <img alt="$\lambda$" src="svgs/fd8be73b54f5436a5cd2e73ba9b6bfa9.svg" valign=0.0px width="9.58908224999999pt" height="11.4155283pt"/> must be
an isomorphism. In other words, up to isomorphism, there only one field of order <img alt="$q=p^n.$" src="svgs/5297e7f20346292016582946713d72f9.svg" valign=-3.1963519500000044px width="51.63044699999999pt" height="14.116037099999998pt"/>

But what of the construction above? Different choices of the irreducible <img alt="$f(x)$" src="svgs/7997339883ac20f551e7f35efff0a2b9.svg" valign=-4.109589000000009px width="31.99783454999999pt" height="16.438356pt"/> lead
obviously to element-wise differences but not, we have argued, overall structural dissimilarity
in the finite field.  It is instructive to think about this in terms of extension fields.

In field theory, the potentially confusion notation <img alt="$L/K$" src="svgs/76b5555a7ab7c6106ad9a8da5c6d2c28.svg" valign=-4.109589000000009px width="34.54345784999999pt" height="16.438356pt"/> means that <img alt="$L$" src="svgs/ddcb483302ed36a59286424aa5e0be17.svg" valign=0.0px width="11.18724254999999pt" height="11.232861749999998pt"/> is a field that
contains <img alt="$K$" src="svgs/d6328eaebbcd5c358f426dbea4bdbf70.svg" valign=0.0px width="15.13700594999999pt" height="11.232861749999998pt"/> as a subfield; i.e., <img alt="$L$" src="svgs/ddcb483302ed36a59286424aa5e0be17.svg" valign=0.0px width="11.18724254999999pt" height="11.232861749999998pt"/> is an
[extension field](https://encyclopediaofmath.org/wiki/Extension_of_a_field)
of <img alt="$K$" src="svgs/d6328eaebbcd5c358f426dbea4bdbf70.svg" valign=0.0px width="15.13700594999999pt" height="11.232861749999998pt"/> (important: <img alt="$L/K$" src="svgs/76b5555a7ab7c6106ad9a8da5c6d2c28.svg" valign=-4.109589000000009px width="34.54345784999999pt" height="16.438356pt"/> is not a quotient of fields).

For instance, the field
<img alt="$\mathbb{Z}/p[x] / \langle 1 + x^2 + 3x^3\rangle$" src="svgs/9746574a81479c795f4b26e528256ce7.svg" valign=-4.109589000000009px width="157.14049229999998pt" height="17.4904653pt"/>
is an extension of its prime field &mdash; note that said prime field is just image
in the quotient group of the constant polynomials (since the multiplicative identity in
<img alt="$\mathbb{Z}/p[x] / \langle 1 + x^2 + 3x^3\rangle$" src="svgs/9746574a81479c795f4b26e528256ce7.svg" valign=-4.109589000000009px width="157.14049229999998pt" height="17.4904653pt"/>
is just <img alt="$1 + \langle 1 + x^2 + 3x^3\rangle$" src="svgs/d857f60dfc38f221a5084c0871288fce.svg" valign=-4.109589000000009px width="131.25553155pt" height="17.4904653pt"/>).

Now, obviously, since it is irreducible, <img alt="$f(x) = 1 + x^2 + 3x^3$" src="svgs/5176d76f6adc7ec6be39077496db221b.svg" valign=-4.109589000000009px width="143.25324915pt" height="17.4904653pt"/> does not have a root
in <img alt="$\mathbb{Z}/p;$" src="svgs/65b68a09ffad5ee62dfbe529b6e5536a.svg" valign=-4.109589000000009px width="32.014941749999984pt" height="16.438356pt"/> but it does have a root in the extension
<img alt="$\mathbb{Z}/p[x] / \langle 1 + x^2 + 3x^3\rangle;$" src="svgs/f1a907a11adcecc2b58f5b361fa3378e.svg" valign=-4.109589000000009px width="161.70671714999997pt" height="17.4904653pt"/>
namely, <img alt="$x + \langle 1 + x^2 + 3x^3\rangle:$" src="svgs/51ec1011ed2fae97fab3ff7c3df9bc48.svg" valign=-4.109589000000009px width="141.56363265pt" height="17.4904653pt"/>

<p align="center"><img alt="$$f(x + \langle 1 + x^2 + 3x^3\rangle) = f(x) + \langle 1 + x^2 + 3x^3\rangle = \langle 1 + x^2 + 3x^3\rangle.$$" src="svgs/b0618bbbf4193ed2d08a1c189e3fcf7d.svg" valign=0.0px width="461.41492650000004pt" height="18.312383099999998pt"/></p>

That <img alt="$x \in \mathbb{Z}/p[x] / \langle 1 + x^2 + 3x^3\rangle$" src="svgs/5452b5cc0b20f0828504a376ea1a3235.svg" valign=-4.109589000000009px width="186.62661765pt" height="17.4904653pt"/> is a root of
<img alt="$f(x) = 1 + x^2 + 3x^3$" src="svgs/5176d76f6adc7ec6be39077496db221b.svg" valign=-4.109589000000009px width="143.25324915pt" height="17.4904653pt"/> is a
tautology; still, we can verify it concretely. Continuing the previous interactive session:
```pycon
>>> f.of(t)  # 0 mod 1 + t^2 + 3t^3
```
Remember that we changed the letter for the indeterminant in **GF** to **t**.

Given a field <img alt="$\mathbb{K}$" src="svgs/9ebeacdd09c18ad447a4e29b9039c3b0.svg" valign=0.0px width="12.785434199999989pt" height="11.324195849999999pt"/> and a non-constant polynomial <img alt="$f(x)\in \mathbb{K}[x]$" src="svgs/6675e9997dd7b0e79699326640cb07bf.svg" valign=-4.109589000000009px width="83.40184214999998pt" height="16.438356pt"/>, one
can always build a field extension

Imagine taking two irreducible polynomials, <img alt="$f_1(x)$" src="svgs/0bc63b8ce7a9477572cb9c4efaf15a2b.svg" valign=-4.109589000000009px width="37.60286804999999pt" height="16.438356pt"/> and f_

The chosen <img alt="$f$" src="svgs/190083ef7a1625fbc75f243cffb9c96d.svg" valign=-3.1963503000000055px width="9.81741584999999pt" height="14.611878599999999pt"/> in the last paragraph is not, in general, unique, of course. But
if one required that the minimal

Once we have a monic polynomial of degree <img alt="$r$" src="svgs/89f2e0d2d24bcf44db73aab8fc03252c.svg" valign=0.0px width="7.87295519999999pt" height="7.0776222pt"/> the Galois field of order <img alt="$p^r$" src="svgs/cdc535338263f4dc40336d4546b6d623.svg" valign=-3.1963519500000044px width="14.728015499999989pt" height="14.116037099999998pt"/> can
be realized as the quotient of the polynomial ring <img alt="$(\mathbb{Z}/p)[x]$" src="svgs/da59cb08e65509b38073548eedb4e008.svg" valign=-4.109589000000009px width="58.76158694999999pt" height="16.438356pt"/> and
the ideal generated by the irreducible monic polynomial.

Above we showed that <img alt="$1 + x^2 + 3x^3$" src="svgs/ea3a9f0df58be04c4230e07c439287f4.svg" valign=-1.3698729000000083px width="89.33778314999999pt" height="14.750749199999998pt"/> is irreducible over <img alt="$\mathbb{Z}/17.$" src="svgs/e2d61fa49b6fc21b56dc34ecbfeb0580.svg" valign=-4.109589000000009px width="40.18279319999999pt" height="16.438356pt"/>
The same is true for the monic polynomial <img alt="$6 + 6x^2 + x^3.$" src="svgs/94b6b0109e94fdec1c70cebe8da962ef.svg" valign=-1.3698729000000083px width="94.72592084999998pt" height="14.750749199999998pt"/>
```pycon
  >>> from polylib import FPolynomial
>>> 1/PF(3) * FPolynomial([1, 0, 1, 3])  # FPolynomial((6 mod 17, 0 mod 17, 6 mod 17, 1 mod 17))
>>> print(1/PF(3) * FPolynomial([1, 0, 1, 3]))  # 6 + 6x^2 + x^3
```
The quotient field
<img alt="$(\mathbb{Z}/17)[x]/\langle 6 + 6x^2 + x^3\rangle$" src="svgs/b2df3c26f0992a7d0d25c869d9a91d66.svg" valign=-4.109589000000009px width="178.09377794999997pt" height="17.4904653pt"/> is in fact a field of
order <img alt="$17^3$" src="svgs/d07c190856b363b8ab7555fd37127d7d.svg" valign=0.0px width="22.990966349999994pt" height="13.380876299999999pt"/>.

We can use **numlib** to get our hands on this Galois field as follows.
```pycon
>>> from numlib import Zmod, FPmod
>>> from polylib import FPolynomial
>>> PF = Zmod(17)
>>> GF = FPmod(FPolynomial([PF(6), PF(0), PF(6), PF(1)]))   # A Galois field
>>> print(GF)  # Z/17[x]/<6 + 6x^2 + x^3>
```

In practice, the symbology of the second to last line is too repetitive.  We see better
ways to instantiate a Galois field later.

First, let us define an element in our new Galois field, and make some computations.
```pycon
>>> f1 = GF([1, 2, 3, 2, 3, 4, 5])
>>> f1  # 11 + 16x^2 mod 6 + 6x^2 + x^3
```
Notice that we passed just a list of integers to GF. We didn't have to bother with
wrapping them in **PF** (because the type of the coefficients is automatically inferred
from the irreducible polynomial). Let us now computed in the Galois field.
```pycon
>>> f1**1000  # 12 + 13x + 14x^2 mod 6 + 6x^2 + x^3
>>> f1**(17**3-1) # 1 mod 6 + 6x^2 + x^3
>>> f2 = GF([8, 12])
>>> f1/f2  # 16 + 10x + 8x^2 mod 6 + 6x^2 + x^3
```
In practice, we often simplify notation as shown in the following program, which
prints out each non-zero and non-identity element of a Galois field of order <img alt="$3^3$" src="svgs/122d7384984d648b31748539d7b3d481.svg" valign=0.0px width="14.771756999999988pt" height="13.380876299999999pt"/>
along with its order.
```python
from numlib import Zmod, FPmod
from polylib import FPolynomial
from itertools import product

PF = Zmod(3)  # the prime field Z/3
t = FPolynomial([0, PF(1)], 't')  # some prefer t for Galois fields
irred = 1 + 2 * t ** 2 + t ** 3  # an irreducible cubic over Z/3

GF = FPmod(irred)  # a Galois field of order 3^3

def find_order(elt):
    """return multiplicative order"""
    for i in range(1, len(list(PF))**irred.degree()):
        if elt ** i == 1:
            return i

orders_ = {}  # dict that maps a non-zero elements of GF to its order
for coeffs in product(PF, repeat=3):  # iterate over all triples
    elt = GF(coeffs)
    if elt != 0:
        orders_[str(elt)] = find_order(elt)

orders = {}  # the reverse dict of order_, with aggregated values
for k, v in orders_.items():
    orders.setdefault(v, []).append(k)

print(f"Orders of non-zero elements of {str(GF)}\n")
print(f"{'order':<8}")
for k, v in orders.items():
    print(f"{k:^5} {', '.join(map(lambda s: ''.join(s.split()), v))}")
```
Before we run the program, we know that the order of each element must divide
<img alt="$3^3-1=26$" src="svgs/afd70bf0b8a36a4d964f6941089327bf.svg" valign=-1.3698729000000083px width="82.26011969999999pt" height="14.750749199999998pt"/> since that's the order of the multiplicative group of units in this
case. Here is the output of the program:
```
Orders of non-zero elements of Z/3[t]/<1 + 2t^2 + t^3>

order
 13   t^2, 2t, 2t+t^2, 2t+2t^2, 1+2t^2, 1+t, 1+t+t^2, 1+2t, 1+2t+t^2, 2+2t^2, 2+t+t^2, 2+2t+t^2
 26   2t^2, t, t+t^2, t+2t^2, 1+t^2, 1+t+2t^2, 1+2t+2t^2, 2+t^2, 2+t, 2+t+2t^2, 2+2t, 2+2t+2t^2
  1   1
  2   2
```

We see that the multiplicative group of units is in fact cyclic. It turns out that
that the group of units is always cyclic (see ...).

Notice that the polynomial
<img alt="$t\in\mathbb{Z}/3[x]/\langle t + 2r^2 +t^3\rangle$" src="svgs/7fb1dc67bdbafbaba9d0e90614bc8253.svg" valign=-4.109589000000009px width="175.85233544999997pt" height="17.4904653pt"/> generates the entire
group of units.  When <img alt="$t$" src="svgs/4f4f4e395762a3af4575de74c019ebb5.svg" valign=0.0px width="5.936097749999991pt" height="10.110901349999999pt"/> is such a generator  This is particularly nice

Knowing this, there is no need to break out **itertools.product** in
the code above because the powers of <img alt="$t$" src="svgs/4f4f4e395762a3af4575de74c019ebb5.svg" valign=0.0px width="5.936097749999991pt" height="10.110901349999999pt"/> run through all non-zero elements of the
Galois field.

#### Exercise
5. Rewrite the program above without using **itertools.product** and observe that you get the same output.
6. Write a program that finds *all* irreducible, monic cubics over <img alt="$\mathbb{Z}/3[x].$" src="svgs/bc299bd56543342f626b12fd87dc0a5b.svg" valign=-4.109589000000009px width="50.491019699999995pt" height="16.438356pt"/> How many of those lead to <img alt="$t$" src="svgs/4f4f4e395762a3af4575de74c019ebb5.svg" valign=0.0px width="5.936097749999991pt" height="10.110901349999999pt"/> be a generator?

A Galois field of order <img alt="$p^r$" src="svgs/cdc535338263f4dc40336d4546b6d623.svg" valign=-3.1963519500000044px width="14.728015499999989pt" height="14.116037099999998pt"/> is often denoted <img alt="$\operatorname{GF}(p^r)$" src="svgs/97833e01f80c4c539094bb84cc9e083d.svg" valign=-4.109589000000009px width="51.96550754999999pt" height="16.438356pt"/> where <img alt="$p$" src="svgs/2ec6e630f199f589a2402fdf3e0289d5.svg" valign=-3.1963502999999895px width="8.270567249999992pt" height="10.2739725pt"/> is
prime and <img alt="$r$" src="svgs/89f2e0d2d24bcf44db73aab8fc03252c.svg" valign=0.0px width="7.87295519999999pt" height="7.0776222pt"/> is a positive integer.

## [Algebraic number fields](https://encyclopediaofmath.org/wiki/Algebraic_number_field)

If <img alt="$K$" src="svgs/d6328eaebbcd5c358f426dbea4bdbf70.svg" valign=0.0px width="15.13700594999999pt" height="11.232861749999998pt"/> is a field containing a subfield <img alt="$k$" src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg" valign=0.0px width="9.075367949999992pt" height="11.4155283pt"/>, then we denote this by <img alt="$K/k$" src="svgs/39b0c9afc9cb06ba72e65dda675e7685.svg" valign=-4.109589000000009px width="31.51833299999999pt" height="16.438356pt"/> and say that
the <img alt="$K$" src="svgs/d6328eaebbcd5c358f426dbea4bdbf70.svg" valign=0.0px width="15.13700594999999pt" height="11.232861749999998pt"/> is an [extension](https://encyclopediaofmath.org/wiki/Extension_of_a_field) <img alt="$k$" src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg" valign=0.0px width="9.075367949999992pt" height="11.4155283pt"/>.
The standard notation <img alt="$K/k$" src="svgs/39b0c9afc9cb06ba72e65dda675e7685.svg" valign=-4.109589000000009px width="31.51833299999999pt" height="16.438356pt"/> is somewhat confusing in that the <img alt="$/$" src="svgs/87f05cbf93b3fa867c09609490a35c99.svg" valign=-4.109589000000009px width="8.219209349999991pt" height="16.438356pt"/> does not, here, mean
quotient. In fact, all ideals of <img alt="$K$" src="svgs/d6328eaebbcd5c358f426dbea4bdbf70.svg" valign=0.0px width="15.13700594999999pt" height="11.232861749999998pt"/> are proper so that the morphisms in the category of
fields are either zero or injections.

Let <img alt="$K$" src="svgs/d6328eaebbcd5c358f426dbea4bdbf70.svg" valign=0.0px width="15.13700594999999pt" height="11.232861749999998pt"/> denote a **field extension** of a ground field <img alt="$k$" src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg" valign=0.0px width="9.075367949999992pt" height="11.4155283pt"/>; i.e., <img alt="$K$" src="svgs/d6328eaebbcd5c358f426dbea4bdbf70.svg" valign=0.0px width="15.13700594999999pt" height="11.232861749999998pt"/> is contains

#### [Quadratic field](https://encyclopediaofmath.org/wiki/Quadratic_field)

We can extend the field, <img alt="$\mathbb{Q}$" src="svgs/0f452ec0bcf578fa387e4857f80f03f4.svg" valign=-2.739730950000001px width="12.785434199999989pt" height="14.0639268pt"/>, of rational numbers by adjoining the complex
number <img alt="$i,$" src="svgs/1e7097ee7ea18cca8abf23e2d617a6f3.svg" valign=-3.1963519500000044px width="10.22945054999999pt" height="14.0378568pt"/> thereby obtaining the
[Gaussian rationals](https://en.wikipedia.org/wiki/Gaussian_rational),
<img alt="$\mathbb{Q}[i].$" src="svgs/806743fc3fb84dd1f13d00bc7aded538.svg" valign=-4.109589000000009px width="32.147331149999985pt" height="16.438356pt"/>

In **numlib**, we get our hands of the Gaussian rationals as follows.
```pycon
>>> from fractions import Fraction
>>> from polylib import FPolynomial
>>> from numlib import FPmod
>>> x = FPolynomial([0, Fraction(1)], 'i')
>>> GQ = FPmod(1+x**2)  # The Gaussian rationals
>>> i = GQ([0, 1])
>>> print(', '.join([str(i**j) for j in range(4)]))
1, i, -1, i  # GQ contains the 4th roots of unity as its only finite-order units.
```

More generally, one can implement any degree 2 extension a &mdash; so called
**quadratic field** &mdash; of the rationals by adjoining <img alt="$\sqrt{d}$" src="svgs/395dc24959bcf673cda05c9c16f6ff43.svg" valign=-1.7717139000000102px width="22.25463569999999pt" height="16.438356pt"/> where <img alt="$d$" src="svgs/2103f85b8b1477f430fc407cad462224.svg" valign=0.0px width="8.55596444999999pt" height="11.4155283pt"/>
is a square-free integer; for instance:
```pycon
>>> x = FPolynomial([0, Fraction(1)])
>>> QF = FPmod(5+x**2)  # The Gaussian rationals
```

## [Elliptic Curves](https://encyclopediaofmath.org/wiki/Elliptic_curve)

Let us define an elliptic curve, <img alt="$E$" src="svgs/84df98c65d88c6adf15d4645ffa25e47.svg" valign=0.0px width="13.08219659999999pt" height="11.232861749999998pt"/>,  with Weierstrass normal form
<img alt="$y^2= x^3 + 17x + 20$" src="svgs/2261788873d14dd272cdb92343f9c839.svg" valign=-3.1963503000000086px width="137.1649521pt" height="16.5772266pt"/>  over, say, <img alt="$\mathbb{Z}/43.$" src="svgs/4f4bb62712aa4196478bb8f1133f62d3.svg" valign=-4.109589000000009px width="40.18279319999999pt" height="16.438356pt"/>
```pycon
>>> from numlib import Zmod, EllCurve
>>> PF = Zmod(43)  # a prime field
>>> E = EllCurve(PF(17), PF(20))
>>> E
y^2 = 20 + 17x + x^3 over Z/43
>>> E.discriminant() # let's check that the curve is non-singular
21 + <43>
>>> E.j-invariant()
21 + <43>
```
We can define points on the curve using integers &mdash; their correct type
will be inferred from the coefficients (17 and 20) used to define the curve:
```pycon
>>> E(25, 26)
(25, 26) on y^2 = 20 + 17x + x^3 over Z/43
>>> print(E(25, 26))
(25, 26)
>>> E(1,2)
ValueError: (1, 2) is not on y^2 = 20 + 17x + x^3 ...
```
Let us find all points on the curve:
```pycon
>>> from itertools import product
>>> count = 0
>>> for coeffs in product(PF,PF):
...     try:
...         print(E(*coeffs))
...         count += 1
...     except:
...         pass
...
(1, 9)
(1, 34)
(4, 18)
(4, 25)
(5, 12)
  ...
(39, 19)
(39, 24)
(41, 8)
(41, 35)
>>> count
46
```
In total, the curve <img alt="$E$" src="svgs/84df98c65d88c6adf15d4645ffa25e47.svg" valign=0.0px width="13.08219659999999pt" height="11.232861749999998pt"/> consists of 47 points including the point at infinity:
any finite point on the curve is a generator:
```pycon
>>> len(set(n * E(25, 26) for n in range(1, 48)))
47
>>> 47 * E(25, 26)
[0: 1: 0] on y^2 = 20 + 17x + x^3 over Z/43
```
The curve is an Abelian group; an element's additive inverse is itself with the
negated y-coordinate:
```pycon
>>> E(25, 26) + E(25, -26)
[0: 1: 0] on y^2 = 20 + 17x + x^3 over Z/43
```

Exercises:

7. The method of iterating, as above, through the product <img alt="$\operatorname{PF} \times \operatorname{PF}$" src="svgs/cc25678c95d9ad77731747ccf3137f9f.svg" valign=-1.369874549999991px width="62.10050879999999pt" height="12.6027363pt"/> is inefficient in general since it is order <img alt="$\mathcal{O}(n^2)$" src="svgs/c5566036dd2bd924fef1c6263072eb45.svg" valign=-4.109589000000009px width="43.570210199999984pt" height="17.4904653pt"/> in the size of <img alt="$\operatorname{PF}.$" src="svgs/09f485feb71728c385ea8299c839e059.svg" valign=0.0px width="29.22375884999999pt" height="11.232861749999998pt"/> (Also, try/except is not cheap.) Meanwhile, the elliptic curve should have roughly the same order as <img alt="$\operatorname{PF}.$" src="svgs/09f485feb71728c385ea8299c839e059.svg" valign=0.0px width="29.22375884999999pt" height="11.232861749999998pt"/> (Footne to Hasse's thm here).  Implement an <img alt="$\mathcal{O}(n)$" src="svgs/eaf65be831b761bd6436418200a823ad.svg" valign=-4.109589000000009px width="36.19574969999999pt" height="16.438356pt"/> algorithm that finds all points on the curve <img alt="$E.$" src="svgs/4d25e0b0e43d440c7619891f54dba48f.svg" valign=0.0px width="17.64840164999999pt" height="11.232861749999998pt"/>  Hint: first compute the squares of all elements of <img alt="$\operatorname{PF}.$" src="svgs/09f485feb71728c385ea8299c839e059.svg" valign=0.0px width="29.22375884999999pt" height="11.232861749999998pt"/>

8. How many points are on the curve <img alt="$y^2= x^3 + 1113x + 1932$" src="svgs/dbdd4f923720bf67a1dfab8ba8e5d527.svg" valign=-3.1963503000000086px width="170.04179114999997pt" height="16.5772266pt"/> over <img alt="$\mathbb{Z}/2017?$" src="svgs/1324004bc7c64764a79b0c6b8e2b6232.svg" valign=-4.109589000000009px width="59.81757869999999pt" height="16.438356pt"/> Is this curve cyclic?

A more correct notation for the curve above is <img alt="$E(\mathbb{Z}/43);$" src="svgs/25b4326dcd1246d95313a2ab89101ed0.svg" valign=-4.109589000000009px width="66.05040419999999pt" height="16.438356pt"/> then, the name
<img alt="$E$" src="svgs/84df98c65d88c6adf15d4645ffa25e47.svg" valign=0.0px width="13.08219659999999pt" height="11.232861749999998pt"/> can be reserved for the set of points <img alt="$(x,y)$" src="svgs/7392a8cd69b275fa1798ef94c839d2e0.svg" valign=-4.109589000000009px width="38.135511149999985pt" height="16.438356pt"/> in the *algebraic closure* of
<img alt="$\mathbb{Z}/43$" src="svgs/6713273f26b0786147c5f82b4bf58596.svg" valign=-4.109589000000009px width="35.61656834999999pt" height="16.438356pt"/> which satisfy <img alt="$y^2= x^3 + 1113x + 1932$" src="svgs/dbdd4f923720bf67a1dfab8ba8e5d527.svg" valign=-3.1963503000000086px width="170.04179114999997pt" height="16.5772266pt"/>.

In common mathematical parlance, we know that the number of
<img alt="$\mathbb{Z}/43$" src="svgs/6713273f26b0786147c5f82b4bf58596.svg" valign=-4.109589000000009px width="35.61656834999999pt" height="16.438356pt"/>-rational points of <img alt="$E$" src="svgs/84df98c65d88c6adf15d4645ffa25e47.svg" valign=0.0px width="13.08219659999999pt" height="11.232861749999998pt"/> is <img alt="$47$" src="svgs/b3f8830b237658b2e64f9849bc647471.svg" valign=0.0px width="16.438418699999993pt" height="10.5936072pt"/>; together, those
comprise <img alt="$E(\mathbb{Z}/43).$" src="svgs/f2f2c67d3f10438038a6795b7e23caff.svg" valign=-4.109589000000009px width="66.05040419999999pt" height="16.438356pt"/>

It is also good to write <img alt="$E/\mathbb{F}_{43}$" src="svgs/de2395ab1a8b770e830f18bc381dced5.svg" valign=-4.109589000000009px width="44.452166549999994pt" height="16.438356pt"/> to indicating where live the
coefficients <img alt="$a$" src="svgs/44bc9d542a92714cac84e01cbbb7fd61.svg" valign=0.0px width="8.68915409999999pt" height="7.0776222pt"/> and <img alt="$b$" src="svgs/4bdc8d9bcfb35e1c9bfb51fc69687dfc.svg" valign=0.0px width="7.054796099999991pt" height="11.4155283pt"/> that determine the Weierstrass form of the curve.
Note that the <img alt="$/$" src="svgs/87f05cbf93b3fa867c09609490a35c99.svg" valign=-4.109589000000009px width="8.219209349999991pt" height="16.438356pt"/> has nothing to do with quotienting, so we wrote <img alt="$\mathbb{F}_43$" src="svgs/65cfa8c1a8da852b6e50fc55e23739f9.svg" valign=-2.465728650000001px width="25.63935659999999pt" height="13.7899245pt"/>
instead of the notation <img alt="$\mathbb{Z}/{43}$" src="svgs/50fa74787e660074440a008b4517309a.svg" valign=-4.109589000000009px width="35.61656834999999pt" height="16.438356pt"/> where the slash *does* indicate
quotienting.

Now consider the set of points in an algebraically closed field that lie on
some Weiestrass curve with coefficients <img alt="$a$" src="svgs/44bc9d542a92714cac84e01cbbb7fd61.svg" valign=0.0px width="8.68915409999999pt" height="7.0776222pt"/> and <img alt="$b$" src="svgs/4bdc8d9bcfb35e1c9bfb51fc69687dfc.svg" valign=0.0px width="7.054796099999991pt" height="11.4155283pt"/> elements of a subfield <img alt="$k$" src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg" valign=0.0px width="9.075367949999992pt" height="11.4155283pt"/>.
Since the group addition laws are rational expressions of the coordinates of
points lying on the curve, the <img alt="$K$" src="svgs/d6328eaebbcd5c358f426dbea4bdbf70.svg" valign=0.0px width="15.13700594999999pt" height="11.232861749999998pt"/>-rational points of <img alt="$E$" src="svgs/84df98c65d88c6adf15d4645ffa25e47.svg" valign=0.0px width="13.08219659999999pt" height="11.232861749999998pt"/> form a group for
any intermediate field <img alt="$K$" src="svgs/d6328eaebbcd5c358f426dbea4bdbf70.svg" valign=0.0px width="15.13700594999999pt" height="11.232861749999998pt"/>.

Of course, if we expand our point of view to an extension field then we generally
expect to pick up more up more points.  Here is a concrete demonstration:
``` python
import numlib as nl
import polylib as pl

PF = nl.Zmod(43)  # a prime field
irreducible = pl.FPolynomial([PF(5), PF(2), PF(1)], 't')  # 5+2t+t^2 in Z/43[t]
GF = nl.FPmod(irreducible)  # GF(43^2)

E =  nl.EllCurve(GF([17]), GF([20]))  # y^2 = 20+17x+x^3 over Z/43[t]/<5+2t+t^2>

squares = {}  # dict that maps squares to their roots
for y in GF:
    squares.setdefault(y**2, []).append(y)

points_of_E = {E(0,1,0)}
for x in GF:
   f_of_x = E.f.of(x)
   if f_of_x in squares:
       for y in squares[f_of_x]:
           points_of_E.add(E(x,y))

print(len(points_of_E))
```
Notes:
* Notice that <img alt="$\operatorname{GF}$" src="svgs/e8dabd4b46e75cbaa6866d30d94e06e3.svg" valign=0.0px width="23.63021594999999pt" height="11.232861749999998pt"/> is a generator (that runs through all elements of <img alt="$\operatorname{GF}(43^2)).$" src="svgs/89ebb6d48c1e59e09e440090f4064985.svg" valign=-4.109589000000009px width="71.18743994999998pt" height="17.4904653pt"/>
* **E.f** is just the function defining <img alt="$E:$" src="svgs/5ae864f6b5293eaafacf92e58b15aef3.svg" valign=0.0px width="22.21449944999999pt" height="11.232861749999998pt"/>
  ```python
  print(E.f)  # 20 + 17x + x^3
  ```
* A subtlety is that one must define the curve by specifying coefficients in <img alt="$\operatorname{GF}$" src="svgs/e8dabd4b46e75cbaa6866d30d94e06e3.svg" valign=0.0px width="23.63021594999999pt" height="11.232861749999998pt"/>, not <img alt="$\operatorname{PF}.$" src="svgs/09f485feb71728c385ea8299c839e059.svg" valign=0.0px width="29.22375884999999pt" height="11.232861749999998pt"/> Unless one want to roll one's own Galois field by choosing some particular irreducible, consider using **numlib**'s function **GaloisField**.  So replace this
  ```python
  PF = nl.Zmod(43)  # a prime field
  irreducible = pl.FPolynomial([PF(5), PF(2), PF(1)], 't')  # 5+2t+t^2 in Z/43[t]
  GF = nl.FPmod(irreducible)  # GF(43^2)
  ```
  with simply
  ```python
  GF = nl.GaloisField(43, 2) # GF(43^2)
  ```
  in the program above (after which there is no need to import polylib).

The output of the program is: <img alt="$1927.$" src="svgs/b65c4c54020f29b0336cc1166f9c6d3d.svg" valign=0.0px width="37.44306224999999pt" height="10.5936072pt"/>

So over <img alt="$\operatorname{GF}(43^2),$" src="svgs/779b607d4f5e9b80020fd0b4713a8414.svg" valign=-4.109589000000009px width="64.79472284999999pt" height="17.4904653pt"/> <img alt="$E$" src="svgs/84df98c65d88c6adf15d4645ffa25e47.svg" valign=0.0px width="13.08219659999999pt" height="11.232861749999998pt"/> has order <img alt="$1927.$" src="svgs/b65c4c54020f29b0336cc1166f9c6d3d.svg" valign=0.0px width="37.44306224999999pt" height="10.5936072pt"/>  The number of those
points whose coordinates live in <img alt="$\operatorname{GF}(43^2)$" src="svgs/416cc4c78235cd4d312bbdf6879993c0.svg" valign=-4.109589000000009px width="60.22849964999999pt" height="17.4904653pt"/>'s prime field,
<img alt="$\mathbb{F}_{43}$" src="svgs/9ccbd3312f29da838314377f80adaa7e.svg" valign=-2.465728650000001px width="23.15078039999999pt" height="13.7899245pt"/> &mdash; i.e., that are <img alt="$\mathbb{F}_{43}$" src="svgs/9ccbd3312f29da838314377f80adaa7e.svg" valign=-2.465728650000001px width="23.15078039999999pt" height="13.7899245pt"/>-rational &mdash; is
<img alt="$47.$" src="svgs/32caa25fbf6c62c1e9db3b0b24fc1ba6.svg" valign=0.0px width="21.00464354999999pt" height="10.5936072pt"/> Said differently, <img alt="$E$" src="svgs/84df98c65d88c6adf15d4645ffa25e47.svg" valign=0.0px width="13.08219659999999pt" height="11.232861749999998pt"/> has <img alt="$1927$" src="svgs/e5e9463b5d52b61d0b55bf181211abc7.svg" valign=0.0px width="32.876837399999985pt" height="10.5936072pt"/> <img alt="$\mathbb{F}_{43^2}$" src="svgs/93946f515aeeffbe0ba3f22796310a3c.svg" valign=-2.9223628500000003px width="28.744480049999986pt" height="14.2465587pt"/>-rational points, <img alt="$47$" src="svgs/b3f8830b237658b2e64f9849bc647471.svg" valign=0.0px width="16.438418699999993pt" height="10.5936072pt"/>
of which are <img alt="$\mathbb{F}_{43}$" src="svgs/9ccbd3312f29da838314377f80adaa7e.svg" valign=-2.465728650000001px width="23.15078039999999pt" height="13.7899245pt"/>-rational.

Of course, <img alt="$E(\mathbb{F}_{43})$" src="svgs/3eeb4492977915c636c7cc99f368d541.svg" valign=-4.109589000000009px width="49.84029929999999pt" height="16.438356pt"/> is a subgroup of order <img alt="$47$" src="svgs/b3f8830b237658b2e64f9849bc647471.svg" valign=0.0px width="16.438418699999993pt" height="10.5936072pt"/> of
<img alt="$E(\mathbb{F}_{43^2})$" src="svgs/6813ab33f2a8d88d187350b4de6a7d82.svg" valign=-4.109589000000009px width="56.25591344999999pt" height="16.438356pt"/> which, in turn, has order <img alt="$1927 = 41\cdot 47$" src="svgs/5d128dccdd3dce36f9c78c5fc2432ca0.svg" valign=0.0px width="99.54328724999998pt" height="10.5936072pt"/> (and
is cyclic, it turns out).

For convenience, the functionality above is built-in to numlib:
```pycon
>>> import numlib as nl:
>>> GF = nl.GaloisField(43, 2) # GF(43^2)  # a finite field
>>> E =  nl.EllCurve(GF([17]), GF([20]))   # an elliptic curve over GF
>>> for pt in E:
...     print(pt)
...
(-21-21t, -13-21t)
(-21-21t, 13+21t)
(-21-20t, -21+7t)
(-21-20t, 21-7t)
(-21-19t, -16-12t)
(-21-19t, 16+12t)
         ...
```
Note: <img alt="$E$" src="svgs/84df98c65d88c6adf15d4645ffa25e47.svg" valign=0.0px width="13.08219659999999pt" height="11.232861749999998pt"/> in the last code block is a generator that runs through all finite
<img alt="$\operatorname{GF}$" src="svgs/e8dabd4b46e75cbaa6866d30d94e06e3.svg" valign=0.0px width="23.63021594999999pt" height="11.232861749999998pt"/>-rational points on the curve.

``` pycon
>>> len(E)
1926
```
Hence, <img alt="$\#E(\mathbb{F}_{43^2}) = 1927.$" src="svgs/1bed9084d04362b70a2883bae1ff1bf0.svg" valign=-4.109589000000009px width="129.31527839999998pt" height="16.438356pt"/>

---

### References
* Keith Conrad's [Constructing algebraic closures](https://kconrad.math.uconn.edu/blurbs/galoistheory/algclosure.pdf),
  [Perfect fields](https://kconrad.math.uconn.edu/blurbs/galoistheory/perfect.pdf),
  and [Cyclotomic extensions](https://kconrad.math.uconn.edu/blurbs/galoistheory/cyclotomic.pdf)
* [Polynomial ring over a field](https://commalg.subwiki.org/wiki/Polynomial_ring_over_a_field)

### Todo
* improve **factorPR** (a la GNU factor);
  * or, maybe, import [primefac](https://pypi.org/project/primefac/) (but this is only Python2)
  * or, have a look at this [fork of primefac](https://github.com/elliptic-shiho/primefac-fork).
* Consider implementing the
  [Cantor-Zassenhaus algorithm](https://en.wikipedia.org/wiki/Cantor%E2%80%93Zassenhaus_algorithm)
  for factoring.
