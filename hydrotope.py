from __future__ import annotations
from math import factorial
from sympy import Expr, Rational, I
from functools import lru_cache
from itertools import combinations, permutations
from typing import List, Tuple, Iterator, Callable


def Ekernel(n: int, ks: List[Rational]) -> Rational:
    if n == 3:
        return -(abs(ks[0])*abs(ks[1]) + ks[0]*ks[1]) / 2

    val = abs(ks[1])**(n-3) * Ekernel(3, [ks[0], ks[1], sum(ks[2:])]) / factorial(n-2)

    for m in range(1, n-2): # m goes upto n-3
        val -= (abs(ks[1])**m / factorial(m)) * Ekernel(n-m, [ks[0], sum(ks[1:m+2])] + ks[m+2:])

    return val


def Fkernel(n: int, ks: List[Rational]) -> Rational:
    val = 2 * Ekernel(n, ks) / abs(ks[0])

    for m in range(1, n-2): # m goes upto n-3
        sm = sum(ks[1:m+2])
        val -= 2 * Ekernel(m+2, [-sm] + ks[1:m+2]) * Fkernel(n-m, [ks[0], sm] + ks[m+2:])

    return val / abs(ks[1])


def Vertex(n: int, ks: List[Rational], ws: List[Rational]) -> Expr:
    val = Rational(0)
    for p in permutations(range(n)):
        kp = [ks[i] for i in p]
        val += ws[p[0]] * ws[p[1]] * Fkernel(n, kp)

    return -I * val / 2


def Propagator(k: Rational, w: Rational, g:Rational) -> Expr:
    return -I / (w**2/abs(k) - g)


# generates all set partitions of S into k non-empty parts
def SetPartitions(S: List[int], k: int) -> Iterator[Tuple[Tuple[int, ...], ...]]:
    if k == 0:
        if not S:
            yield ()
        return

    if len(S) < k:
        return

    s0 = S[0]

    # s0 starts a new block
    for s in SetPartitions(S[1:], k-1):
        yield ((s0,),) + s

    # s0 joins an existing block
    for s in SetPartitions(S[1:], k):
        for i in range(len(s)):
            yield s[:i] + ((s0,) + s[i],) + s[i+1:]


def BGcurrent(ks: List[Rational], ws: List[Rational], g: Rational) -> Callable[[Tuple[int, ...]], Expr]:
    @lru_cache(maxsize=None)
    def current(subset: Tuple[int, ...]) -> Expr:
        s = list(subset)

        if len(s) == 1:
            return Rational(1)

        sumsk = sum(ks[i] for i in s)
        sumsw = sum(ws[i] for i in s)
        val = Rational(0)

        for m in range(2, len(s) + 1):
            for prt in SetPartitions(s, m):
                vk = [-sumsk] + [sum(ks[i] for i in p) for p in prt]
                vw = [-sumsw] + [sum(ws[i] for i in p) for p in prt]

                curprod = Rational(1)
                for p in prt:
                    curprod *= current(tuple(p))

                val += Vertex(m+1, vk, vw) * curprod

        return val * Propagator(sumsk, sumsw, g)

    return current


def BGamplitude(ks: List[Rational], ws: List[Rational], g: Rational) -> Expr:
    n = len(ks)
    nbut0 = list(range(1, n))
    current = BGcurrent(ks, ws, g)
    val = Rational(0)

    for m in range(2, n):
        for prt in SetPartitions(nbut0, m):
            vk = [ks[0]] + [sum(ks[i] for i in p) for p in prt]
            vw = [ws[0]] + [sum(ws[i] for i in p) for p in prt]

            curprod = Rational(1)
            for p in prt:
                curprod *= current(tuple(p))

            val += Vertex(m+1, vk, vw) * curprod

    return val


# given n signs ss = [σ(1), σ(2), ..., σ(n-1), σ(n) = -σ(1)], and n-2
# frequencies wf = [ω(2), ..., ω(n-1)]; create on-shell momenta and frequencies
# (kᵢ = σᵢωᵢ², ωᵢ), 1 ≤ i ≤ n with Σkᵢ = Σωᵢ = 0. [the restriction σ(n) = -σ(1)
# is only for convenience—it leads to a unique solution.]
def MakeKinematics(wf: List[Rational], ss: List[int], g: Rational) -> Tuple[List[Rational], List[Rational]]:
    if len(ss) != len(wf) + 2:
        raise ValueError("expected 2 more signs than frequencies")

    if ss[0] + ss[-1] != 0:
        raise ValueError("expected opposite first and last signs")

    sumwf = sum(wf)
    sumswf2 = sum(s*w**2 for s, w in zip(ss[1:-1], wf))

    wn = -(ss[0] * sumwf**2 + sumswf2) / (2 * ss[0] * sumwf)
    w1 = -(sumwf + wn)

    ws = [w1] + list(wf) + [wn]
    ks = [s*w**2 / g for s, w in zip(ss, ws)]

    return ks, ws

# implements the RHS of equation (17) arXiv:2606.28280v1
def InclExclFormula(ws: List[Rational]) -> Expr:
    n = len(ws)
    wp = ws[2:]
    np = n-2
    betasq = min(ws[0]**2, ws[1]**2)
    val = Rational(0)
    for r in range(np + 1):
        for c in combinations(range(np), r):
            subwp = [wp[i] for i in c]
            sqsum = sum(x**2 for x in subwp)
            val += (-1)**len(subwp) * max(Rational(0), betasq - sqsum)**(n-3)
    return I * ws[0] * ws[1] * 2**(n-1) * val


ks, ws = MakeKinematics(
    [Rational(1), Rational(2), Rational(3)],
    [-1, -1, 1, 1, 1],
    Rational(1)
)
print(BGamplitude(ks, ws, Rational(1)))
print(InclExclFormula(ws))
