from __future__ import annotations
from math import factorial
from sympy import Expr, Rational, I
from functools import lru_cache
from itertools import combinations, permutations
from typing import List, Tuple, Iterator


def Ekernel(n: int, k: List[Rational]) -> Rational:
    if n == 3:
        return -(abs(k[0])*abs(k[1]) + k[0]*k[1]) / 2

    val = abs(k[1])**(n-3) * Ekernel(3, [k[0], k[1], sum(k[2:])]) / factorial(n-2)

    for m in range(1, n-2): # m goes upto n-3
        val -= (abs(k[1])**m / factorial(m)) * Ekernel(n-m, [k[0], sum(k[1:m+2])] + k[m+2:])

    return val


def Fkernel(n: int, k: List[Rational]) -> Rational:
    val = 2 * Ekernel(n, k) / abs(k[0])

    for m in range(1, n-2): # m goes upto n-3
        sm = sum(k[1:m+2])
        val -= 2 * Ekernel(m+2, [-sm] + k[1:m+2]) * Fkernel(n-m, [k[0], sm] + k[m+2:])

    return val / abs(k[1])


def Vertex(n: int, k: List[Rational], w: List[Rational]) -> Expr:
    val = Rational(0)
    for ip in permutations(range(n)):
        pk = [k[i] for i in ip]
        val += w[ip[0]] * w[ip[1]] * Fkernel(n, pk)

    return -I * val / 2


def Propagator(k: Rational, w: Rational, g:Rational) -> Expr:
    return -I / (w**2/abs(k) - g)


# generates all set partitions of S into k non-empty parts
def SetPartitions(S: List[int], k: int) -> Iterator[Tuple[Tuple[int], ...]]:
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


def BGamplitude(k: List[Rational], w: List[Rational], g: Rational) -> Expr:
    @lru_cache(None)
    def current(subset: Tuple[int, ...]) -> Expr:
        s = list(subset)

        if len(s) == 1:
            return Rational(1)

        sumsk = sum(k[i] for i in s)
        sumsw = sum(w[i] for i in s)
        val = Rational(0)

        for m in range(2, len(s) + 1):
            for prt in SetPartitions(s, m):
                vk = [-sumsk] + [sum(k[i] for i in p) for p in prt]
                vw = [-sumsw] + [sum(w[i] for i in p) for p in prt]

                curprod = Rational(1)
                for p in prt:
                    curprod *= current(tuple(p))

                val += Vertex(m+1, vk, vw) * curprod

        return val * Propagator(sumsk, sumsw, g)

    n = len(k)
    nbut0 = list(range(1, n))
    val = Rational(0)

    for m in range(2, n):
        for prt in SetPartitions(nbut0, m):
            vk = [k[0]] + [sum(k[i] for i in p) for p in prt]
            vw = [w[0]] + [sum(w[i] for i in p) for p in prt]

            curprod = Rational(1)
            for p in prt:
                curprod *= current(tuple(p))

            val += Vertex(m+1, vk, vw) * curprod

    return val


def MakeKinematics(wf: List[Rational], ss: List[int], g: Rational) -> Tuple[List[Rational], List[Rational]]:
    if len(ss) != len(wf) + 2:
        raise ValueError("2 more signs are required than the free frequencies")

    if ss[0] + ss[-1] != 0:
        raise ValueError("first and last signs should be opposite")

    sumwf = sum(wf)
    sf = ss[1:-1]
    sumsw2 = sum(s*w**2 for s, w in zip(sf, wf))

    wn = -(ss[0] * sumwf**2 + sumsw2) / (2 * ss[0] * sumwf)
    w1 = -(sumwf + wn)

    ws = [w1] + list(wf) + [wn]
    ks = [s*w**2 / g for s, w in zip(ss, ws)]

    return ks, ws


def InclExclFormula(w: List[Rational]) -> Expr:
    n = len(w)
    wp = w[2:]
    np = n-2
    betasq = min(w[0]**2, w[1]**2)
    val = Rational(0)
    for r in range(np + 1):
        for ir in combinations(range(np), r):
            subwp = [wp[i] for i in ir]
            sqsum = sum(x**2 for x in subwp)
            val += (-1)**len(subwp) * max(Rational(0), betasq - sqsum)**(n-3)
    return I * w[0] * w[1] * 2**(n-1) * val


ks, ws = MakeKinematics(
    [Rational(1), Rational(2), Rational(3)],
    [-1, -1, 1, 1, 1],
    Rational(1)
)
print(BGamplitude(ks, ws, Rational(1)))
print(InclExclFormula(ws))
