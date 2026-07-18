from __future__ import annotations
from math import factorial
from sympy import Expr, Rational, I, simplify
from functools import lru_cache
from itertools import combinations, permutations
from typing import List, Tuple, Iterator


def Ekernel(n: int, k: List[Rational]) -> Rational:
    if n == 3:
        return -(abs(k[0])*abs(k[1]) + k[0]*k[1]) / 2

    result = abs(k[1])**(n-3) * Ekernel(3, [k[0], k[1], sum(k[2:])]) / factorial(n-2)

    for m in range(1, n-2): # m goes upto n-3
        result -= (abs(k[1])**m / factorial(m)) * Ekernel(n-m, [k[0], sum(k[1:m+2])] + k[m+2:])

    return result


def Fkernel(n: int, k: List[Rational]) -> Rational:
    if n == 3:
        return -1 - k[0]*k[1] / (abs(k[0])*abs(k[1]))

    result = 2 * Ekernel(n, k) / abs(k[0])

    for m in range(1, n-2): # m goes upto n-3
        sigm = sum(k[1:m+2])
        result -= 2 * Ekernel(m+2, [-sigm] + k[1:m+2]) * Fkernel(n-m, [k[0], sigm] + k[m+2:])

    return result / abs(k[1])


def Vertex(n: int, k: List[Rational], w: List[Rational]) -> Expr:
    result = Rational(0)
    for p in permutations(range(n)):
        pk = [k[i] for i in p]
        result += w[p[0]] * w[p[1]] * Fkernel(n, pk)

    return -I * result / 2


def Propagator(w: Rational, k: Rational, g:Rational) -> Expr:
    return -I / (w**2/abs(k) - g)


# generates all set partitions of S into k non-empty parts
def SetPartitions(S: List[int], k: int) -> Iterator[Tuple[Tuple[int], ...]]:
    if len(S) != len(set(S)):
        raise ValueError("items in S must be distinct")

    if k == 0:
        if not S:
            yield ()
        return
    if len(S) < k:
        return
    if len(S) == 1:
        if k == 1:
            yield ((S[0],),)
        return

    s0 = S[0]

    # first, place s0 in its own new block
    for s in SetPartitions(S[1:], k-1):
        yield ((s0,),) + s

    # Then, insert s0 into each existing block
    for s in SetPartitions(S[1:], k):
        for i in range(len(s)):
            yield s[:i] + ((s0,) + s[i],) + s[i+1:]


class BGengine:
    def __init__(self, ks: List[Rational], ws: List[Rational], g: Rational):
        self.k = ks
        self.w = ws
        self.g = g

    @lru_cache(None)
    def current(self, subset: Tuple[int, ...]) -> Expr:
        s = list(subset)

        if len(s) == 1:
            return Rational(1)

        sumk = sum(self.k[i] for i in s)
        sumw = sum(self.w[i] for i in s)
        result = Rational(0)

        for m in range(2, len(s) + 1):
            for prt in SetPartitions(s, m):
                sumsk = [sum(self.k[i] for i in p) for p in prt]
                sumsw = [sum(self.w[i] for i in p) for p in prt]

                vks = [-sumk] + sumsk
                vws = [-sumw] + sumsw

                vertex = Vertex(m+1, vks, vws)
                curprod = Rational(1)
                for p in prt:
                    curprod *= self.current(tuple(p))

                result += vertex * curprod

        return result * Propagator(sumw, sumk, self.g)


def BGamplitude(ks: List[Rational], ws: List[Rational], g: Rational) -> Expr:
    engine = BGengine(ks, ws, g)

    n = len(ks)
    nbut0 = list(range(1, n))
    result = Rational(0)

    for m in range(2, n):
        for prt in SetPartitions(nbut0, m):
            sks = [sum(engine.k[i] for i in p) for p in prt]
            sws = [sum(engine.w[i] for i in p) for p in prt]

            vks = [engine.k[0]] + sks
            vws = [engine.w[0]] + sws

            vertex = Vertex(m+1, vks, vws)
            curprod = Rational(1)
            for p in prt:
                curprod *= engine.current(tuple(p))

            result += vertex * curprod

    return result


def MakeKinematics(n: int, wf: List[Rational], ss: List[int], g: Rational) -> Tuple[List[Rational], List[Rational]]:
    if len(wf) != n-2:
        raise ValueError("n-2 free frequencies required")

    if ss[0] + ss[-1] != 0:
        raise ValueError("sigma_1 + sigma_n should be 0")

    sumwf = sum(wf)
    ssf = ss[1:-1]
    sumssw2 = sum(s*w**2 for s, w in zip(ssf, wf))

    wn = -(ss[0] * sumwf**2 + sumssw2) / (2 * ss[0] * sumwf)
    w1 = -(sumwf + wn)

    ws = [w1] + list(wf) + [wn]
    ks = [ s*w**2 / g for s, w in zip(ss, ws)]

    return ks, ws


def ComputeAmplitude(n: int, wf: List[Rational], ss: List[int]) -> Tuple[List[Rational], List[Rational], Expr]:
    ks, ws = MakeKinematics(n, wf, ss, Rational(1))
    amplitude = BGamplitude(ks, ws, Rational(1))
    return ws, ks, amplitude


def InclExclFormula(n: int, w: List[Rational]) -> Expr:
    beta = min(abs(w[0]), abs(w[1]))
    wp = w[2:]
    value = Rational(0)
    for r in range(len(wp) + 1):
        for idx in combinations(range(len(wp)), r):
            subwp = [wp[i] for i in idx]
            sqsum = sum(x**2 for x in subwp)
            value += (-1)**len(subwp) * max(Rational(0), beta**2 - sqsum)**(n-3)
    return I * w[0] * w[1] * 2**(n-1) * value


ks, ws = MakeKinematics(
    5,
    [Rational(-3), Rational(-2), Rational(8)],
    [-1, -1, -1, 1, 1],
    Rational(1)
)

print(complex(BGamplitude(ks, ws, Rational(1))))

#print(InclExclFormula(5, [-14.723529411764707, -4.6, 14.4, 5.5, -0.5764705882352946]))
print(complex(InclExclFormula(5,
    [
        Rational(-14723529411764707, 10**15),
        Rational(-46, 10),
        Rational(144, 10),
        Rational(55, 10),
        Rational(5764705882352946, 10**16)
    ])))
