from __future__ import annotations
from math import factorial, prod, sqrt, pi
from sympy import symbols, limit, Expr, Rational, I
from functools import lru_cache
from itertools import combinations, permutations, product
from typing import List, Tuple, Iterator, Callable


def Vertexc(n: int, ks: List[Rational], ws: List[Rational]) -> Expr:
    return I * (-1)**(n-1) * factorial(n-3) # (-i)**2 {from mode expansion of ψ} * (-1) {factor in front in (7) 2019ussem 2/4}


def Propagatorc(k: Rational, w: Rational) -> Expr:
    if w**2 == abs(k):
        print('Warning: internal resonance detected!')

    return -I * k**2 / (w**2 - abs(k))


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


def BGcurrent(ks: List[Rational], ws: List[Rational]) -> Callable[[Tuple[int, ...]], Expr]:
    @lru_cache(maxsize=None)
    def current(subset: Tuple[int, ...]) -> Expr:
        subset = list(subset)
        n = len(subset)

        if n == 1:
            return Rational(1)

        kr = sum(ks[i] for i in subset)
        wr = sum(ws[i] for i in subset)
        val = Rational(0)

        for m in range(2, n + 1):
            for prt in SetPartitions(subset, m):
                kps = [sum(ks[i] for i in p) for p in prt]
                wps = [sum(ws[i] for i in p) for p in prt]
                print("current", prt, kps, wps)

                curprod = Rational(1)
                for p in prt:
                    curprod *= current(tuple(p))

                wbyk = [w/k for k,w in zip(kps, wps)]
                val += Vertexc(m+1, [-kr] + kps, [-wr] + wps) * (sum(wbyk)**2 - sum(x**2 for x in wbyk)) * curprod

        return Propagatorc(-kr, wr) * val

    return current


def BGamplitude(ks: List[Rational], ws: List[Rational]) -> Expr:
    n = len(ks)
    nbut0 = list(range(1, n))
    current = BGcurrent(ks, ws)
    val = Rational(0)

    for m in range(2, n): # root attached to (m+1)-point vertex
        for prt in SetPartitions(nbut0, m):
            kps = [sum(ks[i] for i in p) for p in prt]
            wps = [sum(ws[i] for i in p) for p in prt]
            print("amplitude", prt, kps, wps)

            curprod = Rational(1)
            for p in prt:
                curprod *= current(tuple(p))

            wbyk = [w/k for k,w in zip(kps, wps)]
            val += Vertexc(m+1, [ks[0]] + kps, [ws[0]] + wps) * (sum(wbyk)**2 - sum(x**2 for x in wbyk)) * curprod

    return val


def MakeKinematics(wf: List[Rational]) -> Tuple[List[Rational], List[Rational]]:
    ss = [-1] + [1] * (len(wf) + 1)

    sumwf = sum(wf)
    sumswf2 = sum(s*w**2 for s, w in zip(ss[1:-1], wf))

    wn = -(ss[0] * sumwf**2 + sumswf2) / (2 * ss[0] * sumwf)
    w1 = -(sumwf + wn)

    ws = [w1] + list(wf) + [wn]
    ks = [s*w**2 for s, w in zip(ss, ws)]

    return ks, ws


ks, ws = MakeKinematics(
    [Rational(1), Rational(2), Rational(3)],
)
print(ks, ws)
amp = BGamplitude(ks, ws)
print(amp)
