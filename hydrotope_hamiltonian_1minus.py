from __future__ import annotations
from math import factorial, prod
from sympy import Expr, Rational, I
from functools import lru_cache
from itertools import combinations, permutations
from typing import List, Tuple, Iterator, Callable


g = Rational(1)


def omegasqr(k: Rational) -> Rational:
    return g*abs(k)


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


def Ekernel(n: int, ks: List[Rational]) -> Rational:
    if n == 3:
        return -(abs(ks[0])*abs(ks[1]) + ks[0]*ks[1]) / 2

    val = abs(ks[1])**(n-3) * Ekernel(3, [ks[0], ks[1], sum(ks[2:])]) / factorial(n-2)

    for m in range(1, n-2): # m goes upto n-3
        val -= (abs(ks[1])**m / factorial(m)) * Ekernel(n-m, [ks[0], sum(ks[1:m+2])] + ks[m+2:])

    return val


def Vertex(n: int, ks: List[Rational], ws: List[Rational]) -> Expr:
    val = Rational(0)

    for p in permutations(range(n)):
        val +=  (ws[p[0]] / abs(ks[p[0]])) * (ws[p[1]] / abs(ks[p[1]])) * Ekernel(n, [ks[i] for i in p])
#       if ksp[0] >= 0 and ksp[1] >= 0:
#           val += (ws[p[0]] / abs(ks[p[0]])) * (ws[p[1]] / abs(ks[p[1]]))

    return -I * val
#   return I * (-1)**(n-1) * prod([k for k in ks if k >= 0]) * val / (n-2) # (-i)**2 {from mode expansion of ψ} * (-1) {factor in front in (7) 2019ussem 2/4}


def Propagator(k: Rational, w: Rational) -> Expr:
    wksqr = omegasqr(k)

    if w**2 == wksqr:
        print('Warning: internal resonance detected!')

    return -I * abs(k) / (w**2 - wksqr)


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

                val += Vertex(m+1, [-kr] + kps, [-wr] + wps) * curprod

        return Propagator(-kr, wr) * val

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

            amp = Vertex(m+1, [ks[0]] + kps, [ws[0]] + wps) * curprod
            print('amplitude', amp)
            val += amp

    return val


# given n signs ss = [σ(1), σ(2), ..., σ(n-1), σ(n) = -σ(1)], and n-2
# frequencies wf = [ω(2), ..., ω(n-1)]; create on-shell momenta and frequencies
# (kᵢ = σᵢωᵢ², ωᵢ), 1 ≤ i ≤ n with Σkᵢ = Σωᵢ = 0. [the restriction σ(n) = -σ(1)
# is only for convenience—it leads to a unique solution.]
def MakeKinematics(wf: List[Rational]) -> Tuple[List[Rational], List[Rational]]:
    ss = [-1] + [1] * (len(wf) + 1)

    sumwf = sum(wf)
    sumswf2 = sum(s*w**2 for s, w in zip(ss[1:-1], wf))

    wn = -(ss[0] * sumwf**2 + sumswf2) / (2 * ss[0] * sumwf)
    w1 = -(sumwf + wn)

    ws = [w1] + list(wf) + [wn]
    ks = [s*w**2 / g for s, w in zip(ss, ws)]

    return ks, ws


ks, ws = MakeKinematics(
    [Rational(1), Rational(2), Rational(3)],
)
print(ks, ws)
amp = BGamplitude(ks, ws)
print(amp)
