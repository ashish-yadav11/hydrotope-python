from __future__ import annotations
from math import factorial, prod, sqrt, pi
from sympy import symbols, limit, Expr, Rational, I
from functools import lru_cache
from itertools import combinations, permutations, product
from typing import List, Tuple, Iterator, Callable


g = Rational(1)


def omega(k):
    return sqrt(g*abs(k))


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
        ksp = [ks[i] for i in p]
        wsp = [ws[i] for i in p]

        val +=  (-1 if wsp[0] < 0 else 1) * (-1 if wsp[1] < 0 else 1) * abs(omega(ksp[0])/ksp[0]) * abs(omega(ksp[1])/ksp[1]) * Ekernel(n, ksp) # -1 from a*(-k) factor of ψ mode expansion
#       if ksp[0] >= 0 and ksp[1] >= 0:
#           val += (-1 if wsp[0] < 0 else 1) * (-1 if wsp[1] < 0 else 1) * abs(omega(ksp[0])/ksp[0]) * abs(omega(ksp[1])/ksp[1]) # -1 from a*(-k) factor of ψ mode expansion

    return -I * 2**(-n/2) * prod([sqrt(abs(k)/omega(k)) for k in ks]) * val
#   return I * (-1)**(n-1) * 2**(-n/2) * prod([sqrt(abs(k)/omega(k)) for k in ks]) * prod([k for k in ks if k >= 0]) * val / (n-2) # (-i)**2 {from mode expansion of ψ} * (-1) {factor in front in (7) 2019ussem 2/4}


def Propagator(k: Rational, w: Rational, s: int) -> Expr:
    wk = omega(k)

    if w + s*wk == 0:
        print('Warning: internal resonance detected!')

    return -I / (w + s*wk)


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
    def current(subset: Tuple[int, ...], sign: int) -> Expr:
        subset = list(subset)
        n = len(subset)

        kr = sum(ks[i] for i in subset)
        wr = sum(ws[i] for i in subset)
        swr = sign * wr
        val = Rational(0)

        for m in range(2, n + 1):
            for prt in SetPartitions(subset, m):
                prts = [p for p in prt if len(p) == 1]
                kpss = [ks[p[0]] for p in prts]
                wpss = [ws[p[0]] for p in prts]

                prtm = [p for p in prt if len(p) > 1]
                kpms = [sum(ks[i] for i in p) for p in prtm]
                npm = len(prtm)

                for signs in product([1, -1], repeat=npm):
                    wpms = [signs[j] * sum(ws[i] for i in prtm[j]) for j in range(npm)]
                    print("current", prts + prtm, wpss + wpms, npm)

                    curprod = Rational(1)
                    for j in range(npm):
                        curprod *= current(tuple(prtm[j]), signs[j])

                    val += Vertex(m+1, [-kr] + kpss + kpms, [-swr] + wpss + wpms) * curprod

        s = (1 if swr >= 0 else -1)
        return s * Propagator(-kr, wr, -s) * val

    return current


# ks and ws are assumed to be non-zero
def BGamplitude(ks: List[Rational], ws: List[Rational]) -> Expr:
    n = len(ks)
    nbut0 = list(range(1, n))
    current = BGcurrent(ks, ws)
    val = Rational(0)

    for m in range(2, n):
        for prt in SetPartitions(nbut0, m):
            prts = [p for p in prt if len(p) == 1]
            kpss = [ks[p[0]] for p in prts]
            wpss = [ws[p[0]] for p in prts]

            prtm = [p for p in prt if len(p) > 1]
            kpms = [sum(ks[i] for i in p) for p in prtm]
            npm = len(prtm)

            for signs in product([1, -1], repeat=npm):
                wpms = [signs[j] * sum(ws[i] for i in prtm[j]) for j in range(npm)]
                print("amplitude", prts + prtm, wpss + wpms, npm)

                curprod = Rational(1)
                for j in range(npm):
                    curprod *= current(tuple(prtm[j]), signs[j])

                val += Vertex(m+1, [ks[0]] + kpss + kpms, [ws[0]] + wpss + wpms) * curprod

    return val


# given n signs ss = [σ(1), σ(2), ..., σ(n-1), σ(n) = -σ(1)], and n-2
# frequencies wf = [ω(2), ..., ω(n-1)]; create on-shell momenta and frequencies
# (kᵢ = σᵢωᵢ², ωᵢ), 1 ≤ i ≤ n with Σkᵢ = Σωᵢ = 0. [the restriction σ(n) = -σ(1)
# is only for convenience—it leads to a unique solution.]
def MakeKinematics(wf: List[Rational], ss: List[int]) -> Tuple[List[Rational], List[Rational]]:
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
    return I * 2**(n/2 - 1) * sqrt(abs(prod(ws))) * ws[0] * ws[1] * val


ks, ws = MakeKinematics(
    [Rational(1), Rational(2)],
    [-1, -1, 1, 1],
)
print(ks, ws)
amp = BGamplitude(ks, ws)
ief = InclExclFormula(ws)
print(amp, ief)
