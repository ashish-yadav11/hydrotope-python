from __future__ import annotations
from math import factorial, prod, sqrt, pi
from functools import lru_cache
from itertools import combinations, permutations, product
from typing import List, Tuple, Iterator, Callable


g = 1


def omega(k: float) -> float:
    return sqrt(g*abs(k))


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


def Ekernel(n: int, ks: List[float]) -> float:
    if n == 3:
        return -(abs(ks[0])*abs(ks[1]) + ks[0]*ks[1]) / 2

    val = abs(ks[1])**(n-3) * Ekernel(3, [ks[0], ks[1], sum(ks[2:])]) / factorial(n-2)

    for m in range(1, n-2): # m goes upto n-3
        val -= (abs(ks[1])**m / factorial(m)) * Ekernel(n-m, [ks[0], sum(ks[1:m+2])] + ks[m+2:])

    return val


def Vertex(n: int, ks: List[float], ws: List[float], wsgns: [int] = []) -> complex:
    val = 0

    for p in permutations(range(n)):
        ksp = [ks[i] for i in p]
        wsgnp0 = 1 if ws[p[0]] > 0 else -1 if ws[p[0]] < 0 else wsgns[p[0]]
        wsgnp1 = 1 if ws[p[1]] > 0 else -1 if ws[p[1]] < 0 else wsgns[p[1]]
        val +=  wsgnp0 * wsgnp1 * abs(omega(ksp[0])/ksp[0]) * abs(omega(ksp[1])/ksp[1]) * Ekernel(n, ksp) # -1 from a*(-k) factor of ψ mode expansion
#       if ksp[0] >= 0 and ksp[1] >= 0:
#           val += wsgnp0 * wsgnp1 * abs(omega(ksp[0])/ksp[0]) * abs(omega(ksp[1])/ksp[1])

    return -1j * val
#   return 1j * (-1)**(n-1) * prod([k for k in ks if k >= 0]) * val / (n-2) # (-i)**2 {from mode expansion of ψ} * (-1) {factor in front in (7) 2019ussem 2/4}


def Propagator(k: float, w: float, s: int) -> complex:
    wk = omega(k)

    if w - s*wk == 0:
        print('Warning: internal resonance detected!')

    return -1j * s * (1/2) * (abs(k)/wk) / (w - s*wk) # 1/2 to compensate for 2**(-n/2) removal in Vertex, abs(k)/wk for prod(sqrt(abs(k)/wk))


def BGcurrent(ks: List[float], ws: List[float]) -> Callable[[Tuple[int, ...]], complex]:
    @lru_cache(maxsize=None)
    def current(subset: Tuple[int, ...], sign: int) -> complex:
        subset = list(subset)
        n = len(subset)

        kr = sum(ks[i] for i in subset)
        wr = sum(ws[i] for i in subset)
        swr = sign * wr
        val = 0

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

                    curprod = 1
                    for j in range(npm):
                        curprod *= current(tuple(prtm[j]), signs[j])

                    val += Vertex(m+1, [-kr] + kpss + kpms, [-swr] + wpss + wpms, [-sign] + [0]*(m-npm) + list(signs)) * curprod

        return Propagator(-kr, wr, (1 if swr > 0 else -1 if swr < 0 else sign)) * val

    return current


# ks and ws are assumed to be non-zero
def BGamplitude(ks: List[float], ws: List[float]) -> complex:
    n = len(ks)
    nbut0 = list(range(1, n))
    current = BGcurrent(ks, ws)
    val = 0

    for m in range(2, n): # root attached to (m+1)-point vertex
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

                curprod = 1
                for j in range(npm):
                    curprod *= current(tuple(prtm[j]), signs[j])

                val += Vertex(m+1, [ks[0]] + kpss + kpms, [ws[0]] + wpss + wpms, [0]*(n-npm) + list(signs)) * curprod

    return val


# given n signs ss = [σ(1), σ(2), ..., σ(n-1), σ(n) = -σ(1)], and n-2
# frequencies wf = [ω(2), ..., ω(n-1)]; create on-shell momenta and frequencies
# (kᵢ = σᵢωᵢ², ωᵢ), 1 ≤ i ≤ n with Σkᵢ = Σωᵢ = 0. [the restriction σ(n) = -σ(1)
# is only for convenience—it leads to a unique solution.]
def MakeKinematics(wf: List[float], ss: List[int]) -> Tuple[List[float], List[float]]:
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
def InclExclFormula(ws: List[float]) -> complex:
    n = len(ws)
    wp = ws[2:]
    np = n-2
    betasq = min(ws[0]**2, ws[1]**2)
    val = 0
    for r in range(np + 1):
        for c in combinations(range(np), r):
            subwp = [wp[i] for i in c]
            sqsum = sum(x**2 for x in subwp)
            val += (-1)**len(subwp) * max(0, betasq - sqsum)**(n-3)
    return 1j * 2**(n-1) * ws[0] * ws[1] * val # sqrt(abs(prod(ks)) / abs(prod(ws)))


ks, ws = MakeKinematics(
    [1, 2, 3],
    [-1, -1, 1, 1, 1],
)
print(ks, ws)
amp = BGamplitude(ks, ws)
ief = InclExclFormula(ws)
print(amp, ief, amp/ief)
