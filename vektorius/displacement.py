from bisect import bisect
import numpy as np

G = 9.81
RHO = 1.025
PROJECT = "dl-security-test"

CLASS_BP = np.array([
  ["Undefined",  1e-6],
  ["Small",     10000.],
  ["Handy",     25000.],
  ["Handymax",  55000.],
  ["Panamax",   85000.],
  ["Capesize", 200000.],
  ["VLBC",         1e6],
], np.dtype(object))

# Block Coefficient interpolation
SMALL_Cb = np.array([
  [1000.6056476644708, 0.7036036036036035],
  [1994.7005829358768, 0.7184684684684685],
  [3999.621470209705,  0.7477477477477478],
  [6002.119766825648,  0.7626126126126126],
  [8020.289196759784,  0.7707207207207207],
  [10004.315239609357, 0.7756756756756757],
])

HANDY_Cb = np.array([
  [9961.19556348031,   0.7673006134969325],
  [11966.75632385385,  0.7890797546012269],
  [13954.486385203541, 0.8013496932515337],
  [16967.814077185772, 0.8087116564417177],
  [19983.13638973677,  0.8059509202453987],
  [21974.674363081387, 0.7988957055214724],
])

HANDYMAX_Cb = np.array([
  [25000.0,           0.8020506039314754],
  [30000.0,           0.8229292650193415],
  [35000.0,           0.8325945369858687],
  [40000.0,           0.8343937001657851],
  [45000.0,           0.8298330307097181],
  [48726.41509433962, 0.8236093787005605],
  [54952.83018867925, 0.8446451409173442],
])

PANAMAX_Cb = np.array([
  [55051.54639175258, 0.8406008583690986],
  [60000.0,           0.8552789699570815],
  [64536.08247422681, 0.8442060085836909],
  [69020.61855670104, 0.8326180257510729],
  [72061.8556701031,  0.8462660944206007],
  [76030.92783505155, 0.8632618025751072],
  [80051.54639175258, 0.878969957081545],
  [85000.0,           0.8972532188841201],
])

CAPESIZE_Cb = np.array([
  [60125.62814070351,  0.8391304347826087],
  [76959.79899497487,  0.8462450592885377],
  [94045.22613065326,  0.84600790513834],
  [110879.39698492462, 0.842213438735178],
  [127964.824120603,   0.8471936758893281],
  [144798.99497487437, 0.8498023715415021],
  [162135.67839195978, 0.8502766798418973],
  [178969.84924623114, 0.8490909090909091],
  [195552.76381909547, 0.8464822134387352],
  [209874.37185929646, 0.8441106719367589],
])

VLBC_Cb = np.array([
  [199885.05747126436, 0.8281092177646585],
  [214827.58620689652, 0.8311093519388166],
  [230000.0,           0.8327081712062256],
  [239885.05747126436, 0.818910774184892],
  [249770.11494252871, 0.8053468401985778],
  [259885.05747126436, 0.8118601905273044],
  [270000.0,           0.8176731517509728],
  [299885.0574712644,  0.82320649402925],
  [329770.1149425288,  0.8261717429223131],
])

BLOCK_COEFF = {
  "Undefined": np.zeros((2, 2)),
  "Small" :    SMALL_Cb,
  "Handy":     HANDY_Cb,
  "Handymax":  HANDYMAX_Cb,
  "Panamax":   PANAMAX_Cb,
  "Capesize":  CAPESIZE_Cb,
  "VLBC":      VLBC_Cb
}

DESIGN_DRAUGHT = {
  "Undefined": lambda dwt: 0.529*dwt**0.285,
  "Small" :    lambda dwt: 0.529*dwt**0.285,
  "Handy":     lambda dwt: 0.000141*dwt + 6.2,
  "Handymax":  lambda dwt: 0.000101*dwt + 6.84,
  "Panamax":   lambda dwt: 0.0000735*dwt + 8.43,
  "Capesize":  lambda dwt: 0.179*dwt**0.3814,
  "VLBC":      lambda dwt: 0.000015*dwt + 14.95
}

DEPTH = {
  "Small" :    lambda dwt: 5.22 + 0.000485*dwt,
  "Handy":     lambda dwt: 7.84 + 0.000232*dwt,
  "Handymax":  lambda dwt: 9.32 + 0.000158*dwt,
  "Panamax":   lambda dwt: 13.47 + 0.0000777*dwt,
  "Capesize":  lambda dwt: 1.126*dwt**0.2545,
  "VLBC":      lambda dwt: min(6.86 + 0.0000857*dwt, 30)
}

LIGHTWEIGHT = {
  "Small" :    lambda dwt: 0.831*dwt - 0.2,
  "Handy":     lambda dwt: 1.05*(0.153 - 0.00000158*dwt),
  "Handymax":  lambda dwt: 1.05*(0.151 - 0.00000127*dwt),
  "Panamax":   lambda dwt: 1.05*0.079,
  "Capesize":  lambda dwt: 0.0817 - 0.0000000486*dwt,
  "VLBC":      lambda dwt: 1.05*(0.076 - 0.0000000261*dwt)
}


# Derived DWT regression formulas (with 5 significant digits)
SMALL_DWT = lambda L, B: 0.0029893*L**3.0395 - 366.02*np.sqrt(1737. - 67.*B) + 11575.0
HANDY_DWT = lambda L, B: 0.00030717*L**3.0441 + 1894.9*B - 28158.0
HANDYMAX_DWT = lambda L, B: 7.0187e-09*L**5.5157 + 1841.0*B - 34850.0
PANAMAX_DWT = lambda L, B: np.piecewise(
    L,
    [(L < 201.05)*(L >= 193.68), (L <= 226.32)*(L >= 201.05), (L > 226.32)*(L <= 233.68)],
    [(L - 107.00) / 0.0014,     (L - 31.00) / 0.00267,      (L - 180.50) / 0.0005],
)

CAPESIZE_DWT = lambda L, B: 0.0030344*L**3.1056 + np.piecewise(B,
  [(B < 41)*(B >= 38),  (B <= 47)*(B >= 41)],
  [1979.8*B - 55038.1, 3167.7*B - 103740.0]
)

VLBC_DWT = lambda L, B: 1602.29*L - 279210.0

DERIVED_DWT = [
  SMALL_DWT, HANDY_DWT, HANDYMAX_DWT, PANAMAX_DWT, CAPESIZE_DWT, VLBC_DWT,
]

# Breakpoints by length
LENGTH_BP = [
  [0.0, 121.7],   [118.5, 160.2], [160.2, 193.7], [193.7, 233.7],
  [233.7, 305.9], [305.9, 337.9],
]

# Breakpoints by width
WIDTH_BP = [
  [11.0, 19.4],  [19.4, 26.2], [25.7, 32.2], [30.0, 32.2],
  [38.0, 47.0], [50.0, 57.5],
]


def bisect2d(A, B, a, b):
    idx = iter([
        i
        for i, (AA, BB) in enumerate(zip(A, B))
        if AA[0] < a <= AA[1] and BB[0] < b <= BB[1]
    ])
    return next(idx, None)


def get_displacement(dwt, draft, length, width):
    i = bisect(CLASS_BP[:, 1], dwt)
    klass = CLASS_BP[i, 0]
    depth = DEPTH[klass](dwt)
    coeff = BLOCK_COEFF[klass]
    Cb = np.interp(dwt, coeff[:, 0], coeff[:, 1],)
    Td = DESIGN_DRAUGHT[klass](dwt)
    op_Cb = 1 - (1 - Cb) * (Td / float(draft)) ** (0.333)
    dm = LIGHTWEIGHT[klass](dwt)*length*width*depth
    disp = 0.97 * length * width * op_Cb * RHO * draft - dm
    return klass, Cb, Td, disp

