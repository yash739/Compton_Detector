"""Physical and geometric constants of the CZT pixel-detector Monte Carlo (AstroSat-CZTI-like module)."""

ME_KEV = 510.99895                  # electron rest energy (keV)
R0_CM = 2.8179403262e-13            # classical electron radius (cm)

# Cd(0.9)Zn(0.1)Te
RHO_G_CM3 = 5.78
FORMULA = "Cd0.9Zn0.1Te"
MASS_FRACTIONS = {"Cd": 0.4300, "Zn": 0.0278, "Te": 0.5423}     # 0.9 M_Cd : 0.1 M_Zn : 1 M_Te
ELEMENTS = ("Cd", "Zn", "Te")

# detector (CZTI module: 16 x 16 pixels, 2.46 mm pitch, 5 mm thick). A smaller array is enough for nearest-neighbour physics.
PITCH_CM = 0.246
THICKNESS_CM = 0.5
N_PIX = 7                           # 7 x 7 array; events are counted when the lower-energy pixel is in the central 3 x 3 block
FIDUCIAL = (2, 5)                   # pixel index range [2, 5) in each direction

# event selection (CZTI double-event reduction): both pixels above the software threshold, two adjacent pixels, total energy window.
E_THRESHOLD_KEV = 20.0
E_WINDOW_KEV = (100.0, 600.0)
RATIO_MAX = None                    # no upper limit on E_high / E_low (the old IDL cut was 6)
