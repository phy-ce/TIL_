# Holonomy & Parallel Transport: folding a flat circular sector into a cone.
#
# A flat sheet has zero curvature -> parallel transport keeps every arrow
# strictly parallel (all point +Y).  Rolling it ISOMETRICALLY into a cone bends
# the sheet; the arrows stick to the surface and the two seam edges meet, but
# their directions disagree.  That mismatch is the holonomy alpha(C).
#
# Morph math (isometric flat-sector -> cone), with fold parameter t in [0, 1]:
#   phi(t) : pi/2 (flat)            ->  arcsin(THETA / 2pi) (closed cone)
#   psi    = beta / sin(phi)        (arc length preserved  => isometric)
#   P(r, beta) = ( r sinφ cosψ,  r sinφ sinψ,  -r cosφ )
# At t = 0 this is exactly the flat sector in the XY plane (Z = 0).

# Written by claude code
# https://arxiv.org/pdf/2407.10464v1 p4~p5

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
R         = 2.5                       # slant radius of the sector / cone
THETA     = 1.5 * np.pi               # sector opening angle (Theta < 2*pi)
PHI_CONE  = np.arcsin(THETA / (2 * np.pi))   # half-angle of the closed cone
N_ARROWS  = 9                         # arrows along the outer arc
ARROW_LEN = 0.7
NR, NB    = 24, 90                    # surface grid resolution (radial, angular)

COL_BLUE, COL_RED, COL_MID = '#2f7bff', '#ff3b3b', '#444444'


def phi_of(t):
    return (np.pi / 2) * (1 - t) + PHI_CONE * t


def surface_point(r, beta, t):
    """Map flat polar (r, beta) onto the (partially) folded cone."""
    phi = phi_of(t)
    s, c = np.sin(phi), np.cos(phi)
    psi = beta / s
    return (r * s * np.cos(psi), r * s * np.sin(psi), -r * c)


def transport_dir(beta, t):
    """Parallel-transported direction = push-forward of the flat field (0,1,0).

    In the orthonormal surface frame {Pr, Pbeta_hat}, the constant flat vector
    (0,1,0) is sin(beta)*e_r + cos(beta)*e_beta, so its image is the same
    combination of the surface basis vectors.
    """
    phi = phi_of(t)
    s, c = np.sin(phi), np.cos(phi)
    psi = beta / s
    Pr = np.array([s * np.cos(psi), s * np.sin(psi), -c * np.ones_like(psi)])
    Pb = np.array([-np.sin(psi), np.cos(psi), np.zeros_like(psi)])
    d = np.sin(beta) * Pr + np.cos(beta) * Pb
    return d / np.linalg.norm(d, axis=0)


def surface_normal(beta, t):
    phi = phi_of(t)
    s, c = np.sin(phi), np.cos(phi)
    psi = beta / s
    Pr = np.array([s * np.cos(psi), s * np.sin(psi), -c * np.ones_like(psi)])
    Pb = np.array([-np.sin(psi), np.cos(psi), np.zeros_like(psi)])
    n = np.cross(Pr, Pb, axis=0)
    return n / np.linalg.norm(n, axis=0)


# Pre-built parameter grids ---------------------------------------------------
r_lin = np.linspace(0, R, NR + 1)
b_lin = np.linspace(0, THETA, NB + 1)
Rg, Bg = np.meshgrid(r_lin, b_lin, indexing='ij')
betas = np.linspace(0, THETA, N_ARROWS)
arrow_colors = [COL_BLUE] + [COL_MID] * (N_ARROWS - 2) + [COL_RED]

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(10, 8))
fig.subplots_adjust(left=0.0, right=1.0, bottom=0.12, top=1.0)
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1, 1, 1])
ax.set_axis_off()

LIM = R * 1.05
ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_zlim(-R, R * 0.1)
ax.view_init(elev=22, azim=-60)

artists = {'surf': None, 'quiv': None, 'edge_b': None, 'edge_r': None}


def draw(t):
    # --- surface ---
    X, Y, Z = surface_point(Rg, Bg, t)
    if artists['surf'] is not None:
        artists['surf'].remove()
    artists['surf'] = ax.plot_surface(
        X, Y, Z, color='#6f86c9', alpha=0.55, linewidth=0,
        rstride=1, cstride=2, edgecolor='#27324a', antialiased=True)

    # --- arrows along the outer arc (r = R), stuck to the tangent plane ---
    px, py, pz = surface_point(R, betas, t)
    nx, ny, nz = surface_normal(betas, t)
    px, py, pz = px + 0.02 * nx, py + 0.02 * ny, pz + 0.02 * nz
    ux, uy, uz = transport_dir(betas, t)
    if artists['quiv'] is not None:
        artists['quiv'].remove()
    artists['quiv'] = ax.quiver(px, py, pz, ux, uy, uz, colors=arrow_colors,
                                length=ARROW_LEN, normalize=True, linewidth=2.5)

    # --- seam edge guide lines (beta = 0 blue, beta = THETA red) ---
    ex0, ey0, ez0 = surface_point(r_lin, 0.0, t)
    ex1, ey1, ez1 = surface_point(r_lin, THETA, t)
    if artists['edge_b'] is not None:
        artists['edge_b'].remove(); artists['edge_r'].remove()
    artists['edge_b'], = ax.plot(ex0, ey0, ez0, color=COL_BLUE, lw=2, alpha=0.6)
    artists['edge_r'], = ax.plot(ex1, ey1, ez1, color=COL_RED, lw=2, alpha=0.6)

    # --- readouts ---
    bdir = transport_dir(np.array([0.0]), t)[:, 0]
    rdir = transport_dir(np.array([THETA]), t)[:, 0]
    alpha = np.degrees(np.arccos(np.clip(bdir @ rdir, -1, 1)))
    p0 = np.array(surface_point(R, 0.0, t))
    p1 = np.array(surface_point(R, THETA, t))
    seam = np.linalg.norm(p0 - p1)
    ax.set_title(f"Fold t = {t:0.2f}    |    cone half-angle φ = {np.degrees(phi_of(t)):0.1f}°\n"
                 f"seam separation = {seam:0.3f}    |    "
                 f"holonomy α(C) = ∠(blue, red) = {alpha:0.1f}°",
                 fontsize=11)
    fig.canvas.draw_idle()


# ---------------------------------------------------------------------------
# Widgets: slider + auto-fold button
# ---------------------------------------------------------------------------
ax_slider = fig.add_axes([0.18, 0.05, 0.55, 0.03])
slider = Slider(ax_slider, 'Fold progress', 0.0, 1.0, valinit=0.0)
slider.on_changed(draw)

ax_btn = fig.add_axes([0.80, 0.045, 0.12, 0.045])
button = Button(ax_btn, 'Auto-fold')

state = {'playing': False, 'phase': 0.0}
timer = fig.canvas.new_timer(interval=30)


def on_tick():
    state['phase'] += 0.05
    t = 0.5 - 0.5 * np.cos(state['phase'])     # smooth 0..1..0 ping-pong
    slider.set_val(t)                          # triggers draw


def toggle(_event):
    state['playing'] = not state['playing']
    button.label.set_text('Pause' if state['playing'] else 'Auto-fold')
    timer.start() if state['playing'] else timer.stop()


timer.add_callback(on_tick)
button.on_clicked(toggle)

draw(0.0)
plt.show()
