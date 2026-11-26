import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

st.set_page_config(page_title="Wankel Rotary Engine Geometry Demo", layout="wide")

# ------------------------------------------------------------
# Geometry functions
# ------------------------------------------------------------

def housing_points(e, R, n=1200):
    """
    Housing inner wall:
        x = e cos(3a) + R cos(a)
        y = e sin(3a) + R sin(a)
    """
    a = np.linspace(0, 2*np.pi, n)
    x = e*np.cos(3*a) + R*np.cos(a)
    y = e*np.sin(3*a) + R*np.sin(a)
    return x, y, a


def rotate_points(points, angle):
    """
    Rotate local points by angle.
    """
    c = np.cos(angle)
    s = np.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    return points @ rot.T


def rotor_local_shape(R_rotor, bulge=0.78, points_per_side=100):
    """
    Approximate rotor body as a rounded triangular shape.

    This is not the exact production Wankel rotor flank.
    It is good enough for a classroom visual demo.

    Apexes are exactly placed at radius R_rotor.
    The sides are quadratic Bezier curves between apexes.
    """
    vertices = []
    for k in range(3):
        ang = 2*np.pi*k/3
        vertices.append(np.array([R_rotor*np.cos(ang), R_rotor*np.sin(ang)]))

    curve = []

    for i in range(3):
        p0 = vertices[i]
        p2 = vertices[(i+1) % 3]

        mid = 0.5 * (p0 + p2)

        # Direction from center through side midpoint
        norm = np.linalg.norm(mid)
        if norm > 1e-9:
            direction = mid / norm
        else:
            direction = np.array([1.0, 0.0])

        # Control point controls side roundness
        p1 = direction * (bulge * R_rotor)

        t = np.linspace(0, 1, points_per_side)
        for ti in t:
            p = (1-ti)**2*p0 + 2*(1-ti)*ti*p1 + ti**2*p2
            curve.append(p)

    return np.array(curve), np.array(vertices)


def transform_rotor(local_points, local_apexes, e, theta, R_rotor):
    """
    Rotor center:
        C = [e cos(theta), e sin(theta)]

    Rotor orientation:
        phi = theta / 3
    """
    C = np.array([e*np.cos(theta), e*np.sin(theta)])
    phi = theta / 3

    rotor_world = rotate_points(local_points, phi) + C
    apex_world = rotate_points(local_apexes, phi) + C

    return rotor_world, apex_world, C, phi


def circle_points(radius, center=(0, 0), n=300):
    t = np.linspace(0, 2*np.pi, n)
    x = center[0] + radius*np.cos(t)
    y = center[1] + radius*np.sin(t)
    return x, y


def make_csv(hx, hy, rotor_world):
    """
    Export housing and current rotor outline as CSV.
    """
    out = StringIO()
    out.write("curve,x_mm,y_mm\n")

    for x, y in zip(hx, hy):
        out.write(f"housing,{x:.5f},{y:.5f}\n")

    for p in rotor_world:
        out.write(f"rotor_current,{p[0]:.5f},{p[1]:.5f}\n")

    return out.getvalue()


def make_svg(hx, hy, rotor_world, scale_margin=15):
    """
    Basic SVG export of housing and current rotor outline.
    Units are in mm, but most vector programs import SVG as px.
    You can scale after import into CAD.
    """
    all_x = list(hx) + list(rotor_world[:, 0])
    all_y = list(hy) + list(rotor_world[:, 1])

    xmin = min(all_x) - scale_margin
    xmax = max(all_x) + scale_margin
    ymin = min(all_y) - scale_margin
    ymax = max(all_y) + scale_margin

    width = xmax - xmin
    height = ymax - ymin

    def pts_to_svg(xs, ys):
        # SVG y-axis is downward, so use -y
        return " ".join([f"{x:.3f},{-y:.3f}" for x, y in zip(xs, ys)])

    housing_pts = pts_to_svg(hx, hy)
    rotor_pts = pts_to_svg(rotor_world[:, 0], rotor_world[:, 1])

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg"
    viewBox="{xmin:.3f} {-ymax:.3f} {width:.3f} {height:.3f}"
    width="{width:.3f}mm" height="{height:.3f}mm">
    <polyline points="{housing_pts}"
        fill="none" stroke="black" stroke-width="0.6"/>
    <polyline points="{rotor_pts}"
        fill="none" stroke="red" stroke-width="0.6"/>
</svg>
'''
    return svg

def make_dxf_polyline(x, y, layer_name="CURVE"):
    dxf = []
    dxf.append("0")
    dxf.append("SECTION")
    dxf.append("2")
    dxf.append("HEADER")
    dxf.append("0")
    dxf.append("ENDSEC")
    dxf.append("0")
    dxf.append("SECTION")
    dxf.append("2")
    dxf.append("ENTITIES")

    dxf.append("0")
    dxf.append("LWPOLYLINE")
    dxf.append("8")
    dxf.append(layer_name)
    dxf.append("90")
    dxf.append(str(len(x)))
    dxf.append("70")
    dxf.append("1")  # closed polyline

    for xi, yi in zip(x, y):
        dxf.append("10")
        dxf.append(f"{xi:.5f}")
        dxf.append("20")
        dxf.append(f"{yi:.5f}")

    dxf.append("0")
    dxf.append("ENDSEC")
    dxf.append("0")
    dxf.append("EOF")

    return "\n".join(dxf)


# ------------------------------------------------------------
# Sidebar sliders
# ------------------------------------------------------------

st.sidebar.title("Wankel Geometry Controls")

plot_size = st.sidebar.slider(
    "Plot size",
    min_value=4.0,
    max_value=10.0,
    value=5.5,
    step=0.5
)

plot_margin = st.sidebar.slider(
    "Plot margin, mm",
    min_value=10.0,
    max_value=100.0,
    value=40.0,
    step=5.0
)

e = st.sidebar.slider(
    "Eccentricity e, mm",
    min_value=2.0,
    max_value=25.0,
    value=10.0,
    step=0.5
)

R = st.sidebar.slider(
    "Rotor center-to-apex radius R, mm",
    min_value=30.0,
    max_value=150.0,
    value=70.0,
    step=1.0
)

clearance = st.sidebar.slider(
    "Apex clearance, mm",
    min_value=0.0,
    max_value=2.0,
    value=0.5,
    step=0.05
)

theta_deg = st.sidebar.slider(
    "Shaft angle θ, degrees",
    min_value=0,
    max_value=1080,
    value=0,
    step=1
)

bulge = st.sidebar.slider(
    "Rotor side bulge approximation",
    min_value=0.55,
    max_value=0.95,
    value=0.78,
    step=0.01
)

rotor_width = st.sidebar.slider(
    "Rotor/housing width, mm",
    min_value=10.0,
    max_value=80.0,
    value=35.0,
    step=1.0
)

Ns = st.sidebar.slider(
    "Stationary gear teeth",
    min_value=16,
    max_value=80,
    value=32,
    step=2
)

num_rotors = st.sidebar.slider(
    "Number of rotors to show",
    min_value=1,
    max_value=4,
    value=1,
    step=1
)

show_gears = st.sidebar.checkbox("Show gear pitch circles", value=True)
show_orbit = st.sidebar.checkbox("Show eccentric orbit", value=True)
show_apex_traces = st.sidebar.checkbox("Show housing/apex trace", value=True)

theta = np.deg2rad(theta_deg)

# ------------------------------------------------------------
# Derived variables
# ------------------------------------------------------------

R_rotor = R - clearance

# Wankel stationary/internal gear relationship:
# stationary gear pitch radius = 2e
# rotor internal gear pitch radius = 3e
r_stationary = 2 * e
r_rotor_internal = 3 * e

D_stationary = 2 * r_stationary
D_rotor_internal = 2 * r_rotor_internal

Nr = int(1.5 * Ns)
module = D_stationary / Ns

major_width = 2 * (R + e)
minor_height = 2 * (R - e)

# ------------------------------------------------------------
# Main page
# ------------------------------------------------------------

st.title("3D-Printed Wankel Rotary Engine Geometry App")

st.markdown("""
This app visualizes the main geometry for a 3D-printed Wankel rotary engine demo.

The housing curve is generated from:
""")

st.latex(r"""
x = e\cos(3\alpha) + R\cos(\alpha)
""")

st.latex(r"""
y = e\sin(3\alpha) + R\sin(\alpha)
""")

st.markdown("The rotor center follows:")

st.latex(r"""
C(\theta) =
\begin{bmatrix}
e\cos\theta \\
e\sin\theta
\end{bmatrix}
""")

st.markdown("The rotor orientation is:")

st.latex(r"""
\phi = \frac{\theta}{3}
""")

if R <= 3 * e:
    st.warning(
        "R is small relative to e. For a nice Wankel-like shape, try R/e around 6 to 8."
    )

if R_rotor <= 0:
    st.error("Clearance is larger than R. Reduce clearance.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("R/e ratio", f"{R/e:.2f}")

with col2:
    st.metric("Housing major width", f"{major_width:.1f} mm")

with col3:
    st.metric("Housing minor height", f"{minor_height:.1f} mm")

with col4:
    st.metric("Rotor width", f"{rotor_width:.1f} mm")

st.subheader("Gear Parameters")

gear_col1, gear_col2, gear_col3, gear_col4 = st.columns(4)

with gear_col1:
    st.metric("Stationary gear", f"{Ns} teeth")

with gear_col2:
    st.metric("Rotor internal gear", f"{Nr} teeth")

with gear_col3:
    st.metric("Module", f"{module:.3f} mm/tooth")

with gear_col4:
    st.metric("Gear ratio", "2 : 3")

st.markdown("For the selected eccentricity:")

st.latex(rf"""
r_s = 2e = {r_stationary:.2f}\ \text{{mm}}
""")

st.latex(rf"""
r_r = 3e = {r_rotor_internal:.2f}\ \text{{mm}}
""")

st.markdown("So the gear pitch diameters are:")

st.latex(rf"""
D_s = {D_stationary:.2f}\ \text{{mm}}
""")

st.latex(rf"""
D_r = {D_rotor_internal:.2f}\ \text{{mm}}
""")

st.markdown(
    "The rotor internal gear should have **1.5×** as many teeth as the stationary gear."
)

# ------------------------------------------------------------
# Generate geometry
# ------------------------------------------------------------

hx, hy, alpha = housing_points(e, R)
rotor_local, apex_local = rotor_local_shape(R_rotor, bulge=bulge)

rotor_world, apex_world, C, phi = transform_rotor(
    rotor_local,
    apex_local,
    e,
    theta,
    R_rotor
)

# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(plot_size, plot_size), dpi=100)

# Housing
if show_apex_traces:
    ax.plot(hx, hy, color="black", linewidth=2.0, label="Housing inner wall / apex trace")

# Eccentric orbit
if show_orbit:
    ox, oy = circle_points(e)
    ax.plot(ox, oy, "--", color="gray", linewidth=1.0, label="Rotor center orbit")

# Main shaft center
ax.plot(0, 0, "ko", markersize=5)
ax.text(0, 0, "  Shaft center O", fontsize=8)

# Gear pitch circles
if show_gears:
    sx, sy = circle_points(r_stationary)
    ax.plot(sx, sy, color="blue", linestyle=":", linewidth=1.5, label="Stationary gear pitch circle")

# Show selected number of rotors
colors = ["red", "orange", "green", "purple"]

for n in range(num_rotors):
    phase_offset = n * np.pi / 2
    theta_n = theta + phase_offset

    rotor_n, apex_n, Cn, phin = transform_rotor(
        rotor_local,
        apex_local,
        e,
        theta_n,
        R_rotor
    )

    color = colors[n]

    ax.plot(
        rotor_n[:, 0],
        rotor_n[:, 1],
        color=color,
        linewidth=2.0,
        label=f"Rotor {n+1}"
    )

    ax.plot(
        apex_n[:, 0],
        apex_n[:, 1],
        "o",
        color=color,
        markersize=5
    )

    # Rotor center
    ax.plot(Cn[0], Cn[1], "o", color=color, markersize=4)
    ax.text(Cn[0], Cn[1], f"  C{n+1}", fontsize=8, color=color)

    # Internal gear pitch circle
    if show_gears:
        gx, gy = circle_points(r_rotor_internal, center=Cn)
        ax.plot(
            gx,
            gy,
            color=color,
            linestyle=":",
            linewidth=1.2,
            alpha=0.8
        )

# Draw eccentric arm for first rotor
ax.plot([0, C[0]], [0, C[1]], color="gray", linewidth=2, alpha=0.7)

lim = R + e + 20
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect("equal", adjustable="box")
ax.grid(True, alpha=0.3)
ax.set_xlabel("x, mm")
ax.set_ylabel("y, mm")
ax.legend(loc="upper right")

st.pyplot(fig, use_container_width=False)

# ------------------------------------------------------------
# Current state values
# ------------------------------------------------------------

st.subheader("Current Motion State")

state_col1, state_col2, state_col3, state_col4 = st.columns(4)

with state_col1:
    st.metric("Shaft angle θ", f"{theta_deg:.1f}°")

with state_col2:
    st.metric("Rotor angle φ = θ/3", f"{np.rad2deg(phi):.1f}°")

with state_col3:
    st.metric("Rotor center Cx", f"{C[0]:.2f} mm")

with state_col4:
    st.metric("Rotor center Cy", f"{C[1]:.2f} mm")

st.markdown("The three apex points are calculated from:")

st.latex(r"""
A_k(\theta)
=
C(\theta)
+
R
\begin{bmatrix}
\cos\left(\frac{\theta}{3}+\frac{2\pi k}{3}\right) \\
\sin\left(\frac{\theta}{3}+\frac{2\pi k}{3}\right)
\end{bmatrix}
""")

st.markdown("where:")

st.latex(r"""
k = 0, 1, 2
""")

# ------------------------------------------------------------
# Export
# ------------------------------------------------------------

st.subheader("Export Geometry")

csv_data = make_csv(hx, hy, rotor_world)
svg_data = make_svg(hx, hy, rotor_world)

export_col1, export_col2 = st.columns(2)

with export_col1:
    st.download_button(
        label="Download housing + current rotor as CSV",
        data=csv_data,
        file_name="wankel_geometry.csv",
        mime="text/csv"
    )

with export_col2:
    st.download_button(
        label="Download housing + current rotor as SVG",
        data=svg_data,
        file_name="wankel_geometry.svg",
        mime="image/svg+xml"
    )

housing_dxf = make_dxf_polyline(hx, hy, "HOUSING")

rotor_dxf = make_dxf_polyline(
    rotor_local[:, 0],
    rotor_local[:, 1],
    "ROTOR"
)

dxf_col1, dxf_col2 = st.columns(2)

with dxf_col1:
    st.download_button(
        label="Download housing DXF",
        data=housing_dxf,
        file_name="wankel_housing.dxf",
        mime="application/dxf"
    )

with dxf_col2:
    st.download_button(
        label="Download rotor DXF",
        data=rotor_dxf,
        file_name="wankel_rotor.dxf",
        mime="application/dxf"
    )

st.info("""
The SVG/CSV exports are mainly for curve reference. 
For real 3D printing, bring the housing curve into CAD, extrude it, then add bearings, bolt holes, gear teeth, and clearances.
""")