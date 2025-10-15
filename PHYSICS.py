import tkinter as tk
from tkinter import ttk
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import patches
from matplotlib import transforms as mtransforms
import sys

# ---------- Constants ----------
RHO_AIR = 1.225
G = 9.81
SPEED_MIN, SPEED_MAX = 0.0, 300.0
HEIGHT_MIN, HEIGHT_MAX = 0.0, 100.0
HOVER_SNAP_RADIUS = 30  # pixels
CURVE_SNAP_RADIUS = 35  # pixels

# ---------- Tk root & vars (create early) ----------
root = tk.Tk()
root.title("Projectile Motion Simulator")
try:
    # DPI scaling (Windows)
    if hasattr(root, 'tk'):
        root.tk.call('tk', 'scaling', 1.5)
    # Larger default font
    base_font = ('Segoe UI', 10)
    root.option_add('*Font', base_font)
except Exception:
    pass

velocity = tk.DoubleVar(value=30.0)
angle_deg = tk.DoubleVar(value=45.0)
height = tk.DoubleVar(value=9.0)

angle_var = tk.BooleanVar(value=True)
elevation_var = tk.BooleanVar(value=True)
air_var = tk.BooleanVar(value=False)
components_var = tk.BooleanVar(value=False)
compare_var = tk.BooleanVar(value=False)

object_var = tk.StringVar(value="Ball")
mass_var = tk.DoubleVar(value=0.45)
diam_var = tk.DoubleVar(value=0.22)
cd_var = tk.DoubleVar(value=0.47)

# ---------- Matplotlib embedded figure ----------
fig, ax = plt.subplots(figsize=(8,5))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

ax.set_xlabel("Distance (m)")
ax.set_ylabel("Height (m)")
ax.set_title("Projectile Path")

# Hover artists (global references)
hover_point = None
hover_annot = None
hover_cid = None

# persist hover state across redraws
last_hover = {'visible': False, 'xy': None, 'text': ''}

# Component points storage
component_points = []

# (target marker feature removed)

# ---------- globals for animation/playback ----------
anim_running = False
anim_paused = False
anim_dt = 0.02
anim_speed_var = tk.DoubleVar(value=1.0)
anim_marker = None
anim_trail_line = None
trail_x = []
trail_y = []

anim_play_t = None
anim_play_x = None
anim_play_vx = None
anim_play_y = None
anim_play_vy = None
anim_t = 0.0

hover_data = None
last_run = {}
last_run_no_drag = {}  # for comparison mode

redraw_widgets = []
pending_update = False

# ---------- Physics (trajectory calculation) ----------
def compute_trajectory(v0, angle_deg_val, h0_val, use_drag, mass, diameter, cd):
    """
    PROJECTILE MOTION EQUATIONS:
    
    NO DRAG:
    - Horizontal: x(t) = v₀ cos(θ) · t
    - Vertical: y(t) = h₀ + v₀ sin(θ) · t - ½gt²
    - vx(t) = v₀ cos(θ)  [constant]
    - vy(t) = v₀ sin(θ) - gt
    - Time of flight: t = [v₀sinθ + √((v₀sinθ)² + 2gh₀)] / g
    - Range: R = v₀ cos(θ) · t_flight
    - Max height: H = h₀ + (v₀² sin²θ) / (2g)
    - Time to apex: t_apex = (v₀ sinθ) / g
    
    WITH DRAG:
    - Numerical integration using semi-implicit Euler
    - Drag force: F_d = ½ ρ C_d A v²
    - Acceleration: a = -g - (F_d / m) in direction of velocity
    """
    ang = math.radians(angle_deg_val)
    
    if not use_drag:
        # Analytical solution
        vert = v0 * math.sin(ang)
        disc = vert*vert + 2*G*h0_val
        if disc < 0:
            return np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),0.0
        t_flight = (vert + math.sqrt(disc)) / G
        if t_flight <= 0:
            return np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),0.0
        
        t = np.linspace(0, t_flight, max(2, int(np.ceil(t_flight/0.01))))
        vx = np.full_like(t, v0 * math.cos(ang))  # Constant horizontal velocity
        vy = v0 * math.sin(ang) - G * t  # Linear change in vertical velocity
        x = v0 * math.cos(ang) * t
        y = h0_val + v0 * math.sin(ang) * t - 0.5 * G * t**2
        return t, x, y, vx, vy, t_flight

    # Numerical integration with quadratic drag
    area = math.pi * (diameter/2.0)**2
    dt = 0.01
    vx = v0 * math.cos(ang)
    vy = v0 * math.sin(ang)
    x = 0.0
    y = h0_val
    t_vals = [0.0]; x_vals=[x]; y_vals=[y]; vx_vals=[vx]; vy_vals=[vy]
    t_curr = 0.0
    while y >= 0 and t_curr < 120.0:
        v = math.hypot(vx, vy)
        if v > 0:
            drag = 0.5 * RHO_AIR * cd * area * v
            ax_drag = -(drag * vx) / mass
            ay_drag = -(drag * vy) / mass
        else:
            ax_drag = 0.0; ay_drag = 0.0
        vx += ax_drag * dt
        vy += (ay_drag - G) * dt
        x += vx * dt
        y += vy * dt
        t_curr += dt
        t_vals.append(t_curr); x_vals.append(x); y_vals.append(max(y,0.0))
        vx_vals.append(vx); vy_vals.append(vy)
    return np.array(t_vals), np.array(x_vals), np.array(y_vals), np.array(vx_vals), np.array(vy_vals), t_curr

# ---------- Hover handler ----------
def _on_mouse_move(event):
    global hover_point, hover_annot, hover_data, component_points, last_run, last_hover
    if event.inaxes != ax or hover_data is None:
        if hover_annot is not None:
            hover_annot.set_visible(False)
        last_hover['visible'] = False
        last_hover['xy'] = None
        last_hover['text'] = ''
        canvas.draw_idle()
        return

    # Snap to component dots first
    if components_var.get() and component_points:
        comp_xy = np.array([[p['x'], p['y']] for p in component_points])
        txc, tyc = ax.transData.transform(comp_xy).T
        dx = (txc - event.x)
        dy = (tyc - event.y)
        d2c = dx*dx + dy*dy
        ci = int(np.argmin(d2c))
        if d2c[ci] <= (HOVER_SNAP_RADIUS**2):
            p = component_points[ci]
            vx, vy = p['vx'], p['vy']
            speed = float(np.hypot(vx, vy))
            if hover_point is not None:
                hover_point.set_data([p['x']], [p['y']])
            if hover_annot is not None:
                txt = (
                    f"x={p['x']:.3f} m\n"
                    f"speed={speed:.3f} m/s\n"
                    f"vx={vx:.3f} m/s\n"
                    f"vy={vy:.3f} m/s"
                )
                hover_annot.xy = (p['x'], p['y'])
                hover_annot.set_text(txt)
                hover_annot.set_visible(True)
                last_hover['visible'] = True
                last_hover['xy'] = (p['x'], p['y'])
                last_hover['text'] = txt
            canvas.draw_idle()
            return

    # Snap to nearest point on curve
    x = hover_data['x']; y = hover_data['y']
    if x.size == 0:
        if hover_annot is not None:
            hover_annot.set_visible(False)
        last_hover['visible'] = False
        last_hover['xy'] = None
        last_hover['text'] = ''
        canvas.draw_idle()
        return
    if event.xdata is None or event.ydata is None:
        return

    tx, ty = ax.transData.transform(np.column_stack((x, y))).T
    d2 = (tx - event.x)**2 + (ty - event.y)**2
    idx = int(np.argmin(d2))
    if d2[idx] > (CURVE_SNAP_RADIUS**2):
        if hover_annot is not None:
            hover_annot.set_visible(False)
        if hover_point is not None:
            hover_point.set_data([np.nan], [np.nan])
        last_hover['visible'] = False
        last_hover['xy'] = None
        last_hover['text'] = ''
        canvas.draw_idle()
        return

    px = float(x[idx]); py = float(y[idx])
    t_here = float(hover_data['t'][idx])
    speed = float(np.hypot(hover_data['vx'][idx], hover_data['vy'][idx]))

    if hover_point is not None:
        hover_point.set_data([px], [py])
    if hover_annot is not None:
        vx_val = float(hover_data['vx'][idx])
        vy_val = float(hover_data['vy'][idx])
        txt = (
            f"t={t_here:.3f} s\n"
            f"x={px:.3f} m\n"
            f"speed={speed:.3f} m/s\n"
            f"vx={vx_val:.3f} m/s\n"
            f"vy={vy_val:.3f} m/s"
        )
        hover_annot.xy = (px, py)
        hover_annot.set_text(txt)
        hover_annot.set_visible(True)
        last_hover['visible'] = True
        last_hover['xy'] = (px, py)
        last_hover['text'] = txt
    canvas.draw_idle()

# ---------- Helper to (re)create hover artists and reconnect ----------
def _create_hover_artists_and_reconnect():
    global hover_point, hover_annot, hover_cid, last_hover
    
    # Remove existing artists
    if hover_point is not None:
        try:
            hover_point.remove()
        except Exception:
            pass
    if hover_annot is not None:
        try:
            hover_annot.remove()
        except Exception:
            pass

    # Create new hover artists
    hover_point = ax.plot([np.nan], [np.nan], 'o', color='red', markersize=6, zorder=11)[0]
    hover_annot = ax.annotate('', xy=(0,0), xytext=(12,10),
                             textcoords='offset points',
                             bbox=dict(boxstyle='round', fc='w', ec='0.4', alpha=0.95))
    hover_annot.set_visible(False)

    # HUD text box (top-left)
    if hasattr(ax, 'hud_text') and ax.hud_text is not None:
        try:
            ax.hud_text.remove()
        except Exception:
            pass
    ax.hud_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top', ha='left', 
                          fontsize=9, bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.9))

    # Reconnect hover event
    if hover_cid is not None:
        try:
            canvas.mpl_disconnect(hover_cid)
        except Exception:
            pass
    hover_cid = canvas.mpl_connect('motion_notify_event', _on_mouse_move)

    # Restore previous hover if it existed
    if last_hover.get('visible') and last_hover.get('xy') is not None:
        try:
            xy = last_hover['xy']
            hover_annot.xy = xy
            hover_annot.set_text(last_hover.get('text', ''))
            hover_annot.set_visible(True)
            hover_point.set_data([xy[0]], [xy[1]])
        except Exception:
            pass

# ---------- Widget state management ----------
def safe_request_update():
    global pending_update
    if anim_running:
        pending_update = True
    else:
        update_plot()

def safe_set_widget_state(widget, enabled):
    try:
        if hasattr(widget, 'state') and callable(getattr(widget, 'state')):
            if enabled:
                widget.state(['!disabled'])
            else:
                widget.state(['disabled'])
        else:
            widget.configure(state=('normal' if enabled else 'disabled'))
    except Exception:
        pass

def disable_redraw_widgets(enabled: bool):
    for w in redraw_widgets:
        safe_set_widget_state(w, enabled)

# ---------- Update plot & results ----------
def update_plot(*args):
    global hover_data, last_run, last_run_no_drag, component_points, last_hover, anim_running, pending_update
    global target_artist
    
    if anim_running:
        pending_update = True
        return

    reset_animation(silent=True)

    v0 = float(velocity.get())
    angle = float(angle_deg.get()) if angle_var.get() else 0.0
    h0 = float(height.get()) if elevation_var.get() else 0.0
    use_drag = bool(air_var.get())
    m = float(mass_var.get())
    d = float(diam_var.get())
    cd = float(cd_var.get())

    # Compute main trajectory
    t, x, y, vx_arr, vy_arr, t_flight = compute_trajectory(v0, angle, h0, use_drag, m, d, cd)

    # Compute no-drag trajectory for comparison if requested
    t_nd, x_nd, y_nd, vx_nd, vy_nd, t_flight_nd = None, None, None, None, None, None
    if compare_var.get():
        # Always compute no-drag trajectory for comparison
        # Use a standard drag coefficient of 0 for pure no-drag comparison
        t_nd, x_nd, y_nd, vx_nd, vy_nd, t_flight_nd = compute_trajectory(v0, angle, h0, False, m, d, 0.0)

    # Clear and recreate axes
    ax.cla()
    ax.axhline(0, color='k', linewidth=1)
    _create_hover_artists_and_reconnect()

    # Visual cannon
    body = patches.Rectangle((-0.35, h0), 0.35, 0.22, color='dimgray', zorder=2)
    ax.add_patch(body)
    wheel = patches.Circle((-0.15, h0), 0.08, color='black', zorder=3)
    ax.add_patch(wheel)
    barrel_len = 1.0
    ang_rad = math.radians(angle)
    ax.plot([0, barrel_len*math.cos(ang_rad)], [h0, h0+barrel_len*math.sin(ang_rad)], 
            color='tab:gray', linewidth=6, solid_capstyle='round', zorder=4)

    # Plot comparison trajectory (no drag) if enabled
    if compare_var.get() and x_nd is not None and x_nd.size > 0:
        ax.plot(x_nd, y_nd, color='tab:green', linestyle='--', linewidth=3, 
                alpha=0.8, label='Without drag', zorder=5)
        range_nd = float(x_nd[-1]) if x_nd.size else 0.0
        ax.scatter([range_nd], [0], color='tab:green', s=60, zorder=5, alpha=0.8)

    # Plot main trajectory
    if x.size:
        label = 'With drag' if (compare_var.get() and use_drag) else None
        ax.plot(x, y, color='tab:blue', linewidth=2, label=label, zorder=6)

    # Component points and velocity vectors
    component_points = []
    if components_var.get() and x.size:
        if x.size >= 6:
            idxs = np.linspace(1, x.size - 2, 6).astype(int)
        else:
            idxs = np.arange(1, max(x.size - 1, 1))
        pts_x = []; pts_y = []
        for i in idxs:
            px, py = float(x[i]), float(y[i])
            vx_i, vy_i = float(vx_arr[i]), float(vy_arr[i])
            component_points.append({'x': px, 'y': py, 'vx': vx_i, 'vy': vy_i})
            pts_x.append(px); pts_y.append(py)
            
            # Draw velocity component vectors
            scale = 0.5  # Increased scale for better visibility
            # Horizontal velocity component (vx) - red arrow
            if abs(vx_i) > 0.1:  # Only draw if significant
                ax.arrow(px, py, vx_i*scale, 0, head_width=0.4, head_length=0.3, 
                        fc='red', ec='red', alpha=0.8, linewidth=2, zorder=8)
            # Vertical velocity component (vy) - blue arrow  
            if abs(vy_i) > 0.1:  # Only draw if significant
                ax.arrow(px, py, 0, vy_i*scale, head_width=0.4, head_length=0.3, 
                        fc='blue', ec='blue', alpha=0.8, linewidth=2, zorder=8)
        
        ax.scatter(pts_x, pts_y, s=20, c='black', zorder=7)
        
        # Add legend for velocity vectors
        if len(pts_x) > 0:
            # Create dummy lines for legend
            ax.plot([], [], color='red', linewidth=2, label='vx (horizontal)')
            ax.plot([], [], color='blue', linewidth=2, label='vy (vertical)')
        
        # Mark apex
        apex_i = int(np.argmax(y))
        ax.scatter([x[apex_i]], [y[apex_i]], color='orange', s=60, marker='*', 
                  zorder=7, label='Apex')

    # Mark launch and landing points
    horizontal_range = float(x[-1]) if x.size else 0.0
    max_height = float(np.max(y)) if y.size else h0
    ax.scatter([0, horizontal_range], [h0, 0], color='tab:red', s=50, zorder=5)

    # Labels and legend
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Height (m)")
    ax.set_title("Projectile Path")
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_ylim(bottom=0)
    # Expand x-axis to include the longer trajectory when comparing
    try:
        right_limit = horizontal_range
        if compare_var.get():
            # range_nd may exist from comparison block above
            right_limit = max(horizontal_range, float(locals().get('range_nd', 0.0)))
        ax.set_xlim(left=-0.5, right=max(right_limit * 1.1, 5.0))
    except Exception:
        ax.set_xlim(left=-0.5, right=max(horizontal_range * 1.1, 5.0))
    
    if compare_var.get():
        ax.legend(loc='upper right')
    elif components_var.get():
        ax.legend(loc='upper right')

    # Target marker removed

    # Store data for hover
    if x.size > 0:
        hover_data = {'t': t, 'x': x, 'y': y, 'vx': vx_arr, 'vy': vy_arr}
    else:
        hover_data = None

    # Compute projectile metrics
    time_to_apex = 0.0
    apex_x_coord = 0.0
    apex_y_coord = 0.0
    
    if x.size:
        apex_i = int(np.argmax(y))
        time_to_apex = float(hover_data['t'][apex_i])
        apex_x_coord = float(hover_data['x'][apex_i])
        apex_y_coord = float(hover_data['y'][apex_i])

    # Store results
    last_run.update({
        't': t, 'x': x, 'y': y, 'vx': vx_arr, 'vy': vy_arr,
        't_flight': t_flight, 'range': horizontal_range, 'maxh': max_height,
        'time_to_apex': time_to_apex, 'apex_x': apex_x_coord, 'apex_y': apex_y_coord,
        'mass': m
    })

    # Update results panel
    try:
        time_var.set(f"{t_flight:.2f} s")
        range_var.set(f"{horizontal_range:.2f} m")
        maxh_var.set(f"{max_height:.2f} m")
        time_apex_var.set(f"{time_to_apex:.2f} s")
        apex_x_var.set(f"{apex_x_coord:.2f} m")
        apex_y_var.set(f"{apex_y_coord:.2f} m")
    except Exception:
        pass

    canvas.draw_idle()

# (Speed plot feature removed)

# ---------- Show equations window ----------
def show_equations():
    top = tk.Toplevel(root)
    top.title("Projectile Motion Equations")
    top.geometry("600x500")
    
    text_widget = tk.Text(top, wrap=tk.WORD, font=('Courier', 10), padx=10, pady=10)
    text_widget.pack(fill=tk.BOTH, expand=True)
    
    equations_text = """
PROJECTILE MOTION EQUATIONS (Without Air Resistance)
═══════════════════════════════════════════════════════

Initial Velocity Components:
  vx₀ = v₀ cos(θ)    [horizontal component - constant]
  vy₀ = v₀ sin(θ)    [vertical component]

Position Functions:
  x(t) = vx₀ · t
  y(t) = h₀ + vy₀ · t - ½gt²

Velocity Functions:
  vx(t) = vx₀        [constant - no horizontal acceleration]
  vy(t) = vy₀ - gt   [linear decrease due to gravity]

Time of Flight (landing at y=0):
  t = [v₀sinθ + √((v₀sinθ)² + 2gh₀)] / g

Range (horizontal distance):
  R = vx₀ · t_flight = (v₀ cos θ) · t_flight

Maximum Height:
  H_max = h₀ + (v₀² sin²θ) / (2g)

Time to Apex:
  t_apex = (v₀ sin θ) / g

Impact Angle:
  θ_impact = arctan(vy_final / vx_final)

Key Properties:
  • Horizontal velocity remains constant
  • Vertical velocity changes linearly
  • Trajectory forms a parabola
  • For level ground (h₀=0): time up = time down
  • Optimal angle for max range: 45° (level ground)

═══════════════════════════════════════════════════════

WITH AIR RESISTANCE:
  • Drag force: F_d = ½ ρ C_d A v²
  • Both components affected
  • No simple analytical solution
  • Requires numerical integration
  • Range and height reduced
  • Optimal angle < 45°
"""
    
    text_widget.insert('1.0', equations_text)
    text_widget.config(state=tk.DISABLED)
    
    ttk.Button(top, text="Close", command=top.destroy).pack(pady=10)

# ---------- Animation playback ----------
def start_animation():
    global anim_running, anim_paused, anim_marker, anim_trail_line, anim_t, pending_update
    global anim_play_t, anim_play_x, anim_play_y, anim_play_vx, anim_play_vy

    if anim_running:
        return
    update_plot()
    if 't' not in last_run or last_run['t'].size == 0:
        return

    anim_play_t = last_run['t']
    anim_play_x = last_run['x']
    anim_play_y = last_run['y']
    anim_play_vx = last_run['vx']
    anim_play_vy = last_run['vy']
    anim_t = 0.0
    anim_running = True
    anim_paused = False

    # Create marker shape based on object type
    d = float(diam_var.get())
    name = object_var.get()
    
    if name in ("Ball","Rock","Golf Ball","Cannonball","Bullet"):
        anim_marker = patches.Circle((anim_play_x[0], anim_play_y[0]), 
                                    radius=max(d/2,0.08), facecolor='gold', 
                                    edgecolor='black', zorder=10)
        ax.add_patch(anim_marker)
    elif name == "Car":
        w = max(d,0.4)
        h = max(d*0.5, 0.2)
        anim_marker = patches.Rectangle((anim_play_x[0]-w/2, anim_play_y[0]-h/2), 
                                       w, h, facecolor='tab:blue', 
                                       edgecolor='black', zorder=10)
        ax.add_patch(anim_marker)
    elif name == "Arrow":
        s = max(d*2.0, 0.3)
        anim_marker = patches.RegularPolygon((0.0, 0.0), numVertices=3, radius=s, 
                                            orientation=math.radians(90.0), 
                                            facecolor='tab:red', edgecolor='black', zorder=10)
        ax.add_patch(anim_marker)
        place_marker_with_orientation(anim_marker, float(anim_play_x[0]), 
                                     float(anim_play_y[0]), float(anim_play_vx[0]), 
                                     float(anim_play_vy[0]))
    else:
        anim_marker = patches.Circle((anim_play_x[0], anim_play_y[0]), 
                                    radius=max(d/2,0.08), facecolor='gold', 
                                    edgecolor='black', zorder=10)
        ax.add_patch(anim_marker)

    # Trail
    trail_x.clear()
    trail_y.clear()
    trail_x.append(float(anim_play_x[0]))
    trail_y.append(float(anim_play_y[0]))
    anim_trail_line, = ax.plot(trail_x, trail_y, color='gold', linewidth=3, 
                               alpha=0.9, zorder=9)

    # Ensure HUD exists
    if not hasattr(ax, 'hud_text'):
        ax.hud_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top', 
                             ha='left', fontsize=9,
                             bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.9))

    try:
        fire_btn.state(['!disabled'])
        fire_text.set("Pause")
    except Exception:
        pass

    disable_redraw_widgets(False)
    step_animation()

def stop_animation():
    global anim_running, pending_update
    anim_running = False
    try:
        fire_btn.state(['!disabled'])
        fire_text.set("Fire")
    except Exception:
        pass
    disable_redraw_widgets(True)
    if pending_update:
        pending_update = False
        update_plot()

def reset_animation(silent=False):
    global anim_marker, anim_trail_line, anim_play_t, anim_play_x, anim_play_y
    global anim_play_vx, anim_play_vy, anim_t, anim_running, anim_paused
    
    stop_animation()
    anim_paused = False
    
    if anim_marker is not None:
        try:
            anim_marker.remove()
        except Exception:
            pass
    if anim_trail_line is not None:
        try:
            anim_trail_line.remove()
        except Exception:
            pass
    
    anim_marker = None
    anim_trail_line = None
    anim_play_t = anim_play_x = anim_play_y = anim_play_vx = anim_play_vy = None
    anim_t = 0.0
    
    if not silent:
        update_plot()

def step_animation():
    global anim_running, anim_t
    
    if not anim_running:
        return
    if anim_paused:
        root.after(int(1000*anim_dt), step_animation)
        return

    t_arr = anim_play_t
    x_arr = anim_play_x
    y_arr = anim_play_y
    vx_arr = anim_play_vx
    vy_arr = anim_play_vy
    
    if t_arr is None or t_arr.size == 0:
        stop_animation()
        return

    total = float(t_arr[-1])
    if total <= 0:
        stop_animation()
        return

    anim_t = max(0.0, min(anim_t, total))
    frac = anim_t / total
    idxf = frac * (len(t_arr)-1)
    idx = int(round(idxf))
    idx = max(0, min(idx, len(t_arr)-1))

    x = float(x_arr[idx])
    y = float(y_arr[idx])
    vx = float(vx_arr[idx])
    vy = float(vy_arr[idx])
    
    # Update marker position
    try:
        if isinstance(anim_marker, patches.Circle):
            anim_marker.center = (x, y)
        elif isinstance(anim_marker, patches.Rectangle):
            w = anim_marker.get_width()
            h = anim_marker.get_height()
            anim_marker.set_xy((x - w/2, y - h/2))
        else:
            place_marker_with_orientation(anim_marker, x, y, vx, vy)
    except Exception:
        pass

    # Update trail
    if anim_trail_line is not None:
        if len(trail_x) == 0 or (x != trail_x[-1] or y != trail_y[-1]):
            trail_x.append(x)
            trail_y.append(y)
            anim_trail_line.set_data(trail_x, trail_y)

    # Update HUD
    speed = math.hypot(vx, vy)
    if hasattr(ax, 'hud_text'):
        ax.hud_text.set_text(
            f"t = {t_arr[idx]:.3f} s\n"
            f"x = {x:.3f} m\n"
            f"speed = {speed:.3f} m/s"
        )

    canvas.draw_idle()
    anim_t += anim_dt * float(anim_speed_var.get())
    # update scrubber position
    try:
        anim_time_var.set(max(0.0, min(1.0, frac)))
    except Exception:
        pass

    # Stop when finished
    if idx >= len(t_arr)-1:
        stop_animation()
        return

    root.after(int(1000*anim_dt), step_animation)

def place_marker_with_orientation(marker, x, y, vx, vy):
    if marker is None:
        return
    angle = math.degrees(math.atan2(vy, vx)) if (vx != 0 or vy != 0) else 0.0
    base = mtransforms.Affine2D().rotate_deg(angle).translate(x, y)
    marker.set_transform(base + ax.transData)

# ---------- UI Helper functions ----------
def _upd_speed(val=None):
    speed_val.config(text=f"{velocity.get():.1f}")
    safe_request_update()

def _upd_elev(val=None):
    elev_val.config(text=f"{height.get():.1f}")
    safe_request_update()

def _upd_angle(val=None):
    ang_val.config(text=f"{angle_deg.get():.0f}")
    safe_request_update()

def _upd_mass(val=None):
    mass_val.config(text=f"{mass_var.get():.2f}")
    safe_request_update()

def _upd_diam(val=None):
    diam_val.config(text=f"{diam_var.get():.3f}")
    safe_request_update()

def _upd_cd(val=None):
    cd_val.config(text=f"{cd_var.get():.2f}")
    safe_request_update()

def apply_object_defaults(event=None):
    name = object_var.get()
    if name == "Ball":
        cd_var.set(0.47)
        mass_var.set(0.45)
        diam_var.set(0.22)
    elif name == "Arrow":
        cd_var.set(0.30)
        mass_var.set(0.035)
        diam_var.set(0.02)
    elif name == "Car":
        cd_var.set(0.30)
        mass_var.set(2000)
        diam_var.set(2.0)
    elif name == "Rock":
        cd_var.set(0.50)
        mass_var.set(1.0)
        diam_var.set(0.18)
    elif name == "Golf Ball":
        cd_var.set(0.25)
        mass_var.set(0.045)
        diam_var.set(0.043)
    elif name == "Cannonball":
        cd_var.set(0.47)
        mass_var.set(5.4)
        diam_var.set(0.15)
    elif name == "Bullet":
        cd_var.set(0.295)
        mass_var.set(0.012)
        diam_var.set(0.009)
    _upd_mass()
    _upd_diam()
    _upd_cd()
    safe_request_update()

# ---------- UI Building ----------
controls = ttk.Frame(root)
controls.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)

# Speed control
ttk.Label(controls, text="Initial Speed (m/s)").pack(anchor='w')
speed_row = ttk.Frame(controls)
speed_row.pack(fill=tk.X, pady=(0,6))
speed_val = ttk.Label(speed_row, width=8, anchor='e', text=f"{velocity.get():.1f}")
speed_val.pack(side=tk.RIGHT)

speed_scale = ttk.Scale(controls, variable=velocity, from_=SPEED_MIN, to=SPEED_MAX, 
                        orient=tk.HORIZONTAL, command=lambda v: _upd_speed(v))
speed_scale.pack(fill=tk.X)

speed_spin = ttk.Spinbox(controls, from_=SPEED_MIN, to=SPEED_MAX, increment=1.0, width=8)
speed_spin.delete(0,'end')
speed_spin.insert(0,f"{velocity.get():.1f}")

def _speed_spin_commit(event=None):
    try:
        val=float(speed_spin.get())
    except:
        return
    val = max(SPEED_MIN, min(SPEED_MAX, val))
    velocity.set(val)
    speed_spin.delete(0,'end')
    speed_spin.insert(0,f"{val:.1f}")
    _upd_speed()

speed_spin.bind('<Return>', _speed_spin_commit)
speed_spin.bind('<FocusOut>', _speed_spin_commit)
speed_spin.pack(anchor='e', padx=2, pady=(2,6))

redraw_widgets.extend([speed_scale, speed_spin])

# Elevation control
elev_cb = ttk.Checkbutton(controls, text="Use Elevation?", variable=elevation_var, 
                         command=lambda: safe_request_update())
elev_cb.pack(anchor='w')

ttk.Label(controls, text="Cannon Elevation (m)").pack(anchor='w')
elev_row = ttk.Frame(controls)
elev_row.pack(fill=tk.X, pady=(0,6))
elev_val = ttk.Label(elev_row, width=8, anchor='e', text=f"{height.get():.1f}")
elev_val.pack(side=tk.RIGHT)

height_scale = ttk.Scale(controls, variable=height, from_=HEIGHT_MIN, to=HEIGHT_MAX, 
                        orient=tk.HORIZONTAL, command=lambda v: _upd_elev(v))
height_scale.pack(fill=tk.X)

elev_spin = ttk.Spinbox(controls, from_=HEIGHT_MIN, to=HEIGHT_MAX, increment=0.5, width=8)
elev_spin.delete(0,'end')
elev_spin.insert(0,f"{height.get():.1f}")

def _elev_spin_commit(event=None):
    try:
        v=float(elev_spin.get())
    except:
        return
    v=max(HEIGHT_MIN, min(HEIGHT_MAX, v))
    height.set(v)
    elev_spin.delete(0,'end')
    elev_spin.insert(0,f"{v:.1f}")
    _upd_elev()

elev_spin.bind('<Return>', _elev_spin_commit)
elev_spin.bind('<FocusOut>', _elev_spin_commit)
elev_spin.pack(anchor='e', padx=2, pady=(2,6))

redraw_widgets.extend([elev_cb, height_scale, elev_spin])

# Angle control
angle_cb = ttk.Checkbutton(controls, text="Use Angle?", variable=angle_var, 
                          command=lambda: safe_request_update())
angle_cb.pack(anchor='w')

ttk.Label(controls, text="Angle (deg)").pack(anchor='w')
ang_row = ttk.Frame(controls)
ang_row.pack(fill=tk.X, pady=(0,6))
ang_val = ttk.Label(ang_row, width=8, anchor='e', text=f"{angle_deg.get():.0f}")
ang_val.pack(side=tk.RIGHT)

angle_scale = ttk.Scale(controls, variable=angle_deg, from_=0, to=90, 
                       orient=tk.HORIZONTAL, command=lambda v: _upd_angle(v))
angle_scale.pack(fill=tk.X)

ang_spin = ttk.Spinbox(controls, from_=0, to=90, increment=1, width=8)
ang_spin.delete(0,'end')
ang_spin.insert(0,f"{angle_deg.get():.0f}")

def _ang_spin_commit(event=None):
    try:
        v=float(ang_spin.get())
    except:
        return
    v=max(0.0, min(90.0, v))
    angle_deg.set(v)
    ang_spin.delete(0,'end')
    ang_spin.insert(0,f"{v:.0f}")
    _upd_angle()

ang_spin.bind('<Return>', _ang_spin_commit)
ang_spin.bind('<FocusOut>', _ang_spin_commit)
ang_spin.pack(anchor='e', padx=2, pady=(2,10))

redraw_widgets.extend([angle_cb, angle_scale, ang_spin])

# Object & Air Resistance frame
obj_frame = ttk.LabelFrame(controls, text="Object & Air")
obj_frame.pack(fill=tk.X, padx=2, pady=6)

ttk.Label(obj_frame, text="Object:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
object_menu = ttk.Combobox(obj_frame, textvariable=object_var, 
                          values=["Ball","Arrow","Car","Rock","Golf Ball","Cannonball","Bullet"], 
                          state="readonly")
object_menu.grid(row=0, column=1, sticky='ew', padx=5, pady=2)
object_menu.bind('<<ComboboxSelected>>', apply_object_defaults)

air_cb = ttk.Checkbutton(obj_frame, text="Air Resistance", variable=air_var, 
                        command=lambda: safe_request_update())
air_cb.grid(row=1, column=0, sticky='w', padx=5, pady=2)

components_cb = ttk.Checkbutton(obj_frame, text="Show Velocity Vectors", 
                               variable=components_var, 
                               command=lambda: safe_request_update())
components_cb.grid(row=1, column=1, sticky='e', padx=5, pady=2)

compare_cb = ttk.Checkbutton(obj_frame, text="Compare w/o Drag", 
                            variable=compare_var, 
                            command=lambda: safe_request_update())
compare_cb.grid(row=2, column=0, columnspan=2, sticky='w', padx=5, pady=2)


ttk.Label(obj_frame, text="Mass (kg)").grid(row=3, column=0, sticky='w', padx=5, pady=2)
mass_val = ttk.Label(obj_frame, width=8, anchor='e', text=f"{mass_var.get():.2f}")
mass_val.grid(row=3, column=2, sticky='e', padx=5)
mass_scale = ttk.Scale(obj_frame, variable=mass_var, from_=0.01, to=5000, 
                      orient=tk.HORIZONTAL, command=lambda v: _upd_mass())
mass_scale.grid(row=3, column=1, sticky='ew', padx=5, pady=2)

ttk.Label(obj_frame, text="Diameter (m)").grid(row=4, column=0, sticky='w', padx=5, pady=2)
diam_val = ttk.Label(obj_frame, width=8, anchor='e', text=f"{diam_var.get():.3f}")
diam_val.grid(row=4, column=2, sticky='e', padx=5)
diam_scale = ttk.Scale(obj_frame, variable=diam_var, from_=0.005, to=3.0, 
                      orient=tk.HORIZONTAL, command=lambda v: _upd_diam())
diam_scale.grid(row=4, column=1, sticky='ew', padx=5, pady=2)

ttk.Label(obj_frame, text="Drag Coeff Cd").grid(row=5, column=0, sticky='w', padx=5, pady=2)
cd_val = ttk.Label(obj_frame, width=8, anchor='e', text=f"{cd_var.get():.2f}")
cd_val.grid(row=5, column=2, sticky='e', padx=5)
cd_scale = ttk.Scale(obj_frame, variable=cd_var, from_=0.05, to=1.5, 
                    orient=tk.HORIZONTAL, command=lambda v: _upd_cd())
cd_scale.grid(row=5, column=1, sticky='ew', padx=5, pady=2)

obj_frame.grid_columnconfigure(1, weight=1)

redraw_widgets.extend([object_menu, air_cb, components_cb, compare_cb, 
                      mass_scale, diam_scale, cd_scale])

# Action buttons
btns = ttk.Frame(controls)
btns.pack(fill=tk.X, pady=(6,4))

fire_text = tk.StringVar(value="Fire")

def fire_pause_toggle(event=None):
    global anim_running, anim_paused
    if not anim_running:
        start_animation()
    else:
        anim_paused = not anim_paused
        if anim_paused:
            fire_text.set("Resume")
        else:
            fire_text.set("Pause")
            step_animation()

fire_btn = ttk.Button(btns, textvariable=fire_text, command=fire_pause_toggle)
fire_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

ttk.Button(btns, text="Reset", command=reset_animation).pack(side=tk.LEFT, expand=True, 
                                                             fill=tk.X, padx=2)

 

# Playback speed dropdown
spd_row = ttk.Frame(controls)
spd_row.pack(fill=tk.X, pady=(4,4))
ttk.Label(spd_row, text="Playback Speed").pack(side=tk.LEFT)
speed_combo = ttk.Combobox(spd_row, state="readonly",
                          values=["0.25×","0.5×","1×","1.5×","2×","3×"],
                          width=7)
speed_combo.pack(side=tk.RIGHT)
speed_combo.set("1×")

def _on_speed_change(event=None):
    m = {"0.25×":0.25, "0.5×":0.5, "1×":1.0, "1.5×":1.5, "2×":2.0, "3×":3.0}
    anim_speed_var.set(m.get(speed_combo.get(), 1.0))

speed_combo.bind('<<ComboboxSelected>>', _on_speed_change)

# (Speed Plot button removed)

eq_btn = ttk.Frame(controls)
eq_btn.pack(fill=tk.X, pady=(4,8))
ttk.Button(eq_btn, text="Show Equations", command=show_equations).pack(fill=tk.X, padx=2)

# Results panel
results = ttk.LabelFrame(controls, text="Projectile Metrics")
results.pack(fill=tk.X, padx=2, pady=(4,8))

# Time scrubber
scrub_frame = ttk.Frame(controls)
scrub_frame.pack(fill=tk.X, pady=(2,6))
ttk.Label(scrub_frame, text="Scrub Time").pack(anchor='w')
anim_time_var = tk.DoubleVar(value=0.0)
time_scale = ttk.Scale(scrub_frame, variable=anim_time_var, from_=0.0, to=1.0,
                       orient=tk.HORIZONTAL)
time_scale.pack(fill=tk.X)

def _on_scrub_release(event=None):
    # Jump animation time proportionally and render one frame
    if 't' not in last_run or last_run.get('t') is None or last_run['t'].size == 0:
        return
    total = float(last_run['t'][-1])
    frac = max(0.0, min(1.0, float(anim_time_var.get())))
    # set anim_t and draw one frame
    global anim_t
    anim_t = frac * total
    # If animation is not running, ensure data is set
    if not anim_running:
        # prime play arrays
        global anim_play_t, anim_play_x, anim_play_y, anim_play_vx, anim_play_vy
        anim_play_t = last_run['t']; anim_play_x = last_run['x']; anim_play_y = last_run['y']
        anim_play_vx = last_run['vx']; anim_play_vy = last_run['vy']
        # create marker if missing
        if anim_marker is None:
            try:
                d = float(diam_var.get())
                m = patches.Circle((0,0), radius=max(d/2,0.08), facecolor='gold', edgecolor='black', zorder=10)
                ax.add_patch(m)
                globals()['anim_marker'] = m
            except Exception:
                pass
    # Render one immediate step without advancing
    _render_at_current_anim_t()

def _render_at_current_anim_t():
    try:
        t_arr = anim_play_t; x_arr = anim_play_x; y_arr = anim_play_y; vx_arr = anim_play_vx; vy_arr = anim_play_vy
        if t_arr is None or len(t_arr) == 0:
            return
        total = float(t_arr[-1])
        anim_t_local = max(0.0, min(float(globals()['anim_t']), total))
        frac = anim_t_local / total
        idx = int(round(frac * (len(t_arr)-1)))
        idx = max(0, min(idx, len(t_arr)-1))
        x = float(x_arr[idx]); y = float(y_arr[idx]); vx = float(vx_arr[idx]); vy = float(vy_arr[idx])
        if isinstance(anim_marker, patches.Circle):
            anim_marker.center = (x, y)
        elif isinstance(anim_marker, patches.Rectangle):
            w = anim_marker.get_width(); h = anim_marker.get_height(); anim_marker.set_xy((x - w/2, y - h/2))
        else:
            place_marker_with_orientation(anim_marker, x, y, vx, vy)
        if hasattr(ax, 'hud_text'):
            ax.hud_text.set_text(f"t = {t_arr[idx]:.3f} s\n" f"x = {x:.3f} m\n" f"speed = {math.hypot(vx,vy):.3f} m/s")
        canvas.draw_idle()
    except Exception:
        pass

time_scale.bind('<ButtonRelease-1>', _on_scrub_release)

ttk.Label(results, text="Time of Flight:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
time_var = tk.StringVar(value="0.00 s")
ttk.Label(results, textvariable=time_var).grid(row=0, column=1, sticky='e', padx=5, pady=2)

ttk.Label(results, text="Range:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
range_var = tk.StringVar(value="0.00 m")
ttk.Label(results, textvariable=range_var).grid(row=1, column=1, sticky='e', padx=5, pady=2)

ttk.Label(results, text="Max Height:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
maxh_var = tk.StringVar(value="0.00 m")
ttk.Label(results, textvariable=maxh_var).grid(row=2, column=1, sticky='e', padx=5, pady=2)

ttk.Separator(results, orient='horizontal').grid(row=3, columnspan=2, sticky='ew', pady=4)

ttk.Label(results, text="Time to Apex:").grid(row=4, column=0, sticky='w', padx=5, pady=2)
time_apex_var = tk.StringVar(value="0.00 s")
ttk.Label(results, textvariable=time_apex_var).grid(row=4, column=1, sticky='e', padx=5, pady=2)

ttk.Label(results, text="Apex Position:").grid(row=5, column=0, sticky='w', padx=5, pady=2)
apex_x_var = tk.StringVar(value="0.00 m")
apex_y_var = tk.StringVar(value="0.00 m")
apex_label = ttk.Label(results, text="")
apex_label.grid(row=5, column=1, sticky='e', padx=5, pady=2)

def update_apex_display(*args):
    apex_label.config(text=f"({apex_x_var.get()}, {apex_y_var.get()})")

apex_x_var.trace('w', update_apex_display)
apex_y_var.trace('w', update_apex_display)

results.grid_columnconfigure(1, weight=1)

 # (Solver UI removed per request)

# ---------- Keyboard shortcuts ----------
def _on_key(event):
    if event.keysym == "space":
        fire_pause_toggle()
    elif event.keysym.lower() == "r":
        reset_animation()
    elif event.keysym.lower() == "d":
        air_var.set(not air_var.get())
        safe_request_update()
    elif event.keysym.lower() == "v":
        components_var.set(not components_var.get())
        safe_request_update()
    elif event.keysym.lower() == "c":
        compare_var.set(not compare_var.get())
        safe_request_update()
    elif event.keysym.lower() == "e":
        show_equations()

root.bind("<Key>", _on_key)

# (click-to-target handler removed)

def on_close():
    try:
        stop_animation()
    except Exception:
        pass
    try:
        if hover_cid is not None:
            canvas.mpl_disconnect(hover_cid)
    except Exception:
        pass
    try:
        plt.close('all')
    except Exception:
        pass
    try:
        root.destroy()
    except Exception:
        pass
    try:
        sys.exit(0)
    except SystemExit:
        pass

root.protocol("WM_DELETE_WINDOW", on_close)

# ---------- Initialization ----------
_create_hover_artists_and_reconnect()
apply_object_defaults()
update_plot()
disable_redraw_widgets(True)

root.mainloop()