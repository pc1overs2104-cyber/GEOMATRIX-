"""GeoMatrix Explorer - Transformasi Geometri dengan Matriks (Streamlit + NumPy + Plotly)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="GeoMatrix Explorer", page_icon="📐", layout="wide")

BLUE, AMBER, GREEN = "#2563EB", "#F59E0B", "#10B981"
I2 = np.eye(2)
SHAPES = {
    "Persegi satuan": [(0, 0), (1, 0), (1, 1), (0, 1)],
    "Segitiga": [(0, 0), (2, 0), (1, 2)],
    "Huruf F (tak simetris)": [(0, 0), (1, 0), (1, 2), (2, 2), (2, 3), (1, 3), (1, 4), (3, 4), (3, 5), (0, 5)],
}
KINDS = ["Identitas", "Rotasi", "Refleksi sumbu X", "Refleksi sumbu Y", "Refleksi y = x",
         "Refleksi y = -x", "Dilatasi", "Gusuran horizontal", "Gusuran vertikal",
         "Translasi", "Kustom (a, b, c, d)"]
CHALLENGES = {
    "Tanpa tantangan": None,
    "1. Cerminkan terhadap sumbu Y": (np.array([[-1, 0], [0, 1]], float), np.zeros(2), "Lihat kolom 1: ke mana vektor (1,0) pergi?"),
    "2. Putar 90° lalu perbesar 2×": (np.array([[0, -2], [2, 0]], float), np.zeros(2), "Boleh memakai 2 langkah, atau 1 matriks kustom."),
    "3. Gusuran horizontal dengan k = 2": (np.array([[1, 2], [0, 1]], float), np.zeros(2), "Elemen mana yang membuat bangun miring?"),
    "4. Refleksi y = x lalu geser (3, -1)": (np.array([[0, 1], [1, 0]], float), np.array([3, -1.0]), "Butuh 2 langkah. Urutan menentukan hasil!"),
}


def rot(deg):
    r = np.radians(deg)
    return np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])


def bm(M):
    f = lambda v: f"{round(float(v), 3):g}"
    return r"\begin{bmatrix}" + r"\\".join("&".join(f(v) for v in row) for row in M) + r"\end{bmatrix}"


def area(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def make_step(i):
    kind = st.selectbox(f"Langkah {i + 1}", KINDS, key=f"kind{i}")
    A, t, ang, k = I2.copy(), np.zeros(2), 0.0, 1.0
    if kind == "Rotasi":
        ang = st.slider("Sudut θ (°, + = berlawanan arah jarum jam)", -360, 360, 90, 15, key=f"ang{i}")
        A = rot(ang)
    elif kind == "Refleksi sumbu X":
        A = np.array([[1, 0], [0, -1]], float)
    elif kind == "Refleksi sumbu Y":
        A = np.array([[-1, 0], [0, 1]], float)
    elif kind == "Refleksi y = x":
        A = np.array([[0, 1], [1, 0]], float)
    elif kind == "Refleksi y = -x":
        A = np.array([[0, -1], [-1, 0]], float)
    elif kind == "Dilatasi":
        k = st.number_input("Faktor skala k", value=2.0, step=0.5, key=f"dk{i}")
        A = k * I2
    elif kind == "Gusuran horizontal":
        k = st.slider("k", -3.0, 3.0, 1.0, 0.5, key=f"sh{i}")
        A = np.array([[1, k], [0, 1]], float)
    elif kind == "Gusuran vertikal":
        k = st.slider("k", -3.0, 3.0, 1.0, 0.5, key=f"sv{i}")
        A = np.array([[1, 0], [k, 1]], float)
    elif kind == "Translasi":
        c1, c2 = st.columns(2)
        t = np.array([c1.number_input("tx", value=2.0, step=1.0, key=f"tx{i}"),
                      c2.number_input("ty", value=1.0, step=1.0, key=f"ty{i}")])
    elif kind.startswith("Kustom"):
        c = st.columns(4)
        v = [c[j].number_input(n, value=float(d), step=0.5, key=f"{n}{i}") for j, (n, d) in enumerate(zip("abcd", (1, 0, 0, 1)))]
        A = np.array(v).reshape(2, 2)
    st.latex(bm(A) + (r"+" + bm(t.reshape(2, 1)) if t.any() else ""))
    return dict(kind=kind, A=A, t=t, ang=ang, k=k)


def interp(s, S):
    """Matriks antara (0<=s<=1) agar lintasan realistis: rotasi melingkar, translasi lurus."""
    if S["kind"] == "Rotasi":
        return rot(S["ang"] * s), np.zeros(2)
    if S["kind"] == "Dilatasi":
        return (1 + (S["k"] - 1) * s) * I2, np.zeros(2)
    return (1 - s) * I2 + s * S["A"], s * S["t"]


def build_frames(P, steps, n):
    frames, cur = [(P.copy(), "Awal")], P.copy()
    for j, S in enumerate(steps):
        for s in np.linspace(0, 1, n + 1)[1:]:
            As, ts = interp(s, S)
            frames.append((cur @ As.T + ts, f"Langkah {j + 1}"))
        cur = frames[-1][0]
    return frames


def closed(P):
    return np.vstack([P, P[:1]])


# ------------------------------------------------------------------ UI
st.title("📐 GeoMatrix Explorer")
st.caption("Jembatan antara aljabar (matriks) dan geometri (transformasi). Ubah angka, lihat bentuknya berubah.")

left, right = st.columns([1, 2.2], gap="large")

with left:
    st.subheader("1. Bangun datar")
    shape = st.selectbox("Pilih bangun", list(SHAPES) + ["Buat sendiri"])
    base = SHAPES.get(shape, SHAPES["Segitiga"])
    df = st.data_editor(pd.DataFrame(base, columns=["x", "y"]), num_rows="dynamic",
                        hide_index=True, key=f"pts_{shape}", width="stretch")
    P = df.dropna().to_numpy(float)

    st.subheader("2. Rantai transformasi")
    n_steps = st.number_input("Jumlah langkah", 1, 4, 1)
    steps = []
    for i in range(int(n_steps)):
        with st.container(border=True):
            steps.append(make_step(i))

    st.subheader("3. Tampilan")
    speed = st.select_slider("Kecepatan animasi", ["Lambat", "Sedang", "Cepat"], "Sedang")
    show_trail = st.toggle("Jejak (ghosting)", True)
    show_mid = st.toggle("Hasil tiap langkah", True)
    show_basis = st.toggle("Vektor basis (î, ĵ)", False)
    challenge = st.selectbox("Mode tantangan", list(CHALLENGES))

if len(P) < 3:
    right.warning("Minimal 3 titik untuk membentuk bangun datar.")
    st.stop()

H = np.eye(3)
for S in steps:
    Hs = np.eye(3); Hs[:2, :2] = S["A"]; Hs[:2, 2] = S["t"]
    H = Hs @ H
M, T = H[:2, :2], H[:2, 2]
frames = build_frames(P, steps, 18)
final = P @ M.T + T
dur = {"Lambat": 120, "Sedang": 60, "Cepat": 25}[speed]

target = CHALLENGES[challenge]
Ptar = P @ target[0].T + target[1] if target else None

fig = go.Figure()
fig.add_trace(go.Scatter(x=closed(P)[:, 0], y=closed(P)[:, 1], fill="toself", mode="lines+markers",
                         line=dict(color=AMBER, width=3), fillcolor="rgba(245,158,11,0.35)", name="Bayangan"))
fig.add_trace(go.Scatter(x=closed(P)[:, 0], y=closed(P)[:, 1], fill="toself", mode="lines+markers",
                         line=dict(color=BLUE, width=3), fillcolor="rgba(37,99,235,0.45)", name="Bangun asli"))
if show_mid and len(steps) > 1:
    for j in range(1, len(steps)):
        Q = frames[j * 18][0]
        fig.add_trace(go.Scatter(x=closed(Q)[:, 0], y=closed(Q)[:, 1], mode="lines",
                                 line=dict(color="gray", width=1.5, dash="dash"), name=f"Setelah langkah {j}"))
if show_trail:
    allQ = np.array([f[0] for f in frames])
    for v in range(len(P)):
        fig.add_trace(go.Scatter(x=allQ[:, v, 0], y=allQ[:, v, 1], mode="lines", hoverinfo="skip",
                                 line=dict(color="rgba(120,120,120,0.45)", width=1, dash="dot"), showlegend=False))
if Ptar is not None:
    fig.add_trace(go.Scatter(x=closed(Ptar)[:, 0], y=closed(Ptar)[:, 1], mode="lines",
                             line=dict(color=GREEN, width=3, dash="dash"), name="Target"))
if show_basis:
    for col, c, nm in ((0, "#DC2626", "î → kolom 1"), (1, "#7C3AED", "ĵ → kolom 2")):
        fig.add_trace(go.Scatter(x=[0, M[0, col]], y=[0, M[1, col]], mode="lines+markers+text", text=["", nm],
                                 textposition="top right", line=dict(color=c, width=4), name=nm))

pts = np.vstack([P, final, *(f[0] for f in frames[::18]), np.zeros((1, 2))] + ([Ptar] if Ptar is not None else []))
lo, hi = pts.min(0) - 1, pts.max(0) + 1
span = max(hi - lo)
mid = (lo + hi) / 2
rng = [[mid[0] - span / 2, mid[0] + span / 2], [mid[1] - span / 2, mid[1] + span / 2]]

fig.frames = [go.Frame(data=[go.Scatter(x=closed(Q)[:, 0], y=closed(Q)[:, 1])], traces=[0], name=str(i))
              for i, (Q, _) in enumerate(frames)]
fig.update_layout(
    height=640, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
    xaxis=dict(range=rng[0], gridcolor="#E5E7EB", zerolinecolor="black", zerolinewidth=2, dtick=1),
    yaxis=dict(range=rng[1], gridcolor="#E5E7EB", zerolinecolor="black", zerolinewidth=2, dtick=1, scaleanchor="x"),
    legend=dict(orientation="h", y=1.04),
    updatemenus=[dict(type="buttons", x=0.0, y=-0.02, direction="left", buttons=[
        dict(label="▶ Putar", method="animate", args=[None, dict(frame=dict(duration=dur, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
        dict(label="⏸ Jeda", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
    sliders=[dict(active=0, pad=dict(t=40), currentvalue=dict(prefix="Frame: "), steps=[
        dict(method="animate", label=str(i), args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))])
        for i in range(len(frames))])])

with right:
    tab1, tab2, tab3 = st.tabs(["Visualisasi", "Panduan Eksperimen (LKPD)", "Rangkuman Konsep"])
    with tab1:
        st.plotly_chart(fig, width="stretch")
        st.markdown("**Matriks gabungan** (urutan: langkah 1 dilakukan lebih dulu, sehingga $M = M_n \\cdots M_2 M_1$)")
        st.latex(r"M = " + bm(M) + (r"\;,\; \vec{t} = " + bm(T.reshape(2, 1)) if T.any() else ""))
        det = np.linalg.det(M)
        a0, a1 = area(P), area(final)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Determinan", f"{det:.3g}")
        c2.metric("Luas asli", f"{a0:.3g}")
        c3.metric("Luas bayangan", f"{a1:.3g}")
        c4.metric("|det| × luas asli", f"{abs(det) * a0:.3g}")
        if abs(det) < 1e-9:
            st.error("det = 0: bangun 'runtuh' menjadi garis/titik. Matriks ini tidak punya invers.")
        elif det < 0:
            st.info("det < 0: bangun terbalik (orientasi berubah, seperti dicerminkan). Luas berubah |det| kali.")
        else:
            st.success(f"det > 0: orientasi tetap. Luas berubah {det:.3g} kali.")
        if target:
            if np.allclose(final, Ptar, atol=1e-6):
                st.balloons(); st.success("🎉 Tepat! Bayanganmu menimpa target.")
            else:
                st.warning("Belum sama dengan target (garis putus hijau).")
            st.caption("Petunjuk: " + target[2])
        st.download_button("⬇ Unduh visualisasi (HTML)", fig.to_html(include_plotlyjs="cdn"), "geomatrix.html", "text/html")
    with tab2:
        st.markdown("""
**Kegiatan 1 – Prediksi.** Pilih *Persegi satuan*, Langkah 1 = *Kustom*. Tulis dulu prediksimu, lalu coba matriks:
`[1 0; 0 1]`, `[2 0; 0 1]`, `[1 0; 0 3]`, `[0 1; 1 0]`. Apa yang terjadi pada bentuk dan luasnya?

**Kegiatan 2 – Siapa yang menggusur?** Mulai dari `[1 2; 0 1]`. Ubah angka 2 menjadi 3, lalu 0, lalu -2. Elemen mana yang mengatur kemiringan? Nyalakan *Vektor basis*: bagaimana kolom matriks berhubungan dengan panah merah dan ungu?

**Kegiatan 3 – Urutan penting?** Gunakan *Huruf F*. Buat 2 langkah: Rotasi 90° lalu Refleksi sumbu X. Tukar urutannya. Samakah hasilnya? Bandingkan matriks gabungannya.

**Kegiatan 4 – Determinan.** Isi matriks sehingga luas bayangan = 6 kali luas asli. Lalu cari matriks yang membuat determinan = 0. Apa yang terjadi pada bangunnya?

**Kegiatan 5 – Tantangan.** Pilih *Mode tantangan* dan cocokkan bayanganmu dengan target hijau.

**Refleksi:** Tuliskan dengan kata-katamu sendiri: apa arti kolom pertama dan kolom kedua sebuah matriks transformasi?
""")
    with tab3:
        st.latex(r"\begin{bmatrix}x'\\y'\end{bmatrix}=\begin{bmatrix}a&b\\c&d\end{bmatrix}\begin{bmatrix}x\\y\end{bmatrix}")
        st.markdown("""
- **Kolom 1** = bayangan titik (1,0); **kolom 2** = bayangan titik (0,1).
- Rotasi θ: $\\begin{bmatrix}\\cos\\theta&-\\sin\\theta\\\\ \\sin\\theta&\\cos\\theta\\end{bmatrix}$ · Dilatasi k: $kI$ · Refleksi sumbu X: $\\begin{bmatrix}1&0\\\\0&-1\\end{bmatrix}$
- Gusuran horizontal: $\\begin{bmatrix}1&k\\\\0&1\\end{bmatrix}$ · Translasi bukan transformasi linear: $\\vec{p}' = A\\vec{p}+\\vec{t}$
- Komposisi: langkah 1 lalu langkah 2 → $M_2 M_1$ (urutan terbalik saat ditulis).
- $\\text{Luas bayangan} = |\\det M| \\times \\text{luas asli}$; $\\det<0$ → orientasi terbalik; $\\det=0$ → runtuh.
""")
