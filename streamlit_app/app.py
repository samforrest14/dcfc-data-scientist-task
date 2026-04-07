import joblib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import streamlit as st


st.set_page_config(
    page_title="Leverkusen Shot Creation Explorer",
    layout="centered"
)

st.title("Leverkusen Shot Creation Explorer")
st.subheader("Created by Sam Forrest")
st.caption(
    "A lightweight tool to explore how pass destination, defensive context, and delivery style "
    "affect the modelled likelihood that a final-third pass leads directly to a shot."
)

st.markdown(
    """
    The output should be interpreted as **modelled shot-creation likelihood**, not as a
    precise event-level prediction.
    """
)

st.markdown(
    """
    Use the panel on the left to adjust the scenario inputs to visualise how that impacts the shot-creation likelihood.
    """
)

# Load model artefacts
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")
    features = [
        "pass_length",
        "pass_length_sq",
        "pass_angle",
        "into_box",
        "central_not_in_box",
        "is_high_pass",
        "is_cross",
        "under_pressure",
        "nearest_defender_distance",
        "defenders_within_5m",
    ]
    return model, scaler, features


model, scaler, features = load_artifacts()

# Sidebar inputs
st.sidebar.header("Pass Scenario Inputs")

st.sidebar.markdown("### Destination")
zone = st.sidebar.radio(
    "Pass destination zone",
    options=["Into box", "Central not in box", "Wide / other"]
)

st.sidebar.markdown("### Pass Execution")
pass_length = st.sidebar.slider(
    "Pass length",
    min_value=1.0,
    max_value=40.0,
    value=12.0,
    step=0.5
)

pass_angle = st.sidebar.slider(
    "Pass angle",
    min_value=-3.14,
    max_value=3.14,
    value=0.0,
    step=0.05
)

is_high_pass = st.sidebar.checkbox("High pass", value=False)
is_cross = st.sidebar.checkbox("Cross", value=False)

st.sidebar.markdown("### Defensive Context")
under_pressure = st.sidebar.checkbox("Passer under pressure", value=False)

nearest_defender_distance = st.sidebar.slider(
    "Nearest defender distance",
    min_value=0.0,
    max_value=20.0,
    value=6.0,
    step=0.1
)

defenders_within_5m = st.sidebar.slider(
    "Defenders within ~5 units of receiver",
    min_value=0,
    max_value=6,
    value=1,
    step=1
)

# Derive model features
into_box = 1 if zone == "Into box" else 0
central_not_in_box = 1 if zone == "Central not in box" else 0

input_dict = {
    "pass_length": pass_length,
    "pass_length_sq": pass_length ** 2,
    "pass_angle": pass_angle,
    "into_box": into_box,
    "central_not_in_box": central_not_in_box,
    "under_pressure": int(under_pressure),
    "nearest_defender_distance": nearest_defender_distance,
    "defenders_within_5m": defenders_within_5m,
    "is_high_pass": int(is_high_pass),
    "is_cross": int(is_cross),
}

X_input = pd.DataFrame([input_dict])

if hasattr(scaler, "feature_names_in_"):
    X_input = X_input[list(scaler.feature_names_in_)]
else:
    X_input = X_input[features]

X_scaled = scaler.transform(X_input)
shot_prob = model.predict_proba(X_scaled)[0][1]

# Categorise likelihood
if shot_prob >= 0.15:
    status = "High-likelihood scenario"
    colour = "#2ecc71"
elif shot_prob >= 0.07:
    status = "Moderate-likelihood scenario"
    colour = "#f39c12"
else:
    status = "Low-likelihood scenario"
    colour = "#e74c3c"

# Pitch visual
def draw_pitch_zone(selected_zone: str):
    fig, ax = plt.subplots(figsize=(3.8, 5.2))

    pitch_length = 120
    pitch_width = 80

    # Pitch outline
    ax.add_patch(Rectangle((0, 0), pitch_width, pitch_length,
                           fill=False, edgecolor="black", linewidth=2))

    # Halfway line
    ax.plot([0, 80], [60, 60], color="black", linewidth=1)

    # Penalty box
    ax.add_patch(Rectangle((18, 102), 44, 18,
                           fill=False, edgecolor="black", linewidth=1.5))

    # Six-yard box
    ax.add_patch(Rectangle((30, 114), 20, 6, fill=False, edgecolor="black", linewidth=1.5))

    # Goal
    ax.add_patch(Rectangle((36, 120), 8, 2, fill=False, edgecolor="black", linewidth=1.5))

    # Highlight zones
    if selected_zone == "Into box":
        ax.add_patch(Rectangle((18, 102), 44, 18, color="#2ecc71", alpha=0.35))
    elif selected_zone == "Central not in box":
        ax.add_patch(Rectangle((30, 80), 20, 22, color="#3498db", alpha=0.35))
    else:
        ax.add_patch(Rectangle((0, 80), 30, 40, color="#E74C3C", alpha=0.25))
        ax.add_patch(Rectangle((50, 80), 30, 40, color="#E74C3C", alpha=0.25))

    # Focus only on final third and rotate left
    ax.set_xlim(80, 0)
    ax.set_ylim(75, 122)

    ax.set_aspect("equal")
    ax.axis("off")

    return fig

# Main output
st.subheader("Modelled Shot-Creation Likelihood")
st.metric(
    label="Likelihood that the pass leads directly to a shot",
    value=f"{shot_prob:.1%}"
)

st.markdown(
    f"""
    <div style="
        padding: 10px;
        border-radius: 8px;
        background-color: {colour};
        color: white;
        text-align: center;
        font-weight: bold;
        margin-top: 10px;
        margin-bottom: 5px;
    ">
        {status}
    </div>
    """,
    unsafe_allow_html=True
)

st.subheader("Why this scenario scores this way")

insight_lines = []

if into_box:
    insight_lines.append(
        "**Destination:** Into-box passes are the strongest positive driver in the model."
    )
elif central_not_in_box:
    insight_lines.append(
        "**Destination:** Central areas outside the box are also positive, but less valuable than box entries."
    )
else:
    insight_lines.append(
        "**Destination:** Wide or non-central end locations are less favourable for direct shot creation."
    )

if defenders_within_5m >= 3:
    insight_lines.append(
        "**Congestion:** Heavy local congestion reduces the likelihood of the pass leading to a shot."
    )
else:
    insight_lines.append(
        "**Congestion:** The receiver is not in extreme local traffic, which is more favourable than heavy congestion."
    )

if under_pressure:
    insight_lines.append(
        "**Passer pressure:** Pressure on the passer slightly reduces the likelihood of shot creation."
    )
else:
    insight_lines.append(
        "**Passer pressure:** No pressure on the passer gives a slightly cleaner delivery context."
    )

for line in insight_lines:
    st.write(line)

# Mini pitch visual
st.subheader("Final-Third Destination Zone")
pitch_fig = draw_pitch_zone(zone)
st.pyplot(pitch_fig, width="content")

# Expanders for extra detail
with st.expander("Show scenario details"):
    detail_col1, detail_col2 = st.columns(2)

    with detail_col1:
        st.write(f"**Destination zone:** {zone}")
        st.write(f"**Pass length:** {pass_length:.1f}")
        st.write(f"**Pass angle:** {pass_angle:.2f}")
        st.write(f"**High pass:** {'Yes' if is_high_pass else 'No'}")
        st.write(f"**Cross:** {'Yes' if is_cross else 'No'}")

    with detail_col2:
        st.write(f"**Passer under pressure:** {'Yes' if under_pressure else 'No'}")
        st.write(f"**Nearest defender distance:** {nearest_defender_distance:.1f}")
        st.write(f"**Defenders within ~5 units:** {defenders_within_5m}")
        st.write(f"**Into box:** {'Yes' if into_box else 'No'}")
        st.write(f"**Central not in box:** {'Yes' if central_not_in_box else 'No'}")

with st.expander("Show model coefficients"):
    coefs = model.coef_[0]
    coef_df = pd.DataFrame({
        "feature": features,
        "coefficient": coefs
    }).sort_values("coefficient")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(coef_df["feature"], coef_df["coefficient"])
    ax.set_title("Logistic Regression Coefficients")
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("Feature")
    plt.tight_layout()

    st.pyplot(fig)

    st.write(
        """
        Positive coefficients are associated with a higher likelihood that a pass leads
        directly to a shot. Negative coefficients are associated with a lower likelihood.

        The chart is most useful for understanding **directional effects**, not for
        comparing exact magnitudes across very different feature types.
        """
    )

st.markdown("---")

st.info(
    "Use this tool to compare scenarios rather than to interpret any single output as a precise true probability."
)

st.caption(
    "Classification based on modelled likelihood — use comparatively rather than as an absolute measure."
)

st.caption(
    "Built as a lightweight decision-support tool from the final notebook model. "
    "Designed to support tactical interpretation rather than high-confidence event-level prediction."
)