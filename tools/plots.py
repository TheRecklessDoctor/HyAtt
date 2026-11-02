from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


# ============================================================
# Configuration
# ============================================================

# Folder containing both ground-truth and prediction masks.
MASK_DIR = Path("/home/dayan/Documents/PhD/Objective_1/visualisation/images/unetpp_22")

OUTPUT_CSV = Path("isles22_lesion_volumes_unetpp.csv")
OUTPUT_PDF = Path("isles22_volume_agreement_unetpp.pdf")
OUTPUT_PNG = Path("isles22_volume_agreement_unetpp.png")

# Use 0.5 for probability maps or binary masks.
PREDICTION_THRESHOLD = 0.5


# ============================================================
# Volume calculations
# ============================================================

def voxel_volume_mm3(nifti_image):
    """
    Calculate the physical volume of one voxel in cubic millimetres.
    """
    return abs(np.linalg.det(nifti_image.affine[:3, :3]))


def mask_volume_ml(binary_mask, nifti_image):
    """
    Calculate foreground volume in millilitres.
    """
    foreground_voxels = np.count_nonzero(binary_mask)
    volume_mm3 = foreground_voxels * voxel_volume_mm3(nifti_image)

    return volume_mm3 / 1000.0


def concordance_correlation_coefficient(x, y):
    """
    Calculate Lin's concordance correlation coefficient.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    variance_x = np.var(x, ddof=1)
    variance_y = np.var(y, ddof=1)
    covariance_xy = np.cov(x, y, ddof=1)[0, 1]

    denominator = (
        variance_x
        + variance_y
        + (mean_x - mean_y) ** 2
    )

    if denominator == 0:
        return np.nan

    return 2.0 * covariance_xy / denominator


# ============================================================
# Find and pair the masks
# ============================================================

ground_truth_paths = sorted(MASK_DIR.glob("*_gt.nii.gz"))

if not ground_truth_paths:
    raise FileNotFoundError(
        f"No files matching '*_gt.nii.gz' were found in {MASK_DIR}"
    )

records = []

for ground_truth_path in ground_truth_paths:

    # Example:
    # 00_gt.nii.gz -> subject_id = 00
    subject_id = ground_truth_path.name.replace("_gt.nii.gz", "")

    prediction_path = MASK_DIR / f"{subject_id}_pred.nii.gz"

    if not prediction_path.exists():
        print(
            f"Warning: prediction not found for subject {subject_id}: "
            f"{prediction_path.name}"
        )
        continue

    # --------------------------------------------------------
    # Load masks
    # --------------------------------------------------------

    ground_truth_image = nib.load(str(ground_truth_path))
    prediction_image = nib.load(str(prediction_path))

    ground_truth_data = ground_truth_image.get_fdata(
        dtype=np.float32
    )

    prediction_data = prediction_image.get_fdata(
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Check spatial compatibility
    # --------------------------------------------------------

    if ground_truth_image.shape != prediction_image.shape:
        raise ValueError(
            f"Shape mismatch for subject {subject_id}:\n"
            f"  Ground truth: {ground_truth_image.shape}\n"
            f"  Prediction:   {prediction_image.shape}"
        )

    if not np.allclose(
        ground_truth_image.affine,
        prediction_image.affine,
        rtol=1e-4,
        atol=1e-4,
    ):
        raise ValueError(
            f"Affine mismatch for subject {subject_id}. "
            "The prediction must be resampled to the ground-truth "
            "grid before calculating physical volume."
        )

    # --------------------------------------------------------
    # Convert images to binary masks
    # --------------------------------------------------------

    ground_truth_mask = ground_truth_data > 0

    prediction_mask = (
        prediction_data >= PREDICTION_THRESHOLD
    )

    # --------------------------------------------------------
    # Calculate physical volumes
    # --------------------------------------------------------

    reference_volume_ml = mask_volume_ml(
        ground_truth_mask,
        ground_truth_image,
    )

    predicted_volume_ml = mask_volume_ml(
        prediction_mask,
        prediction_image,
    )

    signed_error_ml = predicted_volume_ml - reference_volume_ml
    absolute_error_ml = abs(signed_error_ml)

    if reference_volume_ml > 0:
        relative_error_percent = (
            absolute_error_ml / reference_volume_ml
        ) * 100.0
    else:
        relative_error_percent = np.nan

    records.append(
        {
            "subject_id": subject_id,
            "reference_volume_ml": reference_volume_ml,
            "predicted_volume_ml": predicted_volume_ml,
            "signed_error_ml": signed_error_ml,
            "absolute_error_ml": absolute_error_ml,
            "relative_absolute_error_percent": (
                relative_error_percent
            ),
        }
    )


# ============================================================
# Create subject-level volume table
# ============================================================

df = pd.DataFrame(records)

if len(df) < 3:
    raise ValueError(
        f"Only {len(df)} matched mask pairs were found. "
        "At least three pairs are needed for the analysis."
    )

# Sort numerically where possible.
df["_sort_id"] = pd.to_numeric(
    df["subject_id"],
    errors="coerce",
)

df = (
    df.sort_values(
        by=["_sort_id", "subject_id"],
        na_position="last",
    )
    .drop(columns="_sort_id")
    .reset_index(drop=True)
)

df.to_csv(OUTPUT_CSV, index=False)

print(f"Matched subjects: {len(df)}")
print(f"Saved volume measurements to: {OUTPUT_CSV}")


# ============================================================
# Calculate agreement statistics
# ============================================================

reference = df["reference_volume_ml"].to_numpy()
predicted = df["predicted_volume_ml"].to_numpy()

mean_volume = (reference + predicted) / 2.0
volume_difference = predicted - reference

mean_bias = np.mean(volume_difference)
difference_sd = np.std(volume_difference, ddof=1)

lower_loa = mean_bias - 1.96 * difference_sd
upper_loa = mean_bias + 1.96 * difference_sd

mean_absolute_error = np.mean(
    np.abs(volume_difference)
)

median_absolute_error = np.median(
    np.abs(volume_difference)
)

pearson_r, pearson_p = pearsonr(
    reference,
    predicted,
)

spearman_rho, spearman_p = spearmanr(
    reference,
    predicted,
)

ccc = concordance_correlation_coefficient(
    reference,
    predicted,
)


# ============================================================
# Divide reference volumes into tertiles
# ============================================================

try:
    df["volume_group"] = pd.qcut(
        df["reference_volume_ml"],
        q=3,
        labels=["Small", "Medium", "Large"],
        duplicates="drop",
    )
except ValueError:
    df["volume_group"] = "All lesions"

group_colors = {
    "Small": "#377eb8",
    "Medium": "#ff7f00",
    "Large": "#4daf4a",
    "All lesions": "#377eb8",
}

group_order = ["Small", "Medium", "Large"]

available_groups = [
    group
    for group in group_order
    if group in df["volume_group"].astype(str).unique()
]

if not available_groups:
    available_groups = ["All lesions"]


# ============================================================
# Create figure
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(7.2, 3.2),
)

correlation_axis = axes[0]
bland_altman_axis = axes[1]


# ------------------------------------------------------------
# Panel A: predicted versus reference volume
# ------------------------------------------------------------

for group in available_groups:
    group_mask = (
        df["volume_group"].astype(str) == group
    )

    correlation_axis.scatter(
        df.loc[group_mask, "reference_volume_ml"],
        df.loc[group_mask, "predicted_volume_ml"],
        s=25,
        color=group_colors[group],
        edgecolor="black",
        linewidth=0.3,
        alpha=0.85,
        label=group,
    )

maximum_volume = max(
    float(reference.max()),
    float(predicted.max()),
    1.0,
)

plot_limit = maximum_volume * 1.05

# Perfect-agreement identity line
correlation_axis.plot(
    [0, plot_limit],
    [0, plot_limit],
    color="black",
    linestyle="--",
    linewidth=1.0,
)

correlation_axis.set_xlim(0, plot_limit)
correlation_axis.set_ylim(0, plot_limit)

correlation_axis.set_xlabel(
    "Reference infarct volume (mL)"
)

correlation_axis.set_ylabel(
    "Predicted infarct volume (mL)"
)

correlation_axis.set_title(
    "(a) Volume agreement",
    loc="left",
)

statistics_text = (
    rf"Spearman $\rho$ = {spearman_rho:.3f}"
    "\n"
    rf"CCC = {ccc:.3f}"
    "\n"
    rf"MAE = {mean_absolute_error:.2f} mL"
)

correlation_axis.text(
    0.04,
    0.96,
    statistics_text,
    transform=correlation_axis.transAxes,
    ha="left",
    va="top",
    fontsize=7,
    bbox={
        "boxstyle": "round,pad=0.25",
        "facecolor": "white",
        "edgecolor": "0.75",
        "alpha": 0.9,
    },
)

correlation_axis.legend(
    loc="lower right",
    fontsize=7,
    frameon=True,
)


# ------------------------------------------------------------
# Panel B: Bland-Altman plot
# ------------------------------------------------------------

for group in available_groups:
    group_mask = (
        df["volume_group"].astype(str) == group
    )

    group_reference = df.loc[
        group_mask,
        "reference_volume_ml",
    ].to_numpy()

    group_prediction = df.loc[
        group_mask,
        "predicted_volume_ml",
    ].to_numpy()

    group_mean = (
        group_reference + group_prediction
    ) / 2.0

    group_difference = (
        group_prediction - group_reference
    )

    bland_altman_axis.scatter(
        group_mean,
        group_difference,
        s=25,
        color=group_colors[group],
        edgecolor="black",
        linewidth=0.3,
        alpha=0.85,
    )

# Zero-error line
bland_altman_axis.axhline(
    0,
    color="0.45",
    linestyle=":",
    linewidth=0.9,
)

# Mean bias
bland_altman_axis.axhline(
    mean_bias,
    color="#d62728",
    linestyle="-",
    linewidth=1.1,
)

# Limits of agreement
bland_altman_axis.axhline(
    upper_loa,
    color="#d62728",
    linestyle="--",
    linewidth=1.0,
)

bland_altman_axis.axhline(
    lower_loa,
    color="#d62728",
    linestyle="--",
    linewidth=1.0,
)

x_label = max(float(mean_volume.max()), 1.0) * 0.98

bland_altman_axis.text(
    x_label,
    mean_bias,
    f"Bias: {mean_bias:.2f}",
    color="#d62728",
    fontsize=7,
    ha="right",
    va="bottom",
)

bland_altman_axis.text(
    x_label,
    upper_loa,
    f"Upper LoA: {upper_loa:.2f}",
    color="#d62728",
    fontsize=7,
    ha="right",
    va="bottom",
)

bland_altman_axis.text(
    x_label,
    lower_loa,
    f"Lower LoA: {lower_loa:.2f}",
    color="#d62728",
    fontsize=7,
    ha="right",
    va="top",
)

bland_altman_axis.set_xlabel(
    "Mean predicted and reference volume (mL)"
)

bland_altman_axis.set_ylabel(
    "Predicted $-$ reference volume (mL)"
)

bland_altman_axis.set_title(
    "(b) Bland--Altman analysis",
    loc="left",
)


# ============================================================
# Final formatting and export
# ============================================================

for axis in axes:
    axis.tick_params(
        axis="both",
        labelsize=7,
    )

    axis.grid(
        visible=True,
        linestyle=":",
        linewidth=0.45,
        alpha=0.5,
    )

fig.tight_layout(
    pad=0.7,
    w_pad=1.2,
)

fig.savefig(
    OUTPUT_PDF,
    bbox_inches="tight",
)

fig.savefig(
    OUTPUT_PNG,
    dpi=600,
    bbox_inches="tight",
)

plt.show()


# ============================================================
# Print results
# ============================================================

print("\nVolume agreement results")
print("-" * 48)
print(f"Number of subjects:        {len(df)}")
print(f"Mean reference volume:     {reference.mean():.3f} mL")
print(f"Mean predicted volume:     {predicted.mean():.3f} mL")
print(f"Mean bias:                 {mean_bias:.3f} mL")
print(f"Lower 95% LoA:             {lower_loa:.3f} mL")
print(f"Upper 95% LoA:             {upper_loa:.3f} mL")
print(f"Mean absolute error:       {mean_absolute_error:.3f} mL")
print(f"Median absolute error:     {median_absolute_error:.3f} mL")
print(f"Pearson r:                 {pearson_r:.3f}")
print(f"Pearson p-value:           {pearson_p:.4g}")
print(f"Spearman rho:              {spearman_rho:.3f}")
print(f"Spearman p-value:          {spearman_p:.4g}")
print(f"Concordance coefficient:   {ccc:.3f}")

print("\nVolume-group summary")
print("-" * 48)

print(
    df.groupby(
        "volume_group",
        observed=True,
    )["reference_volume_ml"].agg(
        count="count",
        minimum="min",
        median="median",
        maximum="max",
    )
)