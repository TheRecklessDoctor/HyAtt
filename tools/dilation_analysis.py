import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
from scipy.ndimage import zoom, binary_erosion


# ─────────────────────────────────────────────
# STEP 1: HOOK MANAGER
# ─────────────────────────────────────────────

class ActivationHookManager:
    def __init__(self):
        self._activations: Dict[str, torch.Tensor] = {}
        self._hooks: List = []

    def register(self, module: nn.Module, name: str):
        def hook_fn(mod, inp, out):
            self._activations[name] = out.detach().cpu()

        self._hooks.append(module.register_forward_hook(hook_fn))

    def get(self):
        return self._activations

    def clear(self):
        self._activations = {}

    def remove_all(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


# ─────────────────────────────────────────────
# STEP 2: LABEL FORMAT
# ─────────────────────────────────────────────

def to_dhw_label(label: np.ndarray) -> np.ndarray:
    label = np.asarray(label)

    if label.ndim == 4 and label.shape[0] == 1:
        label = label[0]

    if label.ndim != 3:
        raise ValueError(f"Expected 3D label, got {label.shape}")

    if label.shape[-1] < label.shape[0]:
        label = np.transpose(label, (2, 0, 1))

    return label


# ─────────────────────────────────────────────
# STEP 3: ACTIVATION REDUCTION
# ─────────────────────────────────────────────

def to_dhw_activation(act: torch.Tensor) -> np.ndarray:
    act = act[0].abs().mean(dim=0)
    return act.numpy()


# ─────────────────────────────────────────────
# STEP 4: REGION ACTIVATIONS
# ─────────────────────────────────────────────

def compute_region_activations(act, lesion_mask):
    act = to_dhw_activation(act)

    if lesion_mask.sum() == 0:
        return None

    # lesion_act = float(act[lesion_mask].mean()) if lesion_mask.any() else 0.0
    # bg_act = float(act[~lesion_mask].mean()) if (~lesion_mask).any() else 0.0
    lesion_act = float(np.median(act[lesion_mask])) if lesion_mask.any() else 0.0
    bg_act = float(np.median(act[~lesion_mask])) if (~lesion_mask).any() else 0.0

    return lesion_act, bg_act


# ─────────────────────────────────────────────
# STEP 5: SIZE GROUPING
# ─────────────────────────────────────────────

def get_size_group(volume):
    if volume < 5000:
        return "small"
    elif volume < 15000:
        return "medium"
    else:
        return "large"


# ─────────────────────────────────────────────
# STEP 6: MAIN ANALYSIS
# ─────────────────────────────────────────────

def visualise_dilation_branches(
    model,
    dataloader,
    device,
    n_patients=5
):

    model.eval()
    hook = ActivationHookManager()

    size_groups = ["tiny", "small", "medium", "large"]

    all_stats = {
        "stage0": {"dilation1": {g: [] for g in size_groups}},
        "stage1": {
            "dilation1": {g: [] for g in size_groups},
            "dilation2": {g: [] for g in size_groups},
            "dilation3": {g: [] for g in size_groups},
        },
        "stage2": {
            "dilation1": {g: [] for g in size_groups},
            "dilation2": {g: [] for g in size_groups},
            "dilation3": {g: [] for g in size_groups},
        },
    }

    # ── HOOKS ───────────────────────────────
    hook.register(model.encoder.stages[0].conv_a.attn.value.DW_conv0, "stage0_dilation1")

    hook.register(model.encoder.stages[1].conv_a.attn.value.DW_conv0, "stage1_dilation1")
    hook.register(model.encoder.stages[1].conv_a.attn.value.DW_conv1, "stage1_dilation2")
    hook.register(model.encoder.stages[1].conv_a.attn.value.DW_conv2, "stage1_dilation3")

    hook.register(model.encoder.stages[2].conv_a.attn.value.DW_conv0, "stage2_dilation1")
    hook.register(model.encoder.stages[2].conv_a.attn.value.DW_conv1, "stage2_dilation2")
    hook.register(model.encoder.stages[2].conv_a.attn.value.DW_conv2, "stage2_dilation3")

    count = 0

    with torch.no_grad():
        for batch in dataloader:

            if count >= n_patients:
                break

            dwi = batch["dwi"].float().to(device)
            adc = batch["adc"].float().to(device)

            label = batch["label"].cpu().numpy()
            label = to_dhw_label(label)

            lesion_volume = int(label.sum())
            group = get_size_group(lesion_volume)

            if lesion_volume == 0:
                continue

            hook.clear()

            x = torch.cat([dwi, adc], dim=1).to(device)
            _ = model(x, x)

            acts = hook.get()

            lesion_mask = label.astype(bool)

            for stage in all_stats:
                for branch in all_stats[stage]:

                    key = f"{stage}_{branch}"
                    if key not in acts:
                        continue

                    act = acts[key]

                    ad, ah, aw = act.shape[-3:]
                    ld, lh, lw = lesion_mask.shape

                    if (ad, ah, aw) != (ld, lh, lw):
                        scale = (ad/ld, ah/lh, aw/lw)

                        lesion_mask_r = zoom(
                            lesion_mask.astype(float),
                            scale,
                            order=0
                        ) > 0.5
                    else:
                        lesion_mask_r = lesion_mask

                    result = compute_region_activations(act, lesion_mask_r)

                    if result is None:
                        continue

                    lesion_act, bg_act = result

                    lsi = lesion_act / (bg_act + 1e-8)

                    all_stats[stage][branch][group].append({
                        "lesion": lesion_act,
                        "background": bg_act,
                        "lsi": lsi
                    })

            count += 1
            print(f"Processed {count}/{n_patients}")

    hook.remove_all()


    # ─────────────────────────────────────────────
    # STEP 7: AGGREGATION + BRANCH PREFERENCE
    # ─────────────────────────────────────────────

    final = {}
    branch_preference = {}

    for stage in all_stats:
        final[stage] = {}

        stage_lsi = {}

        for branch in all_stats[stage]:
            final[stage][branch] = {}

            branch_all_lsi = []

            for g in size_groups:
                vals = all_stats[stage][branch][g]

                if len(vals) == 0:
                    continue

                lesion_vals = [v["lesion"] for v in vals]
                bg_vals = [v["background"] for v in vals]
                lsi_vals = [v["lsi"] for v in vals]

                final[stage][branch][g] = {
                    # "lesion_mean": float(np.mean(lesion_vals)),
                    # "background_mean": float(np.mean(bg_vals)),
                    # "lsi_mean": float(np.mean(lsi_vals)),
                    "lesion_median": float(np.median(lesion_vals)),
                    "background_median": float(np.median(bg_vals)),
                    "lsi_median": float(np.median(lsi_vals)),
                    "std": float(np.std(lsi_vals)),
                    "n": len(vals)
                }

                branch_all_lsi.extend(lsi_vals)

            stage_lsi[branch] = np.mean(branch_all_lsi) if len(branch_all_lsi) > 0 else 0.0

        total = sum(stage_lsi.values()) + 1e-8

        bps = {b: stage_lsi[b] / total for b in stage_lsi}

        branch_preference[stage] = {
            "LSI": stage_lsi,
            "BPS": bps
        }


    # ─────────────────────────────────────────────
    # STEP 8: PRINT RESULTS
    # ─────────────────────────────────────────────

    print("\n===== LESION SCALE + BRANCH PREFERENCE =====\n")

    for stage in final:
        print(f"\n{stage.upper()}")

        for branch in final[stage]:
            print(f"\n{branch}")

            for g in size_groups:
                if g not in final[stage][branch]:
                    continue

                v = final[stage][branch][g]
                print(
                    f"  {g:<8} "
                    f"lesion={v['lesion_median']:.3f} "
                    f"bg={v['background_median']:.3f} "
                    f"lsi={v['lsi_median']:.3f} "
                    f"n={v['n']}"
                )

    print("\n===== BRANCH PREFERENCE (LSI / BPS) =====\n")

    for stage in branch_preference:
        print(f"\n{stage.upper()}")

        print("LSI:")
        for b, v in branch_preference[stage]["LSI"].items():
            print(f"  {b}: {v:.4f}")

        print("BPS:")
        for b, v in branch_preference[stage]["BPS"].items():
            print(f"  {b}: {v:.4f}")

    return final, branch_preference

def run_analysis(
    model,
    checkpoint_path,
    dataloader,
    device="cuda",
    n_patients=51
):

    model = model.cuda()
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)

    results, branch_preference = visualise_dilation_branches(
        model=model,
        dataloader=dataloader,
        device=device,
        n_patients=n_patients
    )

    print(branch_preference)

    print("\n===== BRANCH PREFERENCE SUMMARY =====\n")

    for stage in branch_preference:
        print(f"\n{stage.upper()}")

        print("LSI (lesion selectivity index):")
        for branch, val in branch_preference[stage]["LSI"].items():
            print(f"  {branch}: {val:.4f}")

        print("\nBPS (normalized preference):")
        for branch, val in branch_preference[stage]["BPS"].items():
            print(f"  {branch}: {val:.4f}")

    return results
