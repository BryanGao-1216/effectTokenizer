"""Nominal control frequencies for the OXE datasets used by this project.

The main table was checked against the control-frequency registry published by
the official RDT implementation:
https://github.com/thu-ml/RoboticsDiffusionTransformer/blob/main/configs/dataset_control_freq.json

Project-specific aliases are kept here instead of being scattered through the
loader.  LIBERO is sourced from the official environment default (20 Hz):
https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/envs/env_wrapper.py

These are nominal rates because OXE trajectories do not expose a common,
reliable timestamp field. OpenVLA's ``*_no_noops`` LIBERO release is still
treated as 20 Hz: its generation script skips a no-op before calling the next
20 Hz simulator step, producing a time-compressed but regularly stepped replay.
"""

from __future__ import annotations


# Keep this as a plain, standalone table so frequency choices are easy to audit
# and override without touching the data pipeline.
OXE_CONTROL_FREQUENCIES_HZ: dict[str, float] = {
    "fractal20220817_data": 3.0,
    "kuka": 10.0,
    "bridge_oxe": 5.0,
    "bridge_orig": 5.0,
    "bridge_dataset": 5.0,
    "taco_play": 15.0,
    "jaco_play": 10.0,
    "berkeley_cable_routing": 10.0,
    "roboturk": 10.0,
    "nyu_door_opening_surprising_effectiveness": 3.0,
    "viola": 20.0,
    "berkeley_autolab_ur5": 5.0,
    "toto": 30.0,
    "language_table": 10.0,
    "columbia_cairlab_pusht_real": 10.0,
    "stanford_kuka_multimodal_dataset_converted_externally_to_rlds": 20.0,
    "nyu_rot_dataset_converted_externally_to_rlds": 3.0,
    "stanford_hydra_dataset_converted_externally_to_rlds": 10.0,
    "austin_buds_dataset_converted_externally_to_rlds": 20.0,
    "nyu_franka_play_dataset_converted_externally_to_rlds": 3.0,
    "maniskill_dataset_converted_externally_to_rlds": 20.0,
    "furniture_bench_dataset_converted_externally_to_rlds": 10.0,
    # The RTX dataset viewer reports both observation and control at 10 Hz.
    "cmu_franka_exploration_dataset_converted_externally_to_rlds": 10.0,
    "ucsd_kitchen_dataset_converted_externally_to_rlds": 2.0,
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds": 3.0,
    "austin_sailor_dataset_converted_externally_to_rlds": 20.0,
    "austin_sirius_dataset_converted_externally_to_rlds": 20.0,
    "bc_z": 10.0,
    "utokyo_pr2_opening_fridge_converted_externally_to_rlds": 10.0,
    "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds": 10.0,
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds": 10.0,
    "utokyo_xarm_bimanual_converted_externally_to_rlds": 10.0,
    "robo_net": 1.0,
    "berkeley_mvp_converted_externally_to_rlds": 5.0,
    "berkeley_rpt_converted_externally_to_rlds": 30.0,
    "kaist_nonprehensile_converted_externally_to_rlds": 10.0,
    # The RDT registry marks this source as 0 (unknown), so fail loudly if it
    # is ever added to a mixture instead of silently inventing a time base.
    "stanford_mask_vit_converted_externally_to_rlds": 0.0,
    "tokyo_u_lsmo_converted_externally_to_rlds": 10.0,
    "dlr_sara_pour_converted_externally_to_rlds": 10.0,
    "dlr_sara_grid_clamp_converted_externally_to_rlds": 10.0,
    "dlr_edan_shared_control_converted_externally_to_rlds": 5.0,
    "asu_table_top_converted_externally_to_rlds": 12.5,
    "stanford_robocook_converted_externally_to_rlds": 5.0,
    "imperialcollege_sawyer_wrist_cam": 10.0,
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": 20.0,
    "uiuc_d3field": 1.0,
    "utaustin_mutex": 20.0,
    "berkeley_fanuc_manipulation": 10.0,
    "cmu_playing_with_food": 5.0,
    "cmu_play_fusion": 5.0,
    "cmu_stretch": 10.0,
    "berkeley_gnm_recon": 3.0,
    "berkeley_gnm_cory_hall": 5.0,
    "berkeley_gnm_sac_son": 10.0,
    "droid": 15.0,
    "fmb": 10.0,
    "dobbe": 30.0,
    "roboset": 5.0,
    "rh20t": 10.0,
    "tdroid_carrot_in_bowl": 5.0,
    "tdroid_pour_corn_in_pot": 5.0,
    "tdroid_flip_pot_upright": 5.0,
    "tdroid_move_object_onto_plate": 5.0,
    "tdroid_knock_object_over": 5.0,
    "tdroid_cover_object_with_towel": 5.0,
    "droid_wipe": 15.0,
    "libero_spatial_no_noops": 20.0,
    "libero_object_no_noops": 20.0,
    "libero_goal_no_noops": 20.0,
    "libero_10_no_noops": 20.0,
    "libero_90_no_noops": 20.0,
    "rl_bench": 20.0,
}


def get_oxe_control_frequency_hz(dataset_name: str) -> float:
    """Return a validated nominal source rate or fail instead of guessing."""
    try:
        frequency = float(OXE_CONTROL_FREQUENCIES_HZ[dataset_name])
    except KeyError as exc:
        raise KeyError(
            f"No control frequency is registered for OXE dataset {dataset_name!r}. "
            "Add an online-verified value to rlds/oxe/control_frequencies.py."
        ) from exc
    if frequency <= 0:
        raise ValueError(
            f"OXE dataset {dataset_name!r} has invalid control frequency {frequency}."
        )
    return frequency
