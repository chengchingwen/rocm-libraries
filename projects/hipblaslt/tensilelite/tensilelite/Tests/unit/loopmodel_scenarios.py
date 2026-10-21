# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Named TensileLite scenarios for the param-driven entry point (translation ADDON)."""

# Real MXFP8 16x16x128 geometry (fp8, 1B): wave M/N-fan = 8 MI, TDMSplit region 2 -> in-region fan
# 4, substep = DepthU/MI_K = 2.
MXFP8_GEOM = {
    "MatrixInstruction": [16, 16, 128, 1], "DepthU": 256, "ElemBytes": 1,
    "MIWaveTile": [8, 8],
    "GlobalReadVectorWidthA": 64, "GlobalReadVectorWidthB": 64, "LocalReadVectorWidth": 16,
}

SCENARIOS = {
    "baseline": ("Baseline (all defaults: TDM A+B, PGR2, PLR1, no split)", {}),
    # Reduced MT-region split ONLY (no register partition): A/B each split into 2 storage-
    # disjoint regions.  NOT the two-rate [4,2] -- that needs the real geometry (mxfp8_derive).
    "our_split": ("Reduced region split (A0/A1 x B0/B1) -- MT-region only, no register partition",
                  {"TDMSplit": True, "MIWaveTile": [2, 2]}),
    # Groups split each eight-tile M unit 4+4; lo pipelines at W=2 and hi reuses W=1.
    "mxfp8_derive": ("MXFP8 16x16x128 split M-unit groups: S_reg(A)=[lo:2,hi:1] "
                     "(fp A=192,B=256)",
                     {**MXFP8_GEOM, "TDMSplit": [2, 2],
                      "VgprPartitionA": 2, "VgprReuseStrategyA": 1}),
    # --- LoopOrder: axis-word over {K,M,N} ---------------------------------------
    "order_knm": ("LoopOrder='KNM' -- swap M/N blocks",
                  {"TDMSplit": [2, 2], "MIWaveTile": [2, 2], "LoopOrder": "KNM"}),
    "order_mnk": ("LoopOrder='MNK' -- M-outer, K-inner",
                  {"TDMSplit": [2, 2], "MIWaveTile": [2, 2], "LoopOrder": "MNK"}),
    "order_interleave": ("LoopOrder='KMKNMN' -- interleaved regions (split not contiguous with inner)",
                         {"TDMSplit": [2, 2], "MIWaveTile": [2, 2], "LoopOrder": "KMKNMN"}),
    # --- TDMSplit DU (shared K) axis: symmetric + asymmetric residual ------------
    "du_sym": ("TDMSplit DU symmetric [1,1,2,2] -> shared K_Split=2",
               {"TDMSplit": [1, 1, 2, 2]}),
    "du_asym": ("TDMSplit DU asymmetric [1,1,2,1] (2|1) -> K residual on A's copy hop",
                {"TDMSplit": [1, 1, 2, 1]}),
    # --- VGPR reuse: partition + strategy on real geometry -----------------------
    "reg_min_vgpr": ("VgprPartition=2 strategy=1 (min-VGPR: lo unroll, hi in-place)",
                     {**MXFP8_GEOM, "TDMSplit": [2, 2], "VgprPartitionA": 2, "VgprReuseStrategyA": 1}),
    "reg_max_sched": ("VgprPartition=2 strategy=0 (max-sched: both parts unroll W=R)",
                      {**MXFP8_GEOM, "TDMSplit": [2, 2], "VgprPartitionA": 2, "VgprReuseStrategyA": 0}),
    "plr2": ("Deeper local prefetch (PLR=2)", {"PrefetchLocalRead": 2}),
    "halfplr_a": ("HalfPLR on A (masked fan depth)",
                  {"TDMSplit": True, "MIWaveTile": [2, 2], "HalfPLR": 1}),
    "dtv": ("DirectToVgpr A+B (one-hop, VMEM path)", {"DirectToVgprA": True, "DirectToVgprB": True}),
    "padded": ("LDS padding on A",
               {"TDMSplit": True, "MIWaveTile": [2, 2],
                "LdsPadA": 8, "LdsBlockSizePerPad": 128}),
    "1lds": ("Single LDS buffer", {"TDMSplit": True, "MIWaveTile": [2, 2], "1LDSBuffer": 1}),
    "waveseparate": ("Wave-separated global read",
                     {"TDMSplit": True, "MIWaveTile": [2, 2],
                      "WaveSeparateGlobalReadA": 1}),
    # --- MX scenarios: MXSA/MXSB paths + the TDMFuse enum patterns ---------
    "mx_nofuse": ("MX scales present, no fusion (4 separate TDM streams: A,B,MXSA,MXSB) -- one "
                  "wave, since a fuse IS the wave selection",
                  {"TDMSplit": True, "MIWaveTile": [2, 2],
                   "MXBlockA": 32, "MXBlockB": 32, "TDMFuse": 0, "NumWaves": 1}),
    "mx_fuse_ab": ("MX + fuse A/B data (scales separate) [TDMFuse=0]",
                   {"TDMSplit": True, "MIWaveTile": [2, 2],
                    "MXBlockA": 32, "MXBlockB": 32, "TDMFuse": 0, "NumWaves": 4}),
    "mx_fuse_amx": ("MX + fuse {A,MXSA,MXSB} whole (A unsplit), B split [TDMFuse=2, TDMSplit=[1,2]]",
                    {"TDMSplit": [1, 2], "MXBlockA": 32, "MXBlockB": 32, "TDMFuse": 2, "NumWaves": 4}),
    # THE HETEROGENEOUS Phi : "a partition may mix a per-region-paired group with a whole-operand
    # (unsplit) group in the same Phi".
    "mx_fuse_ab_split": ("MX_AB + TDMSplit[2,2]: {A0,B0} and {A1,B1} paired by region index, "
                         "{MXSA,MXSB} whole [TDMFuse=0]",
                         {"TDMSplit": [2, 2], "MIWaveTile": [2, 2],
                          "MXBlockA": 32, "MXBlockB": 32, "TDMFuse": 0, "NumWaves": 4}),
    "mx_fuse_paired": ("MX + paired {A,MXSA},{MXSB,B} [TDMFuse=1]",
                       {"TDMSplit": True, "MIWaveTile": [2, 2],
                        "MXBlockA": 32, "MXBlockB": 32, "TDMFuse": 1, "NumWaves": 4}),
    "mx_fuse_nomx": ("TDMFuse=2 but NO MX present -> should warn + fall back",
                     {"TDMSplit": True, "MIWaveTile": [2, 2], "TDMFuse": 2, "NumWaves": 4}),
}
