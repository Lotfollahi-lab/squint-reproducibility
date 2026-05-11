#!/usr/bin/env python3
"""
Spatial-overview figures for the mmb0-1b_smb1-1b_1p dataset (and any
sibling silver-h5ad-on-disk datasets).

For each dataset-tag this renders ONE figure PER SECTION (not a combined
1×N panel — separate files so they're easy to drop into a figure
layout). For each section:

    <tag>__<section>__cell_type.{svg,png}   coloured by obs["cell_type"]
    <tag>__<section>__niche.{svg,png}       coloured by obs["niche"]
    <tag>__<section>__tissue.{svg,png}      all-grey anatomy view

For mmb0-1b_smb1-1b_1p that produces 6 spatial figures (2 sections ×
3 label types).

Color harmonisation across sections
-----------------------------------
The label vocabularies in mmb-smb's two sources (MERFISH and STARmap+)
are NOT harmonised — "Astro" vs "Astrocyte", "Glut" vs "Excitatory", etc.
This script attempts a best-effort harmonisation so the same biology
gets the same colour in both sections:

  1. Normalise each label (lowercase, strip whitespace + punctuation).
  2. Map normalised labels to a canonical name via the SYNONYMS table
     below. Anything not in SYNONYMS keeps its (title-cased) normalised
     form as its canonical name.
  3. Build a single dataset-wide palette indexed by canonical name.

The script prints the resulting raw → canonical mappings per section
so you can see which labels got merged. Extend SYNONYMS in this file
(or pass `--synonyms-json <path>`) to refine the mapping.

Cells are scattered at `obsm["spatial"]` with aspect locked to equal so
morphology is preserved. PNGs are rasterised; SVGs are vector with
editable text (`svg.fonttype = "none"`).

Default <out_dir>:
    /nfs/team361/sb75/squint-reproducibility/artifacts/data_preparation/<tag>/
"""
from __future__ import annotations

import argparse
import json
import colorsys
import re
import string
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DATASET_TAG = "mmb0-1b_smb1-1b_1p"
DEFAULT_SILVER_ROOT = Path("/nfs/team361/sb75/DATASETS/silver")
DEFAULT_OUT_ROOT = Path(
    "/nfs/team361/sb75/squint-reproducibility/artifacts/data_preparation"
)


# Display names + safe filename slugs for the well-known sections.
# Falls back to the filename stem if a section isn't in the map.
SECTION_LABELS: Dict[str, Tuple[str, str]] = {
    # filename_stem -> (display_title, filename_slug)
    "harmonised_merfish_mouse_brain_239-batch_batch82_shared_genes":
        ("MERFISH mouse brain (batch 82)", "merfish_batch82"),
    "harmonised_starmap_plus_mouse_cns_batch15_shared_genes":
        ("STARmap+ mouse CNS (batch 15)", "starmap_batch15"),
}


# ---------------------------------------------------------------------------
# Label harmonisation — synonyms table
# ---------------------------------------------------------------------------
# Maps a NORMALISED label (lowercase, no whitespace/punct) to a
# canonical display name. Anything not in this table keeps its
# title-cased normalised form. Extend as you discover new labels.
# Override via `--synonyms-json <path>` (a JSON dict mapping
# normalised → canonical).
SYNONYMS: Dict[str, str] = {
    # ---- Glia ----------------------------------------------------------
    "astro":               "Astrocyte",
    "astros":              "Astrocyte",
    "astrocyte":           "Astrocyte",
    "astrocytes":          "Astrocyte",
    "matureastrocyte":     "Astrocyte",
    "astroependymal":      "Astrocyte",
    "astroepen":           "Astrocyte",
    "astroependymalcells": "Astrocyte",
    "27astroepennn":       "Astrocyte",       # Zhang MERFISH class 27
    "oligo":               "Oligodendrocyte",
    "oligos":              "Oligodendrocyte",
    "oligodendrocyte":     "Oligodendrocyte",
    "oligodendrocytes":    "Oligodendrocyte",
    "mol":                 "Oligodendrocyte",
    "mfol":                "Oligodendrocyte",
    "matureoligo":         "Oligodendrocyte",
    "matureoligodendrocyte": "Oligodendrocyte",
    "oligonn":             "Oligodendrocyte",
    "28oligonn":           "Oligodendrocyte",
    "opc":                 "OPC",
    "opcs":                "OPC",
    "oligoprecursor":      "OPC",
    "oligodendrocyteprecursor": "OPC",
    "opcnn":               "OPC",
    "31opcnn":             "OPC",
    "microglia":           "Microglia",
    "micro":               "Microglia",
    "microglial":          "Microglia",
    "micronn":             "Microglia",
    "32immunenn":          "Microglia",       # Zhang MERFISH immune class
    "immune":              "Microglia",
    "immunenn":            "Microglia",
    "tanycyte":            "Tanycyte",
    "tan":                 "Tanycyte",
    "oec":                 "OEC",
    "oecnn":               "OEC",
    "olfactoryensheathing": "OEC",
    "29oecnn":             "OEC",
    # ---- Neurons (broad, glutamatergic) --------------------------------
    "ex":                  "Excitatory neuron",
    "exc":                 "Excitatory neuron",
    "excit":               "Excitatory neuron",
    "excitatory":          "Excitatory neuron",
    "excitatoryneuron":    "Excitatory neuron",
    "excitatoryneurons":   "Excitatory neuron",
    "glut":                "Excitatory neuron",
    "glutamatergic":       "Excitatory neuron",
    "glutaminergic":       "Excitatory neuron",
    "pyramidal":           "Excitatory neuron",
    "principalneuron":     "Excitatory neuron",
    # ---- Neurons (broad, GABAergic) ------------------------------------
    "inh":                 "Inhibitory neuron",
    "inhi":                "Inhibitory neuron",
    "inhibitory":          "Inhibitory neuron",
    "inhibitoryneuron":    "Inhibitory neuron",
    "inhibitoryneurons":   "Inhibitory neuron",
    "gaba":                "Inhibitory neuron",
    "gabaergic":           "Inhibitory neuron",
    "interneuron":         "Inhibitory neuron",
    "interneurons":        "Inhibitory neuron",
    # ---- Neurons (modulatory) ------------------------------------------
    "dopa":                "Dopaminergic neuron",
    "dopaminergic":        "Dopaminergic neuron",
    "da":                  "Dopaminergic neuron",
    "sero":                "Serotonergic neuron",
    "serotonergic":        "Serotonergic neuron",
    "5ht":                 "Serotonergic neuron",
    "chol":                "Cholinergic neuron",
    "cholinergic":         "Cholinergic neuron",
    "neuron":              "Neuron (unspecified)",
    "neurons":             "Neuron (unspecified)",
    # ---- Vascular / mesenchymal ----------------------------------------
    "endo":                "Endothelial",
    "endothelial":         "Endothelial",
    "endothelialcell":     "Endothelial",
    "endothelialcells":    "Endothelial",
    "vec":                 "Endothelial",
    "30vascnn":            "Endothelial",     # Zhang MERFISH vascular class
    "vlmc":                "VLMC",
    "vlmcs":               "VLMC",
    "leptomeningeal":      "VLMC",
    "vsmc":                "Vascular smooth muscle",
    "smc":                 "Vascular smooth muscle",
    "smooth":              "Vascular smooth muscle",
    "smoothmuscle":        "Vascular smooth muscle",
    "peri":                "Pericyte",
    "pericyte":            "Pericyte",
    "pericytes":           "Pericyte",
    "vasc":                "Vascular",
    "vascular":            "Vascular",
    "vascnn":              "Vascular",
    "abc":                 "Arachnoid barrier cell",
    "arachnoid":           "Arachnoid barrier cell",
    "fibroblast":          "Fibroblast",
    "fibroblasts":         "Fibroblast",
    # ---- Ependymal / CSF-producing -------------------------------------
    "ependymal":           "Ependymal",
    "epend":               "Ependymal",
    "epen":                "Ependymal",
    "ependymalcell":       "Ependymal",
    "choroid":             "Choroid plexus",
    "choroidplexus":       "Choroid plexus",
    "choroidplexusepithelial": "Choroid plexus",
    "cpepithelial":        "Choroid plexus",
    "33choroidplexusnn":   "Choroid plexus",  # Zhang MERFISH class 33
    # ---- Immune (peripheral) -------------------------------------------
    "tcell":               "T cell",
    "tcells":              "T cell",
    "bcell":               "B cell",
    "bcells":              "B cell",
    "macrophage":          "Macrophage",
    "macrophages":         "Macrophage",
    "bam":                 "BAM",              # border-associated macrophage
    "34bamnn":             "BAM",
    "borderassociatedmacrophage": "BAM",
    "nk":                  "NK cell",
    "nkcell":              "NK cell",
    # ---- Brain regions (niches) ----------------------------------------
    "cortex":              "Cortex",
    "isocortex":           "Cortex",
    "ctx":                 "Cortex",
    "ctxsp":               "Cortex",
    "cerebralcortex":      "Cortex",
    "neocortex":           "Cortex",
    "ctxpl":               "Cortex",
    "hippocampus":         "Hippocampus",
    "hippocampalformation": "Hippocampus",
    "hip":                 "Hippocampus",
    "hpc":                 "Hippocampus",
    "hippocampalregion":   "Hippocampus",
    "hpf":                 "Hippocampus",
    "striatum":            "Striatum",
    "str":                 "Striatum",
    "cpstr":               "Striatum",
    "caudoputamen":        "Striatum",
    "caudate":             "Striatum",
    "caudateputamen":      "Striatum",
    "thalamus":            "Thalamus",
    "th":                  "Thalamus",
    "thal":                "Thalamus",
    "hypothalamus":        "Hypothalamus",
    "hy":                  "Hypothalamus",
    "hyp":                 "Hypothalamus",
    "midbrain":            "Midbrain",
    "mb":                  "Midbrain",
    "mid":                 "Midbrain",
    "cerebellum":          "Cerebellum",
    "cb":                  "Cerebellum",
    "cer":                 "Cerebellum",
    "cerebellar":          "Cerebellum",
    "cbcortex":            "Cerebellum",
    "amygdala":            "Amygdala",
    "amy":                 "Amygdala",
    "olfactory":           "Olfactory area",
    "olfactorybulb":       "Olfactory area",
    "ob":                  "Olfactory area",
    "olf":                 "Olfactory area",
    "olfactoryarea":       "Olfactory area",
    "obvasm":              "Olfactory area",
    "pallidum":            "Pallidum",
    "pal":                 "Pallidum",
    "globuspallidus":      "Pallidum",
    "hindbrain":           "Hindbrain",
    "hb":                  "Hindbrain",
    "pons":                "Pons",
    "p":                   "Pons",
    "medulla":             "Medulla",
    "my":                  "Medulla",
    "medullaoblongata":    "Medulla",
    "fibertracts":         "Fiber tracts",
    "fiber":               "Fiber tracts",
    "fbtr":                "Fiber tracts",
    "ft":                  "Fiber tracts",
    "corpuscallosum":      "Fiber tracts",
    "cc":                  "Fiber tracts",
    "whitematter":         "White matter",
    "wm":                  "White matter",
    "ventricularsystems":  "Ventricles",
    "ventricles":          "Ventricles",
    "vs":                  "Ventricles",
    "vent":                "Ventricles",
    "ventricularsystem":   "Ventricles",
    "lv":                  "Ventricles",
    "lateralventricle":    "Ventricles",
    "thirdventricle":      "Ventricles",
    "fourthventricle":     "Ventricles",
    "olfactorytract":      "Fiber tracts",
    "stria":               "Fiber tracts",
    "lsx":                 "Striatum",          # lateral septal complex (close to striatum)
    "cnu":                 "Striatum",          # cerebral nuclei
    "tegmentum":           "Midbrain",
    "rhinalcortex":        "Cortex",
}


# Substring-fallback rules. Applied AFTER the exact lookup fails, in
# the listed order. Each rule = (token_in_normalised_label, canonical).
# Designed to catch the MERFISH-style subclass naming convention where
# the cell-class is appended as a single token suffix (e.g.
# "01_IT-ET Glut" → "01itetglut" → contains "glut" → EXCITATORY NEURON).
# Order matters: more specific tokens go first so they win over generic
# ones (e.g. "gaba" before "glut" in case some label has both substrings).
SUBSTRING_FALLBACKS: List[Tuple[str, str]] = [
    # ORDER MATTERS: more specific tokens go first so they win over
    # less specific ones whose substring would also match. E.g.
    # "hypothalam" must be tested BEFORE "thalam", otherwise
    # "hypothalamus" → contains "thalam" → wrongly mapped to Thalamus.
    #
    # The neuron-token block below catches compound labels like
    # "Telencephalon-projecting excitatory neuron" (contains
    # "excitatory") and folds them into the broader "Excitatory neuron"
    # canonical, so the X-projecting variant shares colour with the
    # plain version. Order = most specific to broadest.
    ("excitatoryneuron",   "Excitatory neuron"),
    ("inhibitoryneuron",   "Inhibitory neuron"),
    ("dopaminergicneuron", "Dopaminergic neuron"),
    ("serotonergicneuron", "Serotonergic neuron"),
    ("cholinergicneuron",  "Cholinergic neuron"),
    ("excitatory",         "Excitatory neuron"),
    ("inhibitory",         "Inhibitory neuron"),
    ("dopaminergic",       "Dopaminergic neuron"),
    ("serotonergic",       "Serotonergic neuron"),
    ("cholinergic",        "Cholinergic neuron"),
    ("dopa",       "Dopaminergic neuron"),
    ("sero",       "Serotonergic neuron"),
    ("chol",       "Cholinergic neuron"),
    ("gaba",       "Inhibitory neuron"),
    ("glut",       "Excitatory neuron"),
    ("astro",      "Astrocyte"),
    ("epen",       "Ependymal"),
    ("oligo",      "Oligodendrocyte"),
    ("opc",        "OPC"),
    ("micro",      "Microglia"),
    ("endo",       "Endothelial"),
    ("peri",       "Pericyte"),
    ("vlmc",       "VLMC"),
    ("smc",        "Vascular smooth muscle"),
    ("choroid",    "Choroid plexus"),
    ("hypothalam", "Hypothalamus"),     # MUST precede "thalam"
    ("thalam",     "Thalamus"),
    ("hippoc",     "Hippocampus"),
    ("striatum",   "Striatum"),
    ("cerebell",   "Cerebellum"),
    ("amygdal",    "Amygdala"),
    ("olfact",     "Olfactory area"),
    ("pallid",     "Pallidum"),
    ("ventric",    "Ventricles"),
    ("fibertract", "Fiber tracts"),
    ("whitematt",  "White matter"),
    # `cortex` and similar broad tokens are intentionally LAST — they
    # frequently appear as substrings of more specific labels
    # ("retrosplenial cortex", "rhinal cortex", etc.) which we'd rather
    # see map to Cortex as a coarse-grained niche.
    ("cortex",     "Cortex"),
]

# Curated 64-colour palette from scanpy.plotting.palettes.godsnot_102
# (Zeileis et al.; designed for max categorical distinction on
# scientific figures). Inlined so we don't take a hard scanpy dep.
# We use this ONLY for niches (distinct brain regions should look
# unrelated) and as the fallback for any unknown cell-type label.
GODSNOT_64: List[str] = [
    "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941",
    "#006FA6", "#A30059", "#FFDBE5", "#7A4900", "#0000A6",
    "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
    "#5A0007", "#809693", "#6A3A4C", "#1B4400", "#4FC601",
    "#3B5DFF", "#4A3B53", "#FF2F80", "#61615A", "#BA0900",
    "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA",
    "#D16100", "#DDEFFF", "#000035", "#7B4F4B", "#A1C299",
    "#300018", "#0AA6D8", "#013349", "#00846F", "#372101",
    "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2",
    "#C2FF99", "#001E09", "#00489C", "#6F0062", "#0CBD66",
    "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
    "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459",
    "#456648", "#0086ED", "#886F4C", "#34362D",
]


# ---------------------------------------------------------------------------
# Cell-type category palette
# ---------------------------------------------------------------------------
# Each canonical cell-type label is assigned to a higher-level
# CATEGORY, and each CATEGORY has its own base colour. Sub-types within
# the same category get NEAR-IDENTICAL hues (just shifted in lightness)
# so the viewer immediately sees "all neurons are warm reds/oranges,
# all glia are blues, all vascular cells are teal" without losing the
# ability to distinguish sub-types.
#
# Niches don't go through this — distinct brain regions are categorical
# entities that should look unrelated, so we use the godsnot palette
# directly for them.

CELL_TYPE_CATEGORY: Dict[str, str] = {
    # Excitatory neurons → warm red family
    "Excitatory neuron":           "neuron_excitatory",
    # Inhibitory neurons → warm orange/yellow family
    "Inhibitory neuron":           "neuron_inhibitory",
    # Modulatory neurons → magenta/pink (still "neuron-ish" warm tones,
    # but visually separate from glutamatergic/GABAergic mainlines)
    "Dopaminergic neuron":         "neuron_modulatory",
    "Serotonergic neuron":         "neuron_modulatory",
    "Cholinergic neuron":          "neuron_modulatory",
    "Neuron (unspecified)":        "neuron_modulatory",
    # Astrocytes → bright blue
    "Astrocyte":                   "astrocyte",
    # Oligodendrocyte lineage → deep blue
    "Oligodendrocyte":             "oligo_lineage",
    "OPC":                         "oligo_lineage",
    # Microglia → forest green (immune-ish but resident)
    "Microglia":                   "microglia",
    # Other CNS-resident glia → mint / olive
    "OEC":                         "glia_other",
    "Tanycyte":                    "glia_other",
    # Vascular / mesenchymal → teal family
    "Endothelial":                 "vascular",
    "VLMC":                        "vascular",
    "Pericyte":                    "vascular",
    "Vascular smooth muscle":      "vascular",
    "Vascular":                    "vascular",
    "Arachnoid barrier cell":      "vascular",
    "Fibroblast":                  "vascular",
    # CSF-producing → gold / amber
    "Ependymal":                   "csf_producing",
    "Choroid plexus":              "csf_producing",
    # Peripheral immune → violet
    "Macrophage":                  "peripheral_immune",
    "BAM":                         "peripheral_immune",
    "T cell":                      "peripheral_immune",
    "B cell":                      "peripheral_immune",
    "NK cell":                     "peripheral_immune",
}

# Base hue (H in HLS) and base lightness/saturation for each category.
# Within a category we vary lightness around the base so sub-types are
# distinguishable but obviously share the same hue family.
#
# Neurons get LOW saturation (≈0.10–0.15) so all three categories read
# as "grey-tinted" rather than competing for visual attention with
# glia / vascular / CSF / immune. Sub-categories within the neuron
# family stay distinguishable via lightness fan-out (0.35 → 0.65)
# while the colour itself stays muted. Non-neuronal categories keep
# their saturated hues so the non-neuron biology pops visually.
CATEGORY_BASE_HLS: Dict[str, Tuple[float, float, float]] = {
    # category -> (hue [0..1], lightness, saturation)
    # Neurons — low saturation, all read as greys with a hint of warm tint.
    "neuron_excitatory":   (0.00,  0.50, 0.12),
    "neuron_inhibitory":   (0.05,  0.50, 0.12),
    "neuron_modulatory":   (0.95,  0.50, 0.12),
    # Non-neuronal — saturated hues, well separated.
    "astrocyte":           (0.58,  0.55, 0.80),   # sky blue
    "oligo_lineage":       (0.66,  0.40, 0.70),   # deep blue
    "microglia":           (0.36,  0.40, 0.55),   # forest green
    "glia_other":          (0.46,  0.55, 0.40),   # mint / muted green
    "vascular":            (0.50,  0.45, 0.55),   # teal
    "csf_producing":       (0.13,  0.55, 0.85),   # gold
    "peripheral_immune":   (0.78,  0.50, 0.55),   # violet
}

# Fallback colour for cell-type labels that don't match any category.
# Drawn from godsnot in iteration order so multiple unknown labels get
# distinct fallback colours instead of all sharing the same grey.


def _hls_to_hex(h: float, l: float, s: float) -> str:
    h = h % 1.0
    l = max(0.0, min(1.0, l))
    s = max(0.0, min(1.0, s))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return "#{:02X}{:02X}{:02X}".format(
        int(round(r * 255)), int(round(g * 255)), int(round(b * 255)),
    )


def _category_palette(category: str, n_members: int) -> List[str]:
    """Generate `n_members` colours in the colour family of `category`,
    distributed across a lightness band around the category's base.
    Returns hex strings in HLS-deterministic order."""
    if category not in CATEGORY_BASE_HLS:
        return []
    h, l_center, s = CATEGORY_BASE_HLS[category]
    if n_members <= 0:
        return []
    if n_members == 1:
        return [_hls_to_hex(h, l_center, s)]
    # Spread lightness around the center in a controlled band so the
    # darkest variant is still readable and the lightest is still
    # saturated enough.
    lightness_span = 0.30   # ± 0.15 around the center
    half = lightness_span / 2.0
    lightnesses = np.linspace(l_center - half, l_center + half, n_members)
    return [_hls_to_hex(h, float(l), s) for l in lightnesses]


def build_cell_type_colour_map(canonicals: List[str]) -> Dict[str, str]:
    """Assign colours to cell-type canonicals using category-grouped
    HLS shading. Same biological category → same base hue. Members of
    that category fan out in lightness so they're distinguishable.

    Canonicals NOT in `CELL_TYPE_CATEGORY` (e.g. atlas-specific names
    we don't have synonyms for) fall back to the `GODSNOT_64`
    categorical palette so they still get distinct colours.
    """
    # Group canonicals by category (None = unknown).
    by_category: Dict[Optional[str], List[str]] = {}
    for name in canonicals:
        cat = CELL_TYPE_CATEGORY.get(name)
        by_category.setdefault(cat, []).append(name)

    out: Dict[str, str] = {}
    # 1. Categorised cells — fan out within each category's lightness band.
    for cat, members in by_category.items():
        if cat is None:
            continue
        members_sorted = sorted(members)
        palette = _category_palette(cat, n_members=len(members_sorted))
        for name, colour in zip(members_sorted, palette):
            out[name] = colour
    # 2. Uncategorised cells — fall back to godsnot. Skip indices that
    #    would clash with hues already used by the categories (rough
    #    avoidance only — godsnot is high-variance anyway).
    if None in by_category:
        unknown = sorted(by_category[None])
        for i, name in enumerate(unknown):
            out[name] = GODSNOT_64[i % len(GODSNOT_64)]
    return out


# ---------------------------------------------------------------------------
# Niche category palette (parallels the cell-type palette, but uses
# developmental / anatomical groupings: forebrain telencephalon vs
# diencephalon vs midbrain vs hindbrain vs white matter vs CSF spaces).
# Anatomically-related niches share a hue family, sub-niches vary in
# lightness — same logic as cell types.
# ---------------------------------------------------------------------------

NICHE_CATEGORY: Dict[str, str] = {
    # Telencephalon (cerebrum) — cortical and subcortical structures
    "Cortex":             "telencephalon",
    "Hippocampus":        "telencephalon",
    "Striatum":           "telencephalon",
    "Pallidum":           "telencephalon",
    "Amygdala":           "telencephalon",
    "Olfactory area":     "telencephalon",
    # Diencephalon (interbrain)
    "Thalamus":           "diencephalon",
    "Hypothalamus":       "diencephalon",
    # Mesencephalon (midbrain)
    "Midbrain":           "midbrain",
    # Metencephalon + Myelencephalon (hindbrain group)
    "Cerebellum":         "hindbrain",
    "Pons":               "hindbrain",
    "Medulla":            "hindbrain",
    "Hindbrain":          "hindbrain",
    # White matter / fiber tracts
    "Fiber tracts":       "white_matter",
    "White matter":       "white_matter",
    # CSF spaces
    "Ventricles":         "csf_space",
}

# Base hues for niche categories. Picked to give a clear visual
# separation between developmental groups while staying tonally
# consistent (no clashes between "Cortex" and "Hippocampus" — both
# read as "telencephalon" via shared hue).
NICHE_CATEGORY_BASE_HLS: Dict[str, Tuple[float, float, float]] = {
    "telencephalon":  (0.55, 0.50, 0.70),   # medium blue (cerebrum)
    "diencephalon":   (0.42, 0.50, 0.65),   # cyan-teal
    "midbrain":       (0.30, 0.45, 0.65),   # green
    "hindbrain":      (0.17, 0.50, 0.70),   # yellow-green / olive
    "white_matter":   (0.83, 0.50, 0.55),   # pink/rose
    "csf_space":      (0.07, 0.55, 0.85),   # warm orange (visually
                                            # close to "no tissue here")
}


def _niche_category_palette(category: str, n_members: int) -> List[str]:
    """Same logic as `_category_palette` but reads from
    `NICHE_CATEGORY_BASE_HLS`. Kept separate so cell-type and niche
    category tables stay independently editable."""
    if category not in NICHE_CATEGORY_BASE_HLS:
        return []
    h, l_center, s = NICHE_CATEGORY_BASE_HLS[category]
    if n_members <= 0:
        return []
    if n_members == 1:
        return [_hls_to_hex(h, l_center, s)]
    lightness_span = 0.30
    half = lightness_span / 2.0
    lightnesses = np.linspace(l_center - half, l_center + half, n_members)
    return [_hls_to_hex(h, float(l), s) for l in lightnesses]


def build_niche_colour_map(canonicals: List[str]) -> Dict[str, str]:
    """Niche colour assignment with the same biology-aware grouping the
    cell-type palette uses: members of the same developmental category
    (e.g. all telencephalic structures — Cortex, Hippocampus, Striatum,
    ...) share a hue family, with lightness fanning sub-regions out so
    they remain distinguishable.

    Unknown niche labels fall back to `GODSNOT_64` for distinct
    categorical colours.
    """
    by_category: Dict[Optional[str], List[str]] = {}
    for name in canonicals:
        cat = NICHE_CATEGORY.get(name)
        by_category.setdefault(cat, []).append(name)

    out: Dict[str, str] = {}
    for cat, members in by_category.items():
        if cat is None:
            continue
        members_sorted = sorted(members)
        palette = _niche_category_palette(cat, n_members=len(members_sorted))
        for name, colour in zip(members_sorted, palette):
            out[name] = colour
    if None in by_category:
        unknown = sorted(by_category[None])
        for i, name in enumerate(unknown):
            # Walk godsnot from the END (well separated from category
            # hues already used at the start).
            out[name] = GODSNOT_64[(-1 - i) % len(GODSNOT_64)]
    return out


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def _apply_nature_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 6.5,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })


# ---------------------------------------------------------------------------
# Section discovery
# ---------------------------------------------------------------------------

_FILESYSTEM_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")


def _section_meta(stem: str) -> Tuple[str, str]:
    """Return (display_title, filename_slug) for a section. Falls back
    to the filename stem when not in the known-sections map."""
    if stem in SECTION_LABELS:
        return SECTION_LABELS[stem]
    # Generate a safe slug from the stem.
    slug = _FILESYSTEM_UNSAFE.sub("_", stem).strip("_")
    return stem, slug


def load_sections(silver_dir: Path) -> List[Tuple[str, str, ad.AnnData]]:
    """Read every .h5ad under `silver_dir` (no concat). Returns
    [(display_title, filename_slug, adata), ...]."""
    files = sorted(Path(silver_dir).glob("*.h5ad"))
    if not files:
        raise SystemExit(f"No .h5ad files under {silver_dir}.")
    out: List[Tuple[str, str, ad.AnnData]] = []
    print(f"Loading {len(files)} silver section(s):")
    for f in files:
        a = ad.read_h5ad(f)
        if "spatial" not in a.obsm:
            print(f"  skip {f.name}: missing obsm['spatial']")
            continue
        title, slug = _section_meta(f.stem)
        print(f"  {f.name:60s}  n_obs={a.n_obs:>7d}  -> {title!r} (slug={slug!r})")
        out.append((title, slug, a))
    if not out:
        raise SystemExit("No usable sections (all missing obsm['spatial']).")
    return out


# ---------------------------------------------------------------------------
# Label harmonisation
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[\s" + re.escape(string.punctuation) + r"]+")


def _normalise_label(raw: str) -> str:
    """Lowercase + strip whitespace and punctuation. The result is the
    KEY used to look up `SYNONYMS`; the *canonical* (displayed) name
    comes from SYNONYMS or the title-cased normalised form."""
    return _PUNCT_RE.sub("", str(raw).lower())


def _apply_substring_fallback(
        normalised: str,
        rules: List[Tuple[str, str]],
    ) -> Optional[str]:
    """Return the first canonical name whose token is a substring of
    the normalised label, or None if no rule matches. Rules are checked
    in the order they appear in `SUBSTRING_FALLBACKS` (caller decides
    specificity ordering)."""
    for token, canonical in rules:
        if token in normalised:
            return canonical
    return None


def _resolve_obs_key_for_section(
        adata: ad.AnnData,
        candidate_keys: List[str],
    ) -> Optional[str]:
    """Return the FIRST key in `candidate_keys` that exists in
    `adata.obs.columns` AND has at least one non-NaN entry. Returns
    None if no candidate is populated.

    Used to dispatch per-section: e.g. the niche label lives in
    `obs["ccf_region_name"]` for one section and
    `obs["Sub_molecular_tissue_region"]` for another, so the caller
    passes `["niche", "Sub_molecular_tissue_region", "ccf_region_name"]`
    and this helper picks whichever is actually populated for THIS
    section.
    """
    for k in candidate_keys:
        if k in adata.obs.columns and adata.obs[k].notna().any():
            return k
    return None


def build_canonical_map(
        sections: List[Tuple[str, str, ad.AnnData]],
        obs_keys: List[str],
        synonyms: Dict[str, str],
        substring_fallbacks: List[Tuple[str, str]] = SUBSTRING_FALLBACKS,
    ) -> Tuple[Dict[str, str], List[str]]:
    """Walk each section's labels and build:
      - raw_to_canonical : {raw_label_string -> canonical_display_name}
      - canonical_list   : alphabetically-sorted unique canonical names

    `obs_keys` is a fallback list of obs columns to try per section
    (first one populated wins). This lets the same call handle
    sections that use different label-column names (e.g.
    `Sub_molecular_tissue_region` vs `ccf_region_name`) — same
    canonical name space, same colour map.

    Three-level fallback per raw label, in order:
      1. Exact match on the normalised key in `synonyms`.
      2. Substring match against `substring_fallbacks` (catches the
         MERFISH-style "01_IT-ET Glut" → contains "glut" → Excitatory
         neuron convention).
      3. No match → sentence-case the original (capitalise first
         letter only) so every canonical follows the same formatting.
    """
    raw_to_canonical: Dict[str, str] = {}
    for _, _, a in sections:
        key = _resolve_obs_key_for_section(a, obs_keys)
        if key is None:
            continue
        for raw in pd.unique(a.obs[key].astype(str).to_numpy()):
            if raw in raw_to_canonical:
                continue
            norm = _normalise_label(raw)
            canonical = synonyms.get(norm)
            if canonical is None:
                canonical = _apply_substring_fallback(norm, substring_fallbacks)
            if canonical is None:
                stripped = raw.strip().strip(string.punctuation).strip()
                stripped = re.sub(r"\s+", " ", stripped)
                if not stripped:
                    canonical = raw.strip()
                else:
                    canonical = stripped[0].upper() + stripped[1:].lower()
            raw_to_canonical[raw] = canonical
    canonical_list = sorted(set(raw_to_canonical.values()))
    return raw_to_canonical, canonical_list


# `build_colour_map` removed — replaced by two specialised builders:
#   * build_cell_type_colour_map   (category-grouped HLS shading)
#   * build_niche_colour_map       (godsnot, distinct per brain region)
# Both are defined further up in the module (see CATEGORY_BASE_HLS).


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _figsize_for_section(
        adata: ad.AnnData,
        max_panel_in: float = 5.0,
        legend_w_in: float = 1.6,
        with_legend: bool = True,
    ) -> Tuple[float, float]:
    """Lay out a single-panel figure at preserved aspect ratio."""
    c = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    x_range = float(c[:, 0].max() - c[:, 0].min())
    y_range = float(c[:, 1].max() - c[:, 1].min())
    if y_range <= 0 or x_range <= 0:
        return max_panel_in + (legend_w_in if with_legend else 0.0), max_panel_in
    aspect = x_range / y_range
    h = max_panel_in
    w = min(max_panel_in * 2.0, h * aspect)
    fig_w = w + (legend_w_in if with_legend else 0.0) + 0.2
    fig_h = h + 0.5
    return fig_w, fig_h


def _plot_section_scatter(
        ax: plt.Axes,
        coords: np.ndarray,
        colour_per_cell: np.ndarray,
        title: str,
        marker_size: float,
    ) -> None:
    ax.scatter(
        coords[:, 0], coords[:, 1],
        c=colour_per_cell,
        s=marker_size,
        linewidths=0.0,
        rasterized=True,
        zorder=2,
    )
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=8, fontweight="medium", pad=4)


def _format_legend(
        fig: plt.Figure,
        canonical_to_colour: Dict[str, str],
        title: str,
        max_entries: int = 60,
        overflow_line: Optional[str] = None,
        preserve_input_order: bool = False,
    ) -> None:
    """Render the legend.

    Parameters
    ----------
    canonical_to_colour : dict
        Already-filtered subset to show. Caller is responsible for
        truncating + computing overflow if relevant.
    overflow_line : str, optional
        If given, appended as a final entry (e.g. "… +12 more").
        Renders in neutral grey to signal "not the same as a real
        colour entry".
    preserve_input_order : bool
        When True, render entries in the iteration order of
        `canonical_to_colour` (intended for frequency-sorted dicts
        coming from the caller). When False (default), sort
        alphabetically.
    """
    if preserve_input_order:
        labels = list(canonical_to_colour.keys())
    else:
        labels = sorted(canonical_to_colour.keys(), key=lambda s: s.lower())
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="",
                   color=canonical_to_colour[lbl], markeredgewidth=0,
                   markersize=5, label=lbl)
        for lbl in labels
    ]
    if overflow_line is None and len(handles) > max_entries:
        # Caller didn't pre-truncate; legacy in-helper truncation path.
        overflow_line = f"… +{len(labels) - max_entries} more"
        handles = handles[:max_entries]
    if overflow_line is not None:
        handles.append(
            plt.Line2D([0], [0], marker="o", linestyle="",
                       color="#dddddd", markeredgewidth=0, markersize=5,
                       label=overflow_line)
        )
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.99, 0.5),
        frameon=False,
        fontsize=6.5,
        title=title,
        title_fontsize=7,
        labelspacing=0.4,
        handletextpad=0.4,
        borderaxespad=0.0,
    )


def render_categorical_per_section(
        section_title: str,
        section_slug: str,
        adata: ad.AnnData,
        obs_keys: List[str],
        raw_to_canonical: Dict[str, str],
        canonical_to_colour: Dict[str, str],
        out_dir: Path,
        dataset_tag: str,
        label_kind: str,
        marker_size: float,
        max_legend_entries: Optional[int] = None,
        legend_sort_by: str = "alphabetical",
    ) -> bool:
    """Render ONE figure for ONE section, coloured by whichever of
    `obs_keys` is populated for this section (first one wins). Uses
    the shared canonical→colour map so sections with different niche
    column names still get harmonised colours.

    Legend control:
      - `legend_sort_by="alphabetical"` (default) — list canonicals in
        alphabetical order.
      - `legend_sort_by="frequency"` — order by descending cell count
        in THIS section. Combined with `max_legend_entries`, this
        shows only the top-N most populous niches/cell-types and adds
        a "… +K more" overflow line. The MAP itself still includes
        every cell, only the legend is truncated.
    """
    key = _resolve_obs_key_for_section(adata, obs_keys)
    if key is None:
        print(f"    skip (none of {obs_keys!r} populated in obs of "
              f"{section_title!r})")
        return False
    print(f"    using obs[{key!r}] (out of candidates {obs_keys})")

    raw = adata.obs[key].astype(str).to_numpy()
    # Build canonical per cell, then map to colour.
    canon_per_cell = np.array(
        [raw_to_canonical.get(r, r.strip()) for r in raw],
        dtype=object,
    )
    colours_per_cell = np.array(
        [canonical_to_colour.get(c, "#cccccc") for c in canon_per_cell],
        dtype=object,
    )

    fig_w, fig_h = _figsize_for_section(adata, with_legend=True)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    _plot_section_scatter(
        ax, coords, colours_per_cell, title=section_title,
        marker_size=marker_size,
    )

    # Legend lists canonicals that are actually present in this section.
    # When `legend_sort_by == "frequency"`, also drop the long tail so
    # only the most populous N labels appear (e.g. top 33 niches).
    from collections import Counter
    canon_counts = Counter(canon_per_cell.tolist())
    if legend_sort_by == "frequency":
        # Sort canonicals descending by cell count, then alphabetically
        # for ties so the output is deterministic across reruns.
        present = sorted(
            canon_counts.keys(),
            key=lambda c: (-canon_counts[c], c.lower()),
        )
    else:
        present = sorted(canon_counts.keys(), key=lambda s: s.lower())

    n_total = len(present)
    if max_legend_entries is not None and n_total > max_legend_entries:
        kept = present[:max_legend_entries]
        n_dropped = n_total - max_legend_entries
        section_colour_map = {c: canonical_to_colour[c] for c in kept}
        legend_overflow = f"… +{n_dropped} more"
        print(f"    legend capped to top {max_legend_entries} of "
              f"{n_total} (by {legend_sort_by}); +{n_dropped} hidden")
    else:
        section_colour_map = {c: canonical_to_colour[c] for c in present}
        legend_overflow = None

    _format_legend(
        fig, section_colour_map, title=label_kind.replace("_", " "),
        overflow_line=legend_overflow,
        preserve_input_order=(legend_sort_by == "frequency"),
    )

    fig.subplots_adjust(left=0.02, right=0.78, top=0.92, bottom=0.05)

    base = out_dir / f"{dataset_tag}__{section_slug}__{label_kind}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        out = base.with_suffix(f".{ext}")
        fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
        print(f"    -> {out}")
    plt.close(fig)
    return True


def render_tissue_per_section(
        section_title: str,
        section_slug: str,
        adata: ad.AnnData,
        out_dir: Path,
        dataset_tag: str,
        marker_size: float,
        tissue_grey: str = "#888888",
    ) -> None:
    """Render ONE all-grey anatomy figure for ONE section."""
    fig_w, fig_h = _figsize_for_section(adata, with_legend=False)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    colours = np.full(adata.n_obs, tissue_grey, dtype=object)
    _plot_section_scatter(
        ax, coords, colours, title=section_title, marker_size=marker_size,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.05)

    base = out_dir / f"{dataset_tag}__{section_slug}__tissue"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        out = base.with_suffix(f".{ext}")
        fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
        print(f"    -> {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _print_mapping_summary(
        obs_keys: List[str],
        sections: List[Tuple[str, str, ad.AnnData]],
        raw_to_canonical: Dict[str, str],
    ) -> None:
    """Print the raw→canonical mapping per section so the user can
    inspect what got merged. `→` marks a synonym/substring rewrite;
    `≡` marks "only formatting changed" (sentence-case)."""
    print(f"\n  Label harmonisation for obs[{obs_keys!r}]:")
    for title, _, a in sections:
        key = _resolve_obs_key_for_section(a, obs_keys)
        if key is None:
            print(f"    [{title}]  (none of {obs_keys} populated)")
            continue
        uniq = pd.unique(a.obs[key].astype(str).to_numpy())
        uniq_sorted = sorted(uniq, key=lambda s: s.lower())
        print(f"    [{title}]  using obs[{key!r}] — {len(uniq_sorted)} unique label(s):")
        for r in uniq_sorted:
            canon = raw_to_canonical.get(r, r)
            # Sentence-case version of the raw — matches what
            # `build_canonical_map` produces when no synonym/substring
            # rule matched. If the canonical equals this, only the
            # FORMATTING changed (≡). Otherwise a synonym or
            # substring rule fired (→).
            r_stripped = re.sub(r"\s+", " ", r.strip().strip(string.punctuation).strip())
            r_titled = (r_stripped[0].upper() + r_stripped[1:].lower()) if r_stripped else r.strip()
            arrow = "≡" if canon == r_titled else "→"
            print(f"      {r!r:30s} {arrow} {canon!r}")


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset-tag", type=str, default=DEFAULT_DATASET_TAG)
    p.add_argument("--silver-dir", type=Path, default=None,
                   help="Default: <silver_root>/<dataset_tag>/")
    p.add_argument("--silver-root", type=Path, default=DEFAULT_SILVER_ROOT)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Default: <out_root>/<dataset_tag>/")
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    # Each label flag accepts a comma-separated LIST of obs columns to
    # try, in priority order — per-section, the first one that's
    # populated wins. This handles the mmb-smb case where the two
    # sections use different niche column names:
    #   - MERFISH section:  obs["ccf_region_name"]
    #   - STARmap+ section: obs["Sub_molecular_tissue_region"]
    # Both end up in the SAME canonical/colour map, so e.g. "Cortex"
    # vs "Isocortex" still get the same blue.
    p.add_argument("--cell-label-keys", type=str, default="cell_type,cell_types",
                   help="Comma-separated obs columns to try for the "
                        "cell-type label (first populated wins per "
                        "section).")
    p.add_argument("--niche-label-keys", type=str,
                   default="niche,Sub_molecular_tissue_region,ccf_region_name",
                   help="Comma-separated obs columns to try for the "
                        "niche label. Defaults match the conventions "
                        "used in mmb-smb (Sub_molecular_tissue_region "
                        "for STARmap+, ccf_region_name for MERFISH).")
    p.add_argument("--marker-size", type=float, default=3.0,
                   help="Scatter marker size in pt^2. Default 3.0. "
                        "Tuned for ~50k-cell sections; bump down to "
                        "~1.5 if you see heavy overplotting on denser "
                        "tissues, or up to ~5-6 if cells look too "
                        "small in the rendered PNG.")
    p.add_argument("--niche-legend-top-n", type=int, default=33,
                   help="Cap the niche-figure legend to the N most "
                        "populous niches in that section (rest go into "
                        "a '… +K more' overflow row). The MAP still "
                        "shows every cell; only the legend is "
                        "truncated. Default 33.")
    p.add_argument("--synonyms-json", type=Path, default=None,
                   help="Optional JSON file overriding/extending the "
                        "built-in SYNONYMS table. Format: "
                        "{<normalised_label>: <canonical_name>, ...}.")
    args = p.parse_args(argv)

    if args.silver_dir is None:
        args.silver_dir = args.silver_root / args.dataset_tag
    if args.out_dir is None:
        args.out_dir = args.out_root / args.dataset_tag

    # Parse the comma-separated key lists. Empty entries are dropped so
    # users can write "niche,,ccf_region_name" without weirdness.
    cell_keys = [k.strip() for k in args.cell_label_keys.split(",") if k.strip()]
    niche_keys = [k.strip() for k in args.niche_label_keys.split(",") if k.strip()]
    if not cell_keys:
        raise SystemExit("--cell-label-keys parsed to empty list.")
    if not niche_keys:
        raise SystemExit("--niche-label-keys parsed to empty list.")

    # Load optional synonyms override.
    synonyms = dict(SYNONYMS)
    if args.synonyms_json is not None:
        if not args.synonyms_json.is_file():
            raise SystemExit(f"--synonyms-json={args.synonyms_json} not found.")
        with open(args.synonyms_json) as f:
            extra = json.load(f)
        if not isinstance(extra, dict):
            raise SystemExit("--synonyms-json must be a JSON object {str: str}.")
        synonyms.update({_normalise_label(k): v for k, v in extra.items()})
        print(f"Loaded {len(extra)} additional synonyms from {args.synonyms_json}")

    _apply_nature_style()

    print(f"Dataset tag : {args.dataset_tag}")
    print(f"Silver dir  : {args.silver_dir}")
    print(f"Output dir  : {args.out_dir}")
    print(f"Marker size : {args.marker_size}")
    print(f"Cell-label keys (priority): {cell_keys}")
    print(f"Niche-label keys (priority): {niche_keys}")
    print()
    sections = load_sections(args.silver_dir)

    # --- Build canonical maps + colour maps ----------------------------
    cell_raw_to_canon, cell_canonicals = build_canonical_map(
        sections, obs_keys=cell_keys, synonyms=synonyms,
    )
    niche_raw_to_canon, niche_canonicals = build_canonical_map(
        sections, obs_keys=niche_keys, synonyms=synonyms,
    )

    # Two SEPARATE colour maps. Cell types get category-grouped shading
    # (all neurons in warm hues, all glia in cool blues, etc.); niches
    # use the developmental-region grouping (forebrain/diencephalon/...).
    cell_colour_map = build_cell_type_colour_map(cell_canonicals)
    niche_colour_map = build_niche_colour_map(niche_canonicals)

    _print_mapping_summary(cell_keys, sections, cell_raw_to_canon)
    _print_mapping_summary(niche_keys, sections, niche_raw_to_canon)

    # --- Render per-section figures ------------------------------------
    print()
    print("Rendering per-section figures:")
    for title, slug, a in sections:
        print(f"  Section: {title}  (slug={slug})")
        # cell-type
        render_categorical_per_section(
            section_title=title, section_slug=slug, adata=a,
            obs_keys=cell_keys,
            raw_to_canonical=cell_raw_to_canon,
            canonical_to_colour=cell_colour_map,
            out_dir=args.out_dir,
            dataset_tag=args.dataset_tag,
            label_kind="cell_type",
            marker_size=args.marker_size,
        )
        # niche — legend capped to the top-N most populous niches in
        # this section (the long tail of small region labels would
        # otherwise overflow the legend).
        render_categorical_per_section(
            section_title=title, section_slug=slug, adata=a,
            obs_keys=niche_keys,
            raw_to_canonical=niche_raw_to_canon,
            canonical_to_colour=niche_colour_map,
            out_dir=args.out_dir,
            dataset_tag=args.dataset_tag,
            label_kind="niche",
            marker_size=args.marker_size,
            max_legend_entries=args.niche_legend_top_n,
            legend_sort_by="frequency",
        )
        # tissue (anatomy view, grey, no labels)
        render_tissue_per_section(
            section_title=title, section_slug=slug, adata=a,
            out_dir=args.out_dir,
            dataset_tag=args.dataset_tag,
            marker_size=args.marker_size,
        )

    print()
    print("DONE.")


if __name__ == "__main__":
    sys.exit(main())
