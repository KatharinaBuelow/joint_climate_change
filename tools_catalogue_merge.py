#!/usr/bin/env python3
# tools for merging and plotting the matrix of the combination of different catalogues

import intake
import pandas as pd
import os

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap



#-------------------------
# plotting
#--------------------------

def plot_cmip6_heatmaps(df, plot_dir, selected_variants, ext_plot_name):

    from matplotlib.colors import ListedColormap
    
    # Plotting the heatmaps for the CMIP6 downscaling plans, 
    # Catalogue in Jülich and optionally available at esgf
   
    # Kombination für x-Achse erzeugen
    df["gcm_variant"] = (
        df["driving_source_id"]
        + "\n"
        + df["driving_variant_label"]
    )

    # yes/no in 1/0 umwandeln
    df["value"] = df["in_catalog"].map({
        "yes": 2,
        "no": 1
    })
    if "available at esgf" in df.columns:
        esgf_yes = df["available at esgf"].eq("yes")
        df.loc[esgf_yes & df["in_catalog"].eq("no"), "value"] = 3
        df.loc[esgf_yes & df["in_catalog"].eq("yes"), "value"] = 4

    # Colors:

    cmap = ListedColormap([
        "lightgrey",  # 0 = not planned
      "#08306B",    # 1 = planned, unavailable in both
      "#CB181D",    # 2 = available in catalog only
      "#F4A261",    # 3 = available at ESGF only
      "#2A9D8F",    # 4 = available in both
    ])

    for experiment in sorted(df["driving_experiment_id"].unique()):

        sub = df[df["driving_experiment_id"] == experiment]

        heatmap_df = sub.pivot_table(
            index="gcm_variant",
            columns="source_id",
            values="value",
            aggfunc="max",
            fill_value=0
        )

        plt.figure(
            figsize=(
                max(12, len(heatmap_df.columns) * 0.4),
                max(6, len(heatmap_df.index) * 0.6)
            )
        )

        sns.heatmap(
            heatmap_df,
            cmap=cmap,
            cbar=False,
            linewidths=0.5,
            linecolor="white",
            vmin=0,
            vmax=4
        )
        if experiment in ["historical", "ssp370"]:  # example of multiple experiments to highlight
            ax = plt.gca()
            for label in ax.get_yticklabels():
                if label.get_text() in selected_variants:
                    label.set_color("red")
        if experiment in ["historical"]:
            title = f"{experiment} (also available: CMCC-CM2-SR5_historical_r1i1p1f1_TASK-IMWM_WRF451Q_v1, but EUR-12f)"
        else:
            title = f"{experiment}"
        plt.title(title)
        plt.ylabel("Driving GCM + Variant (red: selected GCMs balanced matrix)")
        plt.xlabel("Source ID")

        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="lightgrey", label="not planned"),
            Patch(facecolor="#08306B", label="planned, unavailable in both"),
            Patch(facecolor="#CB181D", label="available in catalog only"),
            Patch(facecolor="#F4A261", label="available at ESGF only"),
            Patch(facecolor="#2A9D8F", label="available in both"),
        ]

        plt.legend(
            handles=legend_elements,
            bbox_to_anchor=(1.02, 1),
            loc="upper left"
        )

        plt.tight_layout()

        plt.savefig(
            f"{plot_dir}/heatmap_{experiment}_{ext_plot_name}.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.show()

#---------------------------------------------------------------------
# create balanced matrix of data frame df 
#---------------------------------------------------------------------

def create_balanced_matrix(df):
    # create balanced matrix of Simulationschecked in catalogue and esgf

    # only balance matrix
    df_eurbalanced = df.loc[
        df["comments"].str.contains("#EURbalanced", na=False, regex=False)
    ].copy()

    df_eurbalanced['driving_source_id'].unique()

    return df_eurbalanced

#---------------------------------------------------------------------
# Just for fancy plots, I like to high light the simulations of the balanced matrix
# or you also have only the simulations from the balance matrix
#---------------------------------------------------------------------
def get_selected_variants(df_eurbalanced):
    df_eurbalanced["gcm_variant"] = (
        df_eurbalanced["driving_source_id"]
        + "\n"
        + df_eurbalanced["driving_variant_label"]
    )
    selected_variants = df_eurbalanced["gcm_variant"].unique()
    return selected_variants


#---------------------------------------------------------------------
# comparison
#---------------------------------------------------------------------
def create_combined_matrix(df,cdf,match_cols,required_vars):
    
    # Combine the original dataframe with the catalog information
    # ------------------------------------------------------------------
    # Only consider daily data
    # ------------------------------------------------------------------

    cdf_day = cdf[cdf["frequency"] == "day"]

    available = (
        cdf_day.groupby(match_cols)["variable_id"]
        .agg(lambda x: set(x))
        .reset_index()
    )

    available["in_catalog"] = available["variable_id"].apply(
        lambda x: required_vars.issubset(set(x))
    )

    available["missing_variables"] = available["variable_id"].apply(
        lambda x: ",".join(sorted(required_vars - set(x)))
    )

    # ------------------------------------------------------------------
    # Merge with the original CSV
    # ------------------------------------------------------------------
    
    df = df.merge(
        available,
        on=match_cols,
        how="left"
    )

    df["in_catalog"] = df["in_catalog"].fillna(False)
    df["missing_variables"] = df["missing_variables"].fillna(
        "combination_not_found"
    )
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Convert True/False to yes/no
    # ------------------------------------------------------------------

    df["in_catalog"] = df["in_catalog"].map(
        {True: "yes", False: "no"}
    )

    return df

# ------------------------------------------------------------------
#  comparison with esgf
# ------------------------------------------------------------------

def compare_with_esgf(df, esgf_df, match_cols, required_vars):

# Only consider daily data from the ESGF search
    esgf_day = esgf_df[esgf_df["frequency"] == "day"]

    esgf_available = (
           esgf_day.groupby(match_cols)["variable_id"]
           .agg(
               lambda values: {
                   variable.strip()
                   for value in values.dropna()
                   for variable in str(value).split(",")
                   if variable.strip()
               }
           )
           .reset_index()
           .rename(columns={"variable_id": "esgf_variables"})
       )
    esgf_available["available at esgf"] = esgf_available["esgf_variables"].apply(
        lambda variables: required_vars.issubset(variables)
        )

    # Are all required variables available?
    esgf_available["available at esgf"] = esgf_available["esgf_variables"].apply(
        lambda x: required_vars.issubset(x)
        )

    # Which variables are missing at ESGF?
    esgf_available["missing_variables_esgf"] = esgf_available["esgf_variables"].apply(
        lambda x: ",".join(sorted(required_vars - x))
        )

    # Merge with the planned simulations
    df = df.merge(
        esgf_available.drop(columns="esgf_variables"),
        on=match_cols,
        how="left"
    )

    # Combination not found at ESGF
    df["available at esgf"] = df["available at esgf"].fillna(False)
    df["missing_variables_esgf"] = df["missing_variables_esgf"].fillna(
        "combination_not_found"
    )

    # Convert True/False to yes/no
    df["available at esgf"] = df["available at esgf"].map({True: "yes", False: "no"})

    
    return df

