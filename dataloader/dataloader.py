import pandas as pd
import os

class FedBatchDataLoader:
    def __init__(self, filepath, sheet_name, ethanol_to_glucose_factor=1.2):
        self.filepath = filepath
        self.sheet_name = sheet_name
        self.ethanol_to_glucose_factor = ethanol_to_glucose_factor
        self.groups = {}
        self.cleaned_groups = {}
        self.trimmed = {}   # label -> number of leading rows dropped (no DW)

    def load(self):
        df = pd.read_excel(self.filepath, sheet_name=self.sheet_name, header=1)
        df = df.dropna(subset=["Experiment Label"])
        self.groups = {label: group for label, group in df.groupby("Experiment Label")}
        return self.groups

    def clean(self):
        cleaned = {}
        for label, df in self.groups.items():
            df = df[df["Time (h)"] != 0].copy()
            df["Time (h)"] = pd.to_numeric(df["Time (h)"], errors="coerce")
            df["Time (h)"] = df["Time (h)"] - df["Time (h)"].min()
            cleaned[label] = df.reset_index(drop=True)
        self.cleaned_groups = cleaned
        self._handle_edge_cases()
        self._add_glucose_equivalent()
        return self.cleaned_groups

    def _handle_edge_cases(self):
        """Drop leading rows with no biomass measurement.

        The ODE is initialised from the first row (X0 = DW at t0), so a run
        whose first samples have no dry-weight reading cannot be simulated.
        This affects VR5-VR7 (2 leading rows) and M1-M4 (2-3), so the trim is
        applied generically rather than to a hardcoded list. The trailing feed
        row also has no DW but is never touched, since only leading rows are
        removed. Time is re-zeroed on the new first sample.
        """
        for label, df in list(self.cleaned_groups.items()):
            if "DW (g/L)" not in df.columns:
                continue
            dw = pd.to_numeric(df["DW (g/L)"], errors="coerce")
            if dw.isna().empty or not dw.isna().iloc[0]:
                continue
            first_valid = dw.notna().idxmax() if dw.notna().any() else None
            if first_valid is None or first_valid == 0:
                continue
            n_pos = df.index.get_loc(first_valid)
            df = df.iloc[n_pos:].copy()
            df["Time (h)"] = df["Time (h)"] - df["Time (h)"].iloc[0]
            self.cleaned_groups[label] = df.reset_index(drop=True)
            self.trimmed[label] = n_pos

    def _add_glucose_equivalent(self):
        for label, df in self.cleaned_groups.items():
            if "Glucose" in df.columns and "Ethanol" in df.columns:
                df["Glu_eq"] = df["Glucose"] + df["Ethanol"] * self.ethanol_to_glucose_factor
            else:
                print(f"Skipping {label}: 'Glucose' or 'Ethanol' column missing.")

    def get_group(self, label):
        return self.cleaned_groups.get(label, None)
