import pandas as pd

amp = pd.read_csv("amplitude_table_veca_rmin10.0.csv")
master = pd.read_csv("runsW_master.csv")   # Adapt file name here
master.columns = master.columns.str.strip() # Remove blank space
print("MASTER COLUMNS:", list(master.columns)[:30])

# Only take W suite
master["Runs"] = master["Runs"].astype(str).str.strip()
master = master[master["Runs"].eq("W")].copy()

# Align Run -> run 
master = master.rename(columns={"Run": "run"})

#cols = ["run", "Omega_b", "little_h", "n_s", "m_nu"]
cols = ["run", "Omega_m", "sigma_8", "w0", "wa", "Omega_b", "little_h", "n_s", "m_nu"]
missing = [c for c in cols if c not in master.columns]
if missing:
    raise SystemExit("Missing columns in master: " + ", ".join(missing))

out = amp.merge(master[cols], on="run", how="left")

out.to_csv("amplitude_table_veca_rmin10.0_with_cosmo.csv", index=False)

print("Wrote amplitude_table_veca_rmin10.0_with_cosmo.csv")
print("Unique runs in amp:", amp["run"].nunique())
print("Unique runs matched:", out.loc[out["Omega_b"].notna(), "run"].nunique())
