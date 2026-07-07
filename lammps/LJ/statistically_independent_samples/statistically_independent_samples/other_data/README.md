# Additional q_l/q_ls-hard datasets

This directory holds local datasets that are useful stress tests for classical
Steinhardt-style q_l/q_ls descriptors. The data payloads are intentionally
ignored by git; only this README and `download_q_ls_hard_data.py` should be
versioned.

## What Is Here

`martirossyan_complex_crystals/`

Downloaded from the Materials Data Facility / ACDC Globus endpoint for
Martirossyan et al., "Local structural features elucidate crystallization of
complex structures". The dataset contains one-component isotropic-particle
self-assembly trajectories for complex crystals. The transferred files include
one `dump.gsd` trajectory per structure folder, `dump.log`, `training.json`, and
signac metadata.

`noisy_simple_crystals/`

Locally generated noisy FCC/HCP/BCC/SC/diamond structures from
`download_q_ls_hard_data.py generate-noisy-crystals`.

`gispen_lj_nucleation/`

Reserved for the Gispen et al. Lennard-Jones nucleation data if a stable
trajectory archive or journal supplement download endpoint becomes available.
As of the last check, the arXiv source bundle contains only manuscript source
and figures, not the nucleation trajectories.

## Inspection Notebooks

`inspect_martirossyan_complex_crystals.ipynb`

Visualizes sampled frames from the Martirossyan GSD trajectories, summarizes the
structure folders, computes sampled per-particle q_l features, plots q4/q6 and
PCA views, and trains a small random-forest baseline to quantify how well q_l
separates the different complex-crystal folders.

`inspect_noisy_simple_crystals.ipynb`

Visualizes the generated noisy simple-crystal NPZ file, computes sampled
per-particle q_l features, plots q4/q6 and PCA views, and trains a small
random-forest baseline with an additional accuracy-vs-noise plot.

Both notebooks require `freud`; the Martirossyan notebook also requires `gsd`.
Use the Python environment where the existing particle-data analysis stack is
installed.

## Martirossyan Globus Download

The Martirossyan dataset is not a plain HTTP download. The public MDF/ACDC
record exposes it through Globus:

```text
source endpoint: 82f1b5c6-6e9b-11e5-ba47-22000b92c6ec
source path:     /mdf_open/e52f77de-6756-4ca9-8fdb-f4791b395c1f/1.0/
```

Step-by-step:

1. Install the Globus CLI if needed.

```bash
python3 -m pip install globus-cli
```

2. Log in to Globus.

```bash
globus login
```

3. Install Globus Connect Personal on the laptop/workstation that should hold
   the data.

```bash
cd /tmp
wget https://downloads.globus.org/globus-connect-personal/linux/stable/globusconnectpersonal-latest.tgz
tar xzf globusconnectpersonal-latest.tgz
cd globusconnectpersonal-*
```

4. Configure the personal endpoint. The no-GUI setup is the most reliable option
   on headless or remote Linux sessions.

```bash
./globusconnectpersonal -setup
```

5. Start the personal endpoint without the GUI.

```bash
./globusconnectpersonal -start &
```

6. Get the UUID for the local personal endpoint.

```bash
globus endpoint local-id --personal
```

7. From the repository root, submit the transfer. Replace `<LOCAL_ENDPOINT_UUID>`
   with the UUID from the previous command.

```bash
cd /home/jhausberger/Masterthesis/Repos/autoencode_statmech

python3 lammps/LJ/statistically_independent_samples/statistically_independent_samples/other_data/download_q_ls_hard_data.py submit-martirossyan-globus \
  --destination-endpoint <LOCAL_ENDPOINT_UUID> \
  --destination-path /~/Masterthesis/Repos/autoencode_statmech/lammps/LJ/statistically_independent_samples/statistically_independent_samples/other_data/martirossyan_complex_crystals/
```

8. Confirm the data arrived.

```bash
du -sh lammps/LJ/statistically_independent_samples/statistically_independent_samples/other_data/martirossyan_complex_crystals

find lammps/LJ/statistically_independent_samples/statistically_independent_samples/other_data/martirossyan_complex_crystals \
  -maxdepth 2 -name dump.gsd | wc -l
```

The current successful transfer contains 11 `dump.gsd` files and is about 56 MB.

9. Stop Globus Connect Personal after the transfer if it is no longer needed.

```bash
./globusconnectpersonal -stop
```

## Gispen LJ Nucleation Candidate

The Gispen et al. paper is scientifically relevant because it shows that the
polymorph composition of Lennard-Jones nuclei changes strongly with the local
structure detector. However, no stable public trajectory download URL has been
found yet.

This downloads the arXiv source bundle only:

```bash
mkdir -p lammps/LJ/statistically_independent_samples/statistically_independent_samples/other_data/gispen_lj_nucleation/arxiv_source

curl -L \
  https://arxiv.org/e-print/2412.03276 \
  -o lammps/LJ/statistically_independent_samples/statistically_independent_samples/other_data/gispen_lj_nucleation/arxiv_source/2412.03276.tar

tar -tzf lammps/LJ/statistically_independent_samples/statistically_independent_samples/other_data/gispen_lj_nucleation/arxiv_source/2412.03276.tar | sed -n '1,120p'
```

That archive is useful for checking the manuscript source, but it does not
provide the LJ nucleation trajectories. If a journal supplement, Zenodo record,
GitHub repository, or institutional archive is later found, add its direct
download command here and mirror it in `download_q_ls_hard_data.py`.
