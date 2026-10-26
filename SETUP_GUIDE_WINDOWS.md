# Running the Cyberattack Detection Project on Your Windows Laptop

A complete beginner's guide. Follow the steps in order. Total time: ~15 minutes
(plus a few minutes for the model to train).

---

## What you'll end up with

After these steps, running one command produces:
- Real detection results printed in your terminal (macro F1, per-class scores)
- `outputs/alerts.json` — the explained + ranked alerts
- `outputs/dashboard.html` — the interactive dashboard you open in a browser

Everything runs on YOUR laptop and stays there permanently.

---

## STEP 1 — Put the project folder somewhere sensible

Unzip the project package (the folder containing `src`, `data`, `dashboard`,
etc.) and put it somewhere easy to reach, for example:

    C:\Users\<YourName>\Documents\cyberattack-project

Avoid folders synced by OneDrive if you hit permission issues later, but it's
usually fine.

---

## STEP 2 — Put the NSL-KDD data files in place

1. Find `archive.zip` (the Kaggle NSL-KDD download you already have).
2. Unzip it.
3. Copy these two files into the project's `data` folder:
   - `KDDTrain+.txt`
   - `KDDTest+.txt`

After this, your `data` folder should contain those two `.txt` files.
The names must match exactly, including the `+`.

---

## STEP 3 — Check Python is installed

You said you have VS Code. Let's confirm Python works.

1. Open the project folder in VS Code: **File > Open Folder** > pick the project folder.
2. Open a terminal inside VS Code: **Terminal > New Terminal** (or press `` Ctrl+` ``).
3. In that terminal, type:

       python --version

   - If you see something like `Python 3.11.x` — great, skip to Step 4.
   - If you see an error or "not recognized", install Python:
     - Go to https://www.python.org/downloads/
     - Download the latest Windows installer.
     - **IMPORTANT:** on the first screen of the installer, tick the box
       **"Add python.exe to PATH"** before clicking Install. This is the
       single most common thing beginners miss.
     - After installing, CLOSE and REOPEN VS Code, then run `python --version` again.

---

## STEP 4 — Install the required packages

In the same VS Code terminal, run:

    pip install -r requirements.txt

This installs pandas, numpy, and scikit-learn. Wait for it to finish
(you'll see "Successfully installed ..."). If `pip` isn't recognized, try:

    python -m pip install -r requirements.txt

---

## STEP 5 — Run the whole project

Still in the terminal, from the project root folder, run:

    python src/run_pipeline.py

You'll see it work through 8 steps:
- Step 3: preprocessing the data
- Step 4: training two models and picking the best (this takes the longest,
  maybe 2-5 minutes depending on your laptop — be patient)
- Step 5: calibrating confidence
- Step 6: explaining 40 alerts
- Step 7: ranking them
- Step 8: building the dashboard

When it's done, it prints a results summary and the path to your dashboard.

---

## STEP 6 — Open your dashboard

Open the file `outputs/dashboard.html` in any browser:
- In VS Code's file explorer (left panel), find `outputs/dashboard.html`,
  right-click it > **Reveal in File Explorer**, then double-click it.
- Or just double-click `dashboard.html` in Windows File Explorer.

Click any alert to see its explanation, confidence, priority breakdown,
and features. This is what you screenshot for your presentation.

---

## What each file does (so you understand your own project)

| File | What it does | Slide / Step |
|---|---|---|
| `src/preprocess_nslkdd.py` | Cleans data, groups 39 attacks into 5 families, encodes + scales | Step 3 |
| `src/train_nslkdd.py` | Trains Random Forest vs HistGradientBoosting, picks best | Step 4 |
| `src/calibrate_nslkdd.py` | Makes confidence scores trustworthy (ECE) | Step 5 |
| `src/explain_nslkdd.py` | SHAP-style per-alert explanations | Step 6 |
| `src/prioritize_nslkdd.py` | Ranks alerts by danger | Step 7 |
| `src/run_pipeline.py` | Runs ALL of the above + builds dashboard | (the one you run) |
| `dashboard/dashboard_template.html` | The dashboard design (data gets injected) | Step 8 |
| `outputs/` | Where results land after you run | — |

---

## Optional: use the REAL SHAP library (nice for your report)

The project uses a from-scratch SHAP implementation so it runs anywhere.
On your laptop you can optionally install the real, industry-standard library:

    pip install shap

If you want to switch to it, tell me and I'll give you the ~5-line edit.
It's a good thing to mention in your viva ("I used the production SHAP library").

---

## Troubleshooting

**"python is not recognized"** — Python isn't on PATH. Reinstall Python with
the "Add to PATH" box ticked (Step 3), and reopen VS Code.

**"No module named pandas"** — the pip install didn't run or didn't finish.
Re-run Step 4. If it still fails, try `python -m pip install pandas numpy scikit-learn`.

**"FileNotFoundError: KDDTrain+.txt"** — the data files aren't in the `data`
folder, or the names don't match exactly. Recheck Step 2. The `+` matters.

**It seems frozen on Step 4** — it's not frozen, it's training on 126,000
records. Give it up to 5 minutes. You'll know it's done when Step 5 appears.

**Want it faster?** Tell me and I'll give you a "quick mode" that trains on a
smaller sample for testing, then the full run for your final results.
