# 10-minute demo script

Before the talk, run `.venv\Scripts\streamlit run app\dashboard.py` from `website/` and open http://localhost:8501.
Defaults: **Stale + unreliable**, countermeasure **off**, latest time window, Houston / Galveston.

The whole demo happens on the one map. Each dot is a predicted position 15 min ahead, colored by how far off it was.
The line runs to where the vessel's next AIS report actually put it.

| time | segment | what to do / say |
|---|---|---|
| 0:00–1:00 | Problem | A harbor analyst asks where a vessel will be in 15 min. Feeds go stale, and a fast, sloppy source floods the feed. The answers still look reasonable. |
| 1:00–2:00 | Clean baseline | Select **Clean feed**. Most dots are green with short lines. This is the reference. The model is trained once on morning data and frozen. |
| 2:00–3:30 | Degrade it | Click **Stale reports**, then **Unreliable source**, then **Stale + unreliable**. Read the three numbers above the map: the typical miss and the count of >5 km misses jump. Drag the time slider left to right: the feed gets worse later in the day (a simulated progression on real data). |
| 3:30–5:00 | One vessel, end to end | The right panel opens on the largest miss. Walk steps 1→2: what the AI received (old report, wrong speed, a position several km off) and how far off the prediction was. Turn on **Zoom to this vessel** to show the real track against the report. |
| 5:00–6:30 | Countermeasure | Flip **Countermeasure on**. Red dots disappear: estimates are corrected or sent to a reviewer (hollow circles). The numbers show before and after. Step 3 in the panel explains why, in plain words. |
| 6:30–7:30 | Over time | Open **How the error grows as the feed degrades**: the bad-case miss rises sharply with the countermeasure off and stays low with it on. Point out what this costs: about half of estimates go to a reviewer in the worst window. |
| 7:30–8:30 | Limits | Open **Known limits**: an unseen 1.5 km position shift is not caught, valid fast movers get flagged, and age compensation can backfire. |
| 8:30–9:30 | Human oversight | In panel step 4, choose **Defer: request a fresh report**, type a reason, and record it. Open **Review log**: the decision, reason, and measured review seconds (demo only, not an operator study). |
| 9:30–10:00 | Next step | Shadow-mode pilot in one port, with stop criteria (see `docs/DECISION.md`). |

## Recorded fallback

No video was produced automatically. To record one, run the app, start your own recorder (Win+Alt+R or OBS) on the
browser window only, and follow the table above. Save it as `docs/demo_fallback.mp4`. Static fallback: the results tables in the
README and `results/metrics.csv`.
