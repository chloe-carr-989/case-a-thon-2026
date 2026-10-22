"""How busy will this area be? Unique vessels per 0.25 deg grid cell in the next 60 minutes."""
from count_page import render

render(
    task="traffic_density", prefix="density", title="How busy will this area be?",
    caption="Forecast of unique vessels (MMSIs) observed in each 0.25° grid cell during the next 60 minutes, "
            "from the previous 3 hours of AIS. Degrade the feed, then switch on the countermeasure.",
    unit_word="cell", count_word="vessels",
    limits_md="""
- **Target:** unique MMSIs with at least one AIS report in the cell during the next hour, so a vessel counts once
  however often it reports. This is *observed* traffic: vessels outside receiver coverage or not transmitting are missing.
- **Per-area rate:** count ÷ the cell's spherical area. Cells shrink toward the poles, so raw counts of far-apart cells
  are not compared as if the areas were equal.
- **Zeros:** a cell in the training index with no reports in a covered time bucket is a confirmed zero. Time buckets with
  no source data at all are excluded rather than read as empty seas.
- **Split:** the model is trained on the first half of the day and tested on the second. The same cells appear in both,
  and forecasts every 15 minutes overlap, so the case count overstates the independent evidence.
- **Countermeasure limits:** it cannot restore omitted reports, and the persistent synthetic ghost IDs pass its filters.
  Dropping one-message transients also removes some real, sparsely reporting vessels: compare *Clean feed* with the
  countermeasure on and off.
- One day of data: no day-of-week or seasonal pattern can be learned.
""")
