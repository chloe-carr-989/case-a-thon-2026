"""How congested will this port be? AIS occupancy PROXY: vessels dwelling in a study zone in the next hour."""
from count_page import render

render(
    task="port_occupancy", prefix="port", title="How congested will this port be?",
    caption="Answered only as an AIS **occupancy proxy**: unique vessels reporting below 1 kn inside a "
            "port/anchorage study zone during the next 60 minutes. Not berth occupancy, queue length or waiting time.",
    unit_word="zone", count_word="vessels",
    limits_md="""
- **Zones are inferred, not official.** A study zone is a 0.1° cell where at least 10 distinct vessels sat for 30+
  minutes below 1 kn in the training period (top 60). Zones are named by coordinates only. They are not verified
  terminal, berth or anchorage boundaries.
- **Proxy, not congestion.** The data have no berth capacity, queue, waiting-time or terminal records, so the
  forecast says how many vessels will be *sitting* in the zone, not whether the port is delayed.
- **Undercount:** vessels whose reports are missing are not counted, so the reference count is a lower bound.
- The countermeasure and split caveats are the same as for traffic density.
""")
