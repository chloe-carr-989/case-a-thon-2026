# Maritime Intelligence Dashboard: Interface Redesign Spec

## Objective

Redesign the existing maritime AIS visualization to focus on two applications:

1. **Commercial:** Predict vessel arrival times to help ports coordinate unloading and reduce congestion.
2. **Government:** Identify anomalous vessel movements relative to learned sea lanes, supporting maritime safety and defense analysis.

Both applications should use a shared model that learns normal maritime traffic patterns from historical AIS data.

The main focus is shifting from simply displaying ship positions to understanding **where ships are expected to travel, when they should arrive, and when their behavior deviates from normal patterns.**

## 1. Review the Existing Project

Before changing anything:

- Inspect the current codebase, especially the existing maritime map and timeline visualization.
- Look through the `data_vis/` folder, which contains a separate interface designed by my teammate.
- Take inspiration from that interface's colors, typography, and overall visual identity.
- Do not copy its layout or information density. One of its main problems is that it displays too much information at once.
- Preserve the current application's architecture, working functionality, and map interactions wherever possible.

### Design inspiration

Use **Palantir Gotham** as the primary visual reference.

Desired aesthetic:
- Dark navy, charcoal, and muted slate backgrounds.
- Subtle borders and clearly separated panels.
- Compact, professional typography, taking inspiration from the teammate's interface.
- Restrained use of cyan, amber, and red to communicate different types of information.
- Minimal visual clutter.
- Map-first layout, with information appearing contextually when a vessel or anomaly is selected.
- No unnecessary animations, excessive metrics, or decorative elements.

The interface should feel like a professional maritime intelligence tool, not a generic analytics dashboard.

Keep everything responsive, including smaller laptop screens and mobile layouts. Do not add animations.

## 2. Overall Application Structure

Create two primary modes accessible through a simple navigation control:

**Commercial | Government**

Both modes should share the same underlying map, AIS data, vessel tracking, sea-lane model, and timeline controls.

Switching modes should change the information presented and the primary analytical objective, rather than loading an entirely different application.

### Suggested layout

**Top navigation**
- Application name
- Commercial / Government mode switch
- Selected geographic region
- Dataset time range

**Main map**
- Takes up most of the screen.
- Shows vessel positions and trajectories.
- Supports optional sea-lane and weather overlays.
- Supports selecting vessels and inspecting their movements.

**Collapsible side panel**
- Shows information relevant to the current mode.
- Displays details only when needed.
- Avoids showing a large collection of unrelated statistics.

**Bottom timeline**
- Retain the existing ship movement timeline.
- Allow users to scrub through historical AIS observations.
- Show vessel movements and anomaly events as time progresses.

Prioritize the map over everything else.

## 3. Shared Feature: Learning Sea Lanes

The most important new capability is learning typical vessel paths from historical AIS observations.

Instead of treating every ship movement independently, the system should identify common routes, traffic corridors, and movement patterns.

### Sea-lane modeling

Use historical vessel trajectories to estimate:
- Common maritime routes.
- High-density shipping corridors.
- Typical movement direction.
- Expected vessel speed along different route segments.
- Common approaches to ports.
- Areas where ships regularly slow down, stop, or change direction.

Consider approaches such as spatial density estimation, trajectory clustering, or other lightweight methods appropriate for the available data.

Do not assume a simple straight line between two points represents a shipping route.

Where possible, learn patterns separately by relevant vessel categories so that normal tanker movements are not compared indiscriminately with unrelated vessel types.

### Sea-lane visualization

Add a **Sea Lanes** map overlay.

When enabled:
- Show learned traffic corridors as subtle lines or bands.
- Represent traffic density through opacity or line intensity.
- Distinguish historical expected routes from actual vessel trajectories.
- Allow a selected vessel's observed route to be compared against the expected corridor.

The overlay should be visually informative without overwhelming the map.

### Vessel filtering

Exclude pleasure and recreational vessels by default to reduce noise.

Focus on:
- Cargo ships
- Oil tankers
- Other commercial shipping vessels
- Government or military vessels where identifiable from available data

Use AIS vessel type information when available. Handle missing or ambiguous vessel classifications explicitly.

Do not assume every vessel can be reliably categorized as commercial or government from AIS vessel type alone.

## 4. Commercial Mode: Arrival Prediction

### Primary question

**"When is this vessel expected to arrive at port?"**

Commercial mode should focus on predicting vessel arrival times, especially for oil tankers, container ships, and other cargo vessels.

The operational motivation is helping ports coordinate unloading schedules and stagger arrivals to reduce congestion.

### Main features

**Vessel selection**

When a user selects a commercial vessel, display:
- Vessel name and MMSI
- Vessel type
- Current position
- Current speed
- Predicted destination or selected destination, when supported
- Estimated time of arrival
- Estimated remaining travel time
- Prediction uncertainty or confidence interval, when available

**ETA prediction**

Use the vessel's:
- Current location
- Historical trajectory
- Speed and heading
- Expected sea lane
- Distance along its likely route
- Available weather conditions

to estimate arrival time.

Use a clearly defined arrival proxy based on AIS observations if verified port arrival records are unavailable. Do not imply that entering a port is the same as completing unloading.

**Port congestion context**

If supported by the available data, show:
- Number of tracked vessels approaching a selected port
- Predicted arrival order
- Approximate arrival windows
- Potential overlapping arrivals

Do not invent berth availability, unloading duration, or port capacity data.

### Commercial interface

Keep this mode simple.

The default map should emphasize commercial vessel movements and major shipping corridors.

Selecting a ship should open a compact vessel information panel with its ETA as the most prominent metric.

Avoid turning this into a full port management system. The main objective is a credible arrival prediction that can inform scheduling.

## 5. Government Mode: Maritime Anomaly Detection

### Primary question

**"Which vessels are behaving unusually relative to established maritime traffic patterns?"**

Government mode should focus on identifying unusual vessel behavior that may warrant further review for maritime safety or defense purposes.

The system should learn what normal maritime traffic looks like and flag deviations.

### Anomaly detection

Potential anomalies include:
- A vessel moving significantly outside an established sea lane.
- Unexpected changes in direction.
- Unusual speed relative to nearby traffic or historical patterns.
- Extended stops in unusual locations.
- Repeated unexplained route deviations.
- Unusual movement patterns near ports or sensitive maritime areas.
- Reporting gaps or stale AIS observations.

Importantly, not every deviation is suspicious.

A vessel might change course because of weather, congestion, navigational restrictions, or other legitimate reasons.

The system should distinguish between unusual movement and unreliable data wherever possible.

### Weather overlay

Add an optional weather layer to help users understand environmental conditions affecting vessel movement.

Prioritize:
- Wind direction
- Wind speed

If practical and supported by available data, also include:
- Wave conditions
- Storm systems

Display wind using subtle directional arrows or another low-clutter visualization.

The user should be able to toggle weather independently of the sea-lane overlay.

Weather should also contribute to anomaly interpretation where spatially and temporally matched observations are available.

For example:

A ship deviating from its usual route in strong winds may have a plausible environmental explanation.

A similar deviation during calm conditions may warrant closer examination.

Do not automatically classify weather-related deviations as safe or unexplained deviations as malicious.

### Anomaly scoring

Each vessel may receive an anomaly score based on deviation from learned traffic patterns.

Possible factors:
- Distance from expected sea lane
- Speed deviation
- Heading deviation
- Unusual stopping behavior
- Data freshness and completeness
- Environmental context

Normalize scores against comparable vessels and operating conditions where possible.

Show a simple classification such as:
- Normal
- Unusual
- High deviation

These labels describe movement relative to historical patterns, not confirmed threats.

### Government interface

The default view should emphasize:
- Learned sea lanes
- Vessel traffic density
- Anomalous vessel movements
- Optional weather conditions

Flagged vessels should be visually distinguishable without cluttering the map.

Selecting a flagged vessel should show:

**Vessel details**
- Name / MMSI
- Vessel classification
- Last reported location
- Last update time
- Current movement characteristics

**Anomaly explanation**
- Anomaly score
- Expected route
- Actual route
- Primary reasons for the flag
- Relevant weather context
- AIS data quality and freshness

**Human review**
- Ability to mark an alert as reviewed
- Ability to flag a false positive
- Ability to record a brief review note

A hypothetical defense question might be whether an unidentified vessel's movement resembles a particular historical traffic pattern. However, do not infer that a ship is North Korean, military, hostile, or associated with a particular state solely from its route.

If flag or registry information is unavailable or unverified, display that uncertainty explicitly.

The purpose is to prioritize vessel movements for human investigation, not automatically attribute identity or intent.

## 6. Weather and Anomaly Visualization

The map should support three independently controlled overlays:

1. **Sea Lanes:** Expected maritime traffic corridors.
2. **Traffic Density:** Concentration of historical vessel movements.
3. **Weather:** Wind direction and speed.

Keep these controls compact, preferably in a small map toolbar.

Avoid displaying every overlay simultaneously by default.

For vessel selection, show:
- Actual trajectory as a solid line.
- Expected route as a dashed line.
- Deviations using restrained highlighting.

When a vessel is selected, reduce the visual prominence of unrelated vessels.

## 7. Timeline and Historical Playback

Preserve the current timeline-based visualization.

The user should be able to:
- Move through the available AIS data chronologically.
- Observe vessel movement over time.
- See when vessels deviate from learned sea lanes.
- Select a vessel and inspect its trajectory.
- Compare historical movement against expected movement.

If anomaly scores are available at multiple timestamps, update the displayed score as the timeline advances.

Avoid using future observations when calculating what the system would have known at an earlier timestamp.

## 8. Data Quality and Case Competition Requirements

This project is for a case competition about **unintentional data poisoning**, specifically how stale or low-quality information can degrade AI predictions and decisions.

The redesigned interface needs to support that story.

The sea-lane model and ETA/anomaly detection system should not just work on clean data. We need to demonstrate what happens when the underlying AIS information becomes unreliable.

### Data degradation experiments

Support four experimental conditions:

1. Reference / nominal data
2. Stale data only
3. Low-quality data only
4. Combined stale and low-quality data

Examples:
- Delayed AIS position updates
- Missing observations
- Incorrect coordinates
- Inaccurate speed or heading values
- Duplicate records
- An unreliable data source contributing disproportionately many observations

These defects should be introduced in controlled experiments and clearly distinguished from defects already present in the original dataset.

### Countermeasure

The application should eventually demonstrate how data-quality-aware processing improves results.

Potential approaches include:
- Freshness weighting
- Outlier filtering
- Duplicate detection
- Source-quality weighting
- Confidence reduction for incomplete trajectories

The interface should make it possible to compare model behavior before and after the countermeasure on the same evaluation cases.

### Evaluation

For commercial mode, prioritize:
- ETA prediction error
- Error across time periods
- Prediction coverage
- Performance under degraded data

For government mode, prioritize:
- Anomaly detection performance
- False positives and false negatives, where defensible labels exist
- Changes in anomaly scores under degraded data
- Alerts caused by stale or inaccurate AIS observations
- Performance after applying the countermeasure

Do not claim real-world threat detection accuracy without verified labels.

### Interface treatment

Do not overcrowd the main dashboard with experimental metrics.

Instead, create a compact, collapsible **Model Evaluation** section that can show:
- Selected data condition
- Baseline versus corrected model
- Relevant performance metric
- Data freshness / quality indicators

This should be accessible for the competition demonstration but should not dominate the normal user experience.

## 9. Implementation Priorities

Prioritize a clean, working prototype over implementing every possible feature.

### Priority 1: Interface restructure
- Inspect the existing application and `data_vis/` reference.
- Introduce Commercial / Government navigation.
- Apply the Gotham-inspired visual style.
- Simplify existing panels and remove redundant information.
- Preserve the current map and timeline.

### Priority 2: Sea-lane visualization
- Add a toggle for learned sea lanes.
- Display historical vessel traffic density.
- Support actual-versus-expected route comparison.

### Priority 3: Government anomaly interface
- Highlight unusual vessels.
- Add a compact anomaly details panel.
- Show anomaly explanations and data freshness.
- Add weather overlay controls.

### Priority 4: Commercial arrival interface
- Display ETA predictions for selected commercial vessels.
- Show expected routes and estimated arrival windows.
- Include port approach traffic where data permits.

### Priority 5: Evaluation
- Add the model evaluation controls.
- Support baseline and countermeasure comparisons.
- Make data degradation visible in the timeline and results.

## 10. Technical Constraints

- Work within the existing framework and architecture.
- Reuse existing map components and dependencies where practical.
- Do not introduce a new frontend framework.
- Avoid unnecessary dependencies.
- Keep rendering performant with large AIS datasets.
- Preserve existing functionality unless it directly conflicts with the new design.
- Avoid hardcoded predictions, fabricated anomaly scores, or fake weather data.
- If model outputs are not implemented yet, build the interface around clearly defined data contracts and honest unavailable states.
- Separate frontend visualization from model computation.
- Do not assume all ships have complete metadata or continuous AIS observations.
- Use existing data and model outputs wherever available.
- Do not rewrite unrelated files.

## 11. Expected Result

The finished application should have two clear workflows.

**Commercial workflow**

Select a cargo ship or oil tanker → view its learned sea lane → inspect predicted arrival time → understand potential scheduling implications.

**Government workflow**

View maritime traffic → enable sea lanes and weather → identify unusual vessel movements → inspect why a vessel was flagged → determine whether the alert requires human review.

The shared technical foundation is learning normal vessel movement patterns and understanding deviations from those patterns.

The interface should make this immediately obvious.

**Most importantly, keep the design minimal.** The current application should evolve into a focused maritime intelligence dashboard, not a collection of every possible maritime metric.

Start by inspecting the codebase and identifying the smallest set of changes needed to implement the redesigned interface. Then make the changes, preserving existing functionality and keeping the implementation modular.