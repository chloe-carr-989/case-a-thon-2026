# Maritime Map Visualization Updates

Update the existing maritime traffic map's visual styling to make sea lanes, traffic density, and individual ships easy to distinguish. Keep the current dark maritime aesthetic, map behavior, data processing, and interactions. Change only the visualization styles and legend needed for this request.

## Problem

Sea lanes, traffic density squares, and cargo ship markers currently use similar cyan colors. They blend into one visual network, especially around busy channels, making it hard to distinguish where ships typically travel, how concentrated the traffic is, and where ships are currently located.

## Color and layer changes

| Layer | Proposed styling | Purpose |
| --- | --- | --- |
| Historical sea lanes | Muted slate blue `#426879`, thin continuous lines, approximately 20–35% opacity | Low-priority background route context |
| Traffic density | Dark teal to bright cyan gradient (see below) | Show relative concentration of vessel activity |
| Cargo ships | Off-white/light gray `#C9D4E0` | Make current vessel positions distinct from density and lanes |
| Tankers | Amber `#F0AD4E` | Preserve strong contrast with other layers |
| ETA issued | Retain a clearly visible white-edged marker | Distinguish issued ETAs from ordinary ships |
| ETA held for review | Hollow marker with a distinct white or amber outline | Distinguish review status without relying solely on fill color |

Use cyan **only for traffic density**. Sea lanes should look blue-gray, not bright cyan, and cargo ships should not be cyan.

## Traffic density scale

Use a sequential color scale where the brightest cyan is reserved for genuinely high-density cells:

- Very low: `#102F3A`
- Low: `#164555`
- Low–medium: `#1D6075`
- Medium: `#247E99`
- Medium–high: `#2A9FBE`
- High: `#43BFDC`
- Very high: `#85E5F5`

Consider a logarithmic or percentile-based normalization if the density distribution is skewed. Avoid making sparse cells appear bright. Preserve an interpretable low-to-high progression, and update the legend to show it.

## Visual priority

1. **Ships and ETA indicators:** Most prominent. Users should quickly identify current ships and actionable status.
2. **Traffic density:** Visible enough to distinguish congested areas, but not overpowering.
3. **Historical sea lanes:** Subdued geographic context behind the other two layers.

The layers should answer separate questions:

- **Sea lanes:** Where do vessels typically travel?
- **Traffic density:** How much vessel activity occurs in each area?
- **Ships:** Where are vessels right now?

## Implementation constraints

- Preserve the existing dark background and overall minimalist appearance.
- Keep the current layer types and data sources. Do not remove sea lanes or density just because they are related.
- Update the map legend so swatches, labels, and marker styles accurately match the rendered layers.
- Maintain adequate contrast on the dark background, including at dense intersections.
- Do not change unrelated application logic, layout, animation, or interactions.
- Keep existing responsive behavior.

## Acceptance criteria

- Cargo ships are visibly distinct from historical lanes and density cells.
- Only traffic density uses bright cyan, with brightness meaningfully increasing as density increases.
- Historical sea lanes remain recognizable without dominating the map.
- Tankers remain amber and ETA states remain distinguishable.
- The legend accurately reflects all updated styles.
