# TRMG2 GMNS scenario — review package

Validation: gmns-ready 0 errors / 4 warnings (2 = geometry pending shapefile, 1 = km/h rounding cosmetic, 1 = vdf_alpha up to 4.0 which matches TRMG2's own ff_speed_alpha_beta.csv); dtalite_qa validate clean.

- Directed links: 75,939; zones: 3,247; area types {'Rural': 990, 'Suburban': 1201, 'Urban': 710, 'Downtown': 346}

- v0 AM sov demand: 312,753 vehicles; avg congested travel time 20.9 min

## Reviewer checklist

1. network_inventory.csv — lane-miles & capacity ranges per facility/area type vs your expectation of the Triangle network.
2. od_district_matrix.csv — district-to-district pattern plausibility (RTP/Raleigh/Durham pulls, external districts show as NoPolygon).
3. trip_length_distribution.csv — v0 scaffold gravity gives avg 20.9 min; TRMG2's calibrated models will differ.
4. top_loaded_links.csv — check the top-25 are the region's known freeway segments (TRMG2 link/node IDs included for lookup).
5. Known gaps before parity testing: real lengths + geometry (ITRE shapefile), real OD (od_veh_trips export), turn prohibitions.
