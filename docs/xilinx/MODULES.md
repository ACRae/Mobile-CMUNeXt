# Modules

## Layer C3D

Interval: 3,180,814

+-----------------------------------------------------------------------------------------+
| Design Summary                                                                          |
| impl_1                                                                                  |
| xczu3eg-sbva484-1-e                                                                     |
+-----------------------------------------------------------+-----------+--------+--------+
| Criteria                                                  | Guideline | Actual | Status |
+-----------------------------------------------------------+-----------+--------+--------+
| LUT                                                       | 70%       | 16.69% | OK     |
| FD                                                        | 50%       | 7.08%  | OK     |
| LUTRAM+SRL                                                | 25%       | 5.55%  | OK     |
| CARRY8                                                    | 25%       | 7.99%  | OK     |
| MUXF7                                                     | 15%       | 0.01%  | OK     |
| DSP                                                       | 80%       | 39.72% | OK     |
| RAMB/FIFO                                                 | 80%       | 25.69% | OK     |
| DSP+RAMB+URAM (Avg)                                       | 70%       | 32.70% | OK     |
| BUFGCE* + BUFGCTRL                                        | 24        | 0      | OK     |
| DONT_TOUCH (cells/nets)                                   | 0         | 0      | OK     |
| MARK_DEBUG (nets)                                         | 0         | 0      | OK     |
| Control Sets                                              | 1323      | 118    | OK     |
| Average Fanout for modules > 100k cells                   | 4         | 1.12   | OK     |
| Max Average Fanout for modules > 100k cells               | 4         | 0      | OK     |
| Non-FD high fanout nets > 10k loads                       | 0         | 0      | OK     |
+-----------------------------------------------------------+-----------+--------+--------+
| TIMING-6 (No common primary clock between related clocks) | 0         | 0      | OK     |
| TIMING-7 (No common node between related clocks)          | 0         | 0      | OK     |
| TIMING-8 (No common period between related clocks)        | 0         | 0      | OK     |
| TIMING-14 (LUT on the clock tree)                         | 0         | 0      | OK     |
| TIMING-35 (No common node in paths with the same clock)   | 0         | 0      | OK     |
+-----------------------------------------------------------+-----------+--------+--------+
| Number of paths above max LUT budgeting (0.350ns)         | 0         | 0      | OK     |
| Number of paths above max Net budgeting (0.239ns)         | 0         | 0      | OK     |
+-----------------------------------------------------------+-----------+--------+--------+

#=== Post-Implementation Resource usage ===
SLICE:            0
LUT:          11775
FF:            9992
DSP:            143
BRAM:           111
URAM:             0
LATCH:            0
SRL:           1454
CLB:           2385

#=== Final timing ===
CP required:                     10.000
CP achieved post-synthesis:      6.947
CP achieved post-implementation: 7.751


## Layer PW

Interval: 4,785,282

+-----------------------------------------------------------------------------------------+
| Design Summary                                                                          |
| impl_1                                                                                  |
| xczu3eg-sbva484-1-e                                                                     |
+-----------------------------------------------------------+-----------+--------+--------+
| Criteria                                                  | Guideline | Actual | Status |
+-----------------------------------------------------------+-----------+--------+--------+
| LUT                                                       | 70%       | 11.15% | OK     |
| FD                                                        | 50%       | 1.82%  | OK     |
| LUTRAM+SRL                                                | 25%       | 0.01%  | OK     |
| CARRY8                                                    | 25%       | 10.23% | OK     |
| MUXF7                                                     | 15%       | 0.00%  | OK     |
| DSP                                                       | 80%       | 19.17% | OK     |
| RAMB/FIFO                                                 | 80%       | 22.22% | OK     |
| DSP+RAMB+URAM (Avg)                                       | 70%       | 20.70% | OK     |
| BUFGCE* + BUFGCTRL                                        | 24        | 0      | OK     |
| DONT_TOUCH (cells/nets)                                   | 0         | 0      | OK     |
| MARK_DEBUG (nets)                                         | 0         | 0      | OK     |
| Control Sets                                              | 1323      | 50     | OK     |
| Average Fanout for modules > 100k cells                   | 4         | 1.22   | OK     |
| Max Average Fanout for modules > 100k cells               | 4         | 0      | OK     |
| Non-FD high fanout nets > 10k loads                       | 0         | 0      | OK     |
+-----------------------------------------------------------+-----------+--------+--------+
| TIMING-6 (No common primary clock between related clocks) | 0         | 0      | OK     |
| TIMING-7 (No common node between related clocks)          | 0         | 0      | OK     |
| TIMING-8 (No common period between related clocks)        | 0         | 0      | OK     |
| TIMING-14 (LUT on the clock tree)                         | 0         | 0      | OK     |
| TIMING-35 (No common node in paths with the same clock)   | 0         | 0      | OK     |
+-----------------------------------------------------------+-----------+--------+--------+
| Number of paths above max LUT budgeting (0.350ns)         | 0         | 0      | OK     |
| Number of paths above max Net budgeting (0.239ns)         | 0         | 0      | OK     |
+-----------------------------------------------------------+-----------+--------+--------+



#=== Post-Implementation Resource usage ===
SLICE:            0
LUT:           7869
FF:            2566
DSP:             69
BRAM:            96
URAM:             0
LATCH:            0
SRL:              1
CLB:           1396

#=== Final timing ===
CP required:                     10.000
CP achieved post-synthesis:      4.598
CP achieved post-implementation: 5.385


## Layer DW

Interval:  1,754,375

+-----------------------------------------------------------------------------------------+
| Design Summary                                                                          |
| impl_1                                                                                  |
| xczu3eg-sbva484-1-e                                                                     |
+-----------------------------------------------------------+-----------+--------+--------+
| Criteria                                                  | Guideline | Actual | Status |
+-----------------------------------------------------------+-----------+--------+--------+
| LUT                                                       | 70%       | 15.93% | OK     |
| FD                                                        | 50%       | 2.52%  | OK     |
| LUTRAM+SRL                                                | 25%       | 12.12% | OK     |
| CARRY8                                                    | 25%       | 2.21%  | OK     |
| MUXF7                                                     | 15%       | 0.18%  | OK     |
| DSP                                                       | 80%       | 9.72%  | OK     |
| RAMB/FIFO                                                 | 80%       | 8.56%  | OK     |
| DSP+RAMB+URAM (Avg)                                       | 70%       | 9.14%  | OK     |
| BUFGCE* + BUFGCTRL                                        | 24        | 0      | OK     |
| DONT_TOUCH (cells/nets)                                   | 0         | 0      | OK     |
| MARK_DEBUG (nets)                                         | 0         | 0      | OK     |
| Control Sets                                              | 1323      | 126    | OK     |
| Average Fanout for modules > 100k cells                   | 4         | 2.43   | OK     |
| Max Average Fanout for modules > 100k cells               | 4         | 0      | OK     |
| Non-FD high fanout nets > 10k loads                       | 0         | 0      | OK     |
+-----------------------------------------------------------+-----------+--------+--------+
| TIMING-6 (No common primary clock between related clocks) | 0         | 0      | OK     |
| TIMING-7 (No common node between related clocks)          | 0         | 0      | OK     |
| TIMING-8 (No common period between related clocks)        | 0         | 0      | OK     |
| TIMING-14 (LUT on the clock tree)                         | 0         | 0      | OK     |
| TIMING-35 (No common node in paths with the same clock)   | 0         | 0      | OK     |
+-----------------------------------------------------------+-----------+--------+--------+
| Number of paths above max LUT budgeting (0.350ns)         | 0         | 0      | OK     |
| Number of paths above max Net budgeting (0.239ns)         | 0         | 0      | OK     |
+-----------------------------------------------------------+-----------+--------+--------+

#=== Post-Implementation Resource usage ===
SLICE:            0
LUT:          11237
FF:            3554
DSP:             35
BRAM:            37
URAM:             0
LATCH:            0
SRL:             86
CLB:           2077

#=== Final timing ===
CP required:                     10.000
CP achieved post-synthesis:      3.400
CP achieved post-implementation: 6.396


## Overall

| Resource    |             C3D |             PW |              DW |           **Total** | Guideline % |
| ------------| --------------: | -------------: | --------------: | ------------------: | ----------: |
| **LUT**     | 11,775 (16.69%) | 7,869 (11.15%) | 11,237 (15.93%) | **30,881 (43.77%)** |         70% |
| **FF**      |   9,992 (7.08%) |  2,566 (1.82%) |   3,554 (2.52%) | **16,112 (11.42%)** |         50% |
| **DSP**     |    143 (39.72%) |    69 (19.17%) |      35 (9.72%) |    **247 (68.61%)** |         80% |
| **BRAM**    |    111 (25.69%) |    96 (22.22%) |      37 (8.56%) |    **244 (56.48%)** |         80% |
