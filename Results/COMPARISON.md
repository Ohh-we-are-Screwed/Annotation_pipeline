====================================================================================================
CONFIGURATION (read from each run's own stage manifests)
====================================================================================================

  yolo11x_dinov2
    proposal_2d    yolo11x.pt  sha 7bc158aa95c0  6/80 source classes mapped
    class map      coco_to_phrase_nuscenes.yaml  sha 4abfc4592600
    reachable      6 phrases: a pedestrian, a bicycle, a car, a motorcycle, a bus, a truck
    UNREACHABLE    4 phrases: a road barrier, a traffic cone, a construction vehicle, a trailer
    reid_embedding facebook/dinov2-small @ ed25f3a31f01  enabled=True

  yolo11x_dinov3
    proposal_2d    yolo11x.pt  sha 7bc158aa95c0  6/80 source classes mapped
    class map      coco_to_phrase_nuscenes.yaml  sha 4abfc4592600
    reachable      6 phrases: a pedestrian, a bicycle, a car, a motorcycle, a bus, a truck
    UNREACHABLE    4 phrases: a road barrier, a traffic cone, a construction vehicle, a trailer
    reid_embedding facebook/dinov3-vits16-pretrain-lvd1689m @ 114c13799502  enabled=True

  yolov8x-oiv7_dinov2
    proposal_2d    yolov8x-oiv7.pt  sha 89acc72b5b4d  13/601 source classes mapped
    class map      oiv7_to_phrase_nuscenes.yaml  sha e621c8afed0d
    reachable      6 phrases: a bicycle, a pedestrian, a bus, a car, a motorcycle, a truck
    UNREACHABLE    4 phrases: a road barrier, a traffic cone, a construction vehicle, a trailer
    reid_embedding facebook/dinov2-small @ ed25f3a31f01  enabled=True

  yolov8x-oiv7_dinov3
    proposal_2d    yolov8x-oiv7.pt  sha 89acc72b5b4d  13/601 source classes mapped
    class map      oiv7_to_phrase_nuscenes.yaml  sha e621c8afed0d
    reachable      6 phrases: a bicycle, a pedestrian, a bus, a car, a motorcycle, a truck
    UNREACHABLE    4 phrases: a road barrier, a traffic cone, a construction vehicle, a trailer
    reid_embedding facebook/dinov3-vits16-pretrain-lvd1689m @ 114c13799502  enabled=True

====================================================================================================
METRICS
====================================================================================================

| metric | yolo11x_dinov2 | yolo11x_dinov3 | yolov8x-oiv7_dinov2 | yolov8x-oiv7_dinov3 |
|---|---|---|---|---|
| **3D boxes (vs human 3D answer key)** |  |  |  |  |
| precision — localization | 75.1% | 75.1% | 82.0% | 82.0% |
| precision — class-aware | 71.7% | 71.7% | 78.8% | 78.8% |
| GT recall (reachable classes) | 72.2% | 72.2% | 38.3% | 38.3% |
| ATE (m, lower better) | 0.618 | 0.618 | 0.692 | 0.692 |
| ASE (lower better) | 0.777 | 0.777 | 0.682 | 0.682 |
| AOE (rad, lower better) | 0.395 | 0.395 | 0.279 | 0.279 |
| boxes shipped | 6,104 | 6,104 | 2,965 | 2,965 |
| matched to a GT box | 4,586 | 4,586 | 2,432 | 2,432 |
| **2D proposals (vs human 2D answer key)** |  |  |  |  |
| precision — localization | 62.5% | 62.5% | 83.6% | 83.6% |
| precision — class-aware | 59.7% | 59.7% | 80.9% | 80.9% |
| GT recall (all GT) | 36.6% | 36.6% | 16.6% | 16.6% |
| GT recall (>=32px) | 38.8% | 38.8% | 18.2% | 18.2% |
| predictions counted | 10,150 | 10,150 | 3,438 | 3,438 |
| **Paint / lift geometry** |  |  |  |  |
| GT coverage rate | 74.7% | 74.7% | 43.9% | 43.9% |
| painted points inside a GT box | 87.8% | 87.8% | 89.3% | 89.3% |
| enrichment over base rate | 6.93 | 6.93 | 7.05 | 7.05 |
| points painted | 482,074 | 482,074 | 461,589 | 461,589 |
| **Tracking (Stage 7)** |  |  |  |  |
| tracks total | 2,956 | 2,958 | 1,147 | 1,148 |
| tracks >=3 hits | 430 | 424 | 246 | 245 |
| matched pairs | 3,148 | 3,146 | 1,818 | 1,817 |
| appearance-trusted pairs | 3,121 | 3,119 | 1,818 | 1,817 |
| yaw flips applied | 991 | 995 | 567 | 565 |
| **Runtime** |  |  |  |  |
| stage 3 proposals (s) | 80.5 | 80.2 | 104.0 | 104.1 |
| stage 7 track (s) | 85.7 | 86.7 | 65.2 | 66.3 |

| **3D per-class precision_localization (boxes)** |  |  |  |  |
|   a bicycle | 22.8% (162) | 22.8% (162) | 26.4% (72) | 26.4% (72) |
|   a bus | 65.4% (159) | 65.4% (159) | 74.8% (111) | 74.8% (111) |
|   a car | 75.0% (3,316) | 75.0% (3,316) | 81.7% (2,202) | 81.7% (2,202) |
|   a motorcycle | 88.7% (71) | 88.7% (71) | 100.0% (22) | 100.0% (22) |
|   a pedestrian | 82.2% (1,900) | 82.2% (1,900) | 95.5% (356) | 95.5% (356) |
|   a truck | 67.1% (496) | 67.1% (496) | 84.2% (202) | 84.2% (202) |
