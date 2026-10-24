# Facts that exist only in the session log (the folders were deleted or the output was shown in the student's terminal)

## A. Laps the student saved with scripts/collect.py on 2026-10-06 (folder name under data/collected, frames; lap time = frames / 50)
Listed while they still existed. 36 folders in total.
```
20261006_222106_951401 2182   20261006_222221_854948 2195   20261006_222315_036798 2187   20261006_222405_717365 2163
20261006_222455_449375 2148   20261006_222542_793261 2118   20261006_222834_863196 2111   20261006_222936_624634 2176
20261006_223042_178023 2065   20261006_223156_050637 2072   20261006_223409_094498 1990   20261006_223456_415724 2042
20261006_223546_185264 2099   20261006_223633_882927 2072   20261006_223951_105081 2081   20261006_224036_256167 2040
20261006_224119_162532 2040   20261006_224235_101338 1990   20261006_224318_648168 2053   20261006_224547_668857 2099
20261006_224631_101195 2040   20261006_224742_138427 2074   20261006_224826_465398 2043   20261006_224951_792313 2135
20261006_225129_977821 2033   20261006_225214_080995 2035   20261006_225258_145239 2093   20261006_225420_689787 2051
20261006_225511_839890 2065   20261006_225851_948369 2068   20261006_225936_079485 2014   20261006_230019_483440 2040
20261006_230102_347470 1996   20261006_230146_447961 2029   20261006_230331_507785 2019   20261006_230415_685685 2054
```
- Step 1 (student's rule "lap time >= 42 s"): the 9 folders with >= 2100 frames were deleted WITHOUT being replayed
  (so their centre score / progress / off-road frames are unknown).
- Step 2 (student's rule "centre score <= 28"): the remaining 27 were replayed (laps_metrics.json in this folder holds the
  per-lap metrics of those 27: lap_time, center, progress, start_hold, off_road_frames, ...); 6 with center <= 28 were deleted.
- The 21 left were moved to assignment1/data/stage2/ and are the training data of the final model.

## B. End of the final model's training and the student's own inference run (pasted by the student from the terminal)
```
Epoch   100     [Train] loss: 2.286439  ETA: +0.000000s
모델 저장 완료: /home/sukyun02/workspace/aiad/assignment1/models/20261006_231048.pth
제출할 구조 파일: /home/sukyun02/workspace/aiad/assignment1/models/20261006_231048.network.py
(.venv) ➜  assignment1 git:(main) ✗ python scripts/infer.py configs/stage2.yaml
[추론] 장치: CUDA (NVIDIA GeForce RTX 5070 Ti Laptop GPU) | PyTorch 2.10.0+cu128
설정: model_path=/home/sukyun02/workspace/aiad/assignment1/models/20261006_231048.pth | display=True | score=False | max_speed=22.5 | throttle=0.375 | steering_limit=0.45 | steering_rate=1.5
주행 종료: 출발점 복귀 완료
주행 결과: 진행률 70.66/100 | 중앙선 유지 33.24/100 | 랩타임 40.94s
```
The config at training time (configs/stage2.yaml): start_model scratch, data_dir data/stage2, epochs 100, batch_size 256,
learning_rate 0.001. (The student has since changed `epochs` to 50 for a later, different training run.)
The loss printed per epoch is the SUM of the batch losses of that epoch (src/training.py).

## C. Targets told by the student (from the professor): lap time within 42 s, centre score 30 or more.

## D. The 7 provided laps (data/collected/20260923_*) replayed: lap 43.44 43.24 43.02 43.06 43.00 42.72 42.86,
progress 100.00 and 0 off-road frames in all 7 (pilot/work/stage2/out/curate_provided.json).
