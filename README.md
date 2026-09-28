# 冰封纪要 · 现场录音的说话人角色归属

极地冰芯钻探队的班后会录音转写完以后，台账要给每一句话挂上「谁说的、什么角色」。
角色只有三种：`engineer`（钻探工程）、`climatologist`（气候科学）、`unknown`。
判定靠两条线索：说话人分离结果给出 `speaker_id`，转写文本里的关键词给出角色倾向。

## 目录

- `backend/speaker_diarizer.py`：说话人分离与角色归属。关键词打分、说话人画像累积、转写片段与说话人段的对齐、汇总。
- `tests/`：现有测试，用替身 pipeline 覆盖既有正常路径。
- `conftest.py`：把仓库根加进 `sys.path`，测试统一用 `from backend import speaker_diarizer` 导入。

## 角色是怎么判出来的

1. `_classify_role_by_keywords(text)` 数一段文本命中了多少个工程类关键词、多少个气候类关键词，
   返回 `(角色, 该角色命中数 / 总命中数)`；一个都没命中返回 `('unknown', 0.5)`。
2. `_update_speaker_profile(speaker_id, text)` 把这段文本记进该说话人的画像，并把上一步的置信度
   累加到 `role_votes` 里对应的那个桶。
3. `_get_speaker_role(speaker_id)` 取 `role_votes` 里得分最高的桶，置信度 = 该桶 / 全部桶之和。
4. `_align_transcript_with_speakers(...)` 按时间重叠把每条转写片段挂到重叠最久的那个说话人段上，
   边挂边更新画像、边取角色，输出段落 `confidence = 说话人段 confidence * (0.6 + 0.4 * 角色置信度)`。
5. `get_speaker_summary()` 汇总每个说话人的角色、置信度、段落数与字数。

## 音频后端

`librosa` 与 `pyannote.audio` 都只在懒加载分支里 import，本机没有安装、也不允许安装。
所以测试一律给 `diarizer.diarization_pipeline` 注入一个替身：`load_pipeline()` 见到非 None 会直接返回它，
不会走到任何音频后端。替身要能被调用一次拿到一个对象，该对象提供 `itertracks(yield_label=True)`，
产出 `(turn, None, speaker)`，其中 `turn.start` / `turn.end` 是 float。也可以只测纯逻辑方法。

## 运行

    ~/venvs/base/bin/python -m pytest tests/ -q

解释器与依赖版本见 `requirements-task.txt`。不要 pip install，不要联网。

### 本次做题人填写

（把你自己拿主意的那一处写在这里：选了哪种做法、为什么不选另一种、代价是什么。）
