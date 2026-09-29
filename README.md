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
2. `_update_speaker_profile(speaker_id, text, count_for_role=True)` 把这段文本记进该说话人的画像；
   只有命中了关键词的段落才把分类置信度累加到 `role_votes` 里对应的角色桶——
   一个关键词都没命中的中性段落不再往 `unknown` 桶里加 0.5，不会稀释角色得分。
   `count_for_role=False`（低置信对齐段落）时文本仍计入段落数与字数，并累加
   `needs_review_segments`，但不参与角色得分累积。
3. `_get_speaker_role(speaker_id)` 只在 `engineer` / `climatologist` 两个桶之间比较，
   `unknown` 桶不参与最高分比较；两者得分差小于 1e-9（含都为 0）时显式返回
   `('unknown', 0.5)`，不依赖 dict 插入序。角色置信度 = 该角色得分 / 两个角色得分之和。
4. `_align_transcript_with_speakers(...)` 按时间重叠把每条转写片段挂到重叠最久的那个说话人段上。
   对齐置信度 = 最佳说话人段 `confidence` × (重叠时长 / 转写片段时长)，片段时长 <= 0 时记 0；
   低于 `review_threshold`（默认 0.6，构造参数与 `create_diarizer` 均可配）的段落标记
   `needs_review=True` 且不参与角色计分；完全没有重叠的段落走 `speaker_id='UNKNOWN'`、
   `confidence=0.5`，同样标 `needs_review=True`。对齐循环结束后，用本次调用累积的画像
   把最终角色就地回填到该说话人的全部段落上——同一次 `diarize_audio` 里同一个
   `speaker_id` 只有一个角色结论。输出段落
   `confidence = 说话人段 confidence * (0.6 + 0.4 * 最终角色置信度)`，不受阈值影响。
5. `get_speaker_summary()` 汇总每个说话人的角色、置信度、段落数、字数与
   `needs_review_segments`。`total_words` 按 CJK 字符逐字计数、其余连续非空白字符
   按空白切词计数（「本回次进尺 3.2 米」= 5 + 1 + 1 = 7）。

## 音频后端

`librosa` 与 `pyannote.audio` 都只在懒加载分支里 import，本机没有安装、也不允许安装。
所以测试一律给 `diarizer.diarization_pipeline` 注入一个替身：`load_pipeline()` 见到非 None 会直接返回它，
不会走到任何音频后端。替身要能被调用一次拿到一个对象，该对象提供 `itertracks(yield_label=True)`，
产出 `(turn, None, speaker)`，其中 `turn.start` / `turn.end` 是 float。也可以只测纯逻辑方法。

## 运行

    ~/venvs/base/bin/python -m pytest tests/ -q

解释器与依赖版本见 `requirements-task.txt`。不要 pip install，不要联网。

### 本次做题人填写

最终角色回填选了「就地改写」：对齐循环里生成的 `SpeakerSegment` 先带上说话人段的
原始 `confidence` 作为基数，循环结束后统一查一次每个 `speaker_id` 的最终角色，
再在原地改写这批对象的 `speaker_role` 与 `confidence`。

理由：对齐结果只活在这一个方法里，返回前没有任何外部代码持有这些对象的引用，
追溯改写不可观测；同时省掉一份「对齐结果缓存 + 第二遍重建列表」的中间结构，
diff 也最小。不选两遍走的原因：两遍方案要把每段的最佳说话人、重叠量、
`needs_review` 等中间结果全部缓存下来再重新生成输出列表，快照语义虽干净，
但当前调用方（`diarize_and_label`）拿到返回值后只读不写，快照语义没有实际收益，
不值得为它多维护一份中间结构。

代价：如果未来有代码在对齐循环返回前就持有了这些段落的引用（例如把
`_align_transcript_with_speakers` 拆成生成器逐步消费），它会看到对象被追溯改写；
届时应改走两遍重建的方案。
