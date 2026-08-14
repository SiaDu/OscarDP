<!--
Source: /home/sia/.codex/attachments/997142df-1638-42aa-b95c-e9eb37437dfc/pasted-text.txt
Source SHA-256: 64b5171e01e64e030617e4b12d13b49f41654d0f47c04c4f31721d6792228b87
Specification chain:
- /home/sia/.codex/attachments/0ebd8f3b-074a-49b9-b359-2a8827210100/pasted-text.txt (e3c524d90784a99dc81508a603d4bb77dedab44990bad0191fb7c782774d8bb5)
- /home/sia/.codex/attachments/3955e018-2eac-42b7-b58c-eeb3005e5191/pasted-text.txt (7cbb1660f671b4876b4eb7db906fcd8445f96737d659d3783d31cf3c940dc772)
-->

# OscarDP Stage 2 Production Completion Override

## 目标

从当前状态继续完成 OscarDP Stage 2。

本阶段已经结束 reviewer / retrieval / production framework 的开放式研发，接下来优先目标是：

```text
完成电影，而不是继续扩展框架。
```

Primary progress metric:

```text
COMPLETE_MOVIES / 8
TERMINAL_MOVIES / 8
```

代码提交、reviewer 版本、retrieval 版本、测试数量、manifest 数量都不单独视为进度。

---

# 1. 当前状态

当前状态：

```text
COMPLETE_MOVIES = 5/8
TERMINAL_MOVIES = 6/8
```

剩余状态：

```text
tt27714581
生产处理中。
已有 production Batch chunks 正在执行。

tt30343021
尚待 production spot check 和完整 production review。

tt30144839
缺少 screenplay。
保持：
BLOCKED_WITH_EXPLICIT_REASON
```

当前预计最终结果：

```text
7 COMPLETE
1 BLOCKED_WITH_EXPLICIT_REASON
```

如果之后 `tt30144839` screenplay 出现，则再按冻结生产流程补齐至：

```text
8 COMPLETE
```

---

# 2. 冻结当前生产栈

冻结以下 production stack：

```text
v3.2.1-production.3-retrieval-v3-validator-v3
```

从本 override 生效后：

```text
默认禁止继续新增：

- reviewer version
- reviewer prompt version
- retrieval version
- response schema
- validator contract
- calibration framework
- production lifecycle abstraction
- risk-audit abstraction
- evidence-correction framework
- CLI command
```

现有机制能够安全表达的问题，必须优先使用现有机制完成。

不要因为单个 production movie 出现少数 alignment error 就重新进入 reviewer research。

---

# 3. Global reviewer freeze rule

Production reviewer 已进入冻结状态。

孤立 production error 应优先使用：

```text
high-risk audit
source-evidence diagnosis
existing evidence correction
needs_review
explicit ambiguity
```

不得因为单部电影的少数错误重新设计 global reviewer。

Global reviewer / prompt / retrieval redesign 只有在以下条件全部满足时才允许重新打开：

```text
1. 相同的实质性 causal failure 出现在至少 2 部 production movies；

2. 问题是系统性的，而不是孤立 annotation error；

3. 至少一部受影响电影中：
   >= 5% 的 review-required subtitles 受该 failure class 影响，
   或问题造成 structural corruption；

4. 现有：
   audit /
   evidence correction /
   needs_review
   无法安全表达或修正该问题。
```

单部电影本身不能触发新的 global reviewer / retrieval version。

Production spot-check failure 也不能自动触发新的 global calibration cycle。

---

# 4. Parser / loader / validator 例外

不要把 reviewer freeze 错误应用到真正的结构性 bug。

如果单部电影暴露了确认的：

```text
screenplay parser structural corruption
subtitle loader corruption
```

例如：

```text
大量 action 被解析成 dialogue
speaker segmentation 被破坏
scene/block structure 大规模错误
subtitle source 被错误切分
```

则允许进行一次 generic fix，前提是：

```text
- root cause 明确；
- fix 不是 movie-specific hack；
- 添加 targeted regression test；
- 不覆盖历史 artifacts；
- 对已完成电影做必要的 regression check。
```

同样：

```text
validator correctness bug
production lifecycle correctness bug
```

可以在单个案例出现时立即修复。

不要要求这种 correctness bug 也满足“2 movies + 5%”。

---

# 5. Production spot-check evaluator 修正

这是本轮明确允许的最小 correctness fix。

当前 `production_spot_check` 与 `independent_calibration` 必须彻底区分。

## Independent calibration

继续保留：

```text
structural gate
AND
candidate_task_accuracy >= 0.90
AND
candidate_presence_decision_accuracy >= 0.90
```

用于 reviewer promotion。

---

## Production spot check

Production spot check 通常为：

```text
10–15 requests
```

不得再使用 0.90 accuracy 作为硬门槛。

Production spot-check 的 hard gate 只检查：

```text
expected request count 完整
invalid responses = 0
missing predictions = 0
foreign candidate IDs = 0
```

其中：

```text
expected_request_count
```

必须从 frozen pilot manifest / validation metadata 中读取。

禁止固定：

```text
valid_count == 30
```

---

Production spot-check 仍然报告：

```text
candidate_task_accuracy
candidate_presence_decision_accuracy
match exact-block accuracy
no_candidate_match accuracy
confusion matrix
sequence diagnostics
semantic error categories
```

但这些指标是：

```text
descriptive QA evidence
```

不是 reviewer promotion gate。

孤立语义错误不会仅因为：

```text
accuracy < 0.90
```

自动阻断 production。

是否阻断 production 由：

```text
error-class diagnosis
```

决定。

只有确认属于系统性 failure 时才阻断。

---

# 6. 本轮代码修改上限

本轮默认只允许以下 production-framework code changes：

```text
1. 修复 production spot-check evaluator。

2. 修复 terminal-state scheduling。

3. 完成上述修改所需的最小 targeted tests。

4. 若执行过程中发现真实 parser / loader / validator /
   lifecycle correctness bug，可按本 override 的例外规则修复。
```

除非存在明确 correctness blocker，否则禁止新增：

```text
new production CLI
new framework abstraction
new reviewer version
new retrieval version
new QC layer
new calibration framework
new status framework
```

在修改 source code 前必须先回答：

```text
Can this movie be completed correctly using the existing
audit / correction / needs_review mechanisms?
```

如果答案是：

```text
YES
```

则不要改代码。

---

# 7. Terminal-state scheduling

Goal continuation 必须选择：

```text
第一个 non-terminal target movie
```

而不是：

```text
第一个 non-COMPLETE movie
```

Terminal states 为：

```text
COMPLETE
BLOCKED_WITH_EXPLICIT_REASON
```

因此：

```text
tt30144839
```

在 screenplay 仍缺失时必须被跳过。

禁止每次 continuation 重新扫描、重新诊断或消耗 API 处理 `tt30144839`。

同时维护：

```text
COMPLETE_MOVIES / 8
TERMINAL_MOVIES / 8
```

---

# 8. Continuation behavior

每次 Goal continuation 开始时：

```text
1. 读取 stage2_goal_status.json。

2. 找到第一个 non-terminal movie。

3. 从该电影最新 validated artifact 继续。

4. 不重新执行已经完成且 hashes / manifests 已验证的步骤。

5. 不重新处理 COMPLETE movie。

6. 不重新处理 BLOCKED_WITH_EXPLICIT_REASON movie，
   除非 blocker 的输入状态确实发生变化。
```

不要在 continuation 开头重复：

```text
全仓库大范围 inspection
完整历史重新总结
完整 artifacts 重新 hash
完整 calibration replay
```

除非确实有新的 correctness evidence。

---

# 9. Batch 状态规则

OpenAI Batch 必须保持串行 gated execution。

每次提交前：

```text
validate exact input JSONL
validate request count
validate model
validate schema/reviewer version
validate request-specific candidate enum
record input SHA-256
```

Batch 状态若为：

```text
validating
in_progress
finalizing
```

则：

```text
- 原子记录 batch ID 和当前 status；
- 不取消；
- 不重复提交；
- 不修改冻结 input；
- 结束当前 Goal turn。
```

下一次 continuation 时最多正常查询一次状态。

不要在同一个 Goal turn 中反复 polling。

不需要为此实现新的“30 分钟 throttle framework”。

如果用户明确要求检查，可以正常查询一次。

只有：

```text
completed
```

后才能：

```text
fetch
→ hard validate
→ merge/evaluate/apply
```

如果：

```text
failed
expired
cancelled
```

则：

```text
记录 terminal failure
诊断原因
禁止 blind resubmission
```

---

# 10. Hash / immutability efficiency

继续保护：

```text
videos
subtitles
screenplay sources
shots.jsonl
deterministic Stage 2 outputs
historical pilots
historical raw Batch outputs
gold/reference
historical reviewed artifacts
```

禁止覆盖。

但不要每一轮都重新 SHA256 整个大型视频。

对于大型 media file：

第一次 inventory 记录：

```text
SHA-256
file size
mtime
```

后续优先检查：

```text
path
size
mtime
```

若 metadata 未变化，不重复读取几十 GB 文件计算 SHA。

如果 metadata 发生变化，再重新 SHA256。

对于小型关键 JSON / JSONL / manifest，可以继续直接 hash。

---

# 11. Specification revision

将本 production completion override 完整保存至：

```text
docs/stage2_production_completion_override.md
```

记录原始 authoritative specification：

```text
/home/sia/.codex/attachments/0ebd8f3b-074a-49b9-b359-2a8827210100/pasted-text.txt
```

原始 specification SHA-256：

```text
7cbb1660f671b4876b4eb7db906fcd8445f96737d659d3783d31cf3c940dc772
```

禁止修改：

```text
原始 attachment
历史 specification
历史 experiment records
```

`stage2_goal_status.json` 中只需要记录：

```text
active_spec_revision
```

指向当前 completion override。

不要在每条 experiment log 中重复复制全部 governance metadata。

---

# 12. Production execution order

## Movie 1 — tt27714581

优先完成当前正在处理的：

```text
tt27714581
```

保留已经：

```text
completed
fetched
hard validated
```

的 chunks。

禁止重新运行这些 chunks。

当前等待：

```text
chunk 4
batch_id:
batch_6a7929b50f4081908e497a883c690efd
```

如果 status 仍为：

```text
validating
in_progress
finalizing
```

记录状态并结束 turn。

只有 completed 后：

```text
fetch
→ hard validate
```

然后继续串行处理剩余：

```text
chunks 5–8
```

每个 chunk：

```text
local input validation
→ submit
→ status gate
→ fetch when completed
→ hard validation
```

---

全部 chunks 完成后：

```text
merge all validated production responses
```

目标完整 request coverage：

```text
290 requests
```

然后：

```text
version-aware merge/apply
→ new reviewed subtitle alignment
→ new reviewed shot context
```

禁止覆盖 deterministic outputs。

---

# 13. tt27714581 high-risk audit

使用现有 production risk-audit framework。

重点裁决真实可操作类别：

```text
candidate_recall_risk
reviewer_selection_risk
ambiguous_needs_review
```

不要把：

```text
diagnostic warning
sequence movement
low confidence
```

自动当作 confirmed error。

只有 source evidence 明确确认错误时，才生成：

```text
evidence correction plan
```

并写入新的 tagged output，例如：

```text
qc1
```

不要伪装成：

```text
human gold
```

Codex source-evidence correction 必须保留 provenance。

---

# 14. tt27714581 final QC

Final QC 保持现有完整性要求：

```text
0 malformed production responses
0 missing production responses
0 foreign IDs
complete lifecycle coverage
artifact hashes consistent
reviewed alignment valid
reviewed shot context valid
```

允许最多：

```text
5 isolated genuine ambiguities
```

前提是：

```text
- 明确标记 needs_review / ambiguous；
- 不是系统性 failure；
- 不阻碍 shot-context generation。
```

但：

```text
confirmed candidate recall errors = 0
confirmed reviewer selection errors = 0
```

才能 COMPLETE。

完成后：

```text
tt27714581 = COMPLETE
```

立即更新：

```text
stage2_goal_status.json
```

然后进入下一部，不重新设计框架。

---

# 15. Movie 2 — tt30343021

使用已经存在的 deterministic artifacts。

不要重新设计 reviewer。

不要重新做 global calibration。

首先冻结一个：

```text
10–15 request
production spot check
```

要求分层覆盖适当的：

```text
easy
fuzzy
multi
difficult
early
middle
late
fallback
candidate saturation
no_candidate_match
fragment / repeat / vocative
```

根据实际 request pool 选择，不强行制造不存在的类别。

---

运行冻结 production stack：

```text
v3.2.1-production.3-retrieval-v3-validator-v3
```

Production spot check hard gate：

```text
all expected requests returned
0 invalid
0 missing
0 foreign ID
```

Semantic accuracy 仅用于风险诊断。

如果出现少数：

```text
isolated semantic errors
```

优先：

```text
audit
→ source-evidence diagnosis
→ correction / needs_review
```

不要重新设计 reviewer。

只有满足 global failure reopening rule 才允许重新进入 reviewer development。

---

spot check 没有暴露系统性 blocker 后：

```text
prepare remaining/full production review
```

根据账户 token limit：

```text
serial deterministic chunks
```

逐块执行：

```text
preflight
→ submit
→ status gate
→ fetch
→ hard validation
```

随后：

```text
merge
→ apply
→ high-risk audit
→ source-evidence corrections when justified
→ final QC
```

最终：

```text
tt30343021 = COMPLETE
```

---

# 16. tt30144839

当前状态：

```text
BLOCKED_WITH_EXPLICIT_REASON
```

原因：

```text
missing screenplay
```

只要 screenplay 仍不存在：

```text
不要重复扫描
不要运行 reviewer
不要运行 OpenAI Batch
不要消耗新的 Codex work 去重新确认同一个 blocker
```

如果未来 screenplay 实际出现：

```text
重新打开该 movie
```

并使用当前冻结 production flow 处理。

---

# 17. Completed movies

当前已经 COMPLETE 的 5 部电影：

```text
不重新跑
不重新 calibration
不重新 reviewer
不重新 apply
不重新生成 reviewed outputs
```

只在最终 Goal 收敛时检查：

```text
status record
关键 manifest
protected artifact metadata/hash
```

不要重新执行完整 production pipeline。

---

# 18. Production spot-check interpretation

Production spot check 是：

```text
QA sample
```

不是：

```text
independent calibration
```

因此它不能：

```text
promote reviewer
re-promote reviewer
作为 global reviewer generalization claim
```

同样，spot-check semantic error 也不能直接触发：

```text
new reviewer version
```

判断重点是：

```text
是否存在 systematic production failure class
```

而不是：

```text
10/10
13/15
14/15
```

这种小样本百分比本身。

---

# 19. Cross-movie systematic failure

如果 production 期间怀疑需要重开 global architecture，必须先生成 cross-movie diagnosis。

至少证明：

```text
same causal class
appears in >= 2 movies
```

并且满足 materiality 条件。

在满足条件前：

```text
不得新增 reviewer/retrieval/prompt version
不得重新消耗 unused movie 做 calibration
```

---

# 20. Testing policy

本轮最小代码修复添加 targeted tests：

必须覆盖：

```text
10-request production spot check structural gate passes

15-request production spot check structural gate passes

production spot check semantic accuracy < 0.90
does NOT automatically fail

invalid response fails

missing prediction fails

foreign candidate ID fails

manifest/request-count mismatch fails

30-request independent calibration
still applies:
candidate-task >= 0.90
candidate-presence >= 0.90

scheduler skips:
COMPLETE
BLOCKED_WITH_EXPLICIT_REASON

scheduler selects only:
non-terminal movie
```

如果没有真的实现 Batch throttle code，则不要为“30 分钟 throttle”新增测试框架。

只保留现有：

```text
completed-only fetch
no blind resubmission
```

相关测试。

---

# 21. Test execution

代码修改完成后运行：

```bash
.venv/bin/python -m pytest tests/test_stage253.py
```

如果涉及 scheduling / production lifecycle 对应测试文件，也运行相关 targeted tests。

随后运行：

```bash
.venv/bin/python -m pytest
.venv/bin/python -m compileall -q src/oscardp/script_context
git diff --check
```

形成一个有意义的最小 source/tests/docs commit。

完成这个 correctness commit 后：

```text
dataset-only processing
Batch status checks
fetch
validation
audit
evidence correction
QC
```

如果没有修改 source code：

```text
不要反复运行完整 pytest。
```

Goal 最终结束前再运行一次：

```bash
.venv/bin/python -m pytest
.venv/bin/python -m compileall -q src/oscardp/script_context
git diff --check
```

---

# 22. Commit policy

本轮 production completion 阶段：

```text
减少代码 churn。
```

只 commit：

```text
source
tests
documentation
```

不要 commit：

```text
dataset artifacts
Batch payloads
raw Batch outputs
videos
screenplays
subtitles
generated reviewed movie outputs
```

如果 source 没有变化：

```text
不要为了记录 dataset progress 制造空的 code commit。
```

---

# 23. Quota-aware behavior

当前优先级是：

```text
在有限 Codex quota 内完成剩余电影。
```

避免：

```text
speculative refactor
重复读取整个 repo
重复总结 unchanged history
重复 hash 巨型 media
重新运行 immutable deterministic stages
重复 calibration
新增 framework abstraction
为了孤立 annotation 新增 generic code
```

积极复用：

```text
existing manifests
existing validated chunks
existing deterministic outputs
existing reviewer
existing risk audit
existing correction framework
existing QC framework
```

---

# 24. Progress discipline

每次 major checkpoint 内部记录：

```text
COMPLETE_MOVIES / 8
TERMINAL_MOVIES / 8
```

如果连续两个 meaningful generic code changes：

```text
都没有让任何 movie 更接近 COMPLETE
```

立即停止 framework development，并重新判断：

```text
是否可以使用现有 per-movie
audit / evidence correction / needs_review
完成任务。
```

---

# 25. COMPLETE criteria

一部电影可以标记 COMPLETE，当：

```text
source inventory complete

Stage 1 validated

screenplay parse / structural QC passes

deterministic Stage 2 valid

production reviewer coverage complete

0 malformed production responses

0 missing production responses

0 foreign candidate IDs

reviewed subtitle alignment exists

reviewed shot context exists

high-risk audit exists

confirmed candidate recall errors = 0

confirmed reviewer selection errors = 0

isolated genuine ambiguities explicitly recorded

final QC passes

reproducibility manifest exists

protected deterministic/source artifacts unchanged
```

允许少量：

```text
isolated genuine ambiguity
```

存在，但必须明确记录且不得代表系统性错误。

---

# 26. BLOCKED criteria

只有以下情况才使用：

```text
BLOCKED_WITH_EXPLICIT_REASON
```

* required source/input missing；
* source/input corrupt；
* systematic technical failure 无法解决；
* ambiguity 大范围到无法可靠完成 Stage 2；
* required evidence 无法获得。

不要因为：

```text
1–2 isolated ambiguity
```

把 production-ready movie 标记 BLOCKED。

---

# 27. Whole-goal completion

Goal 完成条件：

```text
TERMINAL_MOVIES = 8/8
```

当前预期：

```text
COMPLETE_MOVIES = 7/8
TERMINAL_MOVIES = 8/8

tt30144839 = BLOCKED_WITH_EXPLICIT_REASON
```

如果 screenplay 后续出现：

```text
COMPLETE_MOVIES = 8/8
```

---

# 28. Final validation

Goal 结束前执行一次：

```bash
.venv/bin/python -m pytest
.venv/bin/python -m compileall -q src/oscardp/script_context
git diff --check
```

验证：

```text
working tree state understood
all production statuses terminal
all COMPLETE outputs present
protected artifacts unchanged
Batch terminal states recorded
reviewed outputs versioned
QC manifests present
```

---

# 29. Final report

最终报告必须包括：

```text
final git commit

frozen production reviewer:
v3.2.1-production.3-retrieval-v3-validator-v3

reviewer / retrieval history summary

COMPLETE_MOVIES / 8
TERMINAL_MOVIES / 8

per-movie status table

reviewed alignment paths

reviewed shot-context paths

risk audit paths

QC / manifest paths

Batch IDs

model IDs

request counts

input/output file IDs

paid production run counts

remaining genuine ambiguities

explicit blocker:
tt30144839 missing screenplay

protected artifact verification

Stage 3 recommendations
```

---

# 30. Primary execution principle

From this point onward:

```text
This is a production completion run,
not an open-ended reviewer research project.
```

The default action is:

```text
finish the current movie
→ mark COMPLETE
→ move to the next non-terminal movie
```

not:

```text
discover one isolated error
→ design another framework
→ create another reviewer version
→ create another calibration cycle
```

Use the existing production infrastructure aggressively and finish the remaining movies.
